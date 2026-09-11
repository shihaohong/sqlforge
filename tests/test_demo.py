"""The demo endpoints, against the bundled sample databases.

These run the real sqlite files that ship in the image, so they also assert
that the bundle exists and that gold comparison works - the two things that
make the demo's correctness claim meaningful rather than decorative.
"""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from text2sql.gateway import Settings, create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
DATABASES = REPO_ROOT / "data" / "serving" / "databases"
QUESTIONS = REPO_ROOT / "data" / "serving" / "dev.questions.json"

pytestmark = pytest.mark.skipif(
    not DATABASES.is_dir() or not QUESTIONS.exists(),
    reason="run scripts/build_serving_assets.py to render the demo bundle",
)


def make_client(**overrides) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "SELECT 1"}}]})

    settings = Settings(
        schema_cache=REPO_ROOT / "data" / "serving" / "dev.schemas.json",
        databases_dir=DATABASES,
        questions_cache=QUESTIONS,
        static_dir=Path("/nonexistent"),
        **overrides,
    )
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm.test/v1"
    )
    return TestClient(create_app(settings, upstream=upstream))


@pytest.fixture
def client():
    with make_client() as c:
        yield c


def a_question(db_id: str) -> dict:
    """A real (question, gold_sql) pair for a bundled database."""
    rows = json.loads(QUESTIONS.read_text())
    return next(r for r in rows if r["db_id"] == db_id)


class TestSchemas:
    def test_lists_only_bundled_databases(self, client):
        body = client.get("/v1/demo/schemas").json()
        db_ids = [d["db_id"] for d in body["databases"]]
        assert "concert_singer" in db_ids
        # 105MB and deliberately excluded, so the demo must not offer it.
        assert "wta_1" not in db_ids

    def test_each_database_carries_ddl_and_sample_questions(self, client):
        body = client.get("/v1/demo/schemas").json()
        entry = next(d for d in body["databases"] if d["db_id"] == "concert_singer")
        assert "CREATE TABLE" in entry["schema"]
        assert len(entry["sample_questions"]) == 4

    def test_sample_questions_are_stable_across_calls(self, client):
        first = client.get("/v1/demo/schemas").json()["databases"]
        second = client.get("/v1/demo/schemas").json()["databases"]
        assert first == second


class TestExecute:
    def test_runs_a_select_and_returns_rows(self, client):
        resp = client.post(
            "/v1/demo/execute",
            json={"db_id": "concert_singer", "sql": "SELECT name, age FROM singer LIMIT 2"},
        )
        body = resp.json()
        assert body["ok"] is True
        assert body["preview"]["columns"] == ["Name", "Age"]
        assert body["preview"]["row_count"] == 2

    def test_verdict_is_true_when_the_query_matches_gold(self, client):
        example = a_question("concert_singer")
        resp = client.post(
            "/v1/demo/execute",
            json={
                "db_id": "concert_singer",
                "sql": example["gold_sql"],
                "question": example["question"],
            },
        )
        body = resp.json()
        assert body["matches_gold"] is True
        assert body["gold_sql"] == example["gold_sql"]

    def test_verdict_is_false_when_the_rows_differ(self, client):
        example = a_question("concert_singer")
        resp = client.post(
            "/v1/demo/execute",
            json={
                "db_id": "concert_singer",
                "sql": "SELECT 42",
                "question": example["question"],
            },
        )
        assert resp.json()["matches_gold"] is False

    def test_refuses_sql_the_guardrail_rejects(self, client):
        resp = client.post(
            "/v1/demo/execute", json={"db_id": "concert_singer", "sql": "DROP TABLE singer"}
        )
        body = resp.json()
        assert resp.status_code == 200
        assert body["ok"] is False
        assert body["rejected"] == "not_select"
        assert body["preview"]["rows"] == []

    def test_caps_returned_rows(self, client):
        # A cross join would otherwise return far more rows than the cap.
        resp = client.post(
            "/v1/demo/execute",
            json={"db_id": "concert_singer", "sql": "SELECT * FROM singer, concert, stadium"},
        )
        preview = resp.json()["preview"]
        assert preview["row_count"] == 50
        assert preview["truncated"] is True

    def test_reports_a_sql_error_without_failing_the_request(self, client):
        resp = client.post(
            "/v1/demo/execute",
            json={"db_id": "concert_singer", "sql": "SELECT nope FROM singer"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is False
        assert "no such column" in resp.json()["preview"]["error"]

    def test_unbundled_database_is_404(self, client):
        resp = client.post("/v1/demo/execute", json={"db_id": "wta_1", "sql": "SELECT 1"})
        assert resp.status_code == 404


class TestTokenAndLimits:
    def test_demo_endpoints_require_the_token_when_configured(self):
        with make_client(demo_token="s3cret") as client:
            assert client.get("/v1/demo/schemas").status_code == 401
            ok = client.get("/v1/demo/schemas", headers={"X-Demo-Token": "s3cret"})
            assert ok.status_code == 200

    def test_inference_endpoints_require_the_token_too(self):
        with make_client(demo_token="s3cret") as client:
            resp = client.post("/v1/sql", json={"question": "q", "schema": "CREATE TABLE t (a)"})
            assert resp.status_code == 401

    def test_service_token_is_exempt_from_rate_limits(self):
        with make_client(
            demo_token="demo", service_token="svc", rate_per_minute=1, rate_burst=1
        ) as client:
            for _ in range(5):
                resp = client.post(
                    "/v1/sql",
                    json={"question": "q", "schema": "CREATE TABLE t (a)"},
                    headers={"X-Demo-Token": "svc"},
                )
                assert resp.status_code == 200

    def test_demo_token_is_rate_limited(self):
        with make_client(
            demo_token="demo", service_token="svc", rate_per_minute=1, rate_burst=2
        ) as client:
            headers = {"X-Demo-Token": "demo"}
            payload = {"question": "q", "schema": "CREATE TABLE t (a)"}
            assert client.post("/v1/sql", json=payload, headers=headers).status_code == 200
            assert client.post("/v1/sql", json=payload, headers=headers).status_code == 200
            assert client.post("/v1/sql", json=payload, headers=headers).status_code == 429

    def test_probes_and_metrics_stay_open(self):
        with make_client(demo_token="s3cret") as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/metrics").status_code == 200


class TestCompare:
    def test_reports_503_when_no_credential_is_configured(self, client, monkeypatch):
        # The page must degrade to the local model rather than break.
        client.app.state.claude = None
        resp = client.post(
            "/v1/demo/compare", json={"db_id": "concert_singer", "question": "how many singers?"}
        )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"]
