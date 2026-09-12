"""The demo endpoints, against a miniature serving bundle.

The bundle is built by `tests/conftest.py` rather than taken from the Spider
download, so these run everywhere - including CI, where the real data is
absent and these tests used to skip in silence.

The endpoints exercised here are the ones that execute model-written SQL
against a database, which is the part of the demo that most needs to be
wrong-proof: the guardrail re-check, the row cap, the statement timeout, and
the comparison against a gold query.
"""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from text2sql.gateway import Settings, create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATABASES = REPO_ROOT / "data" / "serving" / "databases"

DB_ID = "mini"


@pytest.fixture
def make_client(bundle):
    """Build a gateway wired to the miniature bundle, with a stubbed vLLM."""

    def factory(**overrides) -> TestClient:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "SELECT 1"}}]})

        settings = Settings(
            schema_cache=bundle / "dev.schemas.json",
            databases_dir=bundle / "databases",
            questions_cache=bundle / "dev.questions.json",
            static_dir=Path("/nonexistent"),
            **overrides,
        )
        upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://vllm.test/v1"
        )
        return TestClient(create_app(settings, upstream=upstream))

    return factory


@pytest.fixture
def client(make_client):
    with make_client() as c:
        yield c


@pytest.fixture
def example(bundle) -> dict:
    """A (question, gold_sql) pair from the bundle."""
    return json.loads((bundle / "dev.questions.json").read_text())[0]


class TestSchemas:
    def test_lists_the_bundled_databases(self, client):
        body = client.get("/v1/demo/schemas").json()
        assert [d["db_id"] for d in body["databases"]] == [DB_ID]

    def test_each_database_carries_ddl_and_sample_questions(self, client):
        body = client.get("/v1/demo/schemas").json()
        entry = body["databases"][0]
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
            json={"db_id": DB_ID, "sql": "SELECT name, age FROM singer LIMIT 2"},
        )
        body = resp.json()
        assert body["ok"] is True
        assert body["preview"]["columns"] == ["name", "age"]
        assert body["preview"]["row_count"] == 2

    def test_verdict_is_true_when_the_query_matches_gold(self, client, example):
        resp = client.post(
            "/v1/demo/execute",
            json={
                "db_id": DB_ID,
                "sql": example["gold_sql"],
                "question": example["question"],
            },
        )
        body = resp.json()
        assert body["matches_gold"] is True
        assert body["gold_sql"] == example["gold_sql"]

    def test_verdict_is_false_when_the_rows_differ(self, client, example):
        resp = client.post(
            "/v1/demo/execute",
            json={
                "db_id": DB_ID,
                "sql": "SELECT 42",
                "question": example["question"],
            },
        )
        assert resp.json()["matches_gold"] is False

    def test_refuses_sql_the_guardrail_rejects(self, client):
        resp = client.post(
            "/v1/demo/execute", json={"db_id": DB_ID, "sql": "DROP TABLE singer"}
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
            json={"db_id": DB_ID, "sql": "SELECT * FROM singer, concert"},
        )
        preview = resp.json()["preview"]
        assert preview["row_count"] == 50
        assert preview["truncated"] is True

    def test_reports_a_sql_error_without_failing_the_request(self, client):
        resp = client.post(
            "/v1/demo/execute",
            json={"db_id": DB_ID, "sql": "SELECT nope FROM singer"},
        )
        assert resp.status_code == 200
        assert resp.json()["ok"] is False
        assert "no such column" in resp.json()["preview"]["error"]

    def test_unbundled_database_is_404(self, client):
        resp = client.post("/v1/demo/execute", json={"db_id": "not_bundled", "sql": "SELECT 1"})
        assert resp.status_code == 404


class TestTokenAndLimits:
    def test_demo_endpoints_require_the_token_when_configured(self, make_client):
        with make_client(demo_token="s3cret") as client:
            assert client.get("/v1/demo/schemas").status_code == 401
            ok = client.get("/v1/demo/schemas", headers={"X-Demo-Token": "s3cret"})
            assert ok.status_code == 200

    def test_inference_endpoints_require_the_token_too(self, make_client):
        with make_client(demo_token="s3cret") as client:
            resp = client.post("/v1/sql", json={"question": "q", "schema": "CREATE TABLE t (a)"})
            assert resp.status_code == 401

    def test_service_token_is_exempt_from_rate_limits(self, make_client):
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

    def test_demo_token_is_rate_limited(self, make_client):
        with make_client(
            demo_token="demo", service_token="svc", rate_per_minute=1, rate_burst=2
        ) as client:
            headers = {"X-Demo-Token": "demo"}
            payload = {"question": "q", "schema": "CREATE TABLE t (a)"}
            assert client.post("/v1/sql", json=payload, headers=headers).status_code == 200
            assert client.post("/v1/sql", json=payload, headers=headers).status_code == 200
            assert client.post("/v1/sql", json=payload, headers=headers).status_code == 429

    def test_probes_and_metrics_stay_open(self, make_client):
        with make_client(demo_token="s3cret") as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/metrics").status_code == 200


class TestCompare:
    def test_missing_credentials_disable_the_comparison_without_a_500(self, monkeypatch, make_client):
        """The SDK raises TypeError - not AnthropicError - with no credential."""

        class Unauthenticated:
            def __init__(self, model: str):
                raise TypeError("Could not resolve authentication method")

        monkeypatch.setattr("text2sql.client.AnthropicClient", Unauthenticated)
        with make_client() as client:
            body = client.get("/v1/demo/schemas")
            assert body.status_code == 200
            assert body.json()["comparison_available"] is False

            resp = client.post(
                "/v1/demo/compare",
                json={"db_id": DB_ID, "question": "how many singers?"},
            )
            assert resp.status_code == 503

    def test_reports_503_when_no_credential_is_configured(self, client, monkeypatch):
        # The page must degrade to the local model rather than break.
        client.app.state.claude = None
        resp = client.post(
            "/v1/demo/compare", json={"db_id": DB_ID, "question": "how many singers?"}
        )
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"]


@pytest.mark.skipif(
    not REAL_DATABASES.is_dir(),
    reason="run scripts/build_serving_assets.py to render the real Spider bundle",
)
class TestRealBundle:
    """Claims about the shipped Spider bundle rather than about the code.

    These genuinely depend on the dataset, so they skip without it - but they
    are now the only tests in this file that do.
    """

    def test_bundles_the_small_databases_and_excludes_the_large_one(self):
        bundled = {p.stem for p in REAL_DATABASES.glob("*.sqlite")}
        assert "concert_singer" in bundled
        # 105MB on its own, and deliberately excluded by the size limit.
        assert "wta_1" not in bundled
        assert len(bundled) == 19

    def test_the_bundle_is_small_enough_to_ship_in_an_image(self):
        total = sum(p.stat().st_size for p in REAL_DATABASES.glob("*.sqlite"))
        assert total < 5 * 1024 * 1024
