"""Gateway tests with a mocked vLLM upstream.

The upstream is an httpx.MockTransport, so these exercise the real FastAPI
app, the real prompt template, and the real guardrail without a GPU.
"""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from text2sql.gateway import Settings, create_app

SCHEMA = "CREATE TABLE singer (id int, name text, age int)"


def chat_response(content: str, prompt_tokens: int = 120, completion_tokens: int = 9) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def sse_stream(deltas: list[str], usage: dict | None = None) -> bytes:
    lines = []
    for delta in deltas:
        lines.append(f"data: {json.dumps({'choices': [{'delta': {'content': delta}}]})}\n\n")
    if usage:
        lines.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def make_client(handler, settings: Settings | None = None) -> TestClient:
    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://vllm.test/v1"
    )
    app = create_app(
        settings or Settings(schema_cache=Path("/nonexistent.json")), upstream=upstream
    )
    return TestClient(app)


@pytest.fixture
def completion_client():
    """Gateway whose upstream always returns a fixed, valid completion."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response("SELECT count(*) FROM singer"))

    with make_client(handler) as client:
        yield client


class TestGenerateSql:
    def test_returns_validated_sql(self, completion_client):
        resp = completion_client.post(
            "/v1/sql", json={"question": "How many singers?", "schema": SCHEMA}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sql"] == "SELECT count(*) FROM singer"
        assert body["valid"] is True
        assert body["reason"] is None
        assert body["usage"] == {"prompt_tokens": 120, "completion_tokens": 9}
        assert body["latency_ms"] > 0

    def test_sends_the_shared_prompt_template(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat_response("SELECT 1"))

        with make_client(handler) as client:
            client.post("/v1/sql", json={"question": "How many singers?", "schema": SCHEMA})

        assert seen["temperature"] == 0.0
        assert [m["role"] for m in seen["messages"]] == ["system", "user"]
        assert SCHEMA in seen["messages"][1]["content"]
        assert "How many singers?" in seen["messages"][1]["content"]

    def test_strips_markdown_fences_from_the_completion(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=chat_response("```sql\nSELECT 1 FROM singer\n```"))

        with make_client(handler) as client:
            body = client.post("/v1/sql", json={"question": "q", "schema": SCHEMA}).json()
        assert body["sql"] == "SELECT 1 FROM singer"

    def test_resolves_db_id_from_the_schema_cache(self, tmp_path):
        cache = tmp_path / "schemas.json"
        cache.write_text(json.dumps({"concert_singer": SCHEMA}))
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=chat_response("SELECT 1"))

        with make_client(handler, Settings(schema_cache=cache)) as client:
            resp = client.post("/v1/sql", json={"question": "q", "db_id": "concert_singer"})
        assert resp.status_code == 200
        assert resp.json()["db_id"] == "concert_singer"
        assert SCHEMA in seen["messages"][1]["content"]

    def test_unknown_db_id_is_404(self, completion_client):
        resp = completion_client.post("/v1/sql", json={"question": "q", "db_id": "nope"})
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": "q"},  # neither db_id nor schema
            {"schema": SCHEMA},  # no question
            {"question": "", "schema": SCHEMA},  # empty question
            {"question": "q", "schema": SCHEMA, "max_tokens": 4},  # below the floor
        ],
    )
    def test_rejects_invalid_requests(self, completion_client, payload):
        assert completion_client.post("/v1/sql", json=payload).status_code == 422


class TestGuardrail:
    def test_withholds_sql_that_fails_the_guardrail(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=chat_response("DROP TABLE singer"))

        with make_client(handler) as client:
            resp = client.post("/v1/sql", json={"question": "q", "schema": SCHEMA})

        assert resp.status_code == 200
        body = resp.json()
        assert body["sql"] is None
        assert body["valid"] is False
        assert body["rejected_sql"] == "DROP TABLE singer"
        assert "not a SELECT" in body["reason"]

    def test_counts_the_rejection_in_metrics(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=chat_response("DROP TABLE singer"))

        with make_client(handler) as client:
            client.post("/v1/sql", json={"question": "q", "schema": SCHEMA})
            metrics = client.get("/metrics").text

        assert 'sqlforge_guardrail_rejections_total{code="not_select"}' in metrics
        assert 'sqlforge_requests_total{outcome="rejected"}' in metrics


class TestUpstreamFailures:
    def test_timeout_is_504(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("too slow", request=request)

        with make_client(handler) as client:
            resp = client.post("/v1/sql", json={"question": "q", "schema": SCHEMA})
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"]

    def test_upstream_error_is_502(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="engine crashed")

        with make_client(handler) as client:
            resp = client.post("/v1/sql", json={"question": "q", "schema": SCHEMA})
        assert resp.status_code == 502


class TestStreaming:
    def test_streams_deltas_then_a_final_verdict(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["stream"] is True
            return httpx.Response(
                200,
                content=sse_stream(
                    ["SELECT ", "count(*) ", "FROM singer"],
                    usage={"prompt_tokens": 120, "completion_tokens": 9},
                ),
                headers={"content-type": "text/event-stream"},
            )

        with (
            make_client(handler) as client,
            client.stream(
                "POST", "/v1/sql/stream", json={"question": "q", "schema": SCHEMA}
            ) as resp,
        ):
            assert resp.status_code == 200
            events = [
                json.loads(line[len("data:") :])
                for line in resp.iter_lines()
                if line.startswith("data:")
            ]

        assert [e["delta"] for e in events[:-1]] == ["SELECT ", "count(*) ", "FROM singer"]
        done = events[-1]
        assert done == {
            "event": "done",
            "sql": "SELECT count(*) FROM singer",
            "valid": True,
            "reason": None,
            "rejected_sql": None,
            "usage": {"prompt_tokens": 120, "completion_tokens": 9},
            "latency_ms": pytest.approx(done["latency_ms"]),
        }

    def test_reports_upstream_failure_as_an_error_event(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="loading model")

        with (
            make_client(handler) as client,
            client.stream(
                "POST", "/v1/sql/stream", json={"question": "q", "schema": SCHEMA}
            ) as resp,
        ):
            events = [
                json.loads(line[len("data:") :])
                for line in resp.iter_lines()
                if line.startswith("data:")
            ]
        assert events[-1]["event"] == "error"


class TestProbes:
    def test_healthz_does_not_touch_the_upstream(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("liveness must not depend on the model server")

        with make_client(handler) as client:
            assert client.get("/healthz").json() == {"status": "ok"}

    def test_readyz_reports_loaded_models(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"id": "sqlforge-3b"}]})

        with make_client(handler) as client:
            body = client.get("/readyz").json()
        assert body == {"status": "ready", "models": ["sqlforge-3b"]}

    def test_readyz_is_503_while_the_model_loads(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="not up yet")

        with make_client(handler) as client:
            assert client.get("/readyz").status_code == 503
