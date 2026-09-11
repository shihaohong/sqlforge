"""FastAPI gateway in front of the vLLM server.

The gateway owns everything that is not model inference: request validation,
schema lookup, prompt construction (the same template training used), output
guardrails, timeouts, and Prometheus metrics. vLLM stays a plain
OpenAI-compatible backend that this service is the only client of.

    POST /v1/sql          question (+ db_id or schema) -> validated SQL
    POST /v1/sql/stream   the same, as server-sent events
    GET  /healthz         liveness: the process is up
    GET  /readyz          readiness: the model is loaded and answering
    GET  /metrics         Prometheus exposition
"""

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field, model_validator

from .demo import DemoAssets, demo_router
from .guardrails import check_sql
from .prompts import build_messages, extract_sql
from .security import SERVICE, RateLimiter, authenticate, client_ip

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUESTS = Counter("sqlforge_requests_total", "Gateway requests by outcome.", ["outcome"])
REJECTIONS = Counter(
    "sqlforge_guardrail_rejections_total", "Completions blocked by a guardrail.", ["code"]
)
TOKENS = Counter("sqlforge_tokens_total", "Tokens reported by the model server.", ["kind"])
LATENCY = Histogram(
    "sqlforge_request_latency_seconds",
    "End-to-end gateway latency.",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
)
UPSTREAM_LATENCY = Histogram(
    "sqlforge_upstream_latency_seconds",
    "Time spent waiting on the model server.",
    buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
)
INFLIGHT = Gauge("sqlforge_inflight_requests", "Requests currently in flight.")
# A spike here means the token leaked or someone is scraping the demo.
DENIED = Counter("sqlforge_denied_total", "Requests refused by token or rate limit.", ["reason"])


@dataclass(frozen=True)
class Settings:
    upstream_url: str = "http://localhost:8000/v1"
    model: str = "sqlforge-3b"
    timeout_s: float = 30.0
    #: Matches the eval harness, so a served query is the query we scored.
    max_tokens: int = 512
    #: db_id -> CREATE TABLE DDL, so the serving box needs no sqlite files.
    schema_cache: Path = REPO_ROOT / "data" / "serving" / "dev.schemas.json"
    #: Sample databases and questions/gold SQL behind the demo endpoints.
    databases_dir: Path = REPO_ROOT / "data" / "serving" / "databases"
    questions_cache: Path = REPO_ROOT / "data" / "serving" / "dev.questions.json"
    #: Built single-page app; served at / when the directory exists.
    static_dir: Path = REPO_ROOT / "web" / "dist"
    #: Shared secret for the demo endpoints. Empty means open, which is right
    #: for local development and never right on a public address.
    demo_token: str = ""
    #: Exempt from rate limits: the eval harness and the load generator.
    service_token: str = ""
    claude_model: str = "claude-haiku-4-5"
    #: Rate limits are per replica; see security.py.
    rate_per_minute: float = 12.0
    rate_burst: int = 6
    daily_limit: int = 2000
    #: The Claude path spends money per call, so it gets its own small budget.
    paid_rate_per_minute: float = 4.0
    paid_burst: int = 2
    paid_daily_limit: int = 200

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            upstream_url=os.environ.get("SQLFORGE_UPSTREAM", cls.upstream_url),
            model=os.environ.get("SQLFORGE_MODEL", cls.model),
            timeout_s=float(os.environ.get("SQLFORGE_TIMEOUT_S", cls.timeout_s)),
            max_tokens=int(os.environ.get("SQLFORGE_MAX_TOKENS", cls.max_tokens)),
            schema_cache=Path(os.environ.get("SQLFORGE_SCHEMA_CACHE", cls.schema_cache)),
            databases_dir=Path(os.environ.get("SQLFORGE_DATABASES_DIR", cls.databases_dir)),
            questions_cache=Path(os.environ.get("SQLFORGE_QUESTIONS_CACHE", cls.questions_cache)),
            static_dir=Path(os.environ.get("SQLFORGE_STATIC_DIR", cls.static_dir)),
            demo_token=os.environ.get("SQLFORGE_DEMO_TOKEN", cls.demo_token),
            service_token=os.environ.get("SQLFORGE_SERVICE_TOKEN", cls.service_token),
            claude_model=os.environ.get("SQLFORGE_CLAUDE_MODEL", cls.claude_model),
            rate_per_minute=float(os.environ.get("SQLFORGE_RATE_PER_MINUTE", cls.rate_per_minute)),
            rate_burst=int(os.environ.get("SQLFORGE_RATE_BURST", cls.rate_burst)),
            daily_limit=int(os.environ.get("SQLFORGE_DAILY_LIMIT", cls.daily_limit)),
            paid_rate_per_minute=float(
                os.environ.get("SQLFORGE_PAID_RATE_PER_MINUTE", cls.paid_rate_per_minute)
            ),
            paid_burst=int(os.environ.get("SQLFORGE_PAID_BURST", cls.paid_burst)),
            paid_daily_limit=int(os.environ.get("SQLFORGE_PAID_DAILY_LIMIT", cls.paid_daily_limit)),
        )


class SchemaRegistry:
    """Resolve a db_id to its CREATE TABLE DDL.

    Prefers a prebuilt JSON cache (`scripts/build_schema_cache.py`); falls back
    to reading the sqlite file directly when the dataset is on the box, which
    is the case in development but not in production.
    """

    def __init__(self, cache_path: Path):
        self._cache: dict[str, str] = {}
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text())

    def get(self, db_id: str) -> str:
        if db_id in self._cache:
            return self._cache[db_id]
        try:
            from .data import schema_ddl

            ddl = schema_ddl(db_id)
        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(404, f"unknown db_id {db_id!r}") from e
        self._cache[db_id] = ddl
        return ddl


class SqlRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    #: Either a known db_id (schema resolved server-side) or the schema itself.
    db_id: str | None = None
    schema_ddl: str | None = Field(default=None, alias="schema", max_length=100_000)
    max_tokens: int | None = Field(default=None, ge=16, le=2048)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _need_a_schema(self) -> "SqlRequest":
        if not self.db_id and not self.schema_ddl:
            raise ValueError("provide either 'db_id' or 'schema'")
        return self


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SqlResponse(BaseModel):
    """A rejected completion never appears in `sql`, only in `rejected_sql`.

    That way a caller can execute `sql` whenever it is non-null without
    re-running the guardrail itself.
    """

    sql: str | None
    valid: bool
    db_id: str | None = None
    reason: str | None = None
    rejected_sql: str | None = None
    usage: Usage = Usage()
    latency_ms: float = 0.0
    upstream_ms: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    app.state.schemas = SchemaRegistry(settings.schema_cache)
    if not hasattr(app.state, "upstream"):
        app.state.upstream = httpx.AsyncClient(
            base_url=settings.upstream_url.rstrip("/"),
            timeout=httpx.Timeout(settings.timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=512),
        )
    try:
        yield
    finally:
        await app.state.upstream.aclose()


def create_app(settings: Settings | None = None, upstream: httpx.AsyncClient | None = None):
    app = FastAPI(title="SQLForge gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings or Settings.from_env()
    settings = app.state.settings
    if upstream is not None:  # tests inject a mock transport
        app.state.upstream = upstream

    limiter = RateLimiter(
        rate_per_minute=settings.rate_per_minute,
        burst=settings.rate_burst,
        daily_limit=settings.daily_limit,
        name="requests",
    )
    paid_limiter = RateLimiter(
        rate_per_minute=settings.paid_rate_per_minute,
        burst=settings.paid_burst,
        daily_limit=settings.paid_daily_limit,
        name="comparisons",
    )

    def guard(request: Request, paid: bool = False) -> str:
        """Authenticate and charge rate limits. Every inference route calls it.

        Inference costs GPU seconds and the comparison costs money, so the
        public routes are all metered; a service token skips the meter.
        """
        try:
            principal = authenticate(request, settings.demo_token, settings.service_token)
        except HTTPException:
            DENIED.labels("unauthorized").inc()
            raise
        if principal == SERVICE:
            return principal
        key = client_ip(request)
        try:
            limiter.check(key)
            if paid:
                paid_limiter.check(key)
        except HTTPException:
            DENIED.labels("paid_rate_limit" if paid else "rate_limit").inc()
            raise
        return principal

    def claude_factory():
        """The Claude client, or None when this deployment cannot authenticate.

        Constructing the SDK succeeds with no credentials at all - it only
        fails when a request is made - so availability is established with
        `models.list()`, a metadata call that costs nothing and generates no
        tokens. The answer is cached: it will not change while the process
        lives, and the demo page asks on every load.
        """
        import anthropic

        if not hasattr(app.state, "claude"):
            # An empty value is not a credential, and leaving it set makes the
            # SDK refuse to look anywhere else. Kubernetes hands over an empty
            # string when the Secret key is blank, which is how "no key
            # configured" reaches this process.
            if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
                os.environ.pop("ANTHROPIC_API_KEY", None)

            from .client import AnthropicClient

            # Construction and the probe share one handler: depending on the
            # SDK version a missing credential surfaces at either point.
            try:
                client = AnthropicClient(model=settings.claude_model)
                client._client.models.list(limit=1)
            # TypeError, and not only AnthropicError: with no resolvable
            # credential the SDK raises a plain TypeError from header
            # validation, which is not part of its error hierarchy.
            except (anthropic.AnthropicError, TypeError) as e:
                print(
                    f"claude comparison disabled ({type(e).__name__}):"
                    " set ANTHROPIC_API_KEY to enable it"
                )
                client = None
            app.state.claude = client
        return app.state.claude

    def resolve_schema(app: FastAPI, req: SqlRequest) -> str:
        return req.schema_ddl if req.schema_ddl else app.state.schemas.get(req.db_id)

    def payload(app: FastAPI, req: SqlRequest, schema: str, stream: bool) -> dict:
        settings: Settings = app.state.settings
        return {
            "model": settings.model,
            "messages": build_messages(schema, req.question),
            "max_tokens": req.max_tokens or settings.max_tokens,
            "temperature": 0.0,
            "stream": stream,
            **({"stream_options": {"include_usage": True}} if stream else {}),
        }

    @app.post("/v1/sql", response_model=SqlResponse)
    async def generate_sql(req: SqlRequest, request: Request) -> SqlResponse:
        guard(request)
        started = time.perf_counter()
        schema = resolve_schema(request.app, req)
        with INFLIGHT.track_inprogress():
            upstream_started = time.perf_counter()
            try:
                resp = await request.app.state.upstream.post(
                    "/chat/completions", json=payload(request.app, req, schema, stream=False)
                )
                resp.raise_for_status()
            except httpx.TimeoutException as e:
                REQUESTS.labels("timeout").inc()
                raise HTTPException(504, "model server timed out") from e
            except httpx.HTTPError as e:
                REQUESTS.labels("upstream_error").inc()
                raise HTTPException(502, f"model server error: {e}") from e
            upstream_ms = (time.perf_counter() - upstream_started) * 1000

        body = resp.json()
        completion = body["choices"][0]["message"]["content"]
        usage = Usage(
            **{k: v for k, v in (body.get("usage") or {}).items() if k in Usage.model_fields}
        )
        TOKENS.labels("prompt").inc(usage.prompt_tokens)
        TOKENS.labels("completion").inc(usage.completion_tokens)

        sql = extract_sql(completion)
        verdict = check_sql(sql)
        REQUESTS.labels("ok" if verdict.ok else "rejected").inc()
        if not verdict.ok:
            REJECTIONS.labels(verdict.code).inc()
        elapsed = time.perf_counter() - started
        LATENCY.observe(elapsed)
        UPSTREAM_LATENCY.observe(upstream_ms / 1000)
        return SqlResponse(
            sql=sql if verdict.ok else None,
            valid=verdict.ok,
            db_id=req.db_id,
            reason=verdict.reason,
            rejected_sql=None if verdict.ok else sql,
            usage=usage,
            latency_ms=elapsed * 1000,
            upstream_ms=upstream_ms,
        )

    @app.post("/v1/sql/stream")
    async def generate_sql_stream(req: SqlRequest, request: Request) -> StreamingResponse:
        """Stream tokens as they are produced, then a final verdict event.

        Guardrails are inherently post-hoc when streaming: a client that acts
        on deltas must wait for the `done` event before trusting the query,
        which is what `valid` in that event is for.
        """
        guard(request)
        schema = resolve_schema(request.app, req)

        async def events() -> AsyncIterator[str]:
            started = time.perf_counter()
            chunks: list[str] = []
            usage = Usage()
            with INFLIGHT.track_inprogress():
                try:
                    async with request.app.state.upstream.stream(
                        "POST",
                        "/chat/completions",
                        json=payload(request.app, req, schema, stream=True),
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[len("data:") :].strip()
                            if data == "[DONE]":
                                break
                            event = json.loads(data)
                            if event.get("usage"):
                                usage = Usage(
                                    **{
                                        k: v
                                        for k, v in event["usage"].items()
                                        if k in Usage.model_fields
                                    }
                                )
                            for choice in event.get("choices") or []:
                                delta = (choice.get("delta") or {}).get("content")
                                if delta:
                                    chunks.append(delta)
                                    yield _sse({"delta": delta})
                except httpx.TimeoutException:
                    REQUESTS.labels("timeout").inc()
                    yield _sse({"event": "error", "reason": "model server timed out"})
                    return
                except httpx.HTTPError as e:
                    REQUESTS.labels("upstream_error").inc()
                    yield _sse({"event": "error", "reason": f"model server error: {e}"})
                    return

            TOKENS.labels("prompt").inc(usage.prompt_tokens)
            TOKENS.labels("completion").inc(usage.completion_tokens)
            sql = extract_sql("".join(chunks))
            verdict = check_sql(sql)
            REQUESTS.labels("ok" if verdict.ok else "rejected").inc()
            if not verdict.ok:
                REJECTIONS.labels(verdict.code).inc()
            elapsed = time.perf_counter() - started
            LATENCY.observe(elapsed)
            yield _sse(
                {
                    "event": "done",
                    "sql": sql if verdict.ok else None,
                    "valid": verdict.ok,
                    "reason": verdict.reason,
                    "rejected_sql": None if verdict.ok else sql,
                    "usage": usage.model_dump(),
                    "latency_ms": elapsed * 1000,
                }
            )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> dict:
        """Ready only when the model server answers: weights are loaded."""
        try:
            resp = await request.app.state.upstream.get("/models", timeout=5.0)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(503, f"model server not ready: {e}") from e
        return {"status": "ready", "models": [m["id"] for m in resp.json().get("data", [])]}

    app.include_router(
        demo_router(
            DemoAssets(
                databases_dir=settings.databases_dir, questions_path=settings.questions_cache
            ),
            SchemaRegistry(settings.schema_cache),
            claude_factory,
            guard,
        )
    )

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Mounted last so it cannot shadow an API route, and only when a build
    # exists: the API is useful on its own and must not depend on `npm run
    # build` having been run.
    if settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="web")

    return app


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"
