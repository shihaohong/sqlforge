"""Endpoints the demo page needs on top of the serving API.

    GET  /v1/demo/schemas   databases on offer, with their DDL and sample questions
    POST /v1/demo/execute   run guardrailed SQL against a bundled database
    POST /v1/demo/compare   answer the same question with Claude, for the side by side

The local model is not called from here: the page streams it through the
gateway's existing /v1/sql/stream, which is the point of the demo - that path
is the production path.

Everything executed here goes through the same guardrail as the serving API,
against read-only connections to the bundled sample databases, with a row cap
and a short statement timeout. The demo cannot reach a database that is not
bundled, so there is nothing to traverse to.
"""

import json
import random
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import anthropic
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .execution import PreviewResult, execute_preview, execute_sql, results_match
from .guardrails import check_sql
from .prompts import build_messages, extract_sql

MAX_PREVIEW_ROWS = 50
EXEC_TIMEOUT_S = 5.0
SAMPLE_QUESTIONS = 4


@dataclass
class DemoAssets:
    """The bundled databases and the questions/gold SQL that go with them."""

    databases_dir: Path
    questions_path: Path

    @cached_property
    def questions(self) -> list[dict]:
        if not self.questions_path.exists():
            return []
        return json.loads(self.questions_path.read_text())

    @cached_property
    def available(self) -> set[str]:
        if not self.databases_dir.exists():
            return set()
        return {p.stem for p in self.databases_dir.glob("*.sqlite")}

    def db_path(self, db_id: str) -> Path:
        if db_id not in self.available:
            raise HTTPException(404, f"no sample database bundled for {db_id!r}")
        return self.databases_dir / f"{db_id}.sqlite"

    def gold_sql(self, db_id: str, question: str) -> str | None:
        """The gold query for an exact question match, if we shipped one."""
        for row in self.questions:
            if row["db_id"] == db_id and row["question"] == question:
                return row.get("gold_sql")
        return None

    def samples(self, db_id: str, n: int = SAMPLE_QUESTIONS) -> list[str]:
        pool = [r["question"] for r in self.questions if r["db_id"] == db_id]
        # Seeded per database so the page shows the same suggestions on every
        # load - a demo that reshuffles under you is disorienting.
        return random.Random(db_id).sample(pool, min(n, len(pool)))


class ExecuteRequest(BaseModel):
    db_id: str
    sql: str = Field(min_length=1, max_length=20_000)
    #: Supplying the question lets the response say whether the rows match
    #: the gold query's, which is what makes the demo persuasive.
    question: str | None = None


class TablePreview(BaseModel):
    columns: list[str] = []
    rows: list[list] = []
    row_count: int = 0
    truncated: bool = False
    error: str | None = None


class ExecuteResponse(BaseModel):
    ok: bool
    preview: TablePreview
    gold_sql: str | None = None
    matches_gold: bool | None = None
    rejected: str | None = None


class CompareRequest(BaseModel):
    db_id: str
    question: str = Field(min_length=1, max_length=4000)


class CompareResponse(BaseModel):
    model: str
    sql: str | None
    valid: bool
    reason: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    usd_per_query: float | None = None
    usd_per_1k_queries: float | None = None


def _preview_of(result: PreviewResult) -> TablePreview:
    if not result.ok:
        return TablePreview(error=result.error)
    return TablePreview(
        columns=result.columns,
        rows=[list(r) for r in result.rows],
        row_count=len(result.rows),
        truncated=result.truncated,
    )


def demo_router(assets: DemoAssets, schemas, claude_factory, guard) -> APIRouter:
    """Build the demo routes.

    `guard` is called first on every route: it applies the shared token and
    the rate limits, so nothing here is reachable unauthenticated.
    `claude_factory` returns a client or None when no API key is configured,
    which keeps the comparison an optional feature rather than a hard
    dependency of the page.
    """
    router = APIRouter(prefix="/v1/demo")

    @router.get("/schemas")
    async def list_schemas(request: Request) -> dict:
        guard(request, paid=False)
        databases = [
            {
                "db_id": db_id,
                "schema": schemas.get(db_id),
                "sample_questions": assets.samples(db_id),
            }
            for db_id in sorted(assets.available)
        ]
        return {
            "databases": databases,
            "comparison_available": claude_factory() is not None,
        }

    @router.post("/execute", response_model=ExecuteResponse)
    async def execute(req: ExecuteRequest, request: Request) -> ExecuteResponse:
        """Execute SQL the model produced, and say whether it was right."""
        guard(request, paid=False)
        db = assets.db_path(req.db_id)

        # The SQL arrives from a client, so it is re-checked here rather than
        # trusted because /v1/sql already checked it: this endpoint is
        # reachable on its own.
        verdict = check_sql(req.sql)
        if not verdict.ok:
            return ExecuteResponse(
                ok=False, preview=TablePreview(error=verdict.reason), rejected=verdict.code
            )

        result = await run_in_threadpool(
            execute_preview, db, req.sql, MAX_PREVIEW_ROWS, EXEC_TIMEOUT_S
        )
        gold = assets.gold_sql(req.db_id, req.question) if req.question else None

        matches = None
        if gold and result.ok:
            matches = await run_in_threadpool(_matches_gold, db, req.sql, gold)
        return ExecuteResponse(
            ok=result.ok, preview=_preview_of(result), gold_sql=gold, matches_gold=matches
        )

    @router.post("/compare", response_model=CompareResponse)
    async def compare(req: CompareRequest, request: Request) -> CompareResponse:
        """Answer the same question with Claude, for the side-by-side."""
        guard(request, paid=True)
        client = claude_factory()
        if client is None:
            raise HTTPException(
                503, "comparison unavailable: no Anthropic API key configured on this deployment"
            )
        schema = schemas.get(req.db_id)
        if schema is None:
            raise HTTPException(404, f"unknown db_id {req.db_id!r}")

        started = time.perf_counter()
        try:
            text, usage = await run_in_threadpool(
                client.complete_with_usage, build_messages(schema, req.question)
            )
        except anthropic.AnthropicError as e:
            # The paid path failing must not take the page down: the local
            # model's answer is still worth showing on its own.
            raise HTTPException(503, f"comparison failed: {type(e).__name__}") from e
        latency_ms = (time.perf_counter() - started) * 1000

        sql = extract_sql(text)
        verdict = check_sql(sql)
        cost = client.cost_for(usage["input_tokens"], usage["output_tokens"])
        return CompareResponse(
            model=client.model,
            sql=sql if verdict.ok else None,
            valid=verdict.ok,
            reason=verdict.reason,
            latency_ms=latency_ms,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            usd_per_query=cost,
            usd_per_1k_queries=cost * 1000 if cost is not None else None,
        )

    return router


def _matches_gold(db: Path, pred_sql: str, gold_sql: str) -> bool:
    """Compare full result sets, the same way the eval harness scores."""
    pred = execute_sql(db, pred_sql, timeout_s=EXEC_TIMEOUT_S)
    gold = execute_sql(db, gold_sql, timeout_s=EXEC_TIMEOUT_S)
    return results_match(pred, gold, gold_sql)
