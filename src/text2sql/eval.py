"""Execution-accuracy evaluation over a Spider split."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .client import ChatClient
from .data import Example, db_path, load_split, schema_ddl
from .execution import execute_sql, results_match
from .prompts import build_messages, extract_sql


@dataclass
class ExampleResult:
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    correct: bool
    pred_error: str | None
    latency_s: float | None = None


@dataclass
class EvalReport:
    split: str
    n: int
    n_correct: int
    n_exec_error: int
    execution_accuracy: float
    results: list[ExampleResult]

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("results")
        return d


def generate_predictions(
    client: ChatClient,
    examples: list[Example],
    split: str,
    workers: int = 4,
    on_progress=None,
) -> list[tuple[str, float]]:
    """Return (pred_sql, latency_s) per example, preserving order."""
    schemas = {ex.db_id: schema_ddl(ex.db_id, split=split) for ex in examples}

    def one(ex: Example) -> tuple[str, float]:
        start = time.perf_counter()
        completion = client.complete(build_messages(schemas[ex.db_id], ex.question))
        latency = time.perf_counter() - start
        if on_progress:
            on_progress()
        return extract_sql(completion), latency

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, examples))


def evaluate(
    examples: list[Example],
    predictions: list[str],
    split: str,
    latencies: list[float] | None = None,
) -> EvalReport:
    results = []
    for i, (ex, pred_sql) in enumerate(zip(examples, predictions, strict=True)):
        db = db_path(ex.db_id, split=split)
        gold_res = execute_sql(db, ex.gold_sql)
        if not gold_res.ok:
            raise RuntimeError(f"Gold SQL failed on {ex.db_id}: {gold_res.error}\n{ex.gold_sql}")
        pred_res = execute_sql(db, pred_sql)
        results.append(
            ExampleResult(
                db_id=ex.db_id,
                question=ex.question,
                gold_sql=ex.gold_sql,
                pred_sql=pred_sql,
                correct=results_match(pred_res, gold_res, ex.gold_sql),
                pred_error=pred_res.error,
                latency_s=latencies[i] if latencies else None,
            )
        )
    n_correct = sum(r.correct for r in results)
    return EvalReport(
        split=split,
        n=len(results),
        n_correct=n_correct,
        n_exec_error=sum(r.pred_error is not None for r in results),
        execution_accuracy=n_correct / len(results) if results else 0.0,
        results=results,
    )


def save_report(report: EvalReport, out_dir: Path, run_name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_name}.summary.json").write_text(json.dumps(report.summary(), indent=2))
    with (out_dir / f"{run_name}.jsonl").open("w") as f:
        for r in report.results:
            f.write(json.dumps(asdict(r)) + "\n")
    return out_dir / f"{run_name}.summary.json"


def load_examples(split: str, limit: int | None = None) -> list[Example]:
    examples = load_split(split)
    return examples[:limit] if limit else examples
