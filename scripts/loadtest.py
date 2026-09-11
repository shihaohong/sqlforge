"""Closed-loop load test: latency and throughput vs. concurrency.

    uv run scripts/loadtest.py --target gateway --concurrency 1,2,4,8,16,32,64

Each level runs `concurrency` workers that each send one request at a time for
`--duration` seconds, after a warmup that is measured but discarded. Closed
loop (a worker waits for its response before sending the next) is the right
model for an inference service: it reports the latency a client actually sees
at a given level of in-flight demand, and throughput saturates instead of
building an unbounded queue the way an open-loop request rate would.

Run it on the serving box against localhost. Driving it over an SSH tunnel
measures the tunnel.

`--target vllm` bypasses the gateway and hits vLLM's /chat/completions with
the same prompts, which isolates the gateway's own overhead.
"""

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
SERVING_DIR = REPO_ROOT / "data" / "serving"

app = typer.Typer(add_completion=False)
# Fixed width so the table renders identically to a terminal and to a log
# file (rich falls back to 80 columns when stdout is not a tty, which
# truncates the cost column).
console = Console(width=110)


@dataclass
class Sample:
    latency_s: float
    ok: bool
    completion_tokens: int = 0
    prompt_tokens: int = 0
    status: int = 0


@dataclass
class LevelResult:
    concurrency: int
    n_requests: int
    n_errors: int
    duration_s: float
    qps: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    output_tokens_per_s: float
    mean_output_tokens: float
    usd_per_1k_queries: float | None = None
    statuses: dict[str, int] = field(default_factory=dict)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; avoids interpolating between two real requests."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(q / 100 * len(ordered))
    return ordered[min(len(ordered), max(1, rank)) - 1]


class Workload:
    """Spider dev questions, replayed in a fixed shuffled order."""

    def __init__(self, split: str, seed: int = 7):
        questions = json.loads((SERVING_DIR / f"{split}.questions.json").read_text())
        self.schemas = json.loads((SERVING_DIR / f"{split}.schemas.json").read_text())
        self.items = list(questions)
        random.Random(seed).shuffle(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def at(self, i: int) -> dict:
        return self.items[i % len(self.items)]


def gateway_request(wl: Workload, item: dict, model: str, max_tokens: int) -> tuple[str, dict]:
    return "/v1/sql", {"question": item["question"], "db_id": item["db_id"]}


def vllm_request(wl: Workload, item: dict, model: str, max_tokens: int) -> tuple[str, dict]:
    from text2sql.prompts import build_messages

    return "/chat/completions", {
        "model": model,
        "messages": build_messages(wl.schemas[item["db_id"]], item["question"]),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


TARGETS = {
    "gateway": ("http://localhost:8080", gateway_request),
    "vllm": ("http://localhost:8000/v1", vllm_request),
}


async def run_level(
    client: httpx.AsyncClient,
    wl: Workload,
    build,
    model: str,
    max_tokens: int,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
) -> LevelResult:
    samples: list[Sample] = []
    collecting = asyncio.Event()
    stop = asyncio.Event()

    async def worker(worker_id: int) -> None:
        i = worker_id
        while not stop.is_set():
            item = wl.at(i)
            i += concurrency  # workers walk disjoint slices, so no request repeats
            path, payload = build(wl, item, model, max_tokens)
            started = time.perf_counter()
            try:
                resp = await client.post(path, json=payload)
                latency = time.perf_counter() - started
                body = resp.json() if resp.status_code == 200 else {}
                usage = body.get("usage") or {}
                sample = Sample(
                    latency_s=latency,
                    ok=resp.status_code == 200,
                    completion_tokens=usage.get("completion_tokens", 0),
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    status=resp.status_code,
                )
            except httpx.HTTPError:
                sample = Sample(latency_s=time.perf_counter() - started, ok=False, status=0)
            if collecting.is_set():
                samples.append(sample)

    workers = [asyncio.create_task(worker(w)) for w in range(concurrency)]
    await asyncio.sleep(warmup_s)
    collecting.set()
    measured_start = time.perf_counter()
    await asyncio.sleep(duration_s)
    measured = time.perf_counter() - measured_start
    stop.set()
    await asyncio.gather(*workers)

    latencies = [s.latency_s * 1000 for s in samples]
    out_tokens = sum(s.completion_tokens for s in samples)
    statuses: dict[str, int] = {}
    for s in samples:
        statuses[str(s.status)] = statuses.get(str(s.status), 0) + 1
    return LevelResult(
        concurrency=concurrency,
        n_requests=len(samples),
        n_errors=sum(not s.ok for s in samples),
        duration_s=measured,
        qps=len(samples) / measured if measured else 0.0,
        mean_ms=statistics.fmean(latencies) if latencies else 0.0,
        p50_ms=percentile(latencies, 50),
        p90_ms=percentile(latencies, 90),
        p99_ms=percentile(latencies, 99),
        max_ms=max(latencies, default=0.0),
        output_tokens_per_s=out_tokens / measured if measured else 0.0,
        mean_output_tokens=out_tokens / len(samples) if samples else 0.0,
        statuses=statuses,
    )


def results_table(levels: list[LevelResult], gpu_hourly: float) -> Table:
    table = Table(title=f"latency vs concurrency (GPU at ${gpu_hourly:.2f}/hr)")
    for col in ("conc", "req", "err", "QPS", "mean", "p50", "p90", "p99", "tok/s", "$/1k"):
        table.add_column(col, justify="right")
    for r in levels:
        table.add_row(
            str(r.concurrency),
            str(r.n_requests),
            str(r.n_errors),
            f"{r.qps:.1f}",
            f"{r.mean_ms:.0f}ms",
            f"{r.p50_ms:.0f}ms",
            f"{r.p90_ms:.0f}ms",
            f"{r.p99_ms:.0f}ms",
            f"{r.output_tokens_per_s:.0f}",
            f"${r.usd_per_1k_queries:.4f}" if r.usd_per_1k_queries is not None else "-",
        )
    return table


@app.command()
def main(
    target: str = typer.Option("gateway", help="'gateway', 'vllm', or a base URL"),
    base_url: str = typer.Option("", help="override the target's default base URL"),
    model: str = typer.Option("sqlforge-3b", help="model name for direct vLLM requests"),
    concurrency: str = typer.Option("1,2,4,8,16,32,64"),
    duration: float = typer.Option(30.0, help="measured seconds per level"),
    warmup: float = typer.Option(5.0, help="discarded seconds per level"),
    max_tokens: int = typer.Option(512),
    split: str = typer.Option("dev"),
    gpu_hourly: float = typer.Option(
        0.85, help="GPU-hour price used for $/1k queries (g2-standard-8 on-demand)"
    ),
    run_name: str = typer.Option(""),
) -> None:
    """Sweep concurrency levels and report latency, throughput, and cost."""
    default_url, build = TARGETS.get(target, (target, gateway_request))
    url = base_url or default_url
    levels = [int(c) for c in concurrency.split(",") if c.strip()]
    wl = Workload(split)
    run_name = run_name or f"loadtest-{target}"

    async def sweep() -> list[LevelResult]:
        out: list[LevelResult] = []
        limit = max(levels) + 8
        async with httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(120.0, connect=10.0),
            limits=httpx.Limits(max_connections=limit, max_keepalive_connections=limit),
            headers={"Authorization": "Bearer none"},
        ) as client:
            for level in levels:
                console.print(f"[cyan]concurrency {level}[/cyan]: warming up {warmup:.0f}s ...")
                result = await run_level(
                    client, wl, build, model, max_tokens, level, duration, warmup
                )
                # Cost per query is GPU time divided across the queries that
                # time served, so higher sustained throughput is cheaper.
                result.usd_per_1k_queries = (
                    gpu_hourly / (result.qps * 3600) * 1000 if result.qps else None
                )
                out.append(result)
                console.print(
                    f"  {result.n_requests} req, {result.qps:.1f} QPS, "
                    f"p50 {result.p50_ms:.0f}ms, p99 {result.p99_ms:.0f}ms, "
                    f"{result.n_errors} errors"
                )
        return out

    results = asyncio.run(sweep())
    console.print(results_table(results, gpu_hourly))

    best = min((r for r in results if r.usd_per_1k_queries), key=lambda r: r.usd_per_1k_queries)
    console.print(
        f"cheapest: ${best.usd_per_1k_queries:.4f}/1k queries at concurrency {best.concurrency} "
        f"({best.qps:.1f} QPS, p99 {best.p99_ms:.0f}ms)"
    )

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RUNS_DIR / f"{run_name}.json"
    out_path.write_text(
        json.dumps(
            {
                "target": target,
                "base_url": url,
                "split": split,
                "duration_s": duration,
                "warmup_s": warmup,
                "max_tokens": max_tokens,
                "gpu_hourly_usd": gpu_hourly,
                "levels": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    console.print(f"saved: {out_path}")


if __name__ == "__main__":
    app()
