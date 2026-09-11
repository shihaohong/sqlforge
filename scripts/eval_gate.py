"""Fail if execution accuracy has regressed against a recorded baseline.

    uv run scripts/eval_gate.py --backend gateway --limit 200
    uv run scripts/eval_gate.py --backend gateway --limit 200 --update-baseline

This is the check that would have caught the worst bug in the project's
history: an unpinned vLLM resolved to an older minor, and the same quantized
artifact served 47.5% instead of 71.2% with no error anywhere. Nothing in a
unit test suite can see that - only running real questions against the real
endpoint and comparing the score to what it used to be.

A fixed prefix of the dev split is used rather than a random sample, so two
runs are comparable: the point is to detect a change in the model, the
prompt, or the serving stack, not to estimate accuracy on unseen data (the
full-set numbers in PLAN.md do that).

The tolerance exists because generation is not perfectly deterministic:
repeating this gate against an unchanged deployment moved the score by a
point (68.0% then 67.0% of 200), since vLLM's continuous batching changes how
requests are grouped and therefore the numerics. Three points absorbs that
while still catching anything that matters - the vLLM downgrade was -23.7,
and a broken prompt costs ten or more. It is a regression gate, not an
equality assertion.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from text2sql.client import make_client
from text2sql.eval import evaluate, generate_predictions, load_examples

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO_ROOT / "runs" / "eval-baseline.json"

app = typer.Typer(add_completion=False)
console = Console()


def main(
    backend: str = typer.Option("gateway", help="'gateway', 'vllm', or a base URL"),
    model: str = typer.Option("sqlforge-3b"),
    split: str = typer.Option("dev"),
    limit: int = typer.Option(200, help="fixed prefix of the split; keep it stable"),
    workers: int = typer.Option(8),
    tolerance: float = typer.Option(
        0.03, help="allowed drop in accuracy before the gate fails, in absolute points"
    ),
    baseline: str = typer.Option(str(BASELINE_PATH), help="where the recorded baseline lives"),
    update_baseline: bool = typer.Option(
        False, help="record this run as the new baseline instead of checking against it"
    ),
) -> None:
    baseline_path = Path(baseline)
    examples = load_examples(split, limit)
    client = make_client(backend, model)

    console.print(f"scoring {len(examples)} {split} examples via {backend} ...")
    preds = generate_predictions(client, examples, split, workers=workers)
    report = evaluate(examples, [sql for sql, _ in preds], split)
    latencies = [latency for _, latency in preds]
    accuracy = report.execution_accuracy

    measured = {
        "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split": split,
        "limit": limit,
        "model": model,
        "n_correct": report.n_correct,
        "n": report.n,
        "execution_accuracy": round(accuracy, 4),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3),
    }

    if update_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(measured, indent=2) + "\n")
        console.print(f"[green]baseline recorded[/green]: {accuracy:.1%} -> {baseline_path}")
        return

    if not baseline_path.exists():
        raise typer.BadParameter(
            f"no baseline at {baseline_path}; run once with --update-baseline to record one"
        )
    recorded = json.loads(baseline_path.read_text())

    if (recorded["split"], recorded["limit"]) != (split, limit):
        # Comparing 200 dev examples against a baseline of 1000 would produce
        # a number that looks like a regression and is not one.
        raise typer.BadParameter(
            f"baseline covers {recorded['limit']} {recorded['split']} examples,"
            f" this run covers {limit} {split}; scores are not comparable"
        )

    delta = accuracy - recorded["execution_accuracy"]
    table = Table(title=f"eval gate ({limit} {split} examples via {backend})")
    for column in ("", "baseline", "this run", "delta"):
        table.add_column(column, justify="right")
    table.add_row(
        "execution accuracy",
        f"{recorded['execution_accuracy']:.1%}",
        f"{accuracy:.1%}",
        f"{delta * 100:+.1f} pts",
    )
    table.add_row(
        "mean latency",
        f"{recorded['mean_latency_s']:.2f}s",
        f"{measured['mean_latency_s']:.2f}s",
        f"{measured['mean_latency_s'] - recorded['mean_latency_s']:+.2f}s",
    )
    console.print(table)
    console.print(f"baseline recorded {recorded['recorded_at']}")

    if delta < -tolerance:
        console.print(
            f"[red]REGRESSION[/red]: accuracy dropped {abs(delta) * 100:.1f} pts,"
            f" more than the {tolerance * 100:.0f}-pt tolerance."
            " Re-run locally before merging; if the change is intended and understood,"
            " record a new baseline with --update-baseline."
        )
        raise typer.Exit(1)

    console.print(f"[green]gate passed[/green] (tolerance {tolerance * 100:.0f} pts)")


if __name__ == "__main__":
    typer.run(main)
