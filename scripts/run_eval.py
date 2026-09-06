"""Evaluate a model (or sanity-check the harness) on Spider execution accuracy.

Examples:
    uv run scripts/run_eval.py sanity --split dev
    uv run scripts/run_eval.py model --backend ollama --model llama3.2:3b --split dev --limit 100
    uv run scripts/run_eval.py model --backend https://api.example.com/v1 --model some-model
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from text2sql.client import make_client
from text2sql.eval import evaluate, generate_predictions, load_examples, save_report

app = typer.Typer(add_completion=False)
console = Console()
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"


@app.command()
def sanity(split: str = "dev", limit: int = 0):
    """Evaluate gold SQL against itself; must score 100%."""
    examples = load_examples(split, limit or None)
    report = evaluate(examples, [ex.gold_sql for ex in examples], split)
    console.print(report.summary())
    assert report.execution_accuracy == 1.0, "gold-vs-gold must be 100%"
    console.print("[green]Harness sanity check passed.[/green]")


@app.command()
def model(
    backend: str = typer.Option(..., help="'ollama', 'vllm', or an OpenAI-compatible base URL"),
    model: str = typer.Option(...),
    split: str = "dev",
    limit: int = 0,
    workers: int = 4,
    run_name: str = typer.Option("", help="defaults to <model>-<split>"),
):
    """Generate predictions with a model and score execution accuracy."""
    examples = load_examples(split, limit or None)
    client = make_client(backend, model)
    run_name = run_name or f"{model.replace('/', '-').replace(':', '-')}-{split}"

    with Progress(console=console) as progress:
        task = progress.add_task(f"generating ({model})", total=len(examples))
        preds = generate_predictions(
            client, examples, split, workers=workers,
            on_progress=lambda: progress.advance(task),
        )

    pred_sqls = [p for p, _ in preds]
    latencies = [lat for _, lat in preds]
    report = evaluate(examples, pred_sqls, split, latencies)
    path = save_report(report, RUNS_DIR, run_name)
    console.print(report.summary())
    console.print(f"mean generation latency: {sum(latencies) / len(latencies):.2f}s")
    if hasattr(client, "cost_usd"):
        cost = client.cost_usd()
        console.print(
            f"tokens: {client.input_tokens} in / {client.output_tokens} out"
            + (f", cost: ${cost:.2f} (${cost / len(examples) * 1000:.2f}/1k queries)" if cost else "")
        )
    console.print(f"saved: {path}")


if __name__ == "__main__":
    app()
