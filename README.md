# SQLForge

> **Status: work in progress.**
> This is a personal learning project exploring the full lifecycle of a production ML system: fine-tuning, quantization, serving, and benchmarking.
> Baselines, the eval harness, the QLoRA fine-tune, and 4-bit quantization are done; serving, load testing, and Kubernetes deployment are still ahead.
> Expect rough edges and unfinished milestones - see [PLAN.md](PLAN.md) for current state.

Fine-tune, quantize, and serve a small (3B) open-weights text-to-SQL model, then benchmark it against frontier API models on quality, latency, and cost.

The core question: can a QLoRA fine-tuned Llama-3.2 3B, self-hosted on a single L4 GPU, beat the cost-matched frontier API model (Claude Haiku 4.5) on Spider execution accuracy at a fraction of the per-query cost?

## Results so far

Evaluated on the Spider 1.0 dev set with execution accuracy (the generated query runs against the database and returns the same rows as the gold query).

| Model | Execution accuracy | Mean latency | $/1k queries |
|---|---|---|---|
| Llama-3.2 3B Instruct (base, zero-shot) | 61.4% (full dev) | 2.66s (local Mac) | n/a |
| Claude Haiku 4.5 (zero-shot) | 74.0% (full dev) | 0.95s | $0.68 |
| Claude Opus 5 (zero-shot, quality ceiling) | 96.7% (first 300) | 2.29s | $5.11 |
| **Fine-tuned 3B (QLoRA, vLLM on L4)** | **72.7% (full dev)** | 1.11s | measured at M3 |
| **Fine-tuned 3B, GPTQ 4-bit (vLLM on L4)** | **71.2% (full dev)** | 0.47s | measured at M3 |

The fine-tune moves the 3B model from 61.4% to 72.7% (+11.3 pts), statistically tied with Haiku 4.5 (74.0% full dev, but the fine-tune wins 74.0% vs 71.7% on a shared 300-example subset), with schema-hallucination errors cut by more than half.
GPTQ 4-bit quantization keeps 71.2% (-1.4 pts) while cutting latency 2.2x and model size 2.9x; AWQ was measured too and rejected at -6.0 pts.
A proper cost/latency benchmark under load (M3) comes next.

See [PLAN.md](PLAN.md) for the full milestone plan, decision log, and detailed results.

## How it works

```
                    train_spider.json (7k examples)
                              |
                    build_train_data.py          <- renders chat-format SFT data with
                              |                     the same prompt template as eval/serving
                    train.py (QLoRA, L4 GPU)
                              |
                    merge -> AWQ quantize -> eval gate
                              |
client -> FastAPI gateway -> vLLM (quantized 3B) -> SQL validation -> response
```

One prompt template (`src/text2sql/prompts.py`) is shared by training, evaluation, and serving.
Schemas are serialized as the `CREATE TABLE` DDL read from the actual sqlite files, so the model always sees exactly the schema its SQL will run against.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Python 3.12 (uv fetches it automatically).
Optional: [ollama](https://ollama.com) for local model baselines, `gcloud` for GPU training, an Anthropic credential (`ant auth login` or `ANTHROPIC_API_KEY`) for API baselines.

```bash
uv sync                 # core + eval dependencies
uv sync --group train   # training stack (only needed on the GPU box)
uv sync --group serve   # serving stack (M3)
```

Download the Spider dataset (questions, gold SQL, and all 166 sqlite databases):

```bash
uvx gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O data/spider_data.zip
cd data && unzip -q spider_data.zip && rm spider_data.zip && cd ..
```

## Evaluating a model

Always validate the harness first.
Gold-vs-gold must score exactly 100%, or no other number means anything:

```bash
uv run scripts/run_eval.py sanity --split dev
```

Evaluate any model behind an OpenAI-compatible endpoint, or Claude via the Anthropic SDK:

```bash
# local model via ollama
uv run scripts/run_eval.py model --backend ollama --model llama3.2:3b --split dev

# Claude baseline (needs an Anthropic credential)
uv run scripts/run_eval.py model --backend claude --model claude-haiku-4-5 --split dev

# any OpenAI-compatible server (vLLM, hosted APIs)
uv run scripts/run_eval.py model --backend http://localhost:8000/v1 --model my-model
```

Useful flags: `--limit N` evaluates a subset, `--workers N` controls request parallelism, `--run-name` names the output.
Each run writes per-example results and a summary (accuracy, latency, token cost where available) to `runs/`.

## Training

Render the SFT dataset locally (6,800 train / 200 val examples, chat format):

```bash
uv run scripts/build_train_data.py
```

Provision and drive the GPU box (spot g2-standard-8 with one NVIDIA L4; edit `PROJECT`/`ZONE` in the script for your own GCP setup):

```bash
scripts/gcp/vm.sh create   # provision spot VM
scripts/gcp/vm.sh sync     # push code + sft data
scripts/gcp/vm.sh setup    # install deps, verify GPU
scripts/gcp/vm.sh train    # launch QLoRA run in tmux
scripts/gcp/vm.sh status   # tail the training log
scripts/gcp/vm.sh fetch    # pull the trained adapter to models/
scripts/gcp/vm.sh delete   # tear down (stops billing)
```

Training checkpoints every 100 steps and auto-resumes from the latest checkpoint, so spot preemptions cost minutes, not hours.

After training, produce the serving artifact on the GPU box (merge the adapter, then GPTQ-quantize to 4-bit with in-domain calibration):

```bash
uv run --group train scripts/merge_adapter.py
uv run --group train --group quant scripts/quantize.py --method gptq
```

Make sure nothing else (like a vLLM server) holds the GPU while quantizing.

## Repository layout

```
src/text2sql/   library: data loading, prompt template, SQL execution, eval, model clients
scripts/        entrypoints: run_eval.py, build_train_data.py, train.py, gcp/vm.sh
tests/          unit tests for the eval harness (SQL extraction, result comparison)
configs/        training and serving configs
deploy/         infrastructure-as-code (M4)
data/           datasets and rendered SFT files (gitignored)
runs/           eval outputs: per-example jsonl + summary json per run
PLAN.md         milestone plan, decision log, detailed results
```

## Development

```bash
uv run pytest -q        # harness unit tests
uv run ruff check .     # lint
```
