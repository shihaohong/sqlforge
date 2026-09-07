# SQLForge - milestone plan and results log

Fine-tune, quantize, and serve a small (3B) text-to-SQL model as a production-grade inference service.

## The headline this project exists to earn

> Fine-tuned Llama-3.2 3B on Spider to XX% execution accuracy (base model: YY%, frontier API model: ZZ%),
> quantized to 4-bit with <N pt accuracy loss, and self-hosted it with vLLM on a single GCP L4 GPU
> at p99 XXX ms and ~$X.XX per 1k queries (~1/20th the cost of the API model).

Every milestone below either moves one of those numbers or makes them trustworthy.
The project is done when the numbers are real.

## Why text-to-SQL

- The metric is unambiguous: **execution accuracy**.
  The generated query runs against the database and returns the right rows, or it does not.
- The economics are a real production story: a specialized small model competing with a large general API model on quality, latency, and cost.
- Standard benchmark (Spider 1.0) with an established third-party evaluation harness, so results are credible.

## Architecture (target state)

```
client -> FastAPI gateway -> vLLM (quantized 3B, L4 GPU) -> SQL validation -> response
              |                                                  |
           metrics                                        sqlglot parse check,
        (latency, tokens)                                 SELECT-only guardrail
```

Training path: Spider train set -> QLoRA fine-tune (W&B tracked) -> merge adapter -> AWQ quantize -> eval gate -> serve.

## Milestones

### M0: Baselines and eval harness (no GPU needed)

- [x] Download Spider 1.0 (train/dev splits + sqlite databases). Used the 2024 official release (`spider_data.zip`), which also includes the formerly held-out test set (2,147 examples): `uvx gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J -O data/spider_data.zip`.
- [x] Build the eval harness: run generated SQL against the sqlite DBs, compute execution accuracy on the dev set (1,034 examples). Gold-vs-gold sanity check passes at 100% (`uv run scripts/run_eval.py sanity`). Unordered multiset comparison with float tolerance; row order enforced when the gold query has ORDER BY. Cross-check against the official test-suite evaluation before publishing final numbers.
- [x] Baseline 1: base Llama-3.2 3B Instruct, zero-shot with schema-as-DDL in prompt (via ollama locally). **61.4%** execution accuracy (635/1034). Failure profile: 280 ran-but-wrong-rows, 117 "no such column", 2 "no such table" - schema grounding is the dominant weakness, which is exactly what fine-tuning on schema-paired examples targets.
- [x] Baseline 2: Claude Haiku 4.5 (cost-matched competitor) **74.0%** on full dev at $0.68/1k queries; Claude Opus 5 (quality ceiling) **96.7%** on the first 300 dev examples at $5.11/1k queries. Same-300-subset scores: Llama 58.7% / Haiku 71.7% / Opus 96.7%.

Exit criteria: one command reproduces a baseline execution-accuracy number and a per-query cost for both models. **M0 complete.**

### M1: QLoRA fine-tune (GCP L4 spot)

- [x] Prompt/format design: schema-as-DDL prompt template in `src/text2sql/prompts.py`, shared verbatim by training, eval, and serving (no train/serve skew possible).
- [x] QLoRA fine-tune Llama-3.2 3B on Spider train (6,800 train / 200 val after shuffle). 4-bit NF4 base, LoRA r=16 alpha=32 on all attention + MLP projections, LR 2e-4 cosine with 25 warmup steps, effective batch 16, 2 epochs (850 steps), completion-only loss, max_length 4096. Final: train loss 0.089, val loss 0.067, val token accuracy 97.7%. ~3h wall clock on one L4.
- [x] Evaluate with the M0 harness (served via vLLM + LoRA adapter on the L4). **72.7%** execution accuracy (752/1034), +11.3 pts over the base model, 1.3 pts under Haiku 4.5. On the shared 300-example subset it scores 74.0% vs Haiku's 71.7%. Failure profile vs base: "no such column" errors cut from 117 to 50, syntax errors near zero (2); the remaining 229 failures run but return wrong rows.
- [ ] Stretch: same recipe on Qwen2.5-Coder 3B for comparison (often stronger at SQL).

Exit criteria: fine-tuned execution accuracy beats base model by a wide margin and is within striking distance of the API baseline. **M1 complete** (+11.3 pts over base; statistically tied with Haiku).

Operational notes from the run: the spot VM was preempted twice (the second time at step 98, just before the first checkpoint), which motivated checkpoint-every-100-steps with auto-resume and, ultimately, recreating the VM as on-demand by keeping the boot disk (`--keep-disks=boot`, ~$0.85/hr vs ~$0.30/hr spot).
Total GPU cost for M1: roughly $4.

### M2: Quantization

- [x] Merge LoRA adapter into base weights (`scripts/merge_adapter.py`). Effectively lossless: 72.6% merged bf16 vs 72.7% adapter-on-4-bit-base (one example).
- [x] Quantize to 4-bit W4A16 with llm-compressor (`scripts/quantize.py`), calibrated on 256 in-domain SFT examples. **AWQ failed the gate**: 66.6%, a 6.0-pt drop. **GPTQ passed**: 71.2%, a 1.4-pt drop, same 2.2GB artifact (2.9x smaller than bf16), and 0.47s mean generation latency vs 1.04s bf16 (2.2x faster).
- [x] Re-run the eval gate: report accuracy delta vs. the merged fp16 model. Target <1-2 pts degradation. GPTQ delta -1.4 pts: within target.

Exit criteria: a quantized artifact with a measured, acceptable accuracy cost. **M2 complete** - `out/merged-gptq` is the serving artifact for M3.

Operational notes: quantizing on the same L4 that was still running the vLLM eval server produced misleading OOMs (vLLM pins ~90% of VRAM by design) - free the GPU before calibration.
GPTQ needed `offload_hessians=True` to fit a 24GB card, and llmcompressor 0.13's per-Linear `sequential_targets` breaks Llama graph tracing (residuals cross subgraph cuts), so granularity stays at the default decoder layer.
A flaky SSH tunnel mid-eval also motivated transport-error retries in the eval client and a self-reconnecting tunnel loop.

### M3: Serving and benchmarks

- [ ] Serve the quantized model with vLLM on a GCP L4 spot instance.
- [ ] FastAPI gateway: request validation, streaming, timeouts, `sqlglot` parse check and SELECT-only guardrail on outputs, Prometheus metrics.
- [ ] Load test (k6 or locust) at increasing concurrency: p50/p99 latency, tokens/sec, max sustainable QPS.
- [ ] Compute $/1k queries from GPU-hour price and measured throughput; compare against the API baseline's per-query cost.

Exit criteria: a benchmark table (concurrency x latency/throughput) and the cost comparison, reproducible by script.

### M4: Kubernetes deployment

- [ ] GKE cluster (Pulumi), GPU node pool, vLLM + gateway as deployments.
- [ ] Health/readiness probes, HPA on a sensible signal (queue depth or GPU utilization).
- [ ] Repeat the load test through the full K8s path.

Exit criteria: `pulumi up` brings up the whole stack from scratch.

### M5: Hardening and writeup

- [ ] Eval regression gate in CI: any model/prompt change re-runs a dev-set subset and fails on regression.
- [ ] Technical writeup: decisions, tradeoffs, and the final numbers table. This is the artifact recruiters and interviewers actually read.

## Results

| Model | Execution accuracy (Spider dev) | p99 latency | $/1k queries |
|---|---|---|---|
| Llama-3.2 3B Instruct (base, zero-shot, ollama on M-series Mac) | 61.4% (635/1034) | mean 2.66s/query | n/a (local) |
| Claude Haiku 4.5 (API baseline, zero-shot) | 74.0% (765/1034) | mean 0.95s/query | $0.68 |
| Claude Opus 5 (quality ceiling, first 300 dev examples) | 96.7% (290/300) | mean 2.29s/query | $5.11 |
| **Fine-tuned 3B (QLoRA adapter, vLLM on L4)** | **72.7% (752/1034)** | mean 1.11s/query (8-way concurrent) | measured at M3 |
| Fine-tuned 3B merged bf16 (vLLM on L4) | 72.6% (751/1034) | mean 1.04s/query | measured at M3 |
| Fine-tuned 3B AWQ 4-bit (rejected: -6.0 pts) | 66.6% (689/1034) | mean 0.47s/query | n/a |
| **Fine-tuned 3B GPTQ 4-bit (serving artifact)** | **71.2% (736/1034)** | mean 0.47s/query | measured at M3 |

Same-subset comparison (first 300 dev examples): Llama-3.2 3B base 58.7%, fine-tuned 74.0%, Haiku 4.5 71.7%, Opus 5 96.7%.
The subset is not harder or easier by construction, but scores differ slightly from full-set numbers, so cross-model comparisons should use matching example sets.

## Repo layout

```
src/text2sql/   library code (data prep, prompting, eval, gateway)
scripts/        entrypoints (download data, run eval, train, benchmark)
configs/        training and serving configs
deploy/         Pulumi IaC for GCP / GKE
data/           datasets and databases (gitignored)
notebooks/      exploration only; nothing load-bearing lives here
```

## Setup

```bash
uv sync                      # core + eval deps
uv sync --group train        # on the GPU box
uv sync --group serve        # on the serving box
```
