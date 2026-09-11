# SQLForge - milestone plan and results log

Fine-tune, quantize, and serve a small (3B) text-to-SQL model as a production-grade inference service.

## The headline this project exists to earn

> Fine-tuned Llama-3.2 3B on Spider to 72.7% execution accuracy (base model: 61.4%, Claude Haiku 4.5: 74.0%),
> quantized it to 4-bit for a 1.4-point accuracy cost, and self-hosted it with vLLM on a single GCP L4 GPU
> at 20 QPS with p50 304 ms / p99 1.5 s and $0.012 per 1k queries - 59x cheaper than the cost-matched API model.

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

Training path: Spider train set -> QLoRA fine-tune (W&B tracked) -> merge adapter -> GPTQ quantize -> eval gate -> serve.

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

- [x] Serve the quantized model with vLLM on the L4 (`scripts/gcp/vm.sh serve`, artifact `out/merged-gptq`, served as `sqlforge-3b`).
- [x] FastAPI gateway (`src/text2sql/gateway.py`): pydantic request validation, server-side schema lookup by `db_id`, SSE streaming, upstream timeouts (504) and errors (502), `sqlglot` parse check plus SELECT-only guardrail on every completion, Prometheus metrics, and `/healthz` / `/readyz` probes.
  Rejected SQL is never returned in the `sql` field, so a caller can execute whatever it receives without re-validating.
- [x] **Scored the full Spider dev set through the gateway: 71.2% (736/1034), identical to the offline GPTQ number.**
  The production path (gateway prompt rendering + guardrails + vLLM) has no accuracy skew against the number the eval harness measured, which is the point of sharing one prompt template.
- [x] Load test at increasing concurrency (`scripts/loadtest.py`, run on the VM against localhost): **20.4 QPS at p50 304 ms / p99 1.5 s (concurrency 8), saturating at 75 QPS / 2,450 output tokens/s (concurrency 128)**, zero errors at every level.
- [x] Cost from GPU-hour price and measured throughput: **$0.0116 per 1k queries at concurrency 8** and $0.0031 at concurrency 128, against **$0.68 for Claude Haiku 4.5** - 59x to 220x cheaper (on-demand L4 at $0.85/hr; on spot at $0.30/hr it is $0.0041 and $0.0011).
- [x] The gateway is free: raw vLLM measured through the same driver is within run-to-run noise at every concurrency level (p50 within 3 ms, QPS within 1%), so validation and guardrails cost nothing next to generation.

Exit criteria: a benchmark table (concurrency x latency/throughput) and the cost comparison, reproducible by script. **M3 complete.**

Benchmark (one L4, GPTQ 4-bit 3B, 512 max output tokens, 30s per level after a 5s warmup, mean 32 output tokens/query):

| Concurrency | QPS | p50 | p90 | p99 | output tok/s | $/1k queries |
|---|---|---|---|---|---|---|
| 1 | 3.3 | 225ms | 603ms | 1268ms | 97 | $0.0723 |
| 2 | 6.0 | 248ms | 619ms | 1306ms | 183 | $0.0394 |
| 4 | 11.4 | 258ms | 660ms | 1160ms | 350 | $0.0208 |
| 8 | 20.4 | 304ms | 723ms | 1518ms | 654 | $0.0116 |
| 16 | 35.0 | 341ms | 828ms | 1693ms | 1120 | $0.0067 |
| 32 | 52.1 | 464ms | 1125ms | 2331ms | 1676 | $0.0045 |
| 64 | 64.4 | 768ms | 1837ms | 3737ms | 2077 | $0.0037 |
| 128 | 75.2 | 1371ms | 3265ms | 6648ms | 2450 | $0.0031 |

Throughput scales almost linearly to concurrency 16 and then flattens as the GPU saturates, so queueing shows up as latency: from 16 to 128 concurrent clients, throughput doubles while p99 grows 4x.
Concurrency 8-16 is the sensible operating point (p99 under 1.7s at a third of a cent per 1k queries); it is also the signal M4's autoscaler should target.

A closed-loop driver rather than k6 or locust: it measures the latency a client sees at a fixed number of in-flight requests, which is how an inference service is sized, and an open-loop request rate above saturation just builds an unbounded queue and reports that as latency.
It also counts the server-reported output tokens per request, which is what makes tokens/sec and $/query real rather than estimated.

Operational notes from the run, each of which cost a debugging cycle (see [FUTURE_LEARNINGS.md](FUTURE_LEARNINGS.md)):
an unpinned `vllm>=0.11` resolved to 0.19 instead of the 0.28 that M2 measured on, and the older engine silently served the same GPTQ artifact at **47.5% instead of 71.2%** - the strongest argument in this project for pinning the inference engine and re-running the eval gate against the thing that actually serves traffic.
The resolution was not a coincidence: `llmcompressor` needs transformers 4.x and vLLM 0.28 needs 5.x, so a single lock quietly downgraded vLLM to fit both; `[tool.uv] conflicts` now forces the two groups to resolve separately.
Model artifacts written by transformers 5.x also record `tokenizer_class: "TokenizersBackend"`, which transformers 4.x cannot load at all, so `normalize_tokenizer_config` rewrites it to `PreTrainedTokenizerFast` when an artifact is saved.
Finally, `sqlglot` raises `TokenError` (not `ParseError`) on an unterminated quote, which the guardrail did not catch: 104 of 1034 requests returned 500 until it caught `SqlglotError` instead.

### M4: Kubernetes deployment

- [x] Prerequisites, all outside the Pulumi program on purpose (they outlive any cluster, and `pulumi destroy` must not be able to delete model weights or the image it deployed): the GPTQ artifact in GCS (`gs://sqlforge-text2sql-serving-sh/models/merged-gptq`, 2.10GiB), an Artifact Registry repo, and the gateway image built by Cloud Build (`deploy/Dockerfile`, multi-stage, non-root, 532MB).
- [x] Pulumi program (`deploy/__main__.py`, Python): zonal GKE cluster with Workload Identity and Managed Prometheus, an `e2-standard-2` system pool (1-3 nodes), a `g2-standard-8` GPU pool **autoscaling 0-1 across all three zones of the region**, the vLLM and gateway Deployments, Services, an HPA, and PodMonitoring for both tiers. `pulumi preview` is clean at 17 resources.
- [x] Health/readiness probes: vLLM gets a long startup probe (a cold start is node provisioning + driver install + a 6GB image pull + a 2.1GB model download + weight load) and a deliberately slack liveness probe so a busy engine is not mistaken for a hung one. The gateway's liveness checks only itself, while readiness checks the model server - restarting a gateway cannot fix a vLLM that is down, but a gateway with no model behind it should leave the Service rather than serve 502s.
- [x] The model reaches the pod through an init container that `gcloud storage rsync`s it from GCS under Workload Identity, so there is no service-account key in the cluster and swapping models is a config change rather than a multi-gigabyte image rebuild.
- [x] `pulumi up` against the real project. The first apply found three bugs that no preview could (below); after fixing them, a `pulumi destroy` (17 resources, 9m17s) followed by a from-scratch `pulumi up` reproduced the whole stack in **one pass, 25m41s, 17 created and 0 errored** - no manual steps.
- [x] Repeat the load test through the full K8s path (`deploy/bench_in_cluster.sh`): **within ±2% of the bare-VM numbers at every concurrency level**, so Kubernetes costs nothing measurable.
- [x] Autoscaling, observed rather than asserted: the GPU pool scaled **0 -> 1** on a pending pod, the HPA scaled the gateway **2 -> 3** under load and **3 -> 2** when idle, and the CPU pool scaled **1 -> 2** to give the benchmark driver its own node.

Exit criteria: `pulumi up` brings up the whole stack from scratch, reproducible by script. **M4 complete.**

| Concurrency | QPS (GKE) | QPS (bare VM) | p50 | p99 | output tok/s | $/1k queries |
|---|---|---|---|---|---|---|
| 1 | 3.2 | 3.3 | 230ms | 1258ms | 95 | $0.0730 |
| 8 | 20.0 | 20.4 | 299ms | 1512ms | 642 | $0.0118 |
| 16 | 34.7 | 35.0 | 355ms | 1680ms | 1112 | $0.0068 |
| 32 | 52.2 | 52.1 | 464ms | 2333ms | 1679 | $0.0045 |
| 64 | 65.7 | 64.4 | 751ms | 3784ms | 2119 | $0.0036 |
| 128 | 74.5 | 75.2 | 1363ms | 6650ms | 2419 | $0.0032 |

Cold start with the GPU pool at zero, which is the price of scale-to-zero: ~90s for the node plus GPU driver, 19s for the init image, **25s to pull the 2.1GB model from GCS**, **3m12s to pull the 8.6GB vLLM image**, then weight load - about 6.5 minutes to first token. On a warm node it is ~2m20s. The image dominates, so caching it (a warm node pool, a smaller runtime image, or GKE image streaming) is where any latency work belongs, not the model download.

The first in-cluster benchmark also produced a lesson rather than a number: throughput *fell* from 60.3 QPS at concurrency 64 to 52.9 at 128, because the driver pod and all three gateway replicas were scheduled onto one 2-vCPU node and starved each other. The driver now requests 2 CPUs with anti-affinity against the gateway, and the numbers above match the bare-VM run. Same mistake as measuring through an SSH tunnel, one layer in.

Three bugs the first apply found, each invisible to `pulumi preview`:
1. **Workload Identity ordering.** The `iam.workloadIdentityUser` binding references the `PROJECT.svc.id.goog` pool, which does not exist until a WI-enabled cluster does. The member is an f-string with no cluster output in it, so Pulumi had no data-flow edge to order on and created the binding in parallel with the still-provisioning cluster.
2. **`master_auth` arrives in an apply as a plain dict**, not the typed `ClusterMasterAuth`. Preview passed either way: before the cluster exists the value is unknown, so Pulumi skips the apply body entirely and that line first executed during the apply.
3. **`enableServiceLinks`.** Kubernetes injects `<SVCNAME>_PORT=tcp://ip:port` for every Service in the namespace, and the Service in front of vLLM is named `vllm` - so the pod received `VLLM_PORT=tcp://10.x.x.x:8000`, which collides with vLLM's own `VLLM_PORT` and stopped the server from starting.

And one application bug that only a multi-node topology could expose: `serve_gateway.py` built its defaults from `Settings()` rather than `Settings.from_env()`, so `SQLFORGE_*` was never read and the gateway dialed `localhost:8000` regardless. On a single box that default was correct, which is exactly why M1-M3 never noticed. `tests/test_serve_gateway.py` now asserts the wiring.

**The GPU quota shapes this milestone.** A GKE GPU node is an ordinary Compute Engine VM and draws from the same GPU quota as the M1-M3 box, and quota caps *concurrently attached* GPUs: this project has `NVIDIA_L4_GPUS = 1` in us-central1 and `GPUS_ALL_REGIONS = 1` globally.
So the GPU pool holds at most one node, the old `t2s-gpu` VM must stay stopped while that node exists, and an HPA on vLLM would have nowhere to scale - a second replica would request a GPU, fail to schedule, trigger a scale-up, and have it refused for quota, leaving the pod `Pending` and the HPA reading `desired 2 / current 1` forever. That is a stuck rollout, not autoscaling.
An increase to 4 L4s has been requested for both quotas (pending); `sqlforge:gpuMaxNodes` and `sqlforge:vllmReplicas` in `Pulumi.dev.yaml` are the only changes needed once it lands.

What autoscales today, then, is the part that can: **the GPU pool scales to zero**, so an idle cluster costs only the system pool and a pending vLLM pod brings the expensive node up on demand. The HPA targets the gateway tier on CPU, which is the honest signal for it - per request the gateway renders a prompt, parses the completion with sqlglot, and runs the guardrail, and that is CPU work. Queue depth (`vllm:num_requests_waiting`) is scraped by Managed Prometheus and is the signal a GPU-tier HPA would use, but scaling the gateway on it would be architecturally wrong: more proxy replicas do not drain a GPU queue.

Operational notes (details in [FUTURE_LEARNINGS.md](FUTURE_LEARNINGS.md)): zonal L4 capacity ran out mid-milestone (`STOCKOUT` refused to start an already-provisioned VM), which is what motivated spreading the GPU pool across all three zones; the deep-learning VM image ships with `devstorage.read_only` scopes, so uploading the artifact to GCS needed a scope change with the instance stopped; and Cloud Build's docker builder runs the legacy engine, so BuildKit cache mounts fail there.

### M5: Interactive demo

A demo neither moves the headline numbers nor makes them more trustworthy, so it does not fit the rule above.
It is here because it *communicates* them: the project's argument is an economic one, and a side-by-side you can click is a far better carrier for it than a table.

- [x] Side-by-side demo page (`web/`): pick a database, ask a question, watch the fine-tuned 3B stream its SQL, see the query execute against the real sqlite file, and see whether the rows match the gold query - with Claude Haiku 4.5 answering the same question beside it, each labelled with measured latency and $/query.
- [x] `GET /v1/demo/schemas`, `POST /v1/demo/execute`, `POST /v1/demo/compare` on the existing gateway; the local model streams through the existing `/v1/sql/stream`, which until now had no consumer and so had never been exercised against a browser.
- [x] Bundled sample databases: 19 of the 20 dev databases, 0.9MB in total (`wta_1` alone is 105MB and is excluded, and the page only offers what it ships).
- [x] Vite + React + Tailwind, built in a Dockerfile stage and served by the gateway itself through `StaticFiles` - one container, one origin, no CORS, no second service. Series colors come from a validated palette (worst-pair CVD delta-E 26.8) and the correctness verdict always pairs its status color with an icon and a word.
- [x] **HTTPS**: an ingress-nginx controller holds a reserved static address and terminates TLS with a Let's Encrypt certificate that cert-manager obtains over HTTP-01 and renews. The hostname is `<ip>.sslip.io`, which resolves an IP embedded in the name, so a real certificate is possible with no DNS zone and no domain purchase. HTTP answers 308 to HTTPS. Server-sent events needed `proxy-buffering: off` on the controller - with buffering on, nginx holds the whole response and token-by-token streaming arrives as one lump; measured after the change, 17 frames spread over 161ms.
- [x] Public behind a shared token and rate limits, verified against the live URL: no token and a wrong token both get 401 on every inference and demo route, while the page itself stays open; 12 requests/min per IP with a burst of 6, a 2000/day cap, and a separate 4/min, 200/day allowance for the paid Claude path.
- [x] Execution safety on top of the existing SELECT-only guardrail: read-only connections, a 5s statement timeout, and a 50-row cap, with the guardrail re-checked server-side because `/v1/demo/execute` is reachable on its own.
- [x] Playwright drives the real page, in a real browser, against both a local gateway and the public URL - the only test that exercises SSE the way a browser does.
- [x] Claude comparison on the public deployment, via an `ANTHROPIC_API_KEY` in the Secret (the `ant auth login` credential the M0 baselines used is a developer credential and cannot authenticate a pod). Setting it exposed one more bug: environment from a `secretKeyRef` is injected at container start, so updating the Secret left the running pods with the old empty value and nothing in the Deployment spec changed to roll them. A hash of the secret values now rides in the pod template, which turns a credential change into a spec change.

Exit criteria: a public URL where a stranger with the token can ask a question and watch both engines answer, with correctness and cost shown. **M5 complete.**

Live through the public HTTPS endpoint, same question to both: the fine-tuned 3B answered in **189ms** and Haiku 4.5 in **725ms**, both matching the gold rows, at **$0.0032 against $0.5490** per 1k queries - **172x**. The paid path's own budget is real too: a six-call burst gets `200 200 429 200 429 429`.

Measured live through the public HTTPS endpoint: 83ms end to end for "How many singers are there?" and 200ms for "What are the names of the stadiums without any concerts?", correct SQL, rows matching the gold query. Locally against the same cluster model with the comparison enabled: 467ms vs Haiku's 1017ms on the same question, both correct, $0.0032 vs $0.5890 per 1k queries - 184x.

Two bugs worth recording, both found only by deploying:
1. **An unauthenticated public endpoint.** `serve_gateway.py` built a fresh `Settings()` from its CLI flags, which reverted every field without a flag - including `demo_token` and the rate limits - to dataclass defaults, and an empty token means no authentication at all. It now `replace()`s over the environment-derived defaults. This is the second bug from the same root as M4's `Settings.from_env()` one: configuration read in one place and rebuilt in another.
2. **A 500 instead of graceful degradation.** With `ANTHROPIC_API_KEY` set to the empty string, the SDK raises a plain `TypeError` from header validation rather than anything in its own error hierarchy, so the handler missed it and `/v1/demo/schemas` returned 500. Exactly the `sqlglot.TokenError` mistake from M3, in a different library.

### M6: Hardening and writeup

- [ ] Eval regression gate in CI: any model/prompt change re-runs a dev-set subset and fails on regression.
- [ ] Technical writeup: decisions, tradeoffs, and the final numbers table. This is the artifact recruiters and interviewers actually read.

## Results

| Model | Execution accuracy (Spider dev) | latency | $/1k queries |
|---|---|---|---|
| Llama-3.2 3B Instruct (base, zero-shot, ollama on M-series Mac) | 61.4% (635/1034) | mean 2.66s/query | n/a (local) |
| Claude Haiku 4.5 (API baseline, zero-shot) | 74.0% (765/1034) | mean 0.95s/query | $0.68 |
| Claude Opus 5 (quality ceiling, first 300 dev examples) | 96.7% (290/300) | mean 2.29s/query | $5.11 |
| **Fine-tuned 3B (QLoRA adapter, vLLM on L4)** | **72.7% (752/1034)** | mean 1.11s/query (8-way concurrent) | n/a (not the serving artifact) |
| Fine-tuned 3B merged bf16 (vLLM on L4) | 72.6% (751/1034) | mean 1.04s/query | n/a (not the serving artifact) |
| Fine-tuned 3B AWQ 4-bit (rejected: -6.0 pts) | 66.6% (689/1034) | mean 0.47s/query | n/a |
| **Fine-tuned 3B GPTQ 4-bit, served through the gateway** | **71.2% (736/1034)** | p50 304ms, p99 1.5s at 20 QPS | **$0.0116** |

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
