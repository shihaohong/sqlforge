# SQLForge

**A QLoRA fine-tuned Llama-3.2 3B, quantized to 4-bit and self-hosted on one NVIDIA L4, answers text-to-SQL within 2.8 points of Claude Haiku 4.5 at 1/58th the cost per query.**

Live demo, both engines side by side: [136.65.15.120.sslip.io](https://136.65.15.120.sslip.io) (token required).
How the system fits together: [ARCHITECTURE.md](ARCHITECTURE.md).
Milestone log and raw numbers: [PLAN.md](PLAN.md).
Everything that went wrong on the way: [FUTURE_LEARNINGS.md](FUTURE_LEARNINGS.md).

## The question

Text-to-SQL is a narrow, well-benchmarked task with an unambiguous metric: run the generated query, and either it returns the same rows as the reference query or it does not.
That makes it a fair place to ask an economic question rather than a modelling one.

> Can a small open-weights model, fine-tuned on the task and self-hosted on one cheap GPU, replace a frontier API model for a specialized workload, and what does it actually cost?

The answer, measured rather than estimated: **yes above roughly 30,000 queries a day, and no below it.**
Utilization, not accuracy, is what decides it.

## Results

Spider 1.0 dev set, 1,034 examples, execution accuracy.
Latency and cost are measured on the deployed service, not estimated from token counts.

| Model | Execution accuracy | Latency | $/1k queries |
|---|---|---|---|
| Llama-3.2 3B Instruct (base, zero-shot) | 61.4% | 2.66s mean (local) | n/a |
| **Fine-tuned 3B (QLoRA)** | **72.7%** | 1.11s mean | n/a (not the serving artifact) |
| **Fine-tuned 3B, GPTQ 4-bit (what is served)** | **71.2%** | **p50 299ms, p99 1.5s at 20 QPS** | **$0.0118** |
| Claude Haiku 4.5 (cost-matched competitor) | 74.0% | 0.95s mean | $0.68 |
| Claude Opus 5 (quality ceiling, first 300) | 96.7% | 2.29s mean | $5.11 |

Fine-tuning moved the 3B model **+11.3 points**, from 61.4% to 72.7%, and cut its dominant failure mode
(hallucinated column names: 117 "no such column" errors down to 50).
4-bit quantization then cost **1.4 points** and bought a 2.2x latency improvement and a 2.9x smaller artifact.

The remaining gap to Haiku 4.5 is **2.8 points**.
On a shared 300-example subset the fine-tune actually wins, 74.0% to 71.7%, which is a caution about subset comparisons rather than a claim of superiority.

Throughput on one L4, measured with a closed-loop driver running inside the cluster:

| Concurrency | QPS | p50 | p99 | output tok/s | $/1k queries |
|---|---|---|---|---|---|
| 1 | 3.2 | 230ms | 1258ms | 95 | $0.0730 |
| 8 | 20.0 | 299ms | 1512ms | 642 | $0.0118 |
| 32 | 52.2 | 464ms | 2333ms | 1679 | $0.0045 |
| 128 | 74.5 | 1363ms | 6650ms | 2419 | $0.0032 |

Throughput scales nearly linearly to concurrency 16 and then flattens as the GPU saturates, so further demand turns into latency.
From 16 to 128 concurrent clients, throughput doubles while p99 grows fourfold.
Concurrency 8 to 16 is the sensible operating point.

## The economics, honestly

The per-query cost of a self-hosted model is a GPU-hour price divided by how many queries that hour served.
That makes it entirely a function of utilization, which is the part cost comparisons usually omit.

| Sustained load | Queries/day | $/1k queries | vs Haiku 4.5 |
|---|---|---|---|
| 0.1 QPS | 8,600 | $2.36 | 3.5x **more** expensive |
| **0.35 QPS** | **30,000** | **$0.67** | **break-even** |
| 1 QPS | 86,400 | $0.236 | 2.9x cheaper |
| 20 QPS | 1.7M | $0.0118 | **58x cheaper** |
| 74.5 QPS (measured peak) | 6.4M | $0.0032 | 215x cheaper |

So the honest headline is conditional.
Below about 30,000 queries a day the API is cheaper, because an idle GPU bills anyway and Haiku does not.
Above it the self-hosted model pulls away fast, and at a genuinely busy 20 QPS it is 58x cheaper.

Two costs the table omits, both small.
Fine-tuning took about **$4** of L4 time, which amortizes after roughly 1.25M queries at the peak rate, or immediately at any rate that would justify self-hosting at all.
The serving stack also carries a small CPU node and a load balancer, on the order of $0.10/hour, which is real but does not move the comparison.

What the API buys that this does not: no operational burden, no cold starts, elastic capacity, and a latency profile that does not degrade when you get busy.
At 128 concurrent clients this service is at p99 6.6s while the API would not be.

## How it was built

Six stages, each ending in a measurement rather than a vibe.

**Baselines and a harness first.**
Before any training, an execution-accuracy harness that runs generated SQL against the real sqlite databases, plus a gold-vs-gold sanity check that must score exactly 100%.
Every number in this document comes from that one harness.
The base model scored 61.4% and Haiku 4.5 scored 74.0%, so the target was defined before anything was fine-tuned.

**One prompt template, shared by everything.**
Training, evaluation, serving, and the demo all render prompts through a single function, and schemas are serialized as the `CREATE TABLE` DDL read from the sqlite file the query will actually run against.
Train/serve skew is one of the easiest ways to silently lose accuracy, so the template is now frozen by a test that fails if a single character changes.

**QLoRA on a spot L4.**
4-bit NF4 base, LoRA rank 16 on all attention and MLP projections, completion-only loss, 2 epochs, about 3 hours and $4.
The spot instance was preempted twice, which is what motivated checkpointing every 100 steps with auto-resume.

**Quantization behind an eval gate.**
Two candidates, both W4A16 via llm-compressor, calibrated on in-domain examples.
**AWQ failed the gate at 66.6%**, a 6.0-point drop, and was rejected.
**GPTQ passed at 71.2%**, a 1.4-point drop, for the same 2.2GB artifact and 2.2x lower latency.
Choosing between them required measuring both, which is the entire argument for having a gate.

**A gateway that owns everything except inference.**
vLLM stays a plain OpenAI-compatible backend, and a FastAPI service in front of it does request validation, schema lookup, prompt rendering, streaming, timeouts, Prometheus metrics, and the output guardrail: every completion must parse as exactly one read-only SELECT.
Measured against raw vLLM under identical load, the gateway costs nothing detectable.
Scoring the full dev set *through* the gateway returns 736/1034, identical to scoring vLLM directly, which is how I know the production path has no skew.

**Kubernetes, with the GPU pool scaled to zero.**
One Pulumi program builds the cluster, a small CPU pool, a GPU pool that autoscales 0 to 1, and both workloads.
A pending vLLM pod is what summons the expensive node, so an idle cluster costs only the CPU pool.
Through the full Kubernetes path the benchmark lands **within ±2% of the bare-VM numbers at every concurrency level**.
Cold start from zero GPUs is about 6.5 minutes, and the dominant term is not the model at all: 25 seconds to pull 2.1GB of weights from GCS against **3m12s to pull the 8.6GB vLLM image**.

## What these numbers do not mean

- **The harness is mine, not the official one.** Execution accuracy here uses my own comparison (unordered multiset, float tolerance, order enforced when the reference query has ORDER BY), validated by a gold-vs-gold check that must score exactly 100%. Cross-checking against Spider's official test-suite evaluation was considered and declined, so these numbers are internally consistent and comparable across this project's runs, but not directly comparable to published leaderboard figures.
- **Subset scores move by several points.** The CI gate scores the first 200 dev examples and gets 68.0%, against 71.2% on the full set. Comparisons are only valid on identical example sets.
- **Generation is not deterministic.** Re-running the gate against an unchanged deployment moved the score by a point, because vLLM's continuous batching changes how requests are grouped and therefore the arithmetic. Any threshold has to absorb that.
- **One GPU is the whole experiment.** The project's quota is a single L4, so the vLLM tier cannot scale horizontally and none of these numbers say anything about multi-GPU behaviour.
- **The accuracy gap is real.** 71.2% against 74.0% means roughly 29 more wrong answers per thousand. Whether that trade is acceptable is a product question, not a benchmark question.

## Four bugs that paid for themselves

The full catalogue is in [FUTURE_LEARNINGS.md](FUTURE_LEARNINGS.md).
These four changed how I work.

**An unpinned dependency cost 24 points of accuracy, silently.**
`vllm>=0.11` resolved to 0.19 rather than the 0.28 the accuracy was measured on, and the older engine served the same quantized artifact at **47.5% instead of 71.2%** with no error anywhere.
The resolver had not misbehaved: `llmcompressor` needs transformers 4.x and vLLM 0.28 needs 5.x, so a single lockfile quietly downgraded vLLM to satisfy both.
The engine version is part of the model, the two groups are now declared as conflicting, and this is the bug the CI eval gate exists to catch.

**A Kubernetes Service name broke the server it pointed at.**
Kubernetes injects `<SVCNAME>_PORT=tcp://ip:port` for every Service in a namespace, the Service was named `vllm`, and vLLM reads `VLLM_PORT` as its own configuration.
The server refused to start with a config error about a value nobody had set.
`enableServiceLinks: false` now disables that mechanism, which fixes the class rather than the instance.

**Configuration read in one place and rebuilt in another, twice.**
Once the gateway ignored its environment entirely and dialed `localhost`, which was invisible for three milestones because on a single box `localhost` was correct.
Then the entrypoint rebuilt its settings from CLI flags, reverting every field without a flag to its default, and the default demo token is empty, so a public endpoint served data with no authentication.
Override configuration, never re-enumerate it, and test the wiring rather than the parts.

**The benchmark measured itself.**
The first in-cluster run showed throughput *falling* from 60 QPS to 53 as concurrency doubled.
The load driver and all three gateway replicas had been scheduled onto one 2-vCPU node and were starving each other.
With the driver given its own node the numbers matched the bare-VM run exactly.
The general form: enumerate what sits between the driver and the thing being measured, and remove everything that is not part of the claim.

## What I would do next

Two of these were considered and consciously dropped, which is worth saying plainly rather than leaving them to look like oversights.

- **Cross-checking against Spider's official test-suite evaluation: decided against.** Every number here comes from one harness, so they are internally consistent and comparable across this project's own runs, which are the comparisons the project actually makes. The cost of that choice is that they are not directly comparable to published leaderboard figures, and the limitation above says so.
- **Closing the 2.8-point gap: not pursued.** The obvious lever is the base model, since Qwen2.5-Coder 3B is generally stronger at SQL and the recipe would transfer unchanged. The remaining 229 failures mostly run and return wrong rows, which points at join and aggregation semantics rather than schema grounding. The project's question was about the economics of self-hosting, though, and that question is answered.
- **Shrink the cold start.** The 8.6GB engine image dominates it, so a slimmer runtime image or GKE image streaming would matter far more than anything about the model.
- **Make the cost curve less cliff-like.** Scale-to-zero handles idle, but the first request after idle pays 6.5 minutes. Keeping a warm node during business hours and scaling to zero overnight would be the pragmatic compromise.
- **Test on harder data.** Spider is a benchmark; BIRD or a private schema set would say more about whether this transfers to real databases with hundreds of tables.

## Reproducing it

[README.md](README.md) has the full commands.
The short version: `uv sync`, download Spider, `uv run scripts/run_eval.py sanity` to prove the harness, then any of the baselines.
Training and serving need one L4; `scripts/gcp/vm.sh` drives a single VM and `deploy/` brings up the Kubernetes stack with one `pulumi up`.

Continuous integration runs lint, 101 tests, the frontend type-check and build, and the frozen prompt contract on every push.
The accuracy gate (`scripts/eval_gate.py`) runs real questions against the live endpoint and fails on a regression beyond 3 points, which is the only check that can see the class of bug described above.
It is verified end to end in GitHub Actions: a dispatched run downloads the dataset, proves the harness scores gold-vs-gold at 100%, scores 200 dev examples against the deployed HTTPS endpoint, and reports the delta against the baseline in about 40 seconds.
