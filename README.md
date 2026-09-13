# SQLForge

> **Status: complete.**
> A personal project taking one small model through the whole lifecycle of a production inference service: fine-tuning, quantization, serving, load testing, Kubernetes, a public demo, and a regression gate.
> All six milestones are done and every number below was measured rather than estimated.
>
> - **[WRITEUP.md](WRITEUP.md)** - start here: the question, the results, and the economics including where self-hosting loses.
> - **[ARCHITECTURE.md](ARCHITECTURE.md)** - the system design: components, contracts, request flows, failure behaviour, and control loops.
> - **[PLAN.md](PLAN.md)** - the milestone log, with the decision and the measurement at each step.
> - **[FUTURE_LEARNINGS.md](FUTURE_LEARNINGS.md)** - 24 things that went wrong and what they taught, symptom first.

Fine-tune, quantize, and serve a small (3B) open-weights text-to-SQL model, then benchmark it against frontier API models on quality, latency, and cost.

The core question: can a QLoRA fine-tuned Llama-3.2 3B, self-hosted on a single L4 GPU, beat the cost-matched frontier API model (Claude Haiku 4.5) on Spider execution accuracy at a fraction of the per-query cost?

## Results

Evaluated on the Spider 1.0 dev set with execution accuracy: the generated query runs against the database and returns the same rows as the gold query, or it does not.
Latency and cost are measured on the deployed service rather than estimated.

| Model | Execution accuracy | Latency | $/1k queries |
|---|---|---|---|
| Llama-3.2 3B Instruct (base, zero-shot) | 61.4% (full dev) | mean 2.66s (local Mac) | n/a |
| Claude Haiku 4.5 (zero-shot) | 74.0% (full dev) | mean 0.95s | $0.68 |
| Claude Opus 5 (zero-shot, quality ceiling) | 96.7% (first 300) | mean 2.29s | $5.11 |
| **Fine-tuned 3B (QLoRA, vLLM on L4)** | **72.7% (full dev)** | mean 1.11s | n/a (not the serving artifact) |
| **Fine-tuned 3B, GPTQ 4-bit, served through the gateway** | **71.2% (full dev)** | p50 299ms, p99 1.5s at 20 QPS | **$0.0118** |

The fine-tune moves the 3B model from 61.4% to 72.7% (+11.3 pts), statistically tied with Haiku 4.5 (74.0% full dev, but the fine-tune wins 74.0% vs 71.7% on a shared 300-example subset), with schema-hallucination errors cut by more than half.
GPTQ 4-bit quantization keeps 71.2% (-1.4 pts) while cutting latency 2.2x and model size 2.9x; AWQ was measured too and rejected at -6.0 pts.

Served on one L4 behind the FastAPI gateway, the quantized model scores **exactly the same 71.2%** as it does offline (736/1034 either way, so the serving path adds no skew), sustains **20 QPS at p50 299ms / p99 1.5s**, saturates at **74.5 QPS / 2,419 output tokens/s**, and costs **$0.0118 per 1k queries against Haiku 4.5's $0.68 - 58x cheaper, 2.8 accuracy points behind**.
The gateway's validation and guardrails are free: raw vLLM measured through the same driver is within run-to-run noise at every concurrency level.

That cost figure is a function of utilization, which is the part such comparisons usually omit.
Break-even against Haiku 4.5 is **0.35 QPS, about 30,000 queries a day**; below that an idle GPU makes the API cheaper.
[WRITEUP.md](WRITEUP.md) works through the economics, [ARCHITECTURE.md](ARCHITECTURE.md) covers how the components fit together, and [PLAN.md](PLAN.md) is the milestone-by-milestone log.

## How it works

```
                    train_spider.json (7k examples)
                              |
                    build_train_data.py          <- renders chat-format SFT data with
                              |                     the same prompt template as eval/serving
                    train.py (QLoRA, L4 GPU)
                              |
                    merge -> GPTQ quantize -> eval gate
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
uv sync --group serve   # gateway stack
uv sync --group vllm    # inference engine (GPU box only)
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

## Serving

Render the two files the serving box needs (schemas keyed by `db_id`, plus the questions the load test replays), then bring up the stack on the GPU box:

```bash
uv run scripts/build_serving_assets.py      # data/serving/dev.{schemas,questions}.json
scripts/gcp/vm.sh setup-serve               # install vLLM + gateway deps
scripts/gcp/vm.sh serve                     # vLLM on out/merged-gptq, port 8000
scripts/gcp/vm.sh gateway                   # FastAPI gateway, port 8080
scripts/gcp/vm.sh tunnel                    # forward both ports to localhost
```

Locally, the gateway runs against any OpenAI-compatible backend, including ollama:

```bash
uv run --group serve scripts/serve_gateway.py --upstream http://localhost:11434/v1 --model llama3.2:3b
```

Ask it for SQL by `db_id` (the gateway looks the schema up) or by passing a `schema` string:

```bash
curl -s localhost:8080/v1/sql -H 'content-type: application/json' \
  -d '{"question":"How many singers are there?","db_id":"concert_singer"}'
# {"sql":"SELECT count(*) FROM singer","valid":true,...}
```

Every completion is parsed with `sqlglot` and checked to be a single read-only SELECT; a rejected query is returned as `rejected_sql` with `valid: false`, never as `sql`.
`POST /v1/sql/stream` streams token deltas as server-sent events and ends with a `done` event carrying the same verdict.
`/healthz` is liveness, `/readyz` checks that the model server answers, and `/metrics` exposes request/rejection/token counters and latency histograms for Prometheus.

Scoring the served path is one command, and it should reproduce the offline accuracy exactly:

```bash
uv run scripts/run_eval.py model --backend gateway --model sqlforge-3b --split dev --workers 8
```

## Benchmarking

The load test sweeps concurrency levels in closed loop and reports p50/p90/p99, throughput, output tokens/sec, and $/1k queries from the GPU-hour price.
Run it on the serving box - driving it through an SSH tunnel measures the tunnel:

```bash
scripts/gcp/vm.sh bench --target gateway --concurrency 1,2,4,8,16,32,64,128 --duration 30
scripts/gcp/vm.sh bench --target vllm                     # same load, bypassing the gateway
scripts/gcp/vm.sh results                                 # pull runs/loadtest-*.json back
```

## The demo

A side-by-side page served by the gateway itself: pick one of 19 bundled Spider databases, ask a question, and watch the fine-tuned 3B stream its SQL while Claude Haiku 4.5 answers the same question beside it.
Both answers are then executed against the real sqlite file and compared to the benchmark's gold query, so "correct" means the rows matched - not that the SQL looked plausible.

```bash
uv run scripts/build_serving_assets.py          # bundles schemas, questions, gold SQL, databases
cd web && npm install && npm run build          # the gateway serves web/dist through StaticFiles
uv run --group serve scripts/serve_gateway.py   # then open http://localhost:8080
```

It is served over HTTPS: an ingress-nginx controller terminates TLS with a Let's Encrypt certificate that cert-manager renews, on a `<ip>.sslip.io` hostname - which resolves an IP embedded in the name, so no domain or DNS zone is needed for a real certificate.
Streaming needs `proxy-buffering: off` on the controller, or nginx holds the whole response and the token-by-token effect is lost.

The page is public but every inference route is behind a shared token and rate limits: 12 requests/min per IP with a burst of 6, a daily cap, and a separate much smaller allowance for the Claude path because each of those calls costs money.
A second *service* token is exempt from the limits, which is what the eval harness and load generator use - the demo token necessarily ships to browsers and cannot be trusted with a GPU.
Generated SQL is re-checked by the guardrail server-side, then run on a read-only connection with a 5s timeout and a 50-row cap.

`npx playwright test` drives the real page in a real browser against either a local gateway or the deployed URL, and captures the screenshots used to review layout:

```bash
cd web && BASE_URL=http://<ip> DEMO_TOKEN=<token> npx playwright test
```

## Deploying to Kubernetes

The whole serving stack is one Pulumi program: a zonal GKE cluster, a small CPU pool for the gateway, and a `g2-standard-8` GPU pool that **autoscales 0-1 across every zone in the region**, so an idle cluster costs only the CPU pool and a pending vLLM pod is what brings the expensive node up.

Prerequisites live outside the program, because `pulumi destroy` should never be able to delete model weights or the image it just deployed:

```bash
# the serving artifact, read by the vLLM pod's init container under Workload Identity
gcloud storage rsync -r out/merged-gptq gs://<bucket>/models/merged-gptq

# the gateway image, built on Cloud Build (nodes are amd64; Macs are not)
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_TAG=$(git rev-parse --short HEAD) .
```

Then bring the stack up and benchmark it from inside the cluster:

```bash
cd deploy
pulumi login gs://<bucket>/pulumi-state
pulumi config set --secret sqlforge:demoToken "$(python3 -c 'import secrets;print(secrets.token_urlsafe(18))')"
pulumi config set --secret sqlforge:serviceToken "$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
pulumi config set --secret sqlforge:anthropicApiKey sk-ant-…   # optional; enables the comparison
pulumi config set sqlforge:expose true                         # LoadBalancer instead of ClusterIP
pulumi up
pulumi stack output demo_url

./bench_in_cluster.sh --concurrency 1,8,32,64 --duration 30
pulumi destroy            # GPU pool scales to zero on its own; this removes the rest
```

A from-scratch `pulumi up` builds all 17 resources in one pass in about 26 minutes, most of which is the GPU node's cold start: ~90s for the node and its driver, 25s to pull the 2.1GB model from GCS, and 3m12s to pull the 8.6GB vLLM image.

`deploy/bench_in_cluster.sh` runs the same load-test driver as a pod on the CPU pool, so it measures Service -> gateway -> vLLM rather than the path from a laptop to the cluster.
Both tiers are scraped by Managed Prometheus, including vLLM's `vllm:num_requests_waiting`.

One constraint worth knowing before you copy this: GPU quota caps *concurrently attached* GPUs, per region and globally, and a GKE GPU node draws from the same allowance as any other VM.
With the default limit of 1, the GPU pool holds one node and the vLLM deployment cannot scale horizontally - so the HPA here targets the gateway tier, and PLAN.md records what changes when more quota lands.

## Current deployment

**Nothing is deployed.** The stack was torn down on 2026-09-12 and the project's cloud spend is now zero: the cluster, both node pools, the reserved IP, the container images and the GCS artifact are all deleted.
Every number in this repository was measured against the live stack before it came down.

Bringing it up from nothing is a single `pulumi up` (about 26 minutes, most of it the GPU node's cold start); tearing it down is a single `pulumi destroy`.

```bash
cd deploy && pulumi stack output demo_url     # the live URL
cd deploy && pulumi destroy                   # removes the cluster and both workloads
```

Running costs, for calibration: the L4 node is about $0.85/hour and only exists while a vLLM pod is scheduled, the CPU node about $0.13/hour, and the zonal control plane is covered by GKE's free tier.

To redeploy, the two prerequisites above have to be rebuilt first, because both deliberately live outside the Pulumi program and were removed by hand: re-upload the serving artifact (regenerate it from the QLoRA adapter with `merge_adapter` then `quantize.py`) and rebuild the gateway image on Cloud Build. The demo hostname is derived from the reserved IP, so a redeploy gets a new address.

One lesson worth keeping from the teardown: `pulumi destroy` fails in about two seconds with no resources touched if `PULUMI_CONFIG_PASSPHRASE` is not set, because the stack's secrets are passphrase-encrypted. It looks like a no-op rather than a failure, so confirm with `gcloud container clusters list` rather than trusting the command came back.

## Repository layout

```
src/text2sql/       library: data loading, prompt template, SQL execution, eval, clients, gateway, guardrails
scripts/            entrypoints: run_eval.py, train.py, quantize.py, serve_gateway.py, loadtest.py, gcp/vm.sh
tests/              unit tests: eval harness, SQL guardrails, gateway (mocked vLLM upstream)
configs/            training and serving configs
deploy/             Pulumi program (GKE cluster, node pools, workloads), gateway Dockerfile, in-cluster benchmark
web/                the demo page (Vite + React + Tailwind) and its Playwright test
data/               datasets, rendered SFT files, serving assets (gitignored)
runs/               eval outputs (per-example jsonl + summary json) and load-test results
PLAN.md             milestone plan, decision log, detailed results
WRITEUP.md          the technical writeup: question, results, economics, tradeoffs
ARCHITECTURE.md     system design: components, contracts, request flows, failure modes
FUTURE_LEARNINGS.md what broke and what to do differently next time
```

## Development

```bash
uv run pytest -q                                  # 103 tests, no dataset required
uv run ruff check . && uv run ruff format --check src scripts tests
cd web && npm run build                           # type-check + bundle the demo
```

Those are exactly what CI runs on every push.
Accuracy is not something CI can check without a GPU, so it has its own gate against a live endpoint:

```bash
SQLFORGE_GATEWAY_URL=https://<host> SQLFORGE_TOKEN=<service-token> \
  uv run scripts/eval_gate.py --limit 200
```

It scores a fixed prefix of the dev split and fails if accuracy falls more than 3 points below `runs/eval-baseline.json`.
Re-record the baseline with `--update-baseline` only when a change is intended and understood.
