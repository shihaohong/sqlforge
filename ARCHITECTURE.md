# Architecture

How the pieces of SQLForge talk to each other, and why the boundaries fall where they do.

[WRITEUP.md](WRITEUP.md) covers what the system achieves and [PLAN.md](PLAN.md) logs how it got there.
This document is about structure: the components, the contracts between them, what happens on each request, and which design choices were load-bearing.

## The shape of it

There are three planes, and they are worth separating because they fail independently and change at different rates.

```
  BUILD PLANE (minutes, on a laptop or in CI)

    Spider dataset ──► build_train_data ──► train.py (QLoRA, L4) ──► adapter
                                                                      │
                                              merge_adapter ◄─────────┘
                                                    │
                                              quantize.py (GPTQ W4A16)
                                                    │
                                                    ▼
                                          gs://…/models/merged-gptq      ← the serving artifact
    src/ + web/ ──► Cloud Build ──► Artifact Registry (gateway image)


  CONTROL PLANE (minutes, deliberate)

    deploy/__main__.py ──► pulumi up ──► GKE: node pools, workloads, Secret, Ingress
                                              │
                       reads image ───────────┘
                       references bucket + reserved IP (owned outside the program)


  DATA PLANE (milliseconds, continuous)

    browser / eval harness / load driver
               │  HTTPS
               ▼
        ingress-nginx  ── TLS termination, Let's Encrypt cert
               │  HTTP, cluster-internal
               ▼
        gateway (2-6 replicas, CPU pool)     ── auth, rate limit, prompt, guardrail, metrics
               │  HTTP /v1/chat/completions
               ▼
        vllm (1 replica, GPU pool 0→1)       ── the model, and nothing else
               │
               ▼
        /model (emptyDir, filled at pod start from GCS)
```

The rule that shaped this: **vLLM stays a stock, unmodified component.** Everything project-specific lives in the gateway.
That is what makes the engine a version-pinned dependency rather than a fork, and it is why "upgrade vLLM" is a config change with an eval gate rather than a merge.

## Components and their contracts

| Component | Talks to | Over | Contract |
|---|---|---|---|
| `web/` (React SPA) | gateway | HTTPS, JSON + SSE | `GET /v1/demo/schemas`, `POST /v1/sql/stream`, `POST /v1/demo/execute`, `POST /v1/demo/compare` |
| `gateway.py` | vLLM | HTTP, OpenAI chat completions | `POST /v1/chat/completions`, streaming or not |
| `gateway.py` | Anthropic API | HTTPS, Anthropic SDK | optional; absence degrades one feature, breaks nothing |
| `demo.py` | sqlite files | local file reads | read-only connections to bundled databases |
| eval harness | any backend | `SqlClient.predict_sql(schema, question) -> sql` | the seam that makes backends interchangeable |
| `loadtest.py` | gateway or vLLM | HTTP | same prompts either way, so the delta is the gateway |
| init container | GCS | `gcloud storage rsync` | Workload Identity, no keys in the cluster |
| Managed Prometheus | gateway + vLLM | scrape `/metrics` | `sqlforge_*` counters, `vllm:*` gauges |

### The one interface that matters most

Every model backend implements the same two-argument method:

```python
class SqlClient:
    def predict_sql(self, schema: str, question: str) -> str: ...
```

`ChatClient` (ollama, vLLM), `AnthropicClient` (Claude), and `GatewayClient` (our own service) all satisfy it.
The eval harness takes any of them, so **one harness scores a local model, an API model, and the production path** without knowing which it has.

This is what made two of the project's most valuable measurements possible at all.
Scoring the deployed gateway against the same harness proved the production path returns 736/1034, identical to scoring vLLM directly, so prompt rendering and extraction in the service introduce no skew.
And it is what let the CI gate point at a live HTTPS endpoint with one environment variable.

Prompt rendering and SQL extraction live *inside* the client implementations rather than in the harness, because the gateway does that work server-side and the harness must not do it twice.

### The seam underneath everything: one prompt template

`prompts.build_messages(schema, question)` is called by training data generation, by the eval harness, by the gateway, and by the load generator.
A schema is always the `CREATE TABLE` DDL read from the sqlite file the query will actually run against, never a summarized or normalized version.

There is no configuration to make these diverge, because divergence is silent: the only symptom of a changed prompt is lower accuracy that nobody attributes to the change.
`tests/test_prompt_contract.py` freezes the rendered strings, and says in its docstring that a failure means re-running the eval gate rather than updating the expected value.

## Request flows

### Synchronous generation

```
client                gateway                                  vLLM
  │  POST /v1/sql        │                                        │
  ├─────────────────────►│ guard(): authenticate, rate limit      │
  │                      │ resolve schema (db_id → DDL)           │
  │                      │ build_messages()                       │
  │                      ├───────────────────────────────────────►│
  │                      │           chat completion (30s timeout)│
  │                      │◄───────────────────────────────────────┤
  │                      │ extract_sql()                          │
  │                      │ check_sql()  ← sqlglot, SELECT-only    │
  │◄─────────────────────┤ 200 {sql | rejected_sql, valid, usage} │
```

Failure mapping is deliberate and narrow:
upstream timeout becomes `504`, any other upstream error `502`, a missing token `401`, exhausted budget `429`, an unknown `db_id` `404`.
A completion the guardrail rejects is **not** an error: it returns `200` with `sql: null` and the offending query under `rejected_sql`, because the request succeeded and it is the model's output that is unusable.
That distinction matters to a caller: `sql` is either safe to execute or absent, never "probably fine".

### Streaming

Same path, except the gateway forwards vLLM's SSE frames as they arrive and emits a terminal `done` event carrying the verdict.

Guardrails are unavoidably post-hoc when streaming, since tokens are shown before the statement is complete.
The contract handles that honestly rather than pretending: a client may render deltas as they arrive, but must wait for `done` before trusting the query, and `valid` in that event is what it waits for.
This path is also why `proxy-buffering: off` is set on the ingress controller: with nginx's default buffering the whole response is held and delivered as one lump, which still "works" while silently destroying the only reason to stream.

### The demo's two-engine flow

```
browser
   ├─► POST /v1/sql/stream      (local model, streamed)   ─┐
   └─► POST /v1/demo/compare    (Claude, one shot)        ─┤ issued concurrently
                                                           │
   then, for each returned query:                          │
       POST /v1/demo/execute ──► guardrail re-check ───────┘
                             ──► read-only sqlite, 5s timeout, 50-row cap
                             ──► compare rows to the gold query
```

The two engines are called concurrently on purpose: the comparison is of latency as much as of correctness, and serializing them would make whichever went second look slow.

`/v1/demo/execute` re-runs the guardrail on SQL the caller supplies rather than trusting that `/v1/sql` already checked it.
The endpoint is independently reachable, so it independently validates. Sharing a `check_sql()` call is cheap; sharing an assumption is not.

## Boundaries that were chosen deliberately

**The gateway owns everything except inference.**
Validation, schema lookup, prompt construction, guardrails, metrics, auth, and rate limiting are all in front of a stock vLLM.
Measured under identical load, that entire layer costs nothing detectable against raw vLLM, so the separation is free.

**The model artifact is delivered at runtime, not baked into an image.**
An init container pulls 2.1GB from GCS into an `emptyDir` under Workload Identity, so there is no service-account key in the cluster and swapping models is a config change rather than a multi-gigabyte image rebuild.
The cost is a slower cold start, and measurement said that cost is not where it looks: the 2.1GB model download takes 25 seconds against 3m12s to pull the 8.6GB vLLM image.

**Infrastructure-as-code owns what it can safely destroy.**
The Pulumi program owns the cluster, node pools, workloads, Secret and Ingress.
It deliberately does *not* own the GCS bucket with the model, the Artifact Registry repository, or the reserved IP address the demo hostname is built from.
The rule: IaC owns what you would happily recreate, and references what you would be upset to lose. `pulumi destroy` must not be able to delete an artifact that cost GPU hours or invalidate a link someone has.

**Two credentials, because they have different threat models.**
The demo token necessarily ships to browsers and is therefore metered: 12 requests/min per IP, a daily cap, and a tighter separate budget for the paid Claude path.
The service token is exempt from rate limits and is what the eval harness and load generator use, because a 200-example run at eight-way concurrency is not abuse.
Same authentication, different capability.

**Rate limiting is per replica, and that is a documented simplification.**
Counters live in each process, so three gateway pods mean three times the configured ceiling.
A shared counter would mean running Redis for a demo. The limits are set low enough that the multiple still lands somewhere sane, and the daily cap is the real backstop.

## Failure and degradation

The system is built so that each dependency failing degrades one thing rather than everything.

| What fails | What happens | What still works |
|---|---|---|
| vLLM down or slow | `502` / `504`; gateway pods go NotReady and leave the Service | the page loads, probes report honestly |
| No Anthropic credential | `/v1/demo/compare` answers `503`, feature flagged off at `/v1/demo/schemas` | the whole local-model path |
| Model produces unsafe SQL | guardrail rejects, `200` with `valid: false` | everything; this is a normal outcome, and counted |
| Client exceeds its budget | `429` with `Retry-After` | other clients, and the service token |
| GPU node absent | vLLM pod pending, autoscaler adds a node (~6.5 min) | nothing until it lands; this is the price of scale-to-zero |

Probe semantics encode the same thinking.
The gateway's **liveness** probe checks only itself, because restarting the gateway cannot fix a vLLM that is down, and a restart loop during a model outage would turn one failure into two.
Its **readiness** probe does check the upstream, so a gateway with no model behind it leaves the Service rather than serving `502`s.
vLLM gets a long **startup** probe (a cold start is node provisioning, an 8.6GB image pull, a model download and a weight load) and a deliberately slack liveness probe, so a busy engine is never mistaken for a hung one.

## Control loops

Three autoscalers, each on a signal it can actually act on.

| Loop | Signal | Range | Why this signal |
|---|---|---|---|
| Cluster autoscaler (GPU pool) | pending pod with a GPU request | 0 → 1 nodes | a pod that cannot schedule is the only honest demand signal for a node that costs $0.85/hr |
| HPA (gateway) | CPU utilization vs request | 2 → 6 replicas | the guardrail's sqlglot parse is genuine per-request CPU work, so CPU tracks load here |
| Cluster autoscaler (CPU pool) | pending gateway pods | 1 → 3 nodes | follows the HPA |

There is deliberately **no** autoscaler on vLLM.
Queue depth (`vllm:num_requests_waiting`) is scraped and is the signal a GPU-tier HPA would use, but the project's quota is one L4, so a second replica would sit `Pending` forever and the HPA would read `desired 2 / current 1` indefinitely.
An autoscaler with nowhere to scale is a stuck rollout wearing a control loop's clothes.

## What I would do differently

- **Put the schema registry behind an interface from the start.** It resolves `db_id` from a JSON cache and falls back to reading sqlite directly, which is fine, but a real system would fetch schemas from the database it is querying and cache them with invalidation. The current design silently assumes schemas never change.
- **Treat rate limiting as a real component.** Per-replica counters were the right call for a demo and the wrong shape for anything else; the interface is already narrow enough to swap, which is the part I would keep.
- **Separate the demo endpoints into their own service.** They share the gateway's process today, so a demo bug can take production traffic with it. They are already a separate router with its own dependencies, so the split would be mechanical, and the fact that it *would* be mechanical is the useful property.
- **Make the cold start a first-class concern earlier.** Scale-to-zero is elegant until the first request after idle waits six and a half minutes. Knowing the image dominates that, the fix is a slimmer runtime image, and I would have measured the breakdown before choosing the topology rather than after.
