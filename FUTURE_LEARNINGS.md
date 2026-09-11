# Future learnings

Things that cost a debugging cycle on this project and would cost one again on the next.
Each entry is the symptom first, because that is what you will recognize next time.

## The inference engine version is part of the model

**Symptom:** the same quantized artifact scored 47.5% execution accuracy instead of the 71.2% measured a few days earlier.
No errors, no warnings - just visibly worse SQL (spurious JOINs, degenerate output long enough to hit the token cap and fail to parse).

**Cause:** `vllm>=0.11` in `pyproject.toml` resolved to vLLM 0.19, while the accuracy number had been measured on vLLM 0.28.
An older engine loading the same 4-bit compressed-tensors checkpoint produced quantitatively different generations.

**Lesson:** pin the serving engine to the minor version the accuracy gate ran on, and treat an engine upgrade as a model change that has to re-pass the gate.
A quantized checkpoint is not self-describing: how the engine dequantizes and which kernels it selects are part of the numerics.
Concretely: record the engine version in the eval run's metadata, so a benchmark can never be attributed to the wrong stack.

## A lockfile will silently downgrade a dependency to satisfy a group it will never share an environment with

**Symptom:** `uv lock` produced vLLM 0.19 even though 0.28 existed and nothing asked for an older version.

**Cause:** the same lockfile also had to satisfy the quantization group, and `llmcompressor` requires transformers 4.x while vLLM 0.28 requires 5.x.
uv found the newest vLLM compatible with *both* and said nothing, because that is a successful resolution.

**Lesson:** when two dependency groups genuinely cannot share an environment, declare it instead of letting the resolver compromise:

```toml
[tool.uv]
conflicts = [[{ group = "quant" }, { group = "vllm" }]]
```

Then each group resolves independently and installing both together fails loudly.
The general rule: a resolver's job is to find *a* solution, not the solution you meant - if a version matters to a measured result, constrain it.

## Model artifacts are only as portable as their tokenizer config

**Symptom:** `RuntimeError: Failed to load the tokenizer` on server startup, suggesting `--trust-remote-code`, with the real cause buried further up: `Tokenizer class TokenizersBackend does not exist or is not currently imported`.

**Cause:** the artifact was saved by transformers 5.x, which records `tokenizer_class: "TokenizersBackend"` in `tokenizer_config.json`.
Transformers 4.x, pinned by the serving stack, has never heard of that class.
`--trust-remote-code` is a red herring; the tokenizer data in `tokenizer.json` was fine all along.

**Lesson:** an artifact crosses environments, so normalize what it records about its own loading (here: rewrite `tokenizer_class` to `PreTrainedTokenizerFast` at save time, `src/text2sql/artifacts.py`).
More generally, when a library writes a version-specific name into an artifact, that name is an interoperability hazard - assert at save time that the serving stack can load it.

## Exception hierarchies do not follow intuition

**Symptom:** 104 of 1034 requests returned HTTP 500 from an endpoint whose whole job was to reject bad SQL safely.

**Cause:** the guardrail caught `sqlglot.ParseError`, but an unterminated quote or backtick raises `TokenError` from the tokenizer, a sibling class - so it escaped the handler.

**Lesson:** catch the library's base error class (`SqlglotError`) at a validation boundary, not the one exception you happened to see in testing.
Anything a model can emit will eventually be emitted, so the boundary has to be total: for every library used on untrusted input, find its error base class and handle that.

## Load-test where the service lives

Driving a load test from a laptop through an SSH tunnel measures the tunnel: home-network RTT and a single multiplexed TCP connection cap throughput long before the GPU does.
Run the driver on the serving box against localhost, and pull the result file back.
The corollary for any latency benchmark: enumerate what sits between the driver and the server, and remove everything that is not part of what you are claiming to measure.

## Closed-loop beats open-loop for sizing an inference service

A fixed request *rate* above saturation builds an unbounded queue, and the latency you report is mostly queue depth, which says nothing useful.
A fixed number of *in-flight* requests (each worker waits for its response before sending the next) reports the latency a client actually sees at that level of demand, and throughput plateaus instead of diverging.
Sweep concurrency, and read the operating point off the knee: here throughput scaled almost linearly to 16, then doubled from 16 to 128 while p99 grew 4x.

## Verify the production path against the offline number

The eval harness talked to the model server directly, while production traffic goes through a gateway that renders prompts and post-processes output server-side.
Making the harness able to target the gateway (`--backend gateway`) took a small refactor - push prompt rendering and SQL extraction behind a `predict_sql(schema, question)` interface that every client implements - and it is what proved the served path scores exactly what was measured offline (736/1034 both ways).
Without that, "71.2%" would have been a claim about a code path no user takes.
It is also what caught both the engine downgrade and the guardrail crash, in one run.

## One GPU means one GPU process

vLLM pins ~90% of VRAM for its KV cache by design, so anything else on the card (a second server, a quantization pass) fails with a misleading OOM.
Stop the server before running anything else on the GPU, and prefer `vm.sh <verb>` wrappers that kill the previous tmux session, so "restart the server" cannot accidentally mean "run two".

## GPU quota is a concurrency cap, and Kubernetes does not get its own pool

**Symptom:** planning an HPA for a GPU deployment on GKE, then discovering it can never scale past one replica.

**Cause:** a GKE GPU node is an ordinary Compute Engine VM, so it draws from the same GPU quota as any hand-rolled VM - there is no separate "GKE GPU" allowance.
And GPU quota limits GPUs *concurrently attached*, enforced both per region (`NVIDIA_L4_GPUS`) and globally (`GPUS_ALL_REGIONS`), which on a newer project default to 1 each.
One `g2-standard-8` carries one L4, so one node consumes the entire allowance.

**Lesson:** check `GPUS_ALL_REGIONS` alongside the per-region metric before designing anything that scales GPUs, and remember that existing GPU VMs compete with cluster nodes for the same number.
The failure mode is quiet and looks like a broken deployment rather than a quota problem: the pod sits `Pending`, the node pool's scale-up is refused, and the HPA reports `desired 2 / current 1` indefinitely.
Request increases early - they are free and asynchronous, so file them before you need them.

## Quota is not capacity

**Symptom:** `gcloud compute instances start` failed on an already-provisioned VM with `sub-state:STOCKOUT`, despite quota being free and the instance having run in that zone an hour earlier.

**Cause:** quota is permission to attach a GPU; capacity is whether the zone physically has one free. They are independent, and scarce accelerators run out zone by zone.

**Lesson:** never pin a GPU workload to a single zone when the region will do.
A GKE node pool takes `--node-locations` spanning every zone in the region, so the autoscaler places the node wherever capacity exists that minute, and regional quota covers all of them equally - it costs nothing to be flexible.
Stopped VMs are worse off than pods here: their disk is zonal, so a stopped VM can only restart in its own zone and simply cannot start until that one zone has capacity.

## A stopped VM is the only VM whose scopes you can change

**Symptom:** `gcloud storage cp` from the GPU box failed with a permissions error, even though the instance's service account had project-wide storage access.

**Cause:** the deep-learning VM image is created with `devstorage.read_only` in its OAuth scopes. Scopes are evaluated *before* IAM, so a read-only scope caps a service account that IAM would otherwise allow to write.

**Lesson:** check `serviceAccounts[].scopes`, not just IAM roles, when a VM cannot reach a Google API.
Changing scopes requires the instance stopped (`gcloud compute instances set-service-account --scopes=cloud-platform`), so it is worth setting them correctly at creation.

## Cloud Build's docker builder is not your local docker

Two things that work locally and fail there:

- **No BuildKit.** `RUN --mount=type=cache,...` fails with "the --mount option requires BuildKit". Layer caching plus `--cache-from` gets most of the benefit, so dropping the cache mounts is usually the right trade rather than fighting the builder.
- **No nested substitutions.** `$PROJECT_ID` is not expanded inside another substitution's default value, which yields the literal image name `us-central1-docker.pkg.dev/${PROJECT_ID}/...` and an unparseable-reference error. Write built-in variables inline in the step arguments instead.

Building in the cloud is still the right call from an arm64 Mac targeting amd64 nodes: no emulation, and the build environment matches the runtime.

## Let the IaC program mint its own Kubernetes credentials

**Symptom:** the Pulumi Kubernetes provider needs a kubeconfig, and the usual one shells out to `gke-gcloud-auth-plugin`, which was not installed - and could not be installed through `gcloud components` on a Homebrew-managed SDK.

**Cause:** the conventional GKE kubeconfig delegates auth to an external binary.

**Lesson:** have the program build a kubeconfig with a short-lived token from the credentials it already runs under (`gcp.organizations.get_client_config().access_token`).
No extra binary locally or in CI, and the token refreshes on every run.
Export that kubeconfig as a secret stack output so `kubectl` and benchmark scripts can use the same path instead of each solving auth again.

## What a cluster's IaC should not own

The model weights in GCS and the container image in Artifact Registry are deliberately created outside the Pulumi program.
Both outlive any individual cluster, and `pulumi destroy` should not be capable of deleting the artifact you spent GPU hours producing or the image currently deployed.
The rule of thumb: IaC owns what you would happily recreate from scratch, and references what you would be upset to lose.
