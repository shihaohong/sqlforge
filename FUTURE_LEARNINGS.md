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

**And the base class is not always enough.** The same mistake recurred later with the Anthropic SDK: given no resolvable credential it raises a plain `TypeError` from header validation, which is not part of `anthropic.AnthropicError` at all, so a handler catching the library's own hierarchy still returned a 500.
A library's declared error hierarchy covers the errors it *means* to raise; argument and configuration mistakes arrive as builtins.
For an optional dependency whose absence must degrade rather than fail, handle the builtins too and wrap construction as well as use - either can be where it surfaces.

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

## `pulumi preview` cannot see inside an apply over unknown outputs

**Symptom:** two clean previews, then `pulumi up` died with `AttributeError: 'dict' object has no attribute 'cluster_ca_certificate'` in a lambda building a kubeconfig.

**Cause:** before the cluster exists, its `master_auth` is an *unknown* output, and Pulumi skips an `apply` body whose inputs are unknown. So the code never ran during preview. Once the cluster was real, the lambda executed for the first time - during the apply - and the value arrived as a plain dict rather than the typed `ClusterMasterAuth`.

**Lesson:** a clean preview says nothing about code inside `.apply()` that depends on resources not yet created; that code is only exercised on a real apply against real state.
Keep apply bodies trivial, and read provider output objects as mappings when both shapes are possible (`gcp.container.ClusterMasterAuth` subclasses `dict` with snake_case keys, so `.get("cluster_ca_certificate")` works for the typed class and the plain dict alike).
The practical consequence: budget for the first apply of a new stack to fail in ways preview cannot predict, and never treat preview as a substitute for a from-scratch apply in a throwaway project.

## Pulumi infers dependencies from data flow, so string-built references have none

**Symptom:** `Error 400: Identity Pool does not exist (PROJECT.svc.id.goog)` when binding `roles/iam.workloadIdentityUser`, on a stack that creates the cluster in the same run.

**Cause:** the project's Workload Identity pool only exists once a cluster with Workload Identity has been created, and the binding's member was an f-string (`f"serviceAccount:{PROJECT}.svc.id.goog[{ns}/{ksa}]"`) containing no output from the cluster. With nothing in the data flow to order them, Pulumi created the binding in parallel with the still-provisioning cluster.

**Lesson:** whenever a resource depends on another's *side effect* rather than on one of its output values, say so with `depends_on` - Pulumi (and Terraform) can only infer what it can see referenced.
Workload Identity is the canonical example: the pool is a side effect of cluster creation, and every binding into it needs the explicit edge.

## Do not pipe a command whose exit code you intend to check

`pulumi up --yes | tail -40` exits 0 even when the update fails, because the shell reports the *last* command's status - so a failed apply looked like a success.
Redirect to a file and echo `$?` (or set `pipefail`) when the status is the thing you care about.
This one is easy to get away with for a long time and then badly misleads at exactly the wrong moment.

## A Kubernetes Service name can poison the container it points at

**Symptom:** vLLM crash-looped in the pod with `ValueError: VLLM_PORT 'tcp://34.118.228.58:8000' appears to be a URI`, having started fine on a VM with identical arguments.

**Cause:** Kubernetes injects Docker-link-style environment variables for every Service in the namespace - `<SVCNAME>_PORT=tcp://<clusterIP>:<port>`, uppercased. The Service in front of the pod was named `vllm`, so the pod received `VLLM_PORT=tcp://...`, which collides with vLLM's own `VLLM_PORT` configuration variable.

**Lesson:** set `enableServiceLinks: false` on pod specs as a matter of habit.
It is legacy compatibility almost nobody uses, the injected variables scale with the number of Services in the namespace, and any of them can collide with an application's own configuration - a class of bug that is invisible locally and depends on what else happens to be deployed alongside.
Renaming the Service also works, but then a future rename reintroduces the bug; disabling the mechanism fixes the class.

## A default that is correct in one topology hides missing configuration

**Symptom:** the gateway pod reported healthy, its `SQLFORGE_UPSTREAM` environment variable was correct, vLLM was reachable from inside that very pod - and `/readyz` still answered "All connection attempts failed".

**Cause:** the CLI entrypoint built its defaults from `Settings()` rather than `Settings.from_env()`, so the environment was never read and the upstream stayed at the dataclass default, `http://localhost:8000/v1`.
On a single box - every earlier milestone - that default was exactly right, so the missing wiring behaved identically to working wiring.

**Lesson:** configuration that is only exercised in one topology is untested configuration, and a default that coincides with the correct value is the most effective way to hide a broken path.
Test the wiring itself (set the env, invoke the entrypoint, assert what reached the settings object) rather than trusting that a value present in the environment must have been read.
The tell here was the mismatch between two facts that could not both be true: the address in the environment was reachable, and the process could not reach its configured address.


## Configuration read in one place and rebuilt in another

**Symptom, twice.** First: a gateway pod whose `SQLFORGE_UPSTREAM` was correct, whose upstream was reachable from inside that very pod, and which still could not connect - it had never read the environment.
Then, after that was fixed: a public endpoint serving data with no token, on a deployment whose token was correctly set in a Kubernetes Secret and correctly present in the container's environment.

**Cause.** The second one is the instructive one. The entrypoint read the environment into its option defaults, then built the settings object from its CLI flags:

```python
settings = Settings(upstream_url=upstream, model=model, timeout_s=timeout_s, ...)
```

The flags are a *subset* of the settings. Every field without a flag - the demo token, the rate limits, the asset paths - silently reverted to its dataclass default, and the default token is empty, which means no authentication.

**Lesson:** override configuration, never rebuild it. `dataclasses.replace(defaults, **explicit_overrides)` keeps everything the caller did not speak to; a constructor call enumerates, and an enumeration silently drops whatever was added later.
The security lesson is sharper than the mechanical one: a field whose default is "off" fails open, so the safe default for a credential check is to refuse when unconfigured, or to assert at startup that a public deployment has a token.
Both bugs were invisible in tests that checked the parts and absent from tests of the wiring - so test the wiring: set the environment, invoke the entrypoint, assert what actually reached the settings object.

## Kubernetes merges list fields, so editing one can duplicate it

**Symptom:** changing a Service's port from 8080 to 80 was rejected with `spec.ports[1].name: Duplicate value: "http"` - an error about a second port that the manifest does not contain.

**Cause:** Kubernetes merges lists like `spec.ports` by a merge key (the port number), so a changed port number reads as *an addition*, leaving the old entry in place. Two entries then share a port name, which is invalid.

**Lesson:** for list fields keyed by value - ports, container env, volume mounts - an edit is an add unless the resource is replaced.
Mark such fields for replacement in the IaC (`replace_on_changes=["spec.ports"]`, `delete_before_replace=True`) rather than discovering it as a validation error, and remember the cost of replacing: a recreated LoadBalancer Service gets a new external IP, so treat that address as an output to read, never a constant to hard-code.

## A real certificate without owning a domain

**Problem:** a demo on a bare load-balancer IP is plain HTTP, so browsers warn - and any token in the URL crosses the network in cleartext, which is the part that actually matters.

**What works:** wildcard-DNS services like `sslip.io` and `nip.io` resolve an IP embedded in the hostname, so `136.65.15.120.sslip.io` points at the load balancer with no DNS zone of your own.
That is a real name, so Let's Encrypt will issue for it over HTTP-01 with cert-manager - no domain purchase, no DNS credentials, and automatic renewal.

**Worth knowing:**
- **Reserve the address first.** The hostname *contains* the IP, so an ephemeral address means the URL changes under you. Promote the existing one in place (`gcloud compute addresses create NAME --addresses <current-ip>`) to keep links working, and remember an unattached reserved address bills a few dollars a month.
- **Move the address between load balancers in two steps.** One apply that both releases the address from the old forwarding rule and claims it for the new one can be ordered either way, and the wrong order fails with "address already in use".
- **Turn off proxy buffering for server-sent events.** With nginx's default buffering the stream is held and delivered as a single lump at the end - the endpoint still "works", the tokens still arrive, and the streaming effect is silently gone. Verify by timestamping frames, not by checking the response body.
- **`installCRDs` vs `crds.enabled`.** Passing both spellings of a renamed Helm value, expecting the chart to ignore the one it does not read, fails on cert-manager - it checks for the deprecated key and refuses. Charts can reject values, so "set both and let it sort itself out" is not a safe migration strategy.

## Updating a Secret does not restart anything

**Symptom:** the API key was correct in the Kubernetes Secret, `pulumi up` reported success, and the application still behaved as though no key were configured - for as long as the pods kept running.

**Cause:** environment variables from a `secretKeyRef` are resolved when the container starts. Changing the Secret changes nothing for a running pod, and because the Deployment's own spec was untouched, there was nothing to trigger a rollout either. (Secrets mounted as *files* do get updated in place, eventually - env vars never do.)

**Lesson:** make the credential part of the pod template so a change to it is a change to the spec:

```python
checksum = Output.all(*secret_values).apply(lambda v: sha256("|".join(v).encode()).hexdigest()[:16])
# ...then annotate the pod template with it
```

This is what Helm charts mean by `checksum/config`, and it belongs in the IaC rather than in a habit of running `kubectl rollout restart` after editing a Secret - an out-of-band restart is invisible to the next person and to the next apply.
The general shape: when configuration lives outside the resource that consumes it, something has to tie the two together, or "applied" and "in effect" quietly diverge.


## A test that skips is not a test that passes

**Symptom:** 101 tests passed locally and CI reported "84 passed, 17 skipped" in green, for months of commits.

**Cause:** the demo's tests needed sample databases rendered from a 841MB dataset the repository does not carry, so they were guarded with `pytest.mark.skipif`. On the author's machine the data is always there and the guard never fires. In CI it always fires, so the endpoints that execute model-written SQL - the guardrail re-check, the row cap, the statement timeout, the gold comparison, the token enforcement - were verified nowhere except the one machine that needed the verification least.

**Lesson:** a skip guard silently converts "untested here" into "green", and it fires exactly where you are not looking.
Build the fixture instead of guarding the test: a 4KB sqlite database created in a session fixture exercised every one of those endpoints and moved CI from 84 tests to 101.
Keep the guard only for assertions that are genuinely about the absent data (that the real bundle excludes a 105MB database, that it carries 19), because those are claims about the dataset rather than about the code.

The general check, worth running on any project: compare the test count CI reports against the count you get locally. A gap is a list of things you believe are covered and are not.

## Auditing for code that was never wired up

Finding one dependency that had been carried but never used prompted a sweep for others. The checks that found something, in the order they were worth running:

- **Declared dependencies against actual imports.** Catches carried-but-unused packages. Allow for tools that are never imported (`ruff`, `vllm` as a CLI) and packages loaded indirectly (`bitsandbytes` via `BitsAndBytesConfig`, `accelerate` via a `Trainer`).
- **Environment variables set by the deployment against those read by the application.** This is the one that matters most, and in both directions: a variable the manifest sets and the code ignores is a silent misconfiguration, which is exactly the bug that made a gateway dial `localhost` in production.
- **Configuration keys declared against those read**, in both the application settings and the IaC config.
- **Metrics defined against metrics incremented**, since a counter nobody increments is a dashboard that reads zero forever.
- **The CI test count against the local one** (above).
- Dead-code tools (`vulture`, `knip`) last: they find the least, and most of what they find is a false positive - a function passed as a callable rather than called reads as unused to a regex and to some analyzers.
