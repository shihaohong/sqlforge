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
