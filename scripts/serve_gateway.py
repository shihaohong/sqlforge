"""Run the FastAPI gateway in front of a vLLM server.

    uv run --group serve scripts/serve_gateway.py --upstream http://localhost:8000/v1

One uvicorn process on purpose: the gateway is I/O-bound (it waits on vLLM,
which serializes work on the single GPU anyway), and one process keeps
Prometheus counters in a single registry instead of splitting them per worker.
"""

from dataclasses import replace
from pathlib import Path

import typer
import uvicorn

from text2sql.gateway import Settings, create_app

# Defaults come from the environment, so SQLFORGE_* works in a container and
# explicit flags still win. Settings() alone would ignore the environment
# entirely - and its default upstream (localhost:8000) is exactly right on a
# single box, which is how that went unnoticed until the gateway ran in a pod
# with vLLM on another node.
# Defaults come from the environment, so SQLFORGE_* works in a container and
# explicit flags still win. Settings() alone would ignore the environment
# entirely - and its default upstream (localhost:8000) is exactly right on a
# single box, which is how that went unnoticed until the gateway ran in a pod
# with vLLM on another node.
DEFAULTS = Settings.from_env()


def main(
    host: str = "0.0.0.0",
    port: int = 8080,
    upstream: str = DEFAULTS.upstream_url,
    model: str = DEFAULTS.model,
    timeout_s: float = DEFAULTS.timeout_s,
    max_tokens: int = DEFAULTS.max_tokens,
    schema_cache: str = str(DEFAULTS.schema_cache),
    log_level: str = "info",
) -> None:
    # replace() over the env-derived defaults, not a fresh Settings(): the
    # flags below are a subset of the settings, and constructing a new object
    # silently reverted everything without a flag - tokens, rate limits, demo
    # asset paths - to dataclass defaults. An empty demo token means an
    # unauthenticated public endpoint, so this one mattered.
    settings = replace(
        DEFAULTS,
        upstream_url=upstream,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        schema_cache=Path(schema_cache),
    )
    uvicorn.run(create_app(settings), host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    typer.run(main)
