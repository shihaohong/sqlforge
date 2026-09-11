"""Run the FastAPI gateway in front of a vLLM server.

    uv run --group serve scripts/serve_gateway.py --upstream http://localhost:8000/v1

One uvicorn process on purpose: the gateway is I/O-bound (it waits on vLLM,
which serializes work on the single GPU anyway), and one process keeps
Prometheus counters in a single registry instead of splitting them per worker.
"""

from pathlib import Path

import typer
import uvicorn

from text2sql.gateway import Settings, create_app

DEFAULTS = Settings()


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
    settings = Settings(
        upstream_url=upstream,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        schema_cache=Path(schema_cache),
    )
    uvicorn.run(create_app(settings), host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    typer.run(main)
