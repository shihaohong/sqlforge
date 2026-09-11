# The gateway image. vLLM runs from its official upstream image, so this is
# the only thing this project builds.
#
# Build from the repository root (the app needs src/, the lockfile, and the
# rendered schema cache):
#   gcloud builds submit --config deploy/cloudbuild.yaml .

# The demo page is built here rather than committed as a dist/ directory, so
# the served assets always match the source in this commit.
FROM node:22-slim AS web

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/tsconfig.json web/vite.config.ts web/index.html ./
COPY web/src ./src
# `npm run build` type-checks first, so a type error fails the image build
# rather than shipping a broken page.
RUN npm run build


FROM python:3.12-slim AS builder

# Pinned uv, copied from its own distroless image: no curl-to-shell install,
# and the version is part of the build's reproducibility.
COPY --from=ghcr.io/astral-sh/uv:0.9.18 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile before the source is copied, so a
# code change reuses this layer instead of re-resolving. No BuildKit cache
# mounts: Cloud Build's docker builder runs the legacy engine, and layer
# caching plus --cache-from already covers rebuilds.
# --no-default-groups skips dev (which pulls in pytest and, through it, the
# whole serve group again) and never touches the linux-only vllm group.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-default-groups --group serve

COPY src/ ./src/
# loadtest.py ships too: the benchmark runs as a pod in the cluster, so it
# measures the service rather than the path from a laptop to the service.
COPY scripts/serve_gateway.py scripts/loadtest.py ./scripts/
RUN uv sync --frozen --no-default-groups --group serve


FROM python:3.12-slim AS runtime

# Non-root by default so the pod's securityContext has nothing to fight.
RUN groupadd --gid 1001 app && useradd --uid 1001 --gid app --no-create-home app

WORKDIR /app
COPY --from=builder --chown=app:app /app /app
# Schemas (db_id -> DDL) and the questions the load test replays, rendered by
# scripts/build_serving_assets.py. Baked in rather than mounted: the schemas
# the gateway serves are versioned with the app that was evaluated on them.
COPY --chown=app:app data/serving/ /app/data/serving/

COPY --from=web --chown=app:app /web/dist /app/web/dist

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SQLFORGE_SCHEMA_CACHE=/app/data/serving/dev.schemas.json

USER app
EXPOSE 8080

# Health is a probe's job, not an entrypoint's; uvicorn binds all interfaces
# because the only client is the cluster.
CMD ["python", "scripts/serve_gateway.py", "--host", "0.0.0.0", "--port", "8080"]
