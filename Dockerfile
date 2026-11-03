# syntax=docker/dockerfile:1.7

# --- build: resolve and install dependencies with uv ---------------------------
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never

# Dependencies first, so source edits don't invalidate this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project --extra claude
COPY src ./src
RUN uv sync --locked --no-dev --extra claude

# --- runtime: slim image, non-root user, state on a volume ---------------------
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 fastrag && mkdir /data && chown fastrag:fastrag /data
WORKDIR /app
COPY --from=build --chown=fastrag:fastrag /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTRAG_DATABASE_PATH=/data/fastrag.db \
    FASTRAG_API_KEYS_FILE=/data/api_keys.json \
    FASTRAG_AUDIT_LOG_PATH=/data/audit.jsonl

USER fastrag
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"
CMD ["fastrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
