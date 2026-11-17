# fastRag API and worker image.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Optional extras, e.g. --build-arg EXTRAS="google s3"
ARG EXTRAS="google"
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev $(for e in $EXTRAS; do printf -- '--extra %s ' "$e"; done)

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 fastrag && mkdir -p /data && chown fastrag /data
WORKDIR /app
COPY --from=build --chown=fastrag /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    FASTRAG_STORAGE_PATH=/data/files \
    FASTRAG_LOCAL_SOURCES_ROOT=/data/sources
USER fastrag
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)" || exit 1
CMD ["fastrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
