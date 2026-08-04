# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, in their own layer: application changes should not
# invalidate the (slow) dependency install.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.13-slim

RUN useradd --create-home --uid 1000 reed

WORKDIR /app
COPY --from=builder --chown=reed:reed /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    REED_DATA_DIR=/data \
    REED_HOST=0.0.0.0 \
    FASTEMBED_CACHE_PATH=/data/.fastembed

RUN mkdir -p /data && chown reed:reed /data
USER reed
VOLUME ["/data"]
EXPOSE 8000

# python is already here; adding curl just for a health check would be waste.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "reed.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
