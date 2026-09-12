# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89

FROM python:3.14.7-slim-bookworm@sha256:9ab8d9c8514b44f90cf0029dd42fdd7e9e211e639c8b995304cc04568dee900f AS builder

COPY --from=docker.io/astral/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded /uv /uvx /bin/

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


FROM python:3.14.7-slim-bookworm@sha256:9ab8d9c8514b44f90cf0029dd42fdd7e9e211e639c8b995304cc04568dee900f

RUN useradd --create-home --uid 1000 reed

WORKDIR /app
COPY --from=builder --chown=reed:reed /app /app

# Reed runs from /app/.venv and installs nothing at runtime, so pip is surface
# with no use — and pip vendors its dependencies, which makes their advisories
# the image's: pip/_vendor/vendor.txt pins msgpack 1.1.2 and setuptools 70.3.0,
# the two HIGH findings the scan gate rejects, while Reed's own lockfile
# resolves the fixed versions of both. Dropping pip keeps the image scan about
# the dependencies Reed actually chose.
RUN set -eux; \
    site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"; \
    rm -rf "$site"/pip "$site"/pip-*.dist-info \
           "$site"/setuptools "$site"/setuptools-*.dist-info \
           "$site"/pkg_resources "$site"/wheel "$site"/wheel-*.dist-info \
           /usr/local/lib/python3.*/ensurepip/_bundled; \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.*; \
    ! command -v pip

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
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "reed.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
