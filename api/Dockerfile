# -------------------------
# Etapa 1: builder
# -------------------------
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends git dvc \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install uv

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /app

COPY .python-version pyproject.toml uv.lock README.md ./
COPY dvc.lock dvc.yaml ./
COPY .dvc .dvc

RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --link-mode=copy --no-dev --no-install-project --only-group api --only-group herdnet-inference

ENV PATH="/app/.venv/bin:$PATH"
RUN uv pip install poethepoet

RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
    git init && \
    dvc pull data/models -r s3-remote && \
    rm -rf .git .dvc

COPY api/ api/
COPY configs/ configs/
COPY oficial_herdnet/ oficial_herdnet/

RUN uv pip install -e .

RUN chown -R api-user:api-user /app


# -------------------------
# Etapa 2: final
# -------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /app

COPY --from=builder --chown=api-user:api-user /app/.venv ./.venv
COPY --from=builder --chown=api-user:api-user /app/api ./api
COPY --from=builder --chown=api-user:api-user /app/configs ./configs
COPY --from=builder --chown=api-user:api-user /app/oficial_herdnet ./oficial_herdnet
COPY --from=builder --chown=api-user:api-user /app/pyproject.toml ./
COPY --from=builder --chown=api-user:api-user /app/data/models ./data/models

ENV PATH="/app/.venv/bin:$PATH"

USER api-user

EXPOSE 8000

CMD ["poe", "serve-frc"]
