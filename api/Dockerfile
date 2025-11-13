# -------------------------
# Etapa 1: builder
# -------------------------
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip install uv

RUN uv pip install --system "dvc[s3]"

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /app

COPY .python-version pyproject.toml uv.lock README.md ./
COPY dvc.lock dvc.yaml ./
COPY .dvc .dvc

RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --link-mode=copy --no-dev --no-install-project --only-group api

ENV PATH="/app/.venv/bin:$PATH"
RUN uv pip install poethepoet

RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install \
    albumentations>=2.0.8 \
    antlr4-python3-runtime==4.9.3 \
    hydra-core \
    opencv-python>=4.12.0.88 \
    pillow \
    scikit-learn>=1.7.2 \
    torchmetrics

RUN uv pip install "animaloc @ git+https://github.com/Alexandre-Delplanque/HerdNet@v0.2.1"

RUN --mount=type=secret,id=aws,target=/root/.aws/credentials \
    git init && \
    dvc pull -r aws-remote --allow-missing \
        data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth \
        data/models/herdnet_v2/best_model.pth \
    && rm -rf .git .dvc

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

RUN mkdir -p data/models/faster-rcnn/resnet50-100-epochs-tbl4 \
 && mkdir -p data/models/herdnet_v2
COPY --from=builder --chown=api-user:api-user /app/.venv ./.venv
COPY --from=builder --chown=api-user:api-user /app/api ./api
COPY --from=builder --chown=api-user:api-user /app/configs ./configs
COPY --from=builder --chown=api-user:api-user /app/oficial_herdnet ./oficial_herdnet
COPY --from=builder --chown=api-user:api-user /app/pyproject.toml ./
COPY --from=builder --chown=api-user:api-user /app/data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth ./data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth
COPY --from=builder --chown=api-user:api-user /app/data/models/herdnet_v2/best_model.pth ./data/models/herdnet_v2/best_model.pth

ENV PATH="/app/.venv/bin:$PATH"

USER api-user

EXPOSE 8000

CMD ["poe", "serve-frc"]
