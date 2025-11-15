# -------------------------
# Etapa 1: builder 
# -------------------------
FROM python:3.13-slim AS builder

# Configuración básica de Python y del entorno virtual administrado por uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Instalar dependencias de sistema necesarias para construir el proyecto
RUN apt-get update \
 && apt-get install --no-install-recommends -y git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Instalar uv, el gestor de paquetes
RUN pip install --no-cache-dir uv

# Directorio de trabajo de la imagen
WORKDIR /app

# Copiar los archivos de dependencias para aprovechar la cache de Docker.
# El contexto del build excluye data/ y otros artefactos pesados mediante .dockerignore
COPY pyproject.toml uv.lock README.md ./
# Para reducir el peso de la imagen en CPU usamos versiones alternativas de pyproject/uv.lock
# específicas para Docker que apuntan a las ruedas CPU.
COPY api/pyproject.cpu.toml ./pyproject.toml
COPY api/uv.cpu.lock ./uv.lock

# Instalar las dependencias requeridas para la API y el modelo dentro del entorno virtual
RUN uv sync --frozen --group api --group herdnet

# Copiar el código de la aplicación necesario para empaquetar la API
COPY api ./api
COPY configs ./configs
COPY oficial_herdnet ./oficial_herdnet
COPY animaloc_improved ./animaloc_improved

# Instalar el proyecto en modo editable y agregar poethepoet al entorno virtual
RUN uv pip install --no-deps --editable . \
 && uv pip install --no-deps poethepoet \
 && uv cache prune


# -------------------------
# Etapa 2: final 
# -------------------------
FROM python:3.13-slim

# Variables de entorno para Python y la aplicación
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Instalar solo las dependencias de sistema necesarias para ejecutar la API
RUN apt-get update \
 && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Crear un usuario no-root para ejecutar la aplicación de forma segura
RUN adduser --system --group --home /app api-user

# Directorio de trabajo
WORKDIR /app

# Copiar el entorno virtual creado en la etapa builder
COPY --from=builder /app/.venv /app/.venv

# Copiar únicamente los artefactos necesarios de la aplicación
COPY --from=builder --chown=api-user:api-user /app/api ./api
COPY --from=builder --chown=api-user:api-user /app/configs ./configs
COPY --from=builder --chown=api-user:api-user /app/oficial_herdnet ./oficial_herdnet
COPY --from=builder --chown=api-user:api-user /app/animaloc_improved ./animaloc_improved
COPY --from=builder --chown=api-user:api-user /app/pyproject.toml ./
COPY --from=builder --chown=api-user:api-user /app/README.md ./

# Copiar el modelo por defecto dentro de la imagen (opción 1)
RUN mkdir -p /app/data/models/herdnet_v2
COPY --chown=api-user:api-user data/models/herdnet_v2/best_model.pth /app/data/models/herdnet_v2/best_model.pth

# Cambiar al usuario no-root
USER api-user

# Exponer el puerto de la API
EXPOSE 8000

# Pequeño cambio para más versatilidad:
ENV MODEL_WEIGHTS_PATH="data/models/herdnet_v2/best_model.pth"
ENV MODEL_CFG_PATH="configs/test/herdnet.yaml"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Este imagen luego se puede ejecutar con un comando externo como por ejemplo:
# docker run -p 8000:8000 -e AWS_PROFILE="dvc-user"  -v $HOME/.aws:/app/.aws \
#  -v ./data/models/:data/models/ \
#  herd-api

# O, si se quiere usar otro modelo se puede hacer override de las variables de entorno correspondientes
# en la línea de comando:
# docker run -p 8000:8000 -e AWS_PROFILE="dvc-user"  -v $HOME/.aws:/app/.aws \
#  -v ./data/models/:data/models/ \
#  -e MODEL_WEIGHTS_PATH="data/models/faster-rcnn/resnet50-100-epochs-tbl4/best_model.pth" \
#  -e MODEL_CFG_PATH="configs/test/faster_rcnn.yaml" \
#  herd-api

# IMPORTANTE Los anteriores comandos asumen que el comando que los ejecuta tiene ya una
# sección [profile dvc-user] definida en su propio archiv $HOME/.aws/config
