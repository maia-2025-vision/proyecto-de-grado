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
RUN adduser --system --group --home /nonexistent api-user

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

# Cambiar al usuario no-root
USER api-user

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar la aplicación usando poethepoet
CMD ["poe", "serve-frc"]
