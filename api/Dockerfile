# -------------------------
# Etapa 1: builder - Instala dependencias y prepara el entorno
# -------------------------
FROM python:3.13-slim AS builder

# Variables de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar git (necesario para dependencias de GitHub como animaloc)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instalar uv, nuestro gestor de paquetes
RUN pip install uv

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias para aprovechar el caché de Docker
COPY pyproject.toml uv.lock README.md ./

# Instalar dependencias de la API y del modelo usando los grupos de pyproject.toml
RUN uv pip install --system --group api --group herdnet

# Instalar poethepoet para usar el task runner
RUN uv pip install --system poethepoet

# Copiar el resto del código fuente del proyecto
COPY . .

# Instalar nuestro proyecto en modo editable
RUN uv pip install --system -e .


# -------------------------
# Etapa 2: final - Crea la imagen de producción ligera
# -------------------------
FROM python:3.13-slim

# Variables de entorno para Python y la aplicación
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Instalar solo las librerías de sistema necesarias para ejecutar (ej. para OpenCV)
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Crear un usuario no-root para correr la aplicación por seguridad
RUN adduser --system --group --no-create-home api-user

# Establecer directorio de trabajo
WORKDIR /app

# Copiar las dependencias instaladas desde la etapa 'builder'
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copiar solo el código de la aplicación necesario desde la etapa 'builder'
# y asignarle el usuario 'api-user' como propietario
COPY --from=builder --chown=api-user:api-user /app/api ./api
COPY --from=builder --chown=api-user:api-user /app/configs ./configs
COPY --from=builder --chown=api-user:api-user /app/oficial_herdnet ./oficial_herdnet
COPY --from=builder --chown=api-user:api-user /app/pyproject.toml ./
COPY --from=builder --chown=api-user:api-user /app/README.md ./

# Cambiar al usuario no-root
USER api-user

# Exponer el puerto de la API
EXPOSE 8000

# Comando para iniciar la aplicación usando poe
CMD ["poe", "serve-frc"]