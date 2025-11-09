# 1. Usar una imagen  de Python
FROM python:3.13-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Instalar uv, el gestor de paquetes del proyecto
RUN pip install uv

# 4. Copiar los archivos de dependencias
COPY pyproject.toml uv.lock ./

# 5. Instalar las dependencias del dashboard usando uv
RUN uv pip install --system --extra dashboard

# 6. Copiar el resto del código de la aplicación del dashboard
COPY dashboard/ ./dashboard/

# 7. Exponer el puerto que Streamlit usa por defecto
EXPOSE 8501

# 8. Definir el comando para correr la aplicación cuando se inicie el contenedor
CMD ["streamlit", "run", "dashboard/streamlit_app.py"]
