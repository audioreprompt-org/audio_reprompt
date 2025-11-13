# /audio_reprompt/app/api/Dockerfile
# ------------------ STAGE 1: BACKEND ------------------
FROM python:3.11-slim

# Establecer la variable de entorno
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias (asumiendo que tienes un requirements.txt en /app/api)
# NOTA: Si no tienes requirements.txt, debes crearlo.
COPY requirements.txt .

# Instalar dependencias
# (Asegúrate de que tus dependencias como uvicorn, fastapi, etc., estén aquí)
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el código de la API al directorio de trabajo
# (Copia main.py, routes, models, clients, etc.)
COPY . .

# Exponer el puerto
EXPOSE 8000