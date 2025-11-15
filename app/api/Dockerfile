FROM python:3.11-slim

# Establecer la variable de entorno
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Crear el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias (asumiendo que tienes un requirements.txt en /app/api)
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API al directorio de trabajo
COPY . .

# Exponer el puerto
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]