FROM python:3.11-slim

# Establecer la variable de entorno
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/weights/all-MiniLM-L6-v2

# Crear el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar el archivo de dependencias e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la distribución e instalar
COPY wheels/ /app/wheels/
RUN pip install  /app/wheels/*.whl


# Descargar modelo para evitar descargarlo en ejecución
RUN python -c "from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    model.save('${MODEL_DIR}')"

# Copiar el código de la API al directorio de trabajo
COPY . .

# Exponer el puerto
EXPOSE 8000

# Entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
