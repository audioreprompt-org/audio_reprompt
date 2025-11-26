# 📦 Audio Reprompt Model Wheels

Este directorio contiene los archivos de distribución de Python (`.whl`) generados a partir del código fuente ubicado en el directorio raíz `models/`.

## 🎯 Propósito
El objetivo de estos archivos es desacoplar la **Lógica de Negocio/Modelado** de la **Capa de API**. En lugar de copiar el código fuente crudo dentro del contenedor de la API, empaquetamos la lógica de `models` como una librería instalable. Esto garantiza que la API (`app/api`) utilice una versión inmutable y versionada del pipeline de procesamiento.

## ⚙️ Generación
Este archivo no debe modificarse manualmente. Se genera automáticamente ejecutando el comando make desde la raíz del proyecto:

    make build-reprompt

Este proceso realiza lo siguiente:
1. Limpia los builds anteriores en `models/dist`.
2. Ejecuta `uv build --wheel` dentro de `models/` para compilar el paquete.
3. Copia el archivo `.whl` resultante a esta carpeta (`app/api/wheels/`).

## 📦 Contenido del Paquete
Según la configuración definida en `models/pyproject.toml`, este wheel **NO contiene todo el código de models**, sino un subconjunto optimizado para la inferencia ligera en la API.

### ✅ Qué incluye (Lógica usada por la API):
La API importa principalmente `models.pipeline` (visto en `routes/audio.py`). Por lo tanto, el wheel incluye:
* **`models.pipeline`**: Contiene la función `transform(prompt)`. Es el núcleo del RAG y el refinamiento de prompts.
* **`config`**: Configuraciones compartidas necesarias para inicializar el pipeline.
* **Dependencias**: Define los requisitos como `openai`, `psycopg`, y `sentence-transformers` necesarios para procesar el texto.

### ❌ Qué excluye (Lógica pesada):
Para mantener el contenedor de la API ligero (basado en `python:3.11-slim`), se excluyen explícitamente los módulos pesados que requieren GPU o que solo se usan para entrenamiento/evaluación (definidos en el `exclude` del toml):
* `models.musicgen`: La generación de audio real ocurre en **RunPod** (GPU), no en la API local.
* `models.clap_score`: Métricas de evaluación.
* `tests`: Tests unitarios del modelo.

## 🚀 Instalación
Este archivo es consumido por el `Dockerfile` de la API durante la construcción:

    # Dockerfile
    COPY wheels/ /app/wheels/
    RUN pip install /app/wheels/*.whl

Esto permite importar el código en la API simplemente usando:

    from models.pipeline import transform
