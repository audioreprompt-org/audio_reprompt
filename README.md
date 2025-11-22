# Audio Reprompt

Este repositorio contiene el codigo y artefactos de la aplicación Audio Reprompt.
Audio Reprompt es parte del trabajo de grado titulado: 
"Ingeniería de Prompts para la Generación de Música Sensorial: Un Enfoque Automático para la Percepción de Sabores."

## Integrantes

- Jorge Luis Sarmiento Herrera
- Valentina Nariño Chicaguy
- Jose Daniel Garcia Davila
- Juana Valentina Mendoza Santamaría
- Mario Fernando Reyes Ojeda

## Asesores

- Brayan Mauricio Rodriguez Rivera
- Felipe Reinoso Carvalho


## Setup

1. Instalar dependencias usando UV, se debe descargar usando el comando

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Instalar la version de Python que se utiliza en el proyecto (3.11) usando UV

```bash
uv python install 3.11
```

3. Instalar las dependencias del proyecto creando un ambiente virtual

```bash
uv venv
```

4. Configurar y activar el ambiente virtual usando el IDE que se utilice.

Nota: Algunos paquetes como torch o torchaudio no se incluyen en el lock de uv o los paquetes registrados
debido a que su distribucion varia de acuerdo al sistema operativo y arquitectura pero se pueden instalar
desde el terminal usando pip.

```bash
pip install torch torchaudio
```

5. Luego de instalar las dependencias, probablemente sea necesario cargar los datos que están en DVC, esto se hace
usando UV

```bash
uv run dvc pull 
```

6. Para usar jupyter

```bash
uv run --locked --with jupyter python -m notebook
```

## Módulo de Configuración

Este módulo de configuración que carga configuraciones desde un archivo YAML ubicado en la raíz del repositorio (`config.yaml`). El módulo de configuración proporciona acceso con seguridad de tipos a todas las configuraciones del proyecto para su uso en notebooks de Python y scripts.

Para más información haga clic [aquí](https://github.com/audioreprompt-org/audio_reprompt/wiki/Configuraci%C3%B3n#uso-b%C3%A1sico)

## Métricas

- **CLAP (texto–audio)** ✅ Implementada. Calcula la similitud coseno entre el embedding del audio y el del prompt.
- **FAD** ⏸️ No implementada en esta fase.
- **Aesthetics** ⏸️ No implementada en esta fase.

## ¿Cómo probar?

La aplicación puede ejecutarse en dos modalidades: **Mocked** (respuesta simulada, ideal para desarrollo y pruebas rápidas) y **Real** (generación con IA, que consume recursos en RunPod). A continuación se describen las formas de levantar el proyecto.


### 1. Despliegue del modelo de Reprompt:

Antes de construir la imagen de Docker para la API, la lógica del modelo debe compilarse en un paquete binario (Python Wheel). El archivo `Makefile` automatiza esta tarea.

``` bash
make make build-reprompt
```

Eso creará una distribución del modelo de audio-reprompt para ser consumido por el API

### 2. Ejecución con Docker 

Esta es la forma más sencilla de probar la integración completa (Frontend + Backend) en un entorno aislado.

1.  **Configurar variables de entorno:**
    Crea un archivo `.env` en la raíz del proyecto (basado en `docker-compose.yml`) o asegúrate de tener las variables exportadas en tu terminal. Necesitarás las claves API para el modo Real y el reprompting:

    ``` bash
    export RUNPOD_API_URL="tu_api_key_de_runpod" # Requerido solo para modo Real
    export RUNPOD_API_KEY="tu_api_key_de_runpod" # Requerido solo para modo Real
    export MOONSHOT_API_KEY="tu_api_key_del_llm" # Requerido para el refinamiento del prompt
    ```

2.  **Levantar los servicios:**
    
    Una vez localizado en la carpeta app

    ``` bash
    docker-compose up --build
    ```

3.  **Acceder a la aplicación:**

      - **Frontend:** Abre tu navegador en [http://localhost:5173](http://localhost:5173).
      - **Swagger API (Backend):** Puedes probar los endpoints directamente en [http://localhost:8000/docs](http://localhost:8000/docs).

### 3. Prueba directa de la API con cURL

Puedes probar el endpoint de generación de audio directamente enviando un prompt:

**Endpoint:** `POST http://localhost:8000/api/audio/generate`

**Ejemplo de petición:**

``` bash
    curl -X 'POST' \
      'http://localhost:8000/api/audio/generate' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
      "prompt": "I am eating a spicy taco in a mexican market with loud trumpet music playing in the background"
    }'
```

**Respuesta esperada:**
Recibirás un objeto JSON con:

  - `audio_id`: Un identificador único.
  - `improved_prompt`: El prompt enriquecido por el LLM.
  - `audio_base64`: La cadena en base64 del archivo de audio generado (listo para reproducir).


### 4. Despliegue del Módulo de Generación de musica en Runpod Serverless:

El módulo de generación de audio `(musicgen/)` está diseñado para ser desplegado como un worker en RunPod Serverless, lo que permite ejecutar la inferencia del modelo MusicGen de forma eficiente.

La carpeta `musicgen/` contiene el Dockerfile y el handler.py necesarios para configurar el entorno del modelo.

#### Desplegar musicgen

Para desplegar el worker de MusicGen y obtener su API Key y Endpoint, sigue estos pasos:

1. En la carpeta `app/musicgen/` construir la imagen:

``` bash
    docker build -t <usuario>/musicgen-worker:latest -f musicgen/Dockerfile .
```

2. Iniciar sesión en alguno de los registros de contenedores (Docker Hub, AWS ECR, etc.) y subir la imagen.

3.  Crear el Endpoint Serverless en RunPod:

      - El proceso detallado para crear el endpoint se encuentra en la documentación oficial de RunPod:
        **[Documentación de RunPod: Build your first worker](https://docs.runpod.io/serverless/workers/custom-worker)**
      - En la consola de RunPod Serverless, selecciona **'Import from Docker Registry'** e ingresa el nombre de la imagen que acabas de subir (ej: `<usuario>/musicgen-worker:latest`).
      - Configura los ajustes de hardware según sea necesario, una GPU con más de 4GB de VRAM es requerida.

4.  Obtener las Credenciales del Endpoint:
      - **Endpoint URL:** La URL de inferencia (ej: `https://api.runpod.ai/v2/xxxxxxxx/runsync`).
      - **API Key (rp\_...):** Tu clave de autorización para este endpoint.

## Licencia

MIT