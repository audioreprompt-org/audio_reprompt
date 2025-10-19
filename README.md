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

### Cómo correr la evaluación

1. Una vez:

```bash
uv run -m models.scripts.evaluate
```

2. Estadisticas:

```bash
uv run -m pytest -m integration metrics/tests/test_clap_backends.py -q
```

Esto:
- Lee la configuración desde config.yaml.
- Busca los prompts y empareja los audios por id.
- Ejecuta las métricas activas y deja los resultados en `data/scores/`

## Licencia

MIT