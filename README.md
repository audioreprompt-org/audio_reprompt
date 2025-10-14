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

## Módulo de Configuración

Este módulo de configuración que carga configuraciones desde un archivo YAML ubicado en la raíz del repositorio (`config.yaml`). El módulo de configuración proporciona acceso con seguridad de tipos a todas las configuraciones del proyecto para su uso en notebooks de Python y scripts.

Para más información haga clic [aquí](https://github.com/audioreprompt-org/audio_reprompt/wiki/Configuraci%C3%B3n#uso-b%C3%A1sico)


## Licencia

MIT