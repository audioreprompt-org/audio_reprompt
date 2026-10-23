# Experimentos de Ablación

## Índice
- [Muestreo Estratificado](#muestreo-estratificado)
- [Arquitectura de Ejecución Remota (EC2 + Kaggle)](#arquitectura-de-ejecución-remota-ec2--kaggle)
  - [Flujo Desatendido End-to-End](#flujo-desatendido-end-to-end)
  - [Protección contra Costos y Guard de Inactividad](#protección-contra-costos-y-guard-de-inactividad)
- [Instrucciones para Ejecutar un Job](#instrucciones-para-ejecutar-un-job)
  - [Opción 1: On-Demand vía Tag de EC2 (Recomendada)](#opción-1-on-demand-vía-tag-de-ec2-recomendada)
  - [Opción 2: On-Demand vía AWS SSM (Sin SSH)](#opción-2-on-demand-vía-aws-ssm-sin-ssh)
  - [Opción 3: Ejecución Local / Workstation](#opción-3-ejecución-local--workstation)
- [Publicación Concurrente en DVC y Git](#publicación-concurrente-en-dvc-y-git)
- [Automatización y Aislamiento en Kaggle](#automatización-y-aislamiento-en-kaggle)
- [Ablaciones de Generación (MCU Re-Prompt)](#ablaciones-de-generación-mcu-re-prompt)
  - [B1. Versión de Prompt](#b1-versión-de-prompt)
  - [B2. Filtro de Dimensiones (Impacto en Audio)](#b2-filtro-de-dimensiones-impacto-en-audio)
  - [B3. Parámetros de Modelo (Penalizaciones y Creatividad)](#b3-parámetros-de-modelo-penalizaciones-y-creatividad)
- [Orden de Ejecución](#orden-de-ejecución)
- [Inputs del Análisis Estadístico](#inputs-del-análisis-estadístico)
- [Comandos Make](#comandos-make)
- [Run ID](#run-id)
- [Estructura de Archivos](#estructura-de-archivos)
- [Convención de Nombres](#convención-de-nombres)

Cada experimento aísla una variable y se evalúa con **CLAP score** (coseno texto↔audio) usando `calculate_clap_score_alignment()` de `models/validate.py`.  
Cada uno produce **tres** archivos con cálculos de score: reprompt↔audio, prompt-original↔audio y evaluación cruzada.

> [!NOTE]
> La evaluación de calidad de Recuperación de Información (IR) dentro del RAG fue actualizada y formalizada de manera independiente en [RAG_Evaluation.md](file:///Users/mario/Documents/re_make_things/repositories/audio_reprompt/RAG_Evaluation.md). Por tanto, las ablaciones descritas en este documento se concentran en la etapa de generación de reprompt y síntesis de audio (Componente B).

## Muestreo Estratificado

Todos los experimentos usan **muestreo estratificado por `taste`** (4 categorías: sweet, sour, salty, bitter).  
Un `seed` fijo garantiza que todas las variantes de un mismo experimento usen **el mismo subset de prompts**.

| Parámetro     | Default | Descripción |
|---------------|---------|-------------|
| `sample_size` | `30`    | Total de prompts (se divide entre categorías: 30 → 7 por taste = 28 efectivos) |
| `seed`        | `42`    | Semilla para reproducibilidad |

## Arquitectura de Ejecución Remota (EC2 + Kaggle)

El pipeline de ablaciones utiliza un modelo híbrido optimizado para costo y estabilidad:
* **EC2 Runner (`t3.xlarge` / Ubuntu 24.04)**: Actúa como orquestador desatendido. Ejecuta la generación de reprompts en PostgreSQL, el cálculo de CLAP scores en CPU, el análisis estadístico y la sincronización/publicación en DVC y Git.
* **Kaggle GPU (Tesla T4 x2)**: Ejecuta la síntesis de audio TTS (`csc-unipd/tasty-musicgen-small`) de forma remota y aislada sin costo adicional de GPU en la nube.
* **Almacenamiento y Control de Versiones**:
  * **DVC (S3)**: Almacena los audios `.wav` y los datasets intermedios bajo `s3://maia-music-reprompt/training`.
  * **Git**: Registra los punteros `.dvc` y los reportes de análisis estadístico mediante una estrategia de publicación optimista con reintentos para soportar múltiples ejecuciones concurrentes.

### Flujo Desatendido End-to-End

```
1. Tag / SSM Trigger
       │
       ▼
2. Encendido de la VM (EC2)
       │
       ▼
3. Boot Service (ablation-autostart.service)
       │  Lee el tag 'AblationExperiment' vía IMDSv2
       ▼
4. Sincronización y Validación (run_ablation_job.sh)
       │  Hard-sync con origin/ablations-experiments + uv sync --frozen
       │  Preflight de DB, Kaggle y permisos de git push
       ▼
5. Pipeline de Ablación (models.ablation_runner --phase full)
       │  Fase 1: Generación de reprompts (Postgres RDS)
       │  Fase 2: Kaggle GPU (dataset y kernel aislados por run_id)
       │  Fase 3: Cálculo de CLAP scores (CPU)
       │  Fase 4: Análisis estadístico y reportes
       ▼
6. Publicación Concurrente (ablation_publish.sh)
       │  dvc add data/ablations + dvc push
       │  git commit pointer (.dvc) + retry con backoff
       ▼
7. Apagado Automático (sudo shutdown -h now)
       │  La instancia pasa al estado 'stopped' (sin costo de cómputo)
```

### Protección contra Costos y Guard de Inactividad

Para evitar cobros accidentales por instancias olvidadas o procesos colgados:
1. **Apagado en Éxito y en Error (`run_ablation_job.sh`)**: Tanto si el pipeline termina exitosamente como si falla un preflight o fase, el trap del script ejecuta un `shutdown -h now` seguro y no reentrante.
2. **Fail-Safe de Systemd (`ExecStopPost`)**: Si el proceso es terminado por `TimeoutStartSec` (5 horas) o muere por OOM, systemd detecta `$SERVICE_RESULT != "success"` y fuerza el apagado.
3. **Idle Guard para Instancias "Zombie" (`ablation-idle-guard.timer`)**:
   * Corre cada 5 minutos.
   * Si no hay ningún servicio `ablation@*.service` activo y el tiempo de actividad (`uptime`) supera los **20 minutos**, apaga la VM automáticamente.
   * **Inmune a esperas de Kaggle**: Dado que monitorea el estado del servicio systemd y no la CPU, la VM nunca se apagará mientras esté esperando que Kaggle termine de sintetizar los audios.
   * **Escape Hatch manual**: Para sesiones de mantenimiento interactivo o depuración manual, se puede desactivar creando el archivo `/tmp/disable_idle_guard`.

---

## Instrucciones para Ejecutar un Job

### Opción 1: On-Demand vía Tag de EC2 (Recomendada)

Este método es 100% desatendido: no requiere conectarse a la máquina ni abrir terminales SSH.

1. **Asignar el tag del experimento deseado** a la instancia EC2 (`B1`, `B2`, `B3`):
   ```bash
   INSTANCE_ID="<id-de-la-instancia-runner>"
   aws ec2 create-tags --resources "$INSTANCE_ID" --tags Key=AblationExperiment,Value=B2
   ```

2. **Encender la instancia**:
   ```bash
   aws ec2 start-instances --instance-ids "$INSTANCE_ID"
   ```

3. **Monitoreo opcional** (la VM se apagará sola al terminar):
   * Ver logs en tiempo real vía SSM:
     ```bash
     aws ssm start-session --target "$INSTANCE_ID"
     # Dentro de la sesión:
     journalctl -u ablation@B2.service -f
     ```

---

### Opción 2: On-Demand vía AWS SSM (Sin SSH)

Si la instancia ya está encendida y se desea disparar un experimento directamente sin claves SSH:

```bash
aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["systemctl start --no-block ablation@B2.service"]'
```

---

### Opción 3: Ejecución Local / Workstation

Si ejecutas los experimentos en tu entorno local o en una VM dedicada de desarrollo:

```bash
# 1. Ejecutar job completo con publicación en DVC y git:
make ablation-job EXPERIMENT=B2

# 2. Ejecutar en local sin sincronizar rama ni publicar (desarrollo/pruebas):
./scripts/run_ablation_job.sh --experiment B2 --no-sync --no-publish

# 3. Personalizar muestra y semilla:
make ablation-job EXPERIMENT=B1 SAMPLE_SIZE=50 SEED=123
```

---

## Publicación Concurrente en DVC y Git

Cuando múltiples jobs o desarrolladores ejecutan ablaciones en paralelo, el archivo `data/ablations.dvc` puede generar conflictos de sincronización.  
El publicador concurrente (`scripts/ablation_publish.sh` y `models/ablation_publish.py`) resuelve esto de forma automática mediante concurrencia optimista:

1. Detecta archivos modificados bajo `data/ablations/` usando `dvc data status`.
2. Guarda los cambios en un stash temporal.
3. Se alinea con la punta remota de la rama (`git fetch` + `checkout`).
4. Descarga los activos remotos más recientes (`dvc pull --force`).
5. Restaura los archivos locales del stash (unión sin pérdidas).
6. Recalcula el hash de DVC (`dvc add data/ablations`) y sube los audios a S3 (`dvc push`).
7. Hace commit del archivo `.dvc` e intenta `git push`. Si detecta un conflicto por otro commit concurrente, aplica un backoff exponencial aleatorio y reintenta.

Para publicar manualmente un run existente:
```bash
make ablation-publish RUN_ID=R20261022_204000
```

---

## Automatización y Aislamiento en Kaggle

La fase de síntesis de audio está orquestada por `models/kaggle_runner.py`:

* **Aislamiento por Run (`KAGGLE_ISOLATE_RUNS=1`)**: Cada ejecución genera un dataset (`ablation-reprompts-<run_id>`) y un notebook (`ablation-audio-gen-<run_id>`) dedicados en Kaggle. Esto previene que ejecuciones paralelas sobreescriban los CSVs de entrada o los audios descargados.
* **Timeout Extendido**: Configurado a **4 horas** (`14400` segundos, configurable vía `KAGGLE_TIMEOUT`) para soportar lotes grandes de audio sin interrupciones.
* **Limpieza Automática**: Tras descargar los archivos `.wav`, los recursos temporales creados en Kaggle son eliminados automáticamente para no saturar la cuenta.

#### Comandos directos de Kaggle (Uso avanzado / depuración):
```bash
# Ejecutar pipeline directo en Kaggle (upload → poll → download):
make kaggle-run CSVS="data/ablations/reprompts/*.csv"

# Monitorear estado de ejecución:
uv run python -m models.kaggle_runner status --run-id <run_id>
```

---

## Evaluación manual / puntual de CSVs

Para auditar o evaluar archivos CSV ya existentes (sin pasar por todo el pipeline de generación):
```bash
# Calcular CLAP scores (requiere audios generados):
make ablation-score-csv CSV="data/ablations/reprompts/pipeline_results_*.csv"

# Calcular métricas NLP ligeras (diversidad léxica, entropía, legibilidad):
make ablation-nlp-csv CSV="data/ablations/reprompts/pipeline_results_*.csv"
```

---

## Ablaciones de Generación (MCU Re-Prompt)

Manejada por `mcu_reprompt()` en `models/music_curator/kimi_mcu.py`.

---

### B1. Versión de Prompt

**Objetivo**: Identificar qué template produce mayor alineación texto-audio.

**Variable**: `prompt_version` en `transform()` / `mcu_reprompt()`.

| Variante | Versión | Diferencias clave |
|----------|---------|-------------------|
| B1-a     | `V1`    | 3 reglas, salida en párrafo |
| B1-b     | `V2`    | 4 reglas: agrega mapeo color/human_response, temperatura→armonía |
| B1-c     | `V3`    | 5 reglas: restricción de orquestación a 30s, salida concisa |
| B1-d     | `V4`    | 5 reglas: solo descriptores musicales, dos oraciones separadas por coma |

#### Pre-requisitos
- PostgreSQL con tabla `crossmodal_food_embeddings` poblada.
- Modelo `all-MiniLM-L6-v2` disponible.
- API keys en `.env`: `MOONSHOT_API_KEY` y `OPENAI_API_KEY`.
- Prompts en `data/raw/user/raw_prompts.csv`.
- Las 4 versiones definidas en `models/music_curator/prompts.py`.

#### Pasos

**Opción A — Pipeline completo (un solo comando):**
```bash
make ablation-B1-full
# Ejecuta: reprompt → Kaggle GPU → CLAP score → análisis estadístico
# El run_id se genera automáticamente
```

**Opción B — Fases individuales:**
1. `make ablation-B1 PHASE=reprompt` → genera CSVs (imprime el `run_id`)
2. `make kaggle-run CSVS="data/ablations/reprompts/*_B1*_R<run_id>.csv"`
3. `make ablation-B1 PHASE=score --run-id R<run_id>`
4. `make ablation-analysis EXPERIMENT=B1 RUN=R<run_id>`

---

### B2. Filtro de Dimensiones (Impacto en Audio)

**Objetivo**: Evaluar cómo el filtro de dimensiones en los descriptores crossmodal afecta la calidad del audio generado evaluado mediante CLAP score texto↔audio.

**Variable**: `filter_dimensions` en `format_crossmodal_descriptors()` (`pipeline.py:30-34`).

| Variante | Filtro | Descripción |
|----------|--------|-------------|
| B2-a     | Solo `emotion`, `taste`, `texture` (actual) | Configuración de producción |
| B2-b     | Todas las dimensiones (sin filtro) | Todas las dimensiones disponibles |

#### Pre-requisitos
- Mismos que B1.

#### Pasos

**Opción A — Pipeline completo:**
```bash
make ablation-B2-full
```

**Opción B — Fases individuales:**
1. `make ablation-B2 PHASE=reprompt`
2. `make kaggle-run CSVS="data/ablations/reprompts/*_B2*_R<run_id>.csv"`
3. `make ablation-B2 PHASE=score --run-id R<run_id>`
4. `make ablation-analysis EXPERIMENT=B2 RUN=R<run_id>`

---

### B3. Parámetros de Modelo (Penalizaciones y Creatividad)

**Objetivo**: Encontrar la configuración que maximice la alineación sin sacrificar diversidad, ajustada para modelos avanzados que omiten temperature/top_p de forma predeterminada, como `kimi-k2-thinking`.

**Variable**: Nivel de creatividad en instrucciones de prompt y variaciones en parámetros de penalización (`presence_penalty` / `frequency_penalty`).

| Variante | Instrucción de Prompt (System) | Penalizaciones (presence/frequency) | Comportamiento esperado |
|----------|--------------------------------|-------------------------------------|-------------------------|
| B3-a     | "Sé determinista y conservador" | `0.0`, `0.0` | Conservador, limitando la entropía |
| B3-b     | *(Default)* | `0.0`, `0.0` | Balanceado (default `config.yaml`) |
| B3-c     | "Sé altamente creativo e imaginativo" | `0.0`, `0.0` | Creativo, mayor riqueza de vocabulario |
| B3-d     | *(Default)* | `0.6` (presence) | Promueve la introducción de nuevos términos |
| B3-e     | *(Default)* | `0.6` (frequency) | Reduce el uso reiterativo de mismos descriptores |

> [!IMPORTANT]
> Dado que hiperparámetros como `temperature`/`top_p` a menudo son ignorados en los modelos de estilo "reasoning/thinking" (como `kimi-k2-thinking`), la ablación B3 ahora explora vías alternativas (prompt engineering explícito y parámetros de penalización en la API) para medir su impacto riguroso en la generación del reprompt sin requerir `gpt-5-nano`.

#### Pre-requisitos
- Mismos que B1.

#### Pasos

**Opción A — Pipeline completo:**
```bash
make ablation-B3-full
```

**Opción B — Fases individuales:**
1. `make ablation-B3 PHASE=reprompt`
2. `make kaggle-run CSVS="data/ablations/reprompts/*_B3*_R<run_id>.csv"`
3. `make ablation-B3 PHASE=score --run-id R<run_id>`
4. `make ablation-analysis EXPERIMENT=B3 RUN=R<run_id>`

---

## Orden de Ejecución

```
B1 (versión de prompt) → B2 (filtro de dimensiones → audio) → B3 (penalizaciones y creatividad)
```
*(Nota: Para la evaluación y diagnóstico de recuperación en el RAG, referirse a [RAG_Evaluation.md](file:///Users/mario/Documents/re_make_things/repositories/audio_reprompt/RAG_Evaluation.md))*

## Inputs del Análisis Estadístico

El análisis (`ablation_analysis.py`) consume datos de **dos fuentes**, filtrados por `run_id`:

| Input | Ubicación | Datos que aporta |
|-------|-----------|------------------|
| Evaluación CSVs | `data/ablations/scores/` | CLAP scores (reprompt↔audio, original↔audio, evaluación cruzada) |
| Reprompt CSVs | `data/ablations/reprompts/` | Columna `taste` para desglose por categoría (sabor) |

El análisis genera:
- Estadísticas descriptivas y t-test pareado (raw vs reprompt)
- Cohen's d (tamaño del efecto)
- Desglose por categoría de taste
- Correlación Kendall τ
- Plots de distribución y comparación
- Reporte `hallazgos_<grupo>.md` en español

## Comandos Make

```bash
# Listar experimentos disponibles
make ablation-list

# ── Ejecución desatendida / orquestada (recomendada) ──
make ablation-job EXPERIMENT=B2              # corre pipeline completo + publica en DVC/git
make ablation-job EXPERIMENT=B1 SAMPLE_SIZE=50 SEED=42 SHUTDOWN=1

# ── Pipeline completo directo (sin publicación ni apagado) ──
make ablation-B2-full             # reprompt → Kaggle GPU → cálculo de score → análisis
make ablation-B1-full
make ablation-B3-full

# ── Publicación concurrente en DVC y git ──
make ablation-publish                                # publica los resultados pendientes
make ablation-publish RUN_ID=R20261022_204000        # publica con run_id específico

# ── Fases individuales (depuración avanzada) ──
make ablation-B2 PHASE=reprompt   # solo genera CSVs (con run ID automático)
make kaggle-run CSVS="data/ablations/reprompts/*_B2*.csv"  # Kaggle
make ablation-B2 PHASE=score --run-id R20260416_194531     # score con run específico
make ablation-analysis EXPERIMENT=B2 RUN=R20260416_194531  # análisis de un run

# Listar runs disponibles para análisis
make ablation-analysis-list

# Puntuar CSVs existentes directamente
make ablation-score-csv CSV="data/ablations/reprompts/mi_archivo.csv"  # CLAP Scores (requiere audios)
make ablation-nlp-csv CSV="data/ablations/reprompts/mi_archivo.csv"    # NLP metrics (diversidad, entropía, legibilidad)
```

## Run ID

Cada ejecución genera un **Run ID** automático con formato `RYYYYMMDD_HHMMSS` (ej: `R20260416_194531`).
Este ID se incluye en todos los nombres de archivo, lo que permite:
- Ejecutar el mismo experimento múltiples veces sin colisiones
- Analizar un run específico con `--run R20260416_194531`
- Listar runs disponibles con `make ablation-analysis-list`

## Estructura de Archivos

```
data/ablations/
├── reprompts/       # CSVs generados (Fase 1: reprompt)
│   ├── pipeline_results_*_B2a_filter_default_R20260416_194531.csv
│   ├── pipeline_results_*_B2b_nofilter_R20260416_194531.csv
│   └── ...
├── audio/           # WAVs descargados de Kaggle (Fase 2)
│   ├── pipeline_results_*_B2a_filter_default_R20260416_194531/
│   │   ├── 24.wav
│   │   └── ...
│   └── pipeline_results_*_B2b_nofilter_R20260416_194531/
│       └── ...
├── scores/          # CLAP scores (Fase 3)
│   ├── clap_score_results_reprompt_outputs_*_R20260416_194531.csv
│   ├── clap_score_results_prompt_outputs_*_R20260416_194531_raw.csv
│   └── clap_score_results_prompt_outputs_*_R20260416_194531_cross.csv
├── analysis/        # Reportes estadísticos y NLP
│   ├── nlp_R20260416_194531/
│   │   ├── nlp_eval_kimi_k2_thinking_..._R20260416_194531.md
│   │   └── nlp_eval_kimi_k2_thinking_..._R20260416_194531.csv
│   └── B2_R20260416_194531/
│       ├── hallazgos_B2.md
│       ├── summary_B2.csv
│       ├── comparison_B2.png
│       └── *.png
├── archive/         # Resultados anteriores archivados
│   └── ...
└── .cache/          # Cache de respuestas LLM (no versionado)
```

## Convención de Nombres

```
pipeline_results_<modelo>_<N>_prompt_<version>_<tag>_<run_id>.csv
```

Ejemplo: `pipeline_results_kimi_k2_thinking_28_prompt_V4_B2a_filter_default_R20260416_194531.csv`

