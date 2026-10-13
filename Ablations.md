# Experimentos de Ablación

## Índice
- [Muestreo Estratificado](#muestreo-estratificado)
- [Flujos de Ejecución](#flujos-de-ejecución)
  - [Generación (Componente B) — tres fases con Kaggle](#generación-componente-b--tres-fases-con-kaggle)
  - [Automatización Kaggle](#automatización-kaggle)
  - [Evaluación manual de archivos CSV](#evaluación-manual-de-archivos-csv)
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

## Flujos de Ejecución

### Generación (Componente B) — tres fases con Kaggle

Las ablaciones de generación requieren GPU para TTS y se evalúan con CLAP scores:

```
Fase 1 (local):  make ablation-B1 PHASE=reprompt                              → genera CSVs
Fase 2 (GPU):    make kaggle-run CSVS="data/ablations/reprompts/*_B1*.csv"     → audio en Kaggle
Fase 3 (local):  make ablation-B1 PHASE=score                                 → cálculo de CLAP scores
```

### Automatización Kaggle

La Fase 2 está automatizada mediante `models/kaggle_runner.py`, que:

1. **Sube** los CSVs de reprompts como un dataset Kaggle (`mfreyeso/ablation-reprompts`)
2. **Genera y empuja** un notebook que ejecuta TTS en GPU con `csc-unipd/tasty-musicgen-small`
3. **Monitorea** la ejecución del kernel cada 60s
4. **Descarga** los archivos `.wav` a `data/ablations/audio/<csv_stem>/`

#### Pre-requisitos Kaggle
- Kaggle CLI instalado: `pip install kaggle`
- Token en `~/.kaggle/kaggle.json` (Kaggle > Account > API > Create New API Token)

#### Comandos
```bash
# Flujo completo (upload → poll → download)
make kaggle-run CSVS="data/ablations/reprompts/*.csv"

# Pasos individuales
make kaggle-upload CSVS="data/ablations/reprompts/pipeline_results_*.csv"
make kaggle-status
make kaggle-download
```

### Evaluación manual de CSVs

También se pueden puntuar archivos CSV directamente (ya sea con CLAP de audio o con métricas NLP ligeras de texto como diversidad léxica, entropía y legibilidad):
```bash
make ablation-score-csv CSV="data/ablations/reprompts/pipeline_results_*.csv"
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

# ── Generación: pipeline completo (un solo comando) ──
make ablation-B2-full             # reprompt → Kaggle GPU → cálculo de score → análisis
make ablation-B1-full
make ablation-B3-full

# ── Generación: fases individuales ──
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

