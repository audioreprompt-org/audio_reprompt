# Experimentos de Ablación

Cada experimento aísla una variable y se evalúa con **CLAP score** (coseno texto↔audio) usando `calculate_clap_score_alignment()` de `models/validate.py`.  
Cada uno produce **tres** archivos de scores: reprompt↔audio, prompt-original↔audio y evaluación cruzada.

## Muestreo Estratificado

Todos los experimentos usan **muestreo estratificado por `taste`** (4 categorías: sweet, sour, salty, bitter).  
Un `seed` fijo garantiza que todas las variantes de un mismo experimento usen **el mismo subset de prompts**.

| Parámetro     | Default | Descripción |
|---------------|---------|-------------|
| `sample_size` | `30`    | Total de prompts (se divide entre categorías: 30 → 7 por taste = 28 efectivos) |
| `seed`        | `42`    | Semilla para reproducibilidad |

## Flujos de Ejecución

### Retrieval (Componente A) — solo reprompts

Las ablaciones de retrieval generan únicamente CSVs de reprompts para análisis textual (comparación entre modelos y configuraciones). **No requieren audio ni CLAP scoring.**

```
make ablation-A1                              → genera CSVs de reprompts
make ablation-retrieval                       → ejecuta todos los A-series
```

### Generation (Componente B) — tres fases con Kaggle

Las ablaciones de generación requieren GPU para TTS y se evalúan con CLAP scores:

```
Fase 1 (local):  make ablation-B1 PHASE=reprompt                              → genera CSVs
Fase 2 (GPU):    make kaggle-run CSVS="data/ablations/reprompts/*_B1*.csv"     → audio en Kaggle
Fase 3 (local):  make ablation-B1 PHASE=score                                 → CLAP scores
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

### Scoreo manual de CSVs

También se pueden puntuar CSVs directamente:
```bash
make ablation-score-csv CSV="data/ablations/reprompts/pipeline_results_*.csv"
```

---

## Componente A — Etapa de Retrieval

> Cada variante de retrieval se ejecuta con **ambos modelos** (`kimi-k2-thinking` y `gpt-5-nano`) para tener métricas comparativas.

### A1. Heurística de Corte (Cross-Modal)

**Objetivo**: Evaluar si `cut_crossmodal_results` mejora la calidad del re-prompt.

**Variable**: `cut_results` en `get_top_k_food_descriptors()` (`rag.py:43`).

| Variante | `cut_results` | Modelos |
|----------|---------------|---------|
| A1-a     | `True` (actual) | kimi-k2, gpt-5-nano |
| A1-b     | `False` | kimi-k2, gpt-5-nano |

#### Pre-requisitos
- PostgreSQL con tabla `crossmodal_food_embeddings` poblada.
- Modelo `all-MiniLM-L6-v2` disponible.
- API keys en `.env`: `MOONSHOT_API_KEY` y `OPENAI_API_KEY`.
- Prompts en `data/raw/user/raw_prompts.csv`.

#### Pasos
1. `make ablation-A1` → genera reprompts con ambos modelos. Se asigna un `run_id` automático.
2. Comparar los CSVs generados entre A1-a y A1-b, por modelo (métricas textuales).

---

### A2. Top-K de Captions de Audio

**Objetivo**: Determinar el `k` óptimo de captions musicales recuperados.

**Variable**: `k` en `get_top_k_audio_captions()` (`pipeline.py:75`).

| Variante | `k` | Modelos |
|----------|-----|---------|
| A2-a     | `5` | kimi-k2, gpt-5-nano |
| A2-b     | `10` (actual) | kimi-k2, gpt-5-nano |
| A2-c     | `50` | kimi-k2, gpt-5-nano |

#### Pre-requisitos
- Mismos que A1.
- Tabla `audio_descriptors` poblada en PostgreSQL.

#### Pasos
1. `make ablation-A2`
2. Comparar distribuciones de reprompts entre k=5, k=10, k=50, por modelo.

---

### A3. Filtro de Dimensiones en Descriptores

**Objetivo**: Evaluar el impacto de restringir `format_crossmodal_descriptors()` a solo `emotion`, `taste` y `texture`.

**Variable**: Filtro de dimensiones en `pipeline.py:30-34`.

| Variante | Filtro | Modelos |
|----------|--------|---------|
| A3-a     | Solo `emotion`, `taste`, `texture` (actual) | kimi-k2, gpt-5-nano |
| A3-b     | Todas las dimensiones (sin filtro) | kimi-k2, gpt-5-nano |

#### Pre-requisitos
- Mismos que A1.

#### Pasos
1. `make ablation-A3`
2. Comparar reprompts entre A3-a y A3-b, por modelo (métricas textuales).

---

## Componente B — Etapa de Generación (MCU Re-Prompt)

Manejada por `mcu_reprompt()` en `models/music_curator/kimi_mcu.py`.  
El modelo a usar se decide a partir de los resultados de las ablaciones de retrieval (Componente A).

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
- Mismos que A1.
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

**Objetivo**: Evaluar cómo el filtro de dimensiones en los descriptores crossmodal afecta la calidad del audio generado. Complementa A3 (que evalúa a nivel textual) con evaluación CLAP texto↔audio.

**Variable**: `filter_dimensions` en `format_crossmodal_descriptors()` (`pipeline.py:30-34`).

| Variante | Filtro | Diferencia con A3 |
|----------|--------|--------------------|
| B2-a     | Solo `emotion`, `taste`, `texture` (actual) | Evaluado con CLAP score |
| B2-b     | Todas las dimensiones (sin filtro) | Evaluado con CLAP score |

> [!NOTE]
> A3 evalúa el mismo parámetro a nivel textual (comparación de reprompts). B2 evalúa su impacto en la calidad del **audio generado** mediante CLAP scores.

#### Pre-requisitos
- Mismos que A1.

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

### B3. Parámetros de Sampling (Temperature & top_p)

**Objetivo**: Encontrar la combinación que maximice alineación sin sacrificar diversidad.

**Variable**: `temperature` y `top_p` en la llamada a chat completions.

| Variante | Temperature | top_p | Comportamiento esperado |
|----------|------------|-------|-------------------------|
| B3-a     | `0.3`      | `0.9` | Conservador, consistente |
| B3-b     | `0.7`      | `0.9` | Balanceado (default `config.yaml`) |
| B3-c     | `1.0`      | `0.9` | Creativo, más diverso |
| B3-d     | `0.7`      | `0.5` | Nucleus más estrecho |
| B3-e     | `0.7`      | `1.0` | Nucleus completo |

> [!IMPORTANT]
> Los parámetros `temperature`/`top_p` se ignoran automáticamente para modelos con `"thinking"` en su nombre.
> B3 solo tiene efecto con `gpt-5-nano`.

#### Pre-requisitos
- Mismos que A1.

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
Fase 1 — Retrieval (cada variante × 2 modelos, solo reprompts)
  A1 (heurística de corte) → A3 (filtro de dimensiones) → A2 (top-k)

Fase 2 — Generación (usar mejor config y modelo de Fase 1)
  B1 (versión de prompt) → B2 (filtro de dimensiones → audio) → B3 (sampling)
```

## Inputs del Análisis Estadístico

El análisis (`ablation_analysis.py`) consume datos de **dos fuentes**, filtrados por `run_id`:

| Input | Ubicación | Datos que aporta |
|-------|-----------|------------------|
| Score CSVs | `data/ablations/scores/` | CLAP scores (reprompt↔audio, raw↔audio, cruzada) |
| Reprompt CSVs | `data/ablations/reprompts/` | Columna `taste` para desglose por categoría |

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

# ── Retrieval (solo reprompts, sin PHASE) ──
make ablation-A1
make ablation-retrieval           # ejecuta A1 + A2 + A3

# ── Generation: pipeline completo (un solo comando) ──
make ablation-B2-full             # reprompt → Kaggle GPU → score → análisis
make ablation-B1-full
make ablation-B3-full

# ── Generation: fases individuales ──
make ablation-B2 PHASE=reprompt   # solo genera CSVs (con run ID auto)
make kaggle-run CSVS="data/ablations/reprompts/*_B2*.csv"  # Kaggle
make ablation-B2 PHASE=score --run-id R20260416_194531     # score con run específico
make ablation-analysis EXPERIMENT=B2 RUN=R20260416_194531  # análisis de un run

# Listar runs disponibles para análisis
make ablation-analysis-list

# Puntuar CSVs existentes directamente
make ablation-score-csv CSV="data/ablations/reprompts/mi_archivo.csv"
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
├── analysis/        # Reportes estadísticos (Fase 4)
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

