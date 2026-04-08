# Experimentos de Ablación

Cada experimento aísla una variable y se evalúa con **CLAP score** (coseno texto↔audio) usando `calculate_clap_score_alignment()` de `models/validate.py`.  
Cada uno produce **tres** archivos de scores: reprompt↔audio, prompt-original↔audio y evaluación cruzada.

---

## Componente A — Etapa de Retrieval

### A1. Heurística de Corte (Cross-Modal)

**Objetivo**: Evaluar si `cut_crossmodal_results` mejora la calidad del re-prompt.

**Variable**: `cut_results` en `get_top_k_food_descriptors()` (`rag.py:43`).

| Variante | `cut_results` |
|----------|---------------|
| A1-a     | `True` (actual) |
| A1-b     | `False` |

#### Pre-requisitos
- PostgreSQL con tabla `crossmodal_food_embeddings` poblada.
- Modelo `all-MiniLM-L6-v2` disponible.
- API keys en `.env`: `MOONSHOT_API_KEY` / `OPENAI_API_KEY`.
- Prompts en `data/raw/user/raw_prompts.csv`.
- Audios generados en `data/tracks/reprompt_audios/` y `data/tracks/raw_prompts_audios/`.
- Pesos del modelo CLAP descargados.

#### Pasos
1. Fijar `cut_results=True` en `pipeline.py:55` → `generate_reprompts(KIMI_K2_THINKING_MODEL, "V4")`.
2. Ejecutar CLAP scoring (3 modos) sobre el CSV generado.
3. Repetir con `cut_results=False`.
4. Comparar scores medios/medianas entre A1-a y A1-b.

---

### A2. Top-K de Captions de Audio

**Objetivo**: Determinar el `k` óptimo de captions musicales recuperados.

**Variable**: `k` en `get_top_k_audio_captions()` (`pipeline.py:75`).

| Variante | `k` |
|----------|-----|
| A2-a     | `5` |
| A2-b     | `10` (actual) |
| A2-c     | `20` |

#### Pre-requisitos
- Mismos que A1.
- Tabla `audio_descriptors` poblada en PostgreSQL.

#### Pasos
1. Modificar `k` en `pipeline.py:75` → generar reprompts.
2. Renombrar CSV de salida incluyendo el valor de `k`.
3. Ejecutar CLAP scoring (3 modos).
4. Repetir para cada valor de `k` y comparar distribuciones.

---

### A3. Filtro de Dimensiones en Descriptores

**Objetivo**: Evaluar el impacto de restringir `format_crossmodal_descriptors()` a solo `emotion`, `taste` y `texture`.

**Variable**: Filtro de dimensiones en `pipeline.py:30-34`.

| Variante | Filtro |
|----------|--------|
| A3-a     | Solo `emotion`, `taste`, `texture` (actual) |
| A3-b     | Todas las dimensiones (sin filtro) |

#### Pre-requisitos
- Mismos que A1.

#### Pasos
1. **A3-a**: ejecutar con filtro actual.
2. **A3-b**: remover condición `if` en línea 33, generar reprompts.
3. Ejecutar CLAP scoring (3 modos) para ambas variantes y comparar.

---

## Componente B — Etapa de Generación (MCU Re-Prompt)

Manejada por `mcu_reprompt()` en `models/music_curator/kimi_mcu.py`.  
Parámetros clave: versión de prompt, modelo LLM y parámetros de sampling.

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
1. Ejecutar `generate_reprompts(KIMI_K2_THINKING_MODEL, "<VERSION>")` para V1–V4.
2. CLAP scoring (3 modos) por cada CSV.
3. Comparar scores entre versiones.

---

### B2. Modelo LLM

**Objetivo**: Medir cómo la elección de LLM impacta la calidad del re-prompt.

**Variable**: `model` en `transform()` / `mcu_reprompt()`.

| Variante | Modelo |
|----------|--------|
| B2-a     | `kimi-k2-thinking-turbo` (Moonshot API) |
| B2-b     | `gpt-5-nano` (OpenAI API) |

#### Pre-requisitos
- Mismos que A1.
- **Ambas** API keys configuradas: `MOONSHOT_API_KEY` y `OPENAI_API_KEY`.

#### Pasos
1. Fijar prompt version (usar mejor de B1 o `V4`).
2. Ejecutar `generate_reprompts(<MODELO>, "V4")` para cada modelo.
3. CLAP scoring (3 modos) y comparar scores, latencia y costo.

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
> `kimi_mcu.py:46` actualmente **no pasa** `temperature` ni `top_p` al API.
> Se requiere modificar `mcu_reprompt()` antes de ejecutar esta ablación.

#### Pre-requisitos
- Mismos que A1.
- **Cambio de código requerido**: agregar `temperature` y `top_p` a la firma de `mcu_reprompt()` y propagarlos por `transform()` y `generate_reprompts()`:
  ```python
  def mcu_reprompt(..., temperature: float = 0.7, top_p: float = 0.9) -> str:
      response = get_client(model).chat.completions.create(
          model=model, messages=messages, temperature=temperature, top_p=top_p,
      )
  ```

#### Pasos
1. Aplicar cambio de código descrito arriba.
2. Generar reprompts para cada combinación (temperatura, top_p).
3. CLAP scoring (3 modos) y comparar distribuciones + inspección cualitativa.

---

## Orden de Ejecución

```
Fase 1 — Retrieval
  A1 (heurística de corte) → A3 (filtro de dimensiones) → A2 (top-k)

Fase 2 — Generación (usar mejor config de Fase 1)
  B1 (versión de prompt) → B2 (modelo) → B3 (sampling)
```

## Registro de Resultados

| Experimento | Variante | CLAP (reprompt↔audio) | CLAP (raw↔audio) | CLAP (cruzada) | Notas |
|-------------|----------|----------------------|-------------------|----------------|-------|
| A1          | A1-a     |                      |                   |                |       |
| A1          | A1-b     |                      |                   |                |       |
| ...         | ...      |                      |                   |                |       |

## Convención de Nombres

```
pipeline_results_<modelo>_<N>_prompt_<version>_<id_experimento>.csv
```

Ejemplo: `pipeline_results_kimi_k2_thinking_turbo_80_prompt_V4_A2c_k20.csv`
