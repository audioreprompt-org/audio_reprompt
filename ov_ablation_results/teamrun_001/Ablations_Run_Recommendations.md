# Recomendaciones de Ejecución para Verificar las Ablaciones B2 y B3

Documento complementario de [Ablations_Analysis.md](Ablations_Analysis.md). Define qué ejecuciones adicionales realizar y cómo evaluarlas tras adoptar $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`) como métrica primaria y habilitar el selector `--variants` en `models/ablation_runner.py`.

---

## 1. Diagnóstico actual por ablación y métrica

| Ablación | Resultado en $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`, Primaria) | Resultado en $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$ (Secundaria) | Acción recomendada |
|---|---|---|---|
| **B1** (`V1..V4`) | Sin diferencia entre versiones: medias `0.1192`, `0.1161`, `0.1112`, `0.1172` (Friedman $p = 0.240$, todos los pares $p_{\text{Holm}} = 1.0$). | **Fuerte a favor de V3**: `0.2246` vs. `0.1345–0.1462` ($\Delta \approx +0.08\text{–}0.09, p_{\text{Holm}} < 0.0001, d_z \approx 0.45\text{–}0.49$). | **No repetir B1.** Se mantiene **V3** porque es la única plantilla cuyas instrucciones musicales son seguidas de forma consistente por el generador de audio. |
| **B2** (Filtro vs. Sin filtro) | Sin diferencia significativa: `B2a (filtro) = 0.1142` vs. `B2b (sin filtro) = 0.1026` ($\Delta = -0.0116, p_t = 0.101, p_w = 0.088$). | Débil a favor de `B2b`: `0.2430` vs. `0.2048` ($\Delta = +0.0382, p_t = 0.038, d_z = 0.19$), del orden del ruido entre runs (`0.027`). | **Opcional / Baja prioridad (§3).** Si se busca confirmar si el filtro afecta la generación, correr 3 semillas adicionales solo con V3. |
| **B3** (Creatividad y `reasoning_effort`) | Sin diferencia significativa tras Holm (Friedman $p = 0.107$). `B3b (default)` obtiene la media más alta (`0.1213`), mientras que el prompt creativo `B3c` cae a `0.1061` ($p = 0.032$ sin corregir, $p_{\text{Holm}} = 0.161$). `B3f (none) = 0.1195`. | Moderada: `B3b (0.2313)` supera a las variantes deterministas `B3a (0.1798, p_Holm=0.021)` y `B3e (0.1876, p_Holm=0.036)`. `B3d` y `B3e` no aportan sobre `B3c` y `B3a`. | **Repetir con subconjunto reducido (`VARIANTS=core+aa`) (§4)** para ahorrar 33% de cómputo (o 17% incluyendo el control A/A `B3g`), descartando `B3d` y `B3e`. |

---

## 2. Consideración clave sobre la métrica primaria antes de lanzar nuevos runs

Al evaluar $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$, todas las variantes de B1, B2 y B3 se concentran en una banda estrecha de **0.103 a 0.121**, con diferencias entre variantes $|\Delta| \le 0.015$ ($d_z \le 0.20$).

- **Por qué ocurre esto**: El prompt original del usuario describe una escena de comida/bebida (*"I'm in the office with my partner and we're gonna drink a glass of tonic water..."*) y no contiene vocabulario musical. LAION-CLAP asigna una similitud coseno baja y casi constante ($\sim 0.11$) entre cualquier texto narrativo de comida y cualquier pista instrumental, independientemente de la variante de reprompt.
- **Implicación para la potencia estadística**:
  - Para detectar un efecto de $|\Delta| \approx 0.015$ ($d_z \approx 0.19$, como `B3c vs B3b` en `_cross.csv`) con $\alpha = 0.05$ corregido por Holm (3 contrastes) y 80% de potencia, se requieren **$N \approx 360\text{–}480$ observaciones pareadas** (es decir, **3 runs nuevos** de `sample_size=120` sumados al run existente de `seed=42`).
  - Para diferencias de $|\Delta| \le 0.008$ ($d_z \le 0.10$, como `B1` o `B3a/B3f vs B3b` en `_cross.csv`), se necesitarían más de 1 000 prompts, lo cual no tiene relevancia práctica.
- **Recomendación de criterio (Regla de Decisión Conjunta `a + b`)**:
  1. **(a) Comparación contra el audio baseline real (`data/tracks/raw_ablations`)**:
     - El preflight (`ensure_raw_baseline`) verifica `clap_score_results_raw_prompts_baseline.csv` primero, luego WAVs locales y finalmente `dvc pull` granular por ID. Si faltan baselines y no se pasa `GENERATE_RAW_AUDIO=1`, el runner falla rápido (`EXIT_BASELINE_MISSING=6`) antes de gastar cuota LLM o Kaggle.
     - Al pasar `GENERATE_RAW_AUDIO=1` (`--generate-raw-audio`), `RAW_AUDIO_SCOPE=sample` *(default)* genera solo los IDs faltantes de la muestra actual (mientras que `RAW_AUDIO_SCOPE=all` genera todos los prompts faltantes de `data/raw/user/*.csv`) en `data/tracks/raw_ablations/{id_prompt}.wav`, y `run_ablation_job.sh` publica `data/tracks` (acotado a `--scope data/tracks/raw_ablations`) únicamente cuando `raw_audio_generated > 0`.
     - Evaluar la ganancia neta de fidelidad $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$ vs. $\text{CLAP}(\text{prompt}, \text{audio}_{\text{prompt}})$ (`_raw.csv` con `source=raw_ablations`), junto con la métrica secundaria $\text{CLAP}(\text{prompt}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`).
  2. **(b) Evaluación conjunta $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$ + Diversidad NLP (`models/scripts/eval_reprompt_nlp.py`)**:
     - Maximizar $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$ por encima del piso de ruido A/A (`B3b − B3g`) verificando simultáneamente que no exista colapso de plantilla inter-prompt (`Distinct-2`, `Dist. Coseno Inter-Prompt`, bajo `Self-BLEU-4`), manteniendo alta riqueza léxica (`MATTR-15`) y densidad de vocabulario musical (`musical_vocab_density`).

---

## 3. Plan de ejecución para B2 (Filtro de dimensiones, solo V3)

**Variantes**: `B2a` (filtro `emotion, taste, texture`) y `B2b` (sin filtro), usando únicamente **`prompt_version="V3"`**.

| Run | Seed | Sample size | Audios generados | Estado |
|---|---|---|---|---|
| 1 | 42 | 120 | 240 | Completado (`R20261116_234152`) |
| 2 | 43 | 120 | 240 | Nuevo |
| 3 | 44 | 120 | 240 | Nuevo |
| 4 | 45 | 120 | 240 | Nuevo |

### Comandos (EC2 desatendido)

```bash
make ablation-ec2-run EXPERIMENT=B2 SEED=43 SAMPLE_SIZE=120
make ablation-ec2-run EXPERIMENT=B2 SEED=44 SAMPLE_SIZE=120
make ablation-ec2-run EXPERIMENT=B2 SEED=45 SAMPLE_SIZE=120
```

**Regla de decisión (sobre los 4 runs agrupados, $N=480$)**:
- Si en `_cross.csv` (`CLAP(prompt_original, audio_reprompt)`) la diferencia `B2b − B2a` sigue siendo nula o negativa (en el run 1 fue $-0.0116, p = 0.101$), **se mantiene `B2a` (filtro de producción)**, ya que incluir todas las dimensiones consume más tokens sin acercar el audio al prompt original del usuario.

---

## 4. Plan de ejecución para B3 con selección de variantes (`VARIANTS`)

Gracias a los cambios implementados en `models/ablation_runner.py`, `scripts/run_ablation_job.sh` y `Makefile`, no es necesario ejecutar las 6 variantes de B3. Se descartan `B3d` (`creativo + effort high`, la más lenta y costosa, con efecto nulo frente a `B3c`) y `B3e` (`determinista + effort low`, efecto nulo frente a `B3a`).

### 4.1 Presets disponibles en `--variants` / `VARIANTS`

| Preset / Valor | Variantes ejecutadas | Audios por run (`N=120`) | Ahorro vs. B3 completo (720 audios) |
|---|---|---|---|
| `VARIANTS=core` | `B3a, B3b, B3c, B3f` | 480 | **−33%** (y mayor ahorro en tiempo de LLM al omitir `B3d`) |
| `VARIANTS=core+aa` *(Recomendado)* | `B3a, B3b, B3c, B3f, B3g` | 600 | **−17%**, incluyendo el control A/A (`B3g`, réplica idéntica de `B3b`) |

### 4.2 Runs recomendados (solo V3)

| Run | Seed | Sample size | Variantes | Estado |
|---|---|---|---|---|
| 1 | 42 | 120 | `B3a..B3f` (6) | Completado (`R20261117_021911`) |
| 2 | 43 | 120 | `core+aa` (5) | Nuevo |
| 3 | 44 | 120 | `core+aa` (5) | Nuevo |
| 4 | 45 | 120 | `core+aa` (5) | Nuevo |

### Comandos (EC2 desatendido)

```bash
# Incluir GENERATE_RAW_AUDIO=1 cuando falten audios baseline (RAW_AUDIO_SCOPE=sample por defecto; o RAW_AUDIO_SCOPE=all para poblar todos los prompts crudos faltantes)
make ablation-ec2-run EXPERIMENT=B3 VARIANTS=core+aa SEED=43 SAMPLE_SIZE=120 GENERATE_RAW_AUDIO=1
make ablation-ec2-run EXPERIMENT=B3 VARIANTS=core+aa SEED=44 SAMPLE_SIZE=120 GENERATE_RAW_AUDIO=1
make ablation-ec2-run EXPERIMENT=B3 VARIANTS=core+aa SEED=45 SAMPLE_SIZE=120 GENERATE_RAW_AUDIO=1
```

*(Para ejecución local sin EC2: `make ablation-B3-full VARIANTS=core+aa SEED=43 SAMPLE_SIZE=120 GENERATE_RAW_AUDIO=1 RAW_AUDIO_SCOPE=sample`).*

### 4.3 Criterios de decisión para B3 (sobre los 4 runs agrupados, corrección de Holm)

1. **Piso de ruido intra-run (`B3b vs B3g`)**: La distribución de diferencias $|\text{B3b} - \text{B3g}|$ en los runs 2–4 define el ruido irreducible de generación (LLM + MusicGen). Ninguna diferencia menor o igual a este piso se interpreta como efecto real.
2. **Prompt creativo (`B3c vs B3b`)**: En el run 1, `B3c` mostró una caída sugestiva en `_cross.csv` ($\Delta = -0.0152, p = 0.032$ sin corregir, $p_{\text{Holm}} = 0.161$). Con $N=480$, si $\Delta < 0$ con $p_{\text{Holm}} < 0.05$, se confirma que forzar alta creatividad degrada la alineación con el prompt original.
3. **Prompt determinista (`B3a vs B3b`)**: Evaluar si se mantiene nulo en `_cross.csv` ($\Delta = -0.0019$) y significativamente inferior en `reprompt` ($\Delta = -0.0515$). En ambos casos, no aporta ventaja sobre `B3b`.
4. **Desactivar CoT (`B3f vs B3b`)**: En el run 1, `B3f` (`reasoning_effort=none`) fue idéntico a `B3b` en `_cross.csv` ($\Delta = -0.0018, p = 0.80$) y ligeramente menor en `reprompt` ($\Delta = -0.0384, p_{\text{Holm}} = 0.075$). Si en los 4 runs agrupados la diferencia en `_cross.csv` se mantiene dentro del margen de equivalencia $|\Delta| < 0.01$ (test TOST, $p < 0.05$), se puede adoptar `reasoning_effort=none` en producción para reducir latencia y costo de tokens sin perder alineación con el prompt del usuario.

---

## 5. Modelo estadístico para el análisis agrupado (Multi-Run)

Una vez completados los runs con `seed=43, 44, 45`, analizar los 4 runs conjuntamente mediante un **modelo lineal mixto** con efectos aleatorios cruzados por `run` y `id_prompt`:

$$\text{CLAP}_{ijk} = \beta_0 + \beta_{\text{variante}(i)} + u_{\text{run}(j)} + v_{\text{prompt}(k)} + \varepsilon_{ijk}$$

acompañado de contrastes pareados de Wilcoxon estratificados por `run` y corrección de Holm ($\alpha = 0.05$).
