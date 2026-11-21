# Análisis Estadístico de las Ablaciones de Generación (B1, B2, B3)

Este documento analiza los resultados de las ablaciones descritas en [Ablations.md](Ablations.md) con una recomputación independiente ([ov_ablation_results/recompute_stats.py](ov_ablation_results/recompute_stats.py)) a partir de los archivos de score en `data/ablations/scores/`, y documenta las mejoras aplicadas a `models/ablation_runner.py`, `models/clap_score/model.py` y `models/ablation_analysis.py`.

- **Métrica primaria (decisoria)**: $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`), que evalúa qué tan bien el audio generado a partir del reprompt se alinea con la petición original del usuario.
- **Métrica secundaria (diagnóstica)**: $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$, que evalúa qué tan fielmente el generador de audio (`tasty-musicgen-small`) siguió las instrucciones textuales del reprompt.
- **Muestra**: 120 prompts por variante (30 por categoría de `taste`: *bitter, salty, sour, sweet*), `seed=42`. Los mismos 120 prompts se evalúan en todas las variantes de un experimento (**diseño de medidas repetidas / pareado**).
- **Runs analizados**: B1 = `R20261107_151324`, B2 = `R20261116_234152`, B3 = `R20261117_021911`. Modelo `gpt_5.6_luna`.
- **Integridad de datos**: 120/120 filas completas por variante, sin duplicados ni `NaN`, alineadas 1:1 por `id_prompt`.

---

## 1. Metodología estadística y arquitectura de scoring

### 1.1 Qué mide cada archivo de score (y por qué se eliminó `_raw.csv`)

En Kaggle (`models/kaggle_runner.py`), solo se sintetiza audio a partir de la columna `reprompt` ($\text{audio}_{\text{reprompt}}$). Anteriormente, `run_score()` invocaba `calculate_clap_score_alignment()` tres veces pasando siempre `audio_dir=exp_audio_dir`:
1. **Default** (`clap_score_results_reprompt_outputs_<tag>.csv`): $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$.
2. **`using_raw_prompts=True`** (`..._raw.csv`): $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$.
3. **`cross_evaluation=True`** (`..._cross.csv`): $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$.

Las opciones 2 y 3 calculaban exactamente la misma métrica sobre el mismo texto y el mismo audio, pero diferían ligeramente (~0.012 de diferencia absoluta media, $r \approx 0.986$) porque `laion_clap` con `enable_fusion=True` realiza recortes aleatorios (*random sub-window cropping*) en audios de 30 s cuando no se fija la semilla antes de cada pasada.

**Correcciones aplicadas en código**:
- Se fijó `set_reproducibility(42)` dentro de `ClapModel.calculate_scores()` (`models/clap_score/model.py`) para garantizar determinismo total.
- Se eliminó la llamada redundante `using_raw_prompts=True` en `models/ablation_runner.py`, conservando únicamente `reprompt` y `cross_evaluation=True` (`_cross.csv`), ahorrando 1/3 del tiempo de CPU en la Fase 3.
- `models/ablation_analysis.py` carga `_cross.csv` como métrica del prompt original (con fallback automático a `_raw.csv` para runs antiguos) y ejecuta comparaciones entre variantes sobre $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$.

### 1.2 Tests estadísticos utilizados

| Pregunta | Test paramétrico | Test no paramétrico (robustez) | Tamaño del efecto / IC |
|---|---|---|---|
| ¿Difieren las variantes entre sí en $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$? (omnibus, $k>2$) | ANOVA de medidas repetidas (con corrección de Greenhouse-Geisser $\varepsilon$) | Test de Friedman ($\chi^2_{k-1}$) | Kendall $W = \chi^2 / (n(k-1))$ |
| ¿Qué pares de variantes difieren? (post-hoc o vs. control) | $t$ de Student pareado sobre $d_i = y_{i,A} - y_{i,B}$, ajustado por **Holm** | Wilcoxon signed-rank ajustado por **Holm** | Cohen's $d_z = \bar{d} / s_d$ e IC95% bootstrap (10 000 remuestreos) |
| Diagnóstico intra-variante: ¿$\text{CLAP}(\text{reprompt}, \text{audio}) > \text{CLAP}(\text{prompt\_original}, \text{audio})$? | $t$ pareado ajustado por Holm | Wilcoxon signed-rank ajustado por Holm | $d_z$ (pareado) y $d_{av}$ (varianza agrupada, reportado en `hallazgos_*.md`) |

---

## 2. B1 — Versión de prompt (V1, V2, V3, V4)

**Hipótesis principal ($H_{1}$)**: Las plantillas de prompt (`V1..V4`) difieren en la alineación del audio generado con el prompt original del usuario, $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$.

### 2.1 Comparación primaria entre variantes: $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`)

- **Medias por variante ($n=120$)**:
  - `B1a (V1)`: **0.1192** ($s = 0.0966$)
  - `B1b (V2)`: **0.1161** ($s = 0.0989$)
  - `B1c (V3)`: **0.1112** ($s = 0.1009$)
  - `B1d (V4)`: **0.1172** ($s = 0.0970$)
- **Tests omnibus ($n=120$)**:
  - **Friedman**: $\chi^2(3) = 4.210$, $p = 0.2397$ (n.s.), Kendall $W = 0.0117$.
  - **ANOVA RM**: $F(3, 357) = 0.351$, $p = 0.788$ (Greenhouse-Geisser $\varepsilon = 0.963$, $p_{\text{GG}} = 0.781$).

| Par | Media A | Media B | $\Delta$ (A − B) | $t$ | $p$ ($t$) | $p_{\text{Holm}}$ ($t$) | $p_{\text{Holm}}$ (Wilcoxon) | $d_z$ |
|---|---|---|---|---|---|---|---|---|
| B1a (V1) vs B1b (V2) | 0.1192 | 0.1161 | +0.0031 | +0.4531 | 0.6513 | **1.0000** n.s. | 1.0000 | +0.0414 |
| B1a (V1) vs B1c (V3) | 0.1192 | 0.1112 | +0.0080 | +1.1996 | 0.2327 | **1.0000** n.s. | 1.0000 | +0.1095 |
| B1a (V1) vs B1d (V4) | 0.1192 | 0.1172 | +0.0019 | +0.2798 | 0.7801 | **1.0000** n.s. | 1.0000 | +0.0255 |
| B1b (V2) vs B1c (V3) | 0.1161 | 0.1112 | +0.0049 | +0.6742 | 0.5015 | **1.0000** n.s. | 1.0000 | +0.0615 |
| B1b (V2) vs B1d (V4) | 0.1161 | 0.1172 | −0.0012 | −0.1577 | 0.8750 | **1.0000** n.s. | 1.0000 | −0.0144 |
| B1c (V3) vs B1d (V4) | 0.1112 | 0.1172 | −0.0061 | −0.9353 | 0.3515 | **1.0000** n.s. | 1.0000 | −0.0854 |

**Inferencia primaria (B1)**: **No se rechaza $H_{1,0}$**. Medida contra el prompt original del usuario, ninguna versión de prompt difiere significativamente ($p_{\text{Holm}} = 1.0$ en los 6 pares, $|d_z| \le 0.11$). Todas las versiones se sitúan alrededor de $\mu \approx 0.11\text{–}0.12$.

### 2.2 Métrica diagnóstica secundaria: $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$

Cuando se evalúa la alineación del audio con el **propio texto del reprompt**:
- **Omnibus**: ANOVA RM $F(3, 357) = 14.05, p = 1.4 \times 10^{-8}$; Friedman $\chi^2(3) = 32.65, p = 3.8 \times 10^{-7}$, $W = 0.091$.

| Variante | Media `_cross` (prompt orig.) | Media `reprompt` | $\Delta$ (rep − orig) | $t$ | $p$ ($t$) | $p_{\text{Holm}}$ ($t$) | $d_z$ | $d_{av}$ | Mejoraron |
|---|---|---|---|---|---|---|---|---|---|
| B1a (V1) | 0.1192 | 0.1345 | +0.0153 | −1.0105 | 0.3143 | 0.3143 n.s. | 0.09 | 0.12 | 49.2% |
| B1b (V2) | 0.1161 | 0.1410 | +0.0250 | −1.7184 | 0.0883 | 0.1767 n.s. | 0.16 | 0.19 | 54.2% |
| B1c (V3) | 0.1112 | **0.2246** | **+0.1134** | −7.5149 | **<0.0001** | **<0.0001** *** | **0.69** | **0.89** | **75.0%** |
| B1d (V4) | 0.1172 | 0.1462 | +0.0290 | −1.9399 | 0.0548 | 0.1643 n.s. | 0.18 | 0.23 | 58.3% |

**Interpretación conjunta (B1)**: `V3` (restricción de orquestación a 30 s y salida concisa) genera descripciones musicales que `tasty-musicgen-small` y CLAP acoplan mucho mejor con el texto del reprompt (+0.078 a +0.090 sobre V1/V2/V4, $p_{\text{Holm}} < 0.0001$), pero esa mayor fidelidad reprompt→audio no incrementa la similitud coseno directa con la frase narrativa original del usuario (que describe comida y entorno, no música).

---

## 3. B2 — Filtro de dimensiones (V3, `emotion/taste/texture` vs. todas)

**Hipótesis principal ($H_{2}$)**: Eliminar el filtro de dimensiones crossmodales (`B2b`) modifica la alineación $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ respecto al filtro de producción (`B2a`).

### 3.1 Comparación primaria entre variantes: $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`)

| Par ($n=120$) | Media B2a (filtro) | Media B2b (sin filtro) | $\Delta$ (B2a − B2b) | $t$ | $p$ ($t$) | $p$ (Wilcoxon) | $d_z$ | IC95% de $\Delta$ |
|---|---|---|---|---|---|---|---|---|
| B2a vs B2b | **0.1142** | **0.1026** | +0.0116 | +1.6538 | **0.1008** n.s. | **0.0882** n.s. | +0.1510 | [−0.002, +0.025] |

**Inferencia primaria (B2)**: **No se rechaza $H_{2,0}$** ($p_t = 0.101, p_w = 0.088$). Sobre la métrica de alineación con el prompt original, quitar el filtro de dimensiones no produce mejora; de hecho, la media puntual sin filtro es ligeramente menor (0.1026 vs. 0.1142, $d_z = -0.15$ en dirección B2b − B2a).

### 3.2 Métrica diagnóstica secundaria: $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$

| Variante | Media `_cross` (prompt orig.) | Media `reprompt` | $\Delta$ (rep − orig) | $t$ | $p$ ($t$) | $d_z$ | $d_{av}$ | Mejoraron |
|---|---|---|---|---|---|---|---|---|
| B2a (filtro default) | 0.1142 | 0.2048 | +0.0906 | −5.6139 | <0.0001 *** | 0.51 | 0.71 | 70.8% |
| B2b (sin filtro) | 0.1026 | 0.2430 | +0.1404 | −8.9092 | <0.0001 *** | 0.81 | 1.06 | 80.0% |

En el score del reprompt, `B2b` supera a `B2a` por $+0.0382$ ($p_t = 0.038, p_w = 0.032, d_z = 0.19$), pero esa diferencia es pequeña, del mismo orden que la variación entre réplicas idénticas de V3 default (`B1c = 0.2246`, `B2a = 0.2048`, `B3b = 0.2313`), y no se traduce en mejor alineación con el prompt original.

---

## 4. B3 — Reasoning effort y creatividad (6 variantes, V3)

**Hipótesis principal ($H_{3}$)**: Las variaciones en instrucciones de creatividad y `reasoning_effort` producen diferencias en $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ respecto a la configuración por defecto (`B3b`).

### 4.1 Comparación primaria entre variantes: $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (`_cross.csv`)

- **Tests omnibus ($n=120, k=6$)**:
  - **Friedman**: $\chi^2(5) = 9.062$, $p = 0.1066$ (n.s.), Kendall $W = 0.0151$.
  - **ANOVA RM**: $F(5, 595) = 1.158$, $p = 0.3285$ (Greenhouse-Geisser $\varepsilon = 0.941$, $p_{\text{GG}} = 0.329$).

Contrastes pareados contra el control `B3b (default)` con corrección de Holm (5 contrastes):

| Contraste vs. Control (`B3b`) | Media Variante | Media `B3b` | $\Delta$ (Var − B3b) | $t$ | $p$ ($t$) | $p_{\text{Holm}}$ ($t$) | $p_{\text{Holm}}$ (Wilcoxon) | $d_z$ |
|---|---|---|---|---|---|---|---|---|
| B3a (determinista) vs B3b | 0.1194 | 0.1213 | −0.0019 | −0.2463 | 0.8058 | **1.0000** n.s. | 0.9875 | −0.0225 |
| B3c (creativo) vs B3b | 0.1061 | 0.1213 | −0.0152 | −2.1663 | 0.0323 | **0.1614** n.s. | 0.1832 | −0.1978 |
| B3d (creativo + high) vs B3b | 0.1084 | 0.1213 | −0.0129 | −1.8288 | 0.0699 | **0.2797** n.s. | 0.1832 | −0.1669 |
| B3e (determinista + low) vs B3b | 0.1131 | 0.1213 | −0.0082 | −1.1161 | 0.2666 | **0.7999** n.s. | 0.3447 | −0.1019 |
| B3f (effort `none`) vs B3b | 0.1195 | 0.1213 | −0.0018 | −0.2565 | 0.7980 | **1.0000** n.s. | 0.9396 | −0.0234 |

**Inferencia primaria (B3)**: **No se rechaza $H_{3,0}$**. El test omnibus no es significativo ($p = 0.107$) y ningún contraste contra `B3b` sobrevive a la corrección de Holm ($p_{\text{Holm}} \ge 0.161$). La variante `B3b` (default) obtiene la media numérica más alta (`0.1213`), mientras que los prompts creativos (`B3c = 0.1061`, `B3d = 0.1084`) tienden a alejar ligeramente el audio del prompt original ($p = 0.032$ sin corregir en `B3c`, no significativo tras Holm). Desactivar el CoT (`B3f = 0.1195`) es indistinguible del default ($\Delta = -0.0018, p = 0.80$).

### 4.2 Métrica diagnóstica secundaria: $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$

| Variante | Media `_cross` (prompt orig.) | Media `reprompt` | $\Delta$ (rep − orig) | $t$ | $p_{\text{Holm}}$ ($t$) | $d_z$ | $d_{av}$ | Contrastes vs. B3b en `reprompt` ($p_{\text{Holm}}$) |
|---|---|---|---|---|---|---|---|---|
| B3a (determinista) | 0.1194 | 0.1798 | +0.0603 | −3.6490 | 0.0004 *** | 0.33 | 0.45 | −0.0515 (**0.021** *) |
| B3b (default, control) | **0.1213** | **0.2313** | +0.1100 | −7.1956 | <0.0001 *** | 0.66 | 0.91 | Control |
| B3c (creativo) | 0.1061 | 0.2208 | +0.1147 | −6.9385 | <0.0001 *** | 0.63 | 0.83 | −0.0105 (1.000 n.s.) |
| B3d (creativo + high) | 0.1084 | 0.2248 | +0.1164 | −6.7884 | <0.0001 *** | 0.62 | 0.82 | −0.0065 (1.000 n.s.) |
| B3e (determinista + low) | 0.1131 | 0.1876 | +0.0745 | −4.5989 | <0.0001 *** | 0.42 | 0.58 | −0.0437 (**0.036** *) |
| B3f (effort `none`) | 0.1195 | 0.1928 | +0.0734 | −4.4882 | <0.0001 *** | 0.41 | 0.58 | −0.0384 (0.075 n.s.) |

En el score del reprompt, `B3b (default)` también lidera (`0.2313`), superando significativamente a los prompts deterministas (`B3a`, `B3e`).

---

## 5. Síntesis global y explicación del techo en $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$

| Ablación | Resultado en $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ (Primaria) | Resultado en $\text{CLAP}(\text{reprompt}, \text{audio}_{\text{reprompt}})$ (Secundaria) | Decisión práctica |
|---|---|---|---|
| **B1** | Sin diferencia entre V1..V4 (Friedman $p=0.24$, $\mu \in [0.111, 0.119]$). | **V3 >> V1, V2, V4** ($\Delta \approx +0.08\text{–}0.09, p_{\text{Holm}} < 0.0001$). | Mantener **V3**. |
| **B2** | Sin diferencia (`B2a=0.1142` vs `B2b=0.1026`, $p=0.101$). | `B2b > B2a` ($\Delta = +0.038, p = 0.038, d_z = 0.19$, del orden del ruido entre runs). | Mantener **B2a (filtro default)** por simplicidad, o verificar con réplicas si interesa el score secundario. |
| **B3** | Sin diferencia significativa tras Holm (Friedman $p=0.107$; `B3b` tiene la media más alta `0.1213`). | **B3b (default) > deterministas (`B3a, B3e`)** ($p_{\text{Holm}} \le 0.036$). Creativo y `effort=high` no aportan. | Mantener **B3b (default)**; descartar `B3d` y `B3e`. |

### ¿Por qué $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$ ronda siempre $\sim 0.11\text{–}0.12$?
Los prompts originales de `data/raw/user/raw_prompts.csv` son descripciones cotidianas de comida y contexto no musical (p. ej. *"I'm in the office with my partner and we're gonna drink a glass of tonic water. The desk is cluttered and the phone is ringing."*), mientras que el audio generado es una pista instrumental de `tasty-musicgen-small`. En el espacio latente de LAION-CLAP, la similitud coseno directa entre una oración sobre comida/oficina y una pieza musical tiene una sensibilidad muy baja (suelo de $\sim 0.11$), porque el puente semántico comida→música ocurre a través de la traducción crossmodal del reprompt, no por coincidencia léxica directa entre la escena y el audio.

---

## 6. Estado de las mejoras en el código de evaluación y análisis

1. **Resuelto — Determinismo en CLAP (`models/clap_score/model.py`)**: Se añadió `set_reproducibility(42)` en `ClapModel.calculate_scores()` para eliminar la variabilidad por recorte aleatorio en modo fusión.
2. **Resuelto — Eliminación de pasada redundante (`models/ablation_runner.py`)**: `run_score()` ahora calcula únicamente el score del reprompt y `cross_evaluation=True` (`_cross.csv`), eliminando `using_raw_prompts=True`.
3. **Resuelto — Selección de subconjuntos de variantes (`models/ablation_runner.py`, `Makefile`, `scripts/run_ablation_job.sh`)**: Se añadió `--variants` (`VARIANTS=...`) con soporte de listas y presets (`core`, `core+aa`) y la variante `B3g` como control A/A.
4. **Resuelto — Comparación entre variantes y reporte corregido (`models/ablation_analysis.py`)**:
   - Usa `_cross.csv` como score del prompt original (con fallback a `_raw.csv`).
   - Añade la Sección 1 en `hallazgos_<grupo>.md` con el test de Friedman y todos los contrastes pareados ($t$ y Wilcoxon) corregidos por Holm sobre $\text{CLAP}(\text{prompt\_original}, \text{audio}_{\text{reprompt}})$, exportando además `pairwise_cross_<grupo>.csv`.
   - Elimina el texto fijo erróneo (`"d > 1.0"` y `"Todas las variantes..."`) y reporta el conteo real de variantes significativas junto con las aclaraciones metodológicas.

## 7. Reproducibilidad

```bash
# Recomputar tablas completas (primaria _cross y secundaria reprompt)
.venv/bin/python ov_ablation_results/recompute_stats.py

# Regenerar reportes hallazgos_B*.md y CSVs de contrastes pareados
.venv/bin/python -m models.ablation_analysis --experiment B1 --run R20261107_151324
.venv/bin/python -m models.ablation_analysis --experiment B2 --run R20261116_234152
.venv/bin/python -m models.ablation_analysis --experiment B3 --run R20261117_021911
```
