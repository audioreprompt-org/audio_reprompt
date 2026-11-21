# B3 — Hallazgos de la Ablación

> Generado automáticamente por `models/ablation_analysis.py`

> Variantes analizadas: B3a, B3b, B3c, B3d, B3e, B3f

## 1. Comparación entre Variantes — CLAP(prompt original ↔ audio del reprompt)

- **Test omnibus (Friedman, n=120)**: χ² = 9.062, p = 0.1066 (n.s.), Kendall W = 0.0151

| Par | Media A | Media B | Δ (A − B) | t | p (t) | p_Holm (t) | p_Holm (Wilcoxon) | d_z |
|-----|---------|---------|-----------|---|-------|------------|-------------------|-----|
| B3a vs B3b | 0.1194 | 0.1213 | -0.0019 | -0.2463 | 0.8058 | **1.0000** n.s. | 0.9875 | -0.0225 |
| B3c vs B3b | 0.1061 | 0.1213 | -0.0152 | -2.1663 | 0.0323 | **0.1614** n.s. | 0.1832 | -0.1978 |
| B3d vs B3b | 0.1084 | 0.1213 | -0.0129 | -1.8288 | 0.0699 | **0.2797** n.s. | 0.1832 | -0.1669 |
| B3e vs B3b | 0.1131 | 0.1213 | -0.0082 | -1.1161 | 0.2666 | **0.7999** n.s. | 0.3447 | -0.1019 |
| B3f vs B3b | 0.1195 | 0.1213 | -0.0018 | -0.2565 | 0.7980 | **1.0000** n.s. | 0.9396 | -0.0234 |

**Hallazgo entre variantes**: Ningún par difiere significativamente tras corrección de Holm (p_Holm ≥ 0.05).


---

## 2. Estadísticas por Variante (reprompt vs. prompt original sobre el mismo audio)

| Métrica | B3a (prompt determinista R20261117 021911) | B3b (default pen0 R20261117 021911) | B3c (prompt creativo R20261117 021911) | B3d (creativo effort high R20261117 021911) | B3e (determinista effort low R20261117 021911) | B3f (random effort none R20261117 021911) |
|---------|---|---|---|---|---|---|
| n | 120 | 120 | 120 | 120 | 120 | 120 |
| Media raw prompt | 0.1194 | 0.1213 | 0.1061 | 0.1084 | 0.1131 | 0.1195 |
| Media reprompt | 0.1798 | 0.2313 | 0.2208 | 0.2248 | 0.1876 | 0.1928 |
| Diferencia media | +0.0603 (+50.5%) | +0.1100 (+90.7%) | +0.1147 (+108.1%) | +0.1164 (+107.4%) | +0.0745 (+65.9%) | +0.0734 (+61.4%) |
| Estadístico t | -3.6490 | -7.1956 | -6.9385 | -6.7884 | -4.5989 | -4.4882 |
| Valor p | **0.0004** *** | **0.0000** *** | **0.0000** *** | **0.0000** *** | **0.0000** *** | **0.0000** *** |
| Cohen's d | **0.4520** | **0.9077** | **0.8300** | **0.8220** | **0.5769** | **0.5827** |
| Casos que mejoraron | 63.3% | 74.2% | 75.8% | 74.2% | 65.0% | 67.5% |

**Hallazgo**: 6 de 6 variantes con mejora significativa (p < 0.05, sin corrección por comparaciones múltiples). Mayor efecto: B3b (default pen0 R20261117 021911) (d = 0.91; d promedia las varianzas de raw y reprompt, el d pareado es menor).

> Este reporte compara cada variante contra su prompt original, no variantes entre sí. Para contrastes entre variantes con Holm, ver `Ablations_Analysis.md`.


---

## 3. Análisis por Categoría de Taste

### B3a (prompt determinista R20261117 021911)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.1118 | 0.2051 | +0.0933 | **+83.5%** | **0.0041** ** | 70.0% |
| salty | 30 | 0.1028 | 0.1751 | +0.0723 | +70.3% | 0.0552 n.s. | 66.7% |
| sweet | 30 | 0.1056 | 0.1490 | +0.0433 | +41.0% | 0.2424 n.s. | 56.7% |
| bitter | 30 | 0.1575 | 0.1899 | +0.0324 | +20.6% | 0.2845 n.s. | 60.0% |

### B3b (default pen0 R20261117 021911)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.0975 | 0.2254 | +0.1279 | **+131.2%** | **0.0002** *** | 76.7% |
| salty | 30 | 0.1119 | 0.2347 | +0.1228 | **+109.7%** | **0.0002** *** | 83.3% |
| bitter | 30 | 0.1478 | 0.2565 | +0.1087 | **+73.6%** | **0.0030** ** | 66.7% |
| sweet | 30 | 0.1280 | 0.2085 | +0.0805 | **+62.9%** | **0.0131** * | 70.0% |

### B3c (prompt creativo R20261117 021911)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sweet | 30 | 0.1216 | 0.2526 | +0.1310 | **+107.7%** | **0.0001** *** | 80.0% |
| sour | 30 | 0.0917 | 0.2104 | +0.1188 | **+129.6%** | **0.0011** ** | 73.3% |
| bitter | 30 | 0.1177 | 0.2273 | +0.1096 | **+93.1%** | **0.0144** * | 73.3% |
| salty | 30 | 0.0936 | 0.1929 | +0.0994 | **+106.2%** | **0.0012** ** | 76.7% |

### B3d (creativo effort high R20261117 021911)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.1000 | 0.2536 | +0.1535 | **+153.5%** | **0.0000** *** | 83.3% |
| bitter | 30 | 0.1335 | 0.2576 | +0.1241 | **+92.9%** | **0.0052** ** | 76.7% |
| salty | 30 | 0.0900 | 0.2072 | +0.1172 | **+130.3%** | **0.0012** ** | 73.3% |
| sweet | 30 | 0.1100 | 0.1808 | +0.0709 | **+64.4%** | **0.0488** * | 63.3% |

### B3e (determinista effort low R20261117 021911)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.0940 | 0.2078 | +0.1139 | **+121.2%** | **0.0008** *** | 73.3% |
| salty | 30 | 0.1074 | 0.1990 | +0.0916 | **+85.2%** | **0.0254** * | 63.3% |
| bitter | 30 | 0.1223 | 0.2068 | +0.0845 | **+69.1%** | **0.0043** ** | 70.0% |
| sweet | 30 | 0.1286 | 0.1369 | +0.0083 | +6.4% | 0.7863 n.s. | 53.3% |

### B3f (random effort none R20261117 021911)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| salty | 30 | 0.0967 | 0.2178 | +0.1211 | **+125.3%** | **0.0028** ** | 73.3% |
| sour | 30 | 0.0933 | 0.1858 | +0.0925 | **+99.2%** | **0.0011** ** | 76.7% |
| bitter | 30 | 0.1539 | 0.2095 | +0.0557 | +36.2% | 0.1038 n.s. | 60.0% |
| sweet | 30 | 0.1341 | 0.1583 | +0.0242 | +18.0% | 0.4677 n.s. | 60.0% |

**Hallazgo**: La(s) categoría(s) **sour** muestra(n) mejora significativa (p < 0.05) en **todas** las variantes.


---

## 4. Correlación Kendall τ por Taste

| Taste | B3a (prompt determinista R20261117 021911) τ | B3b (default pen0 R20261117 021911) τ | B3c (prompt creativo R20261117 021911) τ | B3d (creativo effort high R20261117 021911) τ | B3e (determinista effort low R20261117 021911) τ | B3f (random effort none R20261117 021911) τ |
|-------|---|---|---|---|---|---|
| bitter | 0.131 (Débil) | 0.016 (Débil) | 0.062 (Débil) | 0.007 (Débil) | 0.159 (Débil) | -0.002 (Débil) |
| salty | 0.053 (Débil) | 0.044 (Débil) | 0.214 (Débil) | 0.241 (Débil) | -0.108 (Débil) | -0.021 (Débil) |
| sour | 0.113 (Débil) | 0.117 (Débil) | 0.053 (Débil) | 0.140 (Débil) | 0.030 (Débil) | 0.062 (Débil) |
| sweet | 0.021 (Débil) | -0.039 (Débil) | -0.025 (Débil) | -0.016 (Débil) | 0.048 (Débil) | -0.030 (Débil) |

**Interpretación**: Correlaciones débiles indican que el reprompt transforma sustancialmente la representación texto→audio respecto al prompt original.


---

## Visualizaciones

![comparison_B3](comparison_B3.png)

![dist_gpt_5.6_luna_120_prompt_V3_B3a_prompt_determinista_R20261117_021911](dist_gpt_5.6_luna_120_prompt_V3_B3a_prompt_determinista_R20261117_021911.png)

![dist_gpt_5.6_luna_120_prompt_V3_B3b_default_pen0_R20261117_021911](dist_gpt_5.6_luna_120_prompt_V3_B3b_default_pen0_R20261117_021911.png)

![dist_gpt_5.6_luna_120_prompt_V3_B3c_prompt_creativo_R20261117_021911](dist_gpt_5.6_luna_120_prompt_V3_B3c_prompt_creativo_R20261117_021911.png)

![dist_gpt_5.6_luna_120_prompt_V3_B3d_creativo_effort_high_R20261117_021911](dist_gpt_5.6_luna_120_prompt_V3_B3d_creativo_effort_high_R20261117_021911.png)

![dist_gpt_5.6_luna_120_prompt_V3_B3e_determinista_effort_low_R20261117_021911](dist_gpt_5.6_luna_120_prompt_V3_B3e_determinista_effort_low_R20261117_021911.png)

![dist_gpt_5.6_luna_120_prompt_V3_B3f_random_effort_none_R20261117_021911](dist_gpt_5.6_luna_120_prompt_V3_B3f_random_effort_none_R20261117_021911.png)

![taste_gpt_5.6_luna_120_prompt_V3_B3a_prompt_determinista_R20261117_021911](taste_gpt_5.6_luna_120_prompt_V3_B3a_prompt_determinista_R20261117_021911.png)

![taste_gpt_5.6_luna_120_prompt_V3_B3b_default_pen0_R20261117_021911](taste_gpt_5.6_luna_120_prompt_V3_B3b_default_pen0_R20261117_021911.png)

![taste_gpt_5.6_luna_120_prompt_V3_B3c_prompt_creativo_R20261117_021911](taste_gpt_5.6_luna_120_prompt_V3_B3c_prompt_creativo_R20261117_021911.png)

![taste_gpt_5.6_luna_120_prompt_V3_B3d_creativo_effort_high_R20261117_021911](taste_gpt_5.6_luna_120_prompt_V3_B3d_creativo_effort_high_R20261117_021911.png)

![taste_gpt_5.6_luna_120_prompt_V3_B3e_determinista_effort_low_R20261117_021911](taste_gpt_5.6_luna_120_prompt_V3_B3e_determinista_effort_low_R20261117_021911.png)

![taste_gpt_5.6_luna_120_prompt_V3_B3f_random_effort_none_R20261117_021911](taste_gpt_5.6_luna_120_prompt_V3_B3f_random_effort_none_R20261117_021911.png)
