# B1 — Hallazgos de la Ablación

> Generado automáticamente por `models/ablation_analysis.py`

> Variantes analizadas: B1a, B1b, B1c, B1d

## 1. Comparación entre Variantes — CLAP(prompt original ↔ audio del reprompt)

- **Test omnibus (Friedman, n=120)**: χ² = 4.210, p = 0.2397 (n.s.), Kendall W = 0.0117

| Par | Media A | Media B | Δ (A − B) | t | p (t) | p_Holm (t) | p_Holm (Wilcoxon) | d_z |
|-----|---------|---------|-----------|---|-------|------------|-------------------|-----|
| B1a vs B1b | 0.1192 | 0.1161 | +0.0031 | 0.4531 | 0.6513 | **1.0000** n.s. | 1.0000 | 0.0414 |
| B1a vs B1c | 0.1192 | 0.1112 | +0.0080 | 1.1996 | 0.2327 | **1.0000** n.s. | 1.0000 | 0.1095 |
| B1a vs B1d | 0.1192 | 0.1172 | +0.0019 | 0.2798 | 0.7801 | **1.0000** n.s. | 1.0000 | 0.0255 |
| B1b vs B1c | 0.1161 | 0.1112 | +0.0049 | 0.6742 | 0.5015 | **1.0000** n.s. | 1.0000 | 0.0615 |
| B1b vs B1d | 0.1161 | 0.1172 | -0.0012 | -0.1577 | 0.8750 | **1.0000** n.s. | 1.0000 | -0.0144 |
| B1c vs B1d | 0.1112 | 0.1172 | -0.0061 | -0.9353 | 0.3515 | **1.0000** n.s. | 1.0000 | -0.0854 |

**Hallazgo entre variantes**: Ningún par difiere significativamente tras corrección de Holm (p_Holm ≥ 0.05).


---

## 2. Estadísticas por Variante (reprompt vs. prompt original sobre el mismo audio)

| Métrica | B1a (V1 R20261107 151324) | B1b (V2 R20261107 151324) | B1c (V3 R20261107 151324) | B1d (V4 R20261107 151324) |
|---------|---|---|---|---|
| n | 120 | 120 | 120 | 120 |
| Media raw prompt | 0.1192 | 0.1161 | 0.1112 | 0.1172 |
| Media reprompt | 0.1345 | 0.1410 | 0.2246 | 0.1462 |
| Diferencia media | +0.0153 (+12.8%) | +0.0250 (+21.5%) | +0.1134 (+102.0%) | +0.0290 (+24.7%) |
| Estadístico t | -1.0105 | -1.7184 | -7.5149 | -1.9399 |
| Valor p | **0.3143** n.s. | **0.0883** n.s. | **0.0000** *** | **0.0548** n.s. |
| Cohen's d | **0.1209** | **0.1900** | **0.8852** | **0.2270** |
| Casos que mejoraron | 49.2% | 54.2% | 75.0% | 58.3% |

**Hallazgo**: 1 de 4 variantes con mejora significativa (p < 0.05, sin corrección por comparaciones múltiples). Mayor efecto: B1c (V3 R20261107 151324) (d = 0.89; d promedia las varianzas de raw y reprompt, el d pareado es menor).

> Este reporte compara cada variante contra su prompt original, no variantes entre sí. Para contrastes entre variantes con Holm, ver `Ablations_Analysis.md`.


---

## 3. Análisis por Categoría de Taste

### B1a (V1 R20261107 151324)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| salty | 30 | 0.1182 | 0.1523 | +0.0341 | +28.9% | 0.2901 n.s. | 60.0% |
| sour | 30 | 0.1034 | 0.1215 | +0.0181 | +17.5% | 0.5544 n.s. | 46.7% |
| bitter | 30 | 0.1368 | 0.1545 | +0.0178 | +13.0% | 0.5330 n.s. | 50.0% |
| sweet | 30 | 0.1183 | 0.1095 | -0.0088 | -7.4% | 0.7861 n.s. | 40.0% |

### B1b (V2 R20261107 151324)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.1049 | 0.1662 | +0.0613 | **+58.5%** | **0.0384** * | 60.0% |
| sweet | 30 | 0.1248 | 0.1500 | +0.0253 | +20.2% | 0.3408 n.s. | 53.3% |
| bitter | 30 | 0.1416 | 0.1508 | +0.0093 | +6.5% | 0.7635 n.s. | 50.0% |
| salty | 30 | 0.0930 | 0.0971 | +0.0040 | +4.3% | 0.8989 n.s. | 53.3% |

### B1c (V3 R20261107 151324)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| bitter | 30 | 0.1131 | 0.2507 | +0.1376 | **+121.6%** | **0.0005** *** | 73.3% |
| sweet | 30 | 0.1040 | 0.2188 | +0.1148 | **+110.3%** | **0.0001** *** | 76.7% |
| sour | 30 | 0.0983 | 0.2123 | +0.1139 | **+115.8%** | **0.0023** ** | 70.0% |
| salty | 30 | 0.1292 | 0.2166 | +0.0874 | **+67.7%** | **0.0031** ** | 80.0% |

### B1d (V4 R20261107 151324)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.1124 | 0.1779 | +0.0655 | +58.2% | 0.0554 n.s. | 73.3% |
| salty | 30 | 0.0989 | 0.1307 | +0.0317 | +32.1% | 0.2563 n.s. | 53.3% |
| bitter | 30 | 0.1391 | 0.1638 | +0.0246 | +17.7% | 0.4602 n.s. | 53.3% |
| sweet | 30 | 0.1185 | 0.1124 | -0.0061 | -5.1% | 0.8146 n.s. | 53.3% |


---

## 4. Correlación Kendall τ por Taste

| Taste | B1a (V1 R20261107 151324) τ | B1b (V2 R20261107 151324) τ | B1c (V3 R20261107 151324) τ | B1d (V4 R20261107 151324) τ |
|-------|---|---|---|---|
| bitter | -0.035 (Débil) | 0.177 (Débil) | 0.154 (Débil) | -0.011 (Débil) |
| salty | 0.103 (Débil) | 0.209 (Débil) | 0.145 (Débil) | 0.094 (Débil) |
| sour | 0.205 (Débil) | 0.283 (Débil) (p=0.03) | 0.103 (Débil) | 0.053 (Débil) |
| sweet | 0.058 (Débil) | 0.218 (Débil) | 0.195 (Débil) | 0.264 (Débil) (p=0.04) |

**Interpretación**: Correlaciones débiles indican que el reprompt transforma sustancialmente la representación texto→audio respecto al prompt original.


---

## Visualizaciones

![comparison_B1](comparison_B1.png)

![dist_gpt_5.6_luna_120_prompt_V1_B1a_V1_R20261107_151324](dist_gpt_5.6_luna_120_prompt_V1_B1a_V1_R20261107_151324.png)

![dist_gpt_5.6_luna_120_prompt_V2_B1b_V2_R20261107_151324](dist_gpt_5.6_luna_120_prompt_V2_B1b_V2_R20261107_151324.png)

![dist_gpt_5.6_luna_120_prompt_V3_B1c_V3_R20261107_151324](dist_gpt_5.6_luna_120_prompt_V3_B1c_V3_R20261107_151324.png)

![dist_gpt_5.6_luna_120_prompt_V4_B1d_V4_R20261107_151324](dist_gpt_5.6_luna_120_prompt_V4_B1d_V4_R20261107_151324.png)

![taste_gpt_5.6_luna_120_prompt_V1_B1a_V1_R20261107_151324](taste_gpt_5.6_luna_120_prompt_V1_B1a_V1_R20261107_151324.png)

![taste_gpt_5.6_luna_120_prompt_V2_B1b_V2_R20261107_151324](taste_gpt_5.6_luna_120_prompt_V2_B1b_V2_R20261107_151324.png)

![taste_gpt_5.6_luna_120_prompt_V3_B1c_V3_R20261107_151324](taste_gpt_5.6_luna_120_prompt_V3_B1c_V3_R20261107_151324.png)

![taste_gpt_5.6_luna_120_prompt_V4_B1d_V4_R20261107_151324](taste_gpt_5.6_luna_120_prompt_V4_B1d_V4_R20261107_151324.png)
