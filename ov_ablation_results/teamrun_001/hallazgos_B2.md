# B2 — Hallazgos de la Ablación

> Generado automáticamente por `models/ablation_analysis.py`

> Variantes analizadas: B2a, B2b

## 1. Comparación entre Variantes — CLAP(prompt original ↔ audio del reprompt)

| Par | Media A | Media B | Δ (A − B) | t | p (t) | p_Holm (t) | p_Holm (Wilcoxon) | d_z |
|-----|---------|---------|-----------|---|-------|------------|-------------------|-----|
| B2a vs B2b | 0.1142 | 0.1026 | +0.0116 | 1.6538 | 0.1008 | **0.1008** n.s. | 0.0882 | 0.1510 |

**Hallazgo entre variantes**: Ningún par difiere significativamente tras corrección de Holm (p_Holm ≥ 0.05).


---

## 2. Estadísticas por Variante (reprompt vs. prompt original sobre el mismo audio)

| Métrica | B2a (filter default R20261116 234152) | B2b (nofilter R20261116 234152) |
|---------|---|---|
| n | 120 | 120 |
| Media raw prompt | 0.1142 | 0.1026 |
| Media reprompt | 0.2048 | 0.2430 |
| Diferencia media | +0.0906 (+79.3%) | +0.1404 (+136.9%) |
| Estadístico t | -5.6139 | -8.9092 |
| Valor p | **0.0000** *** | **0.0000** *** |
| Cohen's d | **0.7116** | **1.0553** |
| Casos que mejoraron | 70.8% | 80.0% |

**Hallazgo**: 2 de 2 variantes con mejora significativa (p < 0.05, sin corrección por comparaciones múltiples). Mayor efecto: B2b (nofilter R20261116 234152) (d = 1.06; d promedia las varianzas de raw y reprompt, el d pareado es menor).

> Este reporte compara cada variante contra su prompt original, no variantes entre sí. Para contrastes entre variantes con Holm, ver `Ablations_Analysis.md`.


---

## 3. Análisis por Categoría de Taste

### B2a (filter default R20261116 234152)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| bitter | 30 | 0.1309 | 0.2438 | +0.1129 | **+86.2%** | **0.0027** ** | 70.0% |
| sour | 30 | 0.1075 | 0.2155 | +0.1080 | **+100.5%** | **0.0002** *** | 76.7% |
| salty | 30 | 0.1068 | 0.1950 | +0.0882 | **+82.5%** | **0.0220** * | 66.7% |
| sweet | 30 | 0.1115 | 0.1647 | +0.0532 | +47.7% | 0.1093 n.s. | 70.0% |

### B2b (nofilter R20261116 234152)

| Taste | N | Raw μ | Reprompt μ | Δ μ | Cambio % | p-value | Mejoraron |
|-------|---|-------|------------|-----|----------|---------|-----------|
| sour | 30 | 0.0696 | 0.2431 | +0.1735 | **+249.3%** | **0.0000** *** | 90.0% |
| salty | 30 | 0.0838 | 0.2447 | +0.1609 | **+191.9%** | **0.0000** *** | 83.3% |
| bitter | 30 | 0.1228 | 0.2537 | +0.1309 | **+106.7%** | **0.0000** *** | 76.7% |
| sweet | 30 | 0.1341 | 0.2305 | +0.0964 | **+71.9%** | **0.0103** * | 70.0% |

**Hallazgo**: La(s) categoría(s) **bitter, salty, sour** muestra(n) mejora significativa (p < 0.05) en **todas** las variantes.


---

## 4. Correlación Kendall τ por Taste

| Taste | B2a (filter default R20261116 234152) τ | B2b (nofilter R20261116 234152) τ |
|-------|---|---|
| bitter | 0.113 (Débil) | 0.310 (Moderada) (p=0.02) |
| salty | -0.081 (Débil) | 0.186 (Débil) |
| sour | 0.048 (Débil) | 0.113 (Débil) |
| sweet | 0.058 (Débil) | 0.081 (Débil) |

**Interpretación**: Correlaciones débiles indican que el reprompt transforma sustancialmente la representación texto→audio respecto al prompt original.


---

## Visualizaciones

![comparison_B2](comparison_B2.png)

![dist_gpt_5.6_luna_120_prompt_V3_B2a_filter_default_R20261116_234152](dist_gpt_5.6_luna_120_prompt_V3_B2a_filter_default_R20261116_234152.png)

![dist_gpt_5.6_luna_120_prompt_V3_B2b_nofilter_R20261116_234152](dist_gpt_5.6_luna_120_prompt_V3_B2b_nofilter_R20261116_234152.png)

![taste_gpt_5.6_luna_120_prompt_V3_B2a_filter_default_R20261116_234152](taste_gpt_5.6_luna_120_prompt_V3_B2a_filter_default_R20261116_234152.png)

![taste_gpt_5.6_luna_120_prompt_V3_B2b_nofilter_R20261116_234152](taste_gpt_5.6_luna_120_prompt_V3_B2b_nofilter_R20261116_234152.png)
