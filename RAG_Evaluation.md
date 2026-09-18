# Informe de Evaluación de Calidad de Recuperación de Información (IR)

- [Informe de Evaluación de Calidad de Recuperación de Información (IR)](#informe-de-evaluación-de-calidad-de-recuperación-de-información-ir)
  - [Parte A: Diagnóstico de la Base de Datos](#parte-a-diagnóstico-de-la-base-de-datos)
    - [D1. Estadísticas de Tablas](#d1-estadísticas-de-tablas)
    - [D2. Distribución de Similitud por Pares](#d2-distribución-de-similitud-por-pares)
    - [D3. Dispersión de Similitud de Consultas (Brecha entre Top-1 y Top-K)](#d3-dispersión-de-similitud-de-consultas-brecha-entre-top-1-y-top-k)
    - [D4. Cobertura de Dimensiones Crossmodal](#d4-cobertura-de-dimensiones-crossmodal)
    - [D5. Detección de Concentración (Hubness)](#d5-detección-de-concentración-hubness)
  - [Parte B: Calidad Intrínseca de Embeddings (Estrategia 1)](#parte-b-calidad-intrínseca-de-embeddings-estrategia-1)
    - [Isotropía](#isotropía)
    - [Auto-Similitud (Self-Similarity)](#auto-similitud-self-similarity)
  - [Parte B: Evaluación de Relevancia Sintética (Estrategia 2)](#parte-b-evaluación-de-relevancia-sintética-estrategia-2)
    - [Descriptores de Comida Transmodales (`get_top_k_food_descriptors`)](#descriptores-de-comida-transmodales-get_top_k_food_descriptors)
    - [Desglose por Consulta (Comida Transmodal)](#desglose-por-consulta-comida-transmodal)
    - [Análisis de Recuperación de Subtítulos de Audio (`get_top_k_audio_captions`)](#análisis-de-recuperación-de-subtítulos-de-audio-get_top_k_audio_captions)
      - [Desempeño Cualitativo según Tipo de Consulta](#desempeño-cualitativo-según-tipo-de-consulta)
      - [El Problema del Emparejamiento Literal](#el-problema-del-emparejamiento-literal)
      - [Ejemplos de Recuperación](#ejemplos-de-recuperación)
      - [`"savory umami briny tender warm satisfaction"` — Top-1 Sim: 0.501](#savory-umami-briny-tender-warm-satisfaction--top-1-sim-0501)
      - [`"sour pungent sharp thin disgust"` — Top-1 Sim: 0.501](#sour-pungent-sharp-thin-disgust--top-1-sim-0501)
  - [Evaluación General](#evaluación-general)
    - [Calidad del Espacio Vectorial](#calidad-del-espacio-vectorial)
    - [Calidad de Recuperación (Retrieval Quality)](#calidad-de-recuperación-retrieval-quality)


**Modelo de embeddings**: `all-MiniLM-L6-v2` (384d, denso)
**Métrica de distancia**: coseno (operador `<=>` de pgvector)

---

## Parte A: Diagnóstico de la Base de Datos

### D1. Estadísticas de Tablas

| Tabla                        | Filas  |
| ---------------------------- | ------ |
| `audio_descriptors`          | 13,158 |
| `crossmodal_food_embeddings` | 26,495 |

- Dimensiones de comida (food dimensions) distintas: **7**
- Descriptores de comida distintos: **639**
- Elementos de comida (food items) distintos: **2,070**

### D2. Distribución de Similitud por Pares

| Tabla                        | Media  | Desv. Est. | Mín     | Mediana | Máx    | P10    | P90    |
| ---------------------------- | ------ | ---------- | ------- | ------- | ------ | ------ | ------ |
| `audio_descriptors`          | 0.2244 | 0.1150     | -0.0514 | 0.2120  | 1.0000 | 0.0896 | 0.3700 |
| `crossmodal_food_embeddings` | 0.2553 | 0.1054     | -0.0926 | 0.2496  | 0.9233 | 0.1286 | 0.3838 |

> **Interpretación**:
> - `audio_descriptors`: Se observa una dispersión adecuada (σ=0.1150), lo cual sugiere un fuerte poder discriminativo.
> - `crossmodal_food_embeddings`: Presenta una dispersión adecuada (σ=0.1054), indicando igualmente un fuerte poder discriminativo.

### D3. Dispersión de Similitud de Consultas (Brecha entre Top-1 y Top-K)

| Tabla                        | Similitud Promedio Top-1 | Brecha Promedio (1 vs 5) | Brecha Promedio (1 vs 10) | Brecha Promedio (1 vs 20) |
| ---------------------------- | ------------------------ | ------------------------ | ------------------------- | ------------------------- |
| `audio_descriptors`          | 1.0000                   | 0.2487                   | 0.3106                    | 0.3792                    |
| `crossmodal_food_embeddings` | 1.0000                   | 0.1630                   | 0.2042                    | 0.3000                    |

### D4. Cobertura de Dimensiones Crossmodal

| Dimensión         | Cantidad | Proporción |
| ----------------- | -------- | ---------- |
| `chemical_flavor` | 6,497    | 24.52%     |
| `texture`         | 6,000    | 22.65%     |
| `human_response`  | 5,968    | 22.53%     |
| `temperature`     | 2,014    | 7.60%      |
| `color`           | 2,006    | 7.57%      |
| `emotion`         | 2,006    | 7.57%      |
| `taste`           | 2,004    | 7.56%      |

**Coeficiente de Gini**: 0.2742 (evidencia de cierto desequilibrio en la distribución).

### D5. Detección de Concentración (Hubness)

| Tabla                        | Vecinos Únicos | Ocurrencia Máxima | Ocurrencia Media | Asimetría (Skewness) |
| ---------------------------- | -------------- | ----------------- | ---------------- | -------------------- |
| `audio_descriptors`          | 483            | 2                 | 1.04             | 5.060                |
| `crossmodal_food_embeddings` | 489            | 2                 | 1.02             | 6.460                |

> **Interpretación**:
> - `audio_descriptors`: Se reporta una alta asimetría (5.06) con una ocurrencia máxima de 2. Al analizar los datos a detalle, dado el valor máximo de ocurrencia, esto se considera un artefacto estadístico sin impacto significativo.
> - `crossmodal_food_embeddings`: Se observa una alta asimetría (6.46) con una ocurrencia máxima de 2.

---

## Parte B: Calidad Intrínseca de Embeddings (Estrategia 1)

### Isotropía

| Tabla                        | Isotropía | Dim. Efectivas (95%) | Dim. Efectivas (99%) | Dim. Totales |
| ---------------------------- | --------- | -------------------- | -------------------- | ------------ |
| `audio_descriptors`          | 0.000000  | 197                  | 283                  | 384          |
| `crossmodal_food_embeddings` | 0.000000  | 158                  | 249                  | 384          |

> **Interpretación** (mayor isotropía = mejor utilización del espacio):
> - `audio_descriptors`: El espacio resulta altamente anisotrópico (isotropía≈2.12e-09). Únicamente 197 de las 384 dimensiones (51%) concentran el 95% de la varianza. Los vectores ocupan un segmento estrecho, lo cual es un comportamiento típico y esperado en modelos tipo *sentence-transformers*.
> - `crossmodal_food_embeddings`: El espacio es altamente anisotrópico (isotropía≈2.56e-09). Solo 158 de las 384 dimensiones (41%) conllevan el 95% de la varianza.

### Auto-Similitud (Self-Similarity)

| Tabla                        | Auto-Similitud Promedio | Desv. Est. | Mín     | Máx    |
| ---------------------------- | ----------------------- | ---------- | ------- | ------ |
| `audio_descriptors`          | 0.2166                  | 0.1197     | -0.1389 | 0.9704 |
| `crossmodal_food_embeddings` | 0.2410                  | 0.1048     | -0.1221 | 0.9693 |

> **Interpretación** (menor auto-similitud = mayor dispersión en el espacio):
> - `audio_descriptors`: Baja auto-similitud promedio (0.2166). Existe una adecuada diversidad en el espacio vectorial a pesar de la anisotropía detectada.
> - `crossmodal_food_embeddings`: Baja auto-similitud promedio (0.2410). Presenta una buena diversidad en el espacio vectorial.

---

## Parte B: Evaluación de Relevancia Sintética (Estrategia 2)

### Descriptores de Comida Transmodales (`get_top_k_food_descriptors`)

| Métrica      | Estructural | Embeddings |
| ------------ | ----------- | ---------- |
| Precision@5  | 0.827       | 0.693      |
| nDCG@5       | 0.734       | 0.696      |
| Precision@10 | 0.767       | 0.553      |
| nDCG@10      | 0.712       | 0.704      |
| MRR          | 0.917       | 0.878      |
| MAP@10       | 0.672       | 0.572      |

### Desglose por Consulta (Comida Transmodal)

| Consulta              | nDCG@5 (Estruct.) | nDCG@10 (Estruct.) | P@5 (Emb.) | Similitud Top-1 | Similitud Promedio |
| --------------------- | ----------------- | ------------------ | ---------- | --------------- | ------------------ |
| chocolate cake        | 0.628             | 0.477              | 1.000      | 0.9086          | 0.7858             |
| lemon                 | 1.000             | 0.779              | 1.000      | 0.8782          | 0.7544             |
| black coffee          | 0.765             | 0.727              | 0.600      | 0.6919          | 0.6188             |
| vanilla ice cream     | 0.277             | 0.465              | 1.000      | 0.8158          | 0.7121             |
| cider vinegar         | 0.485             | 0.634              | 0.400      | 0.8484          | 0.7238             |
| granola bar           | 0.530             | 0.695              | 0.400      | 0.9005          | 0.8084             |
| grapefruit slice      | 0.915             | 0.808              | 1.000      | 0.7645          | 0.6938             |
| kale salad            | 0.720             | 0.808              | 0.400      | 0.7304          | 0.6568             |
| caramel popcorn       | 0.769             | 0.644              | 0.800      | 0.9680          | 0.7834             |
| bitter melon stir-fry | 0.747             | 0.713              | 0.600      | 0.6453          | 0.6042             |
| strawberry donut      | 0.692             | 0.723              | 0.800      | 0.7845          | 0.7160             |
| matcha latte          | 0.861             | 0.842              | 1.000      | 0.5231          | 0.4499             |
| sports drink          | 1.000             | 0.801              | 0.200      | 0.8591          | 0.6544             |
| cuttlefish cooked     | 0.869             | 0.772              | 0.200      | 0.9210          | 0.8035             |
| honey glazed bun      | 0.754             | 0.788              | 1.000      | 0.6422          | 0.5875             |

---

### Análisis de Recuperación de Subtítulos de Audio (`get_top_k_audio_captions`)

> [!WARNING]
> **Nota sobre Métricas IR**: Las métricas cuantitativas originales (nDCG, Precision) reportaron un índice perfecto (1.000) de manera errónea, debido a umbrales de relevancia excesivamente permisivos. El análisis presentado a continuación se fundamenta en un desglose cualitativo riguroso de las consultas.

#### Desempeño Cualitativo según Tipo de Consulta

| Tipo de Consulta                | Ejemplo de Palabras              | Relevancia Musical    | Causa Principal                                                                                         |
| ------------------------------- | -------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------- |
| **Cargada de Emociones**        | warm, nostalgic, happiness, soft | **Alta (8–10/10)**    | Estas palabras comparten un dominio semántico en ambos vocabularios (gastronómico y musical).           |
| **Mixta**                       | sweet, smooth, cold + crunchy    | **Moderada (7–9/10)** | Las connotaciones musicales superan a las interpretaciones literales alimenticias.                      |
| **Cargada de Texturas/Sabores** | crunchy, chewy, pungent, bitter  | **Baja (2–3/10)**     | El modelo es susceptible al emparejamiento literal, resultando en la recuperación de efectos de sonido. |

#### El Problema del Emparejamiento Literal

La base de datos de audio incorpora subtítulos de sonidos ambientales (por ejemplo, sonidos de masticación o efectos "crunchy") junto con descripciones de carácter estrictamente musical (por ejemplo, "romantic melody"). Puesto que el modelo subyacente `all-MiniLM-L6-v2` se basa en la similitud semántica superficial, cuando procesa palabras fuertemente asociadas al ámbito sensorial alimenticio (como *crunchy* o *chewy*), tiende a recuperar efectos de sonido literales. Esto compromete por completo la intención de traducción al dominio musical.

Por ejemplo, la consulta `"crunchy chewy nutty sweet nostalgic"` recupera subtítulos como `"crunchy"`, `"crunching sounds"` y `"chewing"`, mitigando la connotación musical pretendida por el término `"nostalgic"`.

#### Ejemplos de Recuperación

---

#### `"savory umami briny tender warm satisfaction"` — Top-1 Sim: 0.501

| Rango | Similitud en BD | Subtítulo           | Evaluación             |
| ----- | --------------- | ------------------- | ---------------------- |
| 1     | 0.501           | tender              | Emparejamiento literal |
| 2     | 0.462           | warmth              | Tonalidad musical      |
| 3     | 0.447           | acoustic delight    | Atributo musical       |
| 4     | 0.444           | sensual             | Tonalidad musical      |
| 5     | 0.438           | pleasant harmonious | Atributo musical       |
| 6     | 0.434           | tropical feeling    | Relación tangencial    |
| 7     | 0.417           | warm                | Tonalidad musical      |
| 8     | 0.410           | sensual music       | Atributo musical       |
| 9     | 0.405           | warm sounding       | Atributo musical       |
| 10    | 0.404           | warm tone           | Atributo musical       |

**Veredicto**: Relevancia de 7/10. Los términos "warm" y "satisfaction" generan resultados adecuados; sin embargo, el término "tender" en el primer rango constituye un emparejamiento literal, careciendo de valor musical.

---

#### `"sour pungent sharp thin disgust"` — Top-1 Sim: 0.501

| Rango | Similitud en BD | Subtítulo                        | Evaluación                              |
| ----- | --------------- | -------------------------------- | --------------------------------------- |
| 1     | 0.501           | unpleasant                       | Estado de ánimo, no musical             |
| 2     | 0.485           | unpleasant sound                 | Descripción genérica                    |
| 3     | 0.474           | soft melodious                   | Opuesto a la intención de la consulta   |
| 4     | 0.454           | sincere to disgusting atmosphere | Relación tangencial                     |
| 5     | 0.445           | bland                            | Sin utilidad                            |
| 6     | 0.434           | sonic delight                    | Opuesto al concepto "disgust"           |
| 7     | 0.424           | percussive bas slime             | Emparejamiento anómalo                  |
| 8     | 0.421           | slurring                         | Efecto de sonido                        |
| 9     | 0.416           | twangy high pitched licks        | Atributo musical, concuerda con "sharp" |
| 10    | 0.392           | grungy                           | Textura musical                         |

**Veredicto**: Relevancia estrictamente musical de 2/10. El resultado "soft melodious" en el tercer rango para una consulta basada en "pungent disgust" representa un fallo evidente en la clasificación.

---

## Evaluación General

### Calidad del Espacio Vectorial

Ambas tablas emplean embeddings **densos** de 384 dimensiones. Estos no son vectores dispersos (sparse); cada dimensión representa una característica latente con valores continuos.
A pesar de la fuerte anisotropía registrada (que es el comportamiento esperado del modelo), existe suficiente dispersión y baja auto-similitud para permitir que el operador de distancia coseno de `pgvector` clasifique las entidades de forma efectiva.

- `audio_descriptors`: Adecuada utilización del espacio funcional (197 dimensiones efectivas).
- `crossmodal_food_embeddings`: Adecuada utilización del espacio funcional (158 dimensiones efectivas).

### Calidad de Recuperación (Retrieval Quality)

- **Recuperación de comida transmodal**: **Adecuada** (nDCG@10 estructural = 0.712). La función es robusta y mitiga con éxito el desequilibrio inherente al tamaño de las categorías, utilizando de manera eficiente el operador SQL `RANK() OVER`.
- **Recuperación de subtítulos de audio**: **Mixta**. Se observan  buenos resultados para términos con alta carga afectiva o emocional (como *nostalgic* o *soft*). No obstante, se presentan problemas críticos de literalidad frente a términos sensoriales específicos (como *crunchy* o *bitter*), ya que el sistema recupera efectos de sonido en lugar de conceptos musicales. 

Para evaluar y optimizar la precisión en la recuperación de los descriptores musicales es sugerido priorizar o estratificar de forma sistemática las dimensiones crossmodales que pueden ser duales (`emotion`, `human response`, `temperature`, `color`), y relegar los descriptores altamente específicos (como los correspondientes a la dimensión `taste` o `texture`) a una fase intermedia de re-mapeo semántico (nuevas variaciones para reprompt).