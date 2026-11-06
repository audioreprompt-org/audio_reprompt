# Informe de Evaluación de Calidad de Recuperación de Información (IR) — RAG

- [Informe de Evaluación de Calidad de Recuperación de Información (IR) — RAG](#informe-de-evaluación-de-calidad-de-recuperación-de-información-ir--rag)
  - [Resumen Ejecutivo y Correcciones Metodológicas](#resumen-ejecutivo-y-correcciones-metodológicas)
  - [Parte A: Diagnóstico de la Base de Datos y Espacio Vectorial](#parte-a-diagnóstico-de-la-base-de-datos-y-espacio-vectorial)
    - [D1. Estadísticas de Tablas](#d1-estadísticas-de-tablas)
    - [D2. Distribución de Similitud Coseno por Pares](#d2-distribución-de-similitud-coseno-por-pares)
    - [D3. Dispersión y Caída de Similitud (Top-1 vs Top-K, Excluyendo Auto-Coincidencia)](#d3-dispersión-y-caída-de-similitud-top-1-vs-top-k-excluyendo-auto-coincidencia)
    - [D4. Cobertura de Dimensiones Transmodales](#d4-cobertura-de-dimensiones-transmodales)
    - [D5. Detección Rigurosa de Concentración (Hubness Leave-One-Out Exhaustivo)](#d5-detección-rigurosa-de-concentración-hubness-leave-one-out-exhaustivo)
  - [Parte B: Dimensionalidad y Utilización del Espacio Latente](#parte-b-dimensionalidad-y-utilización-del-espacio-latente)
    - [Espectro de Autovalores, Rango Numérico y Ratio de Participación](#espectro-de-autovalores-rango-numérico-y-ratio-de-participación)
  - [Parte C: Evaluación de Recuperación de Descriptores de Comida (`get_top_k_food_descriptors`)](#parte-c-evaluación-de-recuperación-de-descriptores-de-comida-get_top_k_food_descriptors)
    - [Configuración de Producción (`cut_results=True`) vs Línea Base Aleatoria](#configuración-de-producción-cut_resultstrue-vs-línea-base-aleatoria)
    - [Ablación: Sin Corte (`cut_results=False`)](#ablación-sin-corte-cut_resultsfalse)
    - [Diagnóstico de Auto-Consistencia del Codificador (No Relevancia)](#diagnóstico-de-auto-consistencia-del-codificador-no-relevancia)
  - [Parte D: Recuperación de Subtítulos de Audio (`get_top_k_audio_captions`)](#parte-d-recuperación-de-subtítulos-de-audio-get_top_k_audio_captions)
    - [Métricas Observables de Similitud](#métricas-observables-de-similitud)
      - [Diagnóstico de Auto-Consistencia (Audio)](#diagnóstico-de-auto-consistencia-audio)
    - [El Desafío de la Contaminación por Efectos de Sonido (SFX)](#el-desafío-de-la-contaminación-por-efectos-de-sonido-sfx)
  - [Parte E: Conclusiones y Próximos Pasos (Limitaciones)](#parte-e-conclusiones-y-próximos-pasos-limitaciones)

---

**Modelo de embeddings**: `all-MiniLM-L6-v2` (384 dimensiones, denso, normalizado L2)  
**Métrica de distancia**: Coseno (operador `<=>` de pgvector)  
**Semilla global**: `42` (ejecución determinista con intervalos de confianza bootstrap al 95%)  

> [!NOTE]
> **Nota sobre índices vectoriales**: Se conoce *a priori* que la base de datos no utiliza índices aproximados (como HNSW o IVFFlat) para las columnas de embeddings debido a que el volumen de datos es moderado (~13k filas en audio y ~26k en comida); PostgreSQL realiza escaneos secuenciales exactos por defecto, garantizando precisión del 100% en la búsqueda por similitud sin pérdidas por aproximación.

---

## Resumen Ejecutivo y Correcciones Metodológicas

La versión previa de este informe presentaba fallos metodológicos que invalidaban varias conclusiones clave:
1. **D3 incorporaba la propia consulta como Top-1**: Dado que las consultas se extraían de la misma tabla, la similitud reportada era artificialmente `1.0000`. Con la corrección (enmascaramiento *leave-one-out*), el Top-1 real se ubica en **0.8391** (audio) y **0.8854** (comida).
2. **D5 (Hubness) era un artefacto de sub-muestreo**: Una muestra de 50 consultas sobre 26,000 vectores forzaba ocurrencias máximas de 2. La evaluación corregida ejecuta un kNN exacto *leave-one-out* sobre **todos los vectores del corpus**, revelando antihubs reales (0.70% a 1.76%) y hubs máximos reales (48 en comida, 140 en audio).
3. **Métrica de isotropía corregida**: El valor reportado previamente (`2e-9`) correspondía al piso epsilon numérico dividido por $\lambda_{max}$, provocado por 3 dimensiones nulas (rango numérico 381/384). Se sustituye por el ratio de participación de varianza (**76.6** en audio, **73.8** en comida) y dimensiones efectivas al 95% de varianza (**224** y **175** de 384), demostrando una adecuada dispersión del espacio.
4. **Juez de embeddings circular reclasificado**: Usar el mismo modelo `MiniLM` para recuperar y juzgar producía nDCG artificiales de 1.0. Se reclasifica como diagnóstico de auto-consistencia y no como veredicto de relevancia.
5. **Juez estructural riguroso y normalización a nivel de corpus**: Se eliminó el emparejamiento por sub-cadenas no delimitadas (*e.g.* "hot" en "photo"), se restringe el descriptor a su dimensión correcta, y nDCG / Recall se normalizan respecto al total de elementos relevantes en el corpus entero con línea base aleatoria e intervalos de confianza bootstrap.

---

## Parte A: Diagnóstico de la Base de Datos y Espacio Vectorial

### D1. Estadísticas de Tablas

| Tabla                        | Filas Totales | Claves de Texto Únicas                      |
| ---------------------------- | ------------- | ------------------------------------------- |
| `audio_descriptors`          | 13,158        | 13,158 subtítulos distintos                 |
| `crossmodal_food_embeddings` | 26,495        | 835 pares distintos (dimensión, descriptor) |

- Dimensiones de comida distintas: **7**
- Descriptores de comida distintos: **639**
- Elementos de comida (*food items*) distintos: **2,070**

### D2. Distribución de Similitud Coseno por Pares

Calculado sobre el triángulo superior estricto (sin auto-pares) de una muestra sembrada de vectores normalizados:

| Tabla                        | Media  | Desv. Est. | Mín     | Mediana | Máx    | P10    | P90    | Pares Cuasi-Duplicados ($\ge 0.9999$) |
| ---------------------------- | ------ | ---------- | ------- | ------- | ------ | ------ | ------ | ------------------------------------- |
| `audio_descriptors`          | 0.2085 | 0.1186     | -0.1852 | 0.1946  | 0.9946 | 0.0695 | 0.3644 | 0                                     |
| `crossmodal_food_embeddings` | 0.2408 | 0.1035     | -0.1676 | 0.2350  | 1.0000 | 0.1158 | 0.3694 | 4                                     |

> **Interpretación**: La media de similitud de pares aleatorios ($\sim 0.21 - 0.24$) es el estándar esperado para *sentence-transformers* contrastivos. No denota un cono estrecho patológico (a diferencia de BERT crudo, donde la media suele rondar $0.6 - 0.9$).

### D3. Dispersión y Caída de Similitud (Top-1 vs Top-K, Excluyendo Auto-Coincidencia)

| Tabla                        | Similitud Top-1 Promedio | Brecha (1 vs 5) | Brecha (1 vs 10) | Brecha (1 vs 20) | Consultas con Cuasi-Duplicados |
| ---------------------------- | ------------------------ | --------------- | ---------------- | ---------------- | ------------------------------ |
| `audio_descriptors`          | 0.8391                   | 0.1199          | 0.1681           | 0.2160           | 0                              |
| `crossmodal_food_embeddings` | 0.8854                   | 0.0636          | 0.1029           | 0.1825           | 11                             |

> **Interpretación**: Al excluir la auto-coincidencia, la similitud Top-1 desciende a $\sim 0.84 - 0.88$. En la tabla transmodal, 11 de 200 consultas de prueba poseen registros idénticos o casi idénticos en la base de datos (por combinaciones repetidas de alimentos y descriptores).

### D4. Cobertura de Dimensiones Transmodales

| Dimensión         | Cantidad | Proporción |
| ----------------- | -------- | ---------- |
| `chemical_flavor` | 6,497    | 24.52%     |
| `texture`         | 6,000    | 22.65%     |
| `human_response`  | 5,968    | 22.53%     |
| `temperature`     | 2,014    | 7.60%      |
| `color`           | 2,006    | 7.57%      |
| `emotion`         | 2,006    | 7.57%      |
| `taste`           | 2,004    | 7.56%      |

- **Coeficiente de Gini**: **0.2742** (máximo teórico para 7 categorías: **0.8571**; valor normalizado: **0.3199**).
- El desequilibrio refleja el diseño del dataset (tres grupos grandes de $\sim 23-25\%$ y cuatro grupos de $\sim 7.6\%$), lo que justifica el uso de `RANK() OVER (PARTITION BY dimension)` en la función de recuperación de producción.

### D5. Detección Rigurosa de Concentración (Hubness Leave-One-Out Exhaustivo)

Calculado mediante kNN exhaustivo por bloques sobre los $N$ puntos de la tabla, con enmascaramiento diagonal (antihubs incluidos):

| Tabla                        | Modo            | Puntos | Consultas | k   | Asimetría (Skewness) | Antihubs ($N_k = 0$) | Ocurrencia Máx | Ocurrencia Media |
| ---------------------------- | --------------- | ------ | --------- | --- | -------------------- | -------------------- | -------------- | ---------------- |
| `audio_descriptors`          | `leave_one_out` | 13,158 | 13,158    | 10  | 3.018                | 1.76%                | 140            | 10.00            |
| `audio_descriptors`          | `queries`       | 13,158 | 10        | 10  | 19.780               | 99.41%               | 4              | 0.01             |
| `crossmodal_food_embeddings` | `leave_one_out` | 26,495 | 26,495    | 10  | 0.345                | 0.70%                | 48             | 10.00            |
| `crossmodal_food_embeddings` | `queries`       | 26,495 | 15        | 10  | 13.440               | 99.44%               | 2              | 0.01             |

> **Interpretación**: En el espacio de comida, el hubness es mínimo (asimetría 0.345, punto máximo aparece en 48 listas). En audio, la concentración es moderada (asimetría 3.018, con subtítulos genéricos como atrayentes que aparecen hasta 140 veces). Menos del $2\%$ de los puntos son antihubs inalcanzables.

---

## Parte B: Dimensionalidad y Utilización del Espacio Latente

### Espectro de Autovalores, Rango Numérico y Ratio de Participación

| Tabla                        | Muestras ($N$) | Dim. Total | Dim. Efectiva (95%) | Dim. Efectiva (99%) | Ratio Part. (Centrado) | Ratio Part. (Sin centrar) | Norma Vector Medio | Rango Numérico | Ratio $\lambda_{min}/\lambda_{max}$ | Ref. Marchenko-Pastur |
| ---------------------------- | -------------- | ---------- | ------------------- | ------------------- | ---------------------- | ------------------------- | ------------------ | -------------- | ----------------------------------- | --------------------- |
| `audio_descriptors`          | 13,158         | 384        | 224                 | 308                 | 76.6                   | 17.0                      | 0.4598             | 381            | $6.70 \times 10^{-19}$              | 0.5015                |
| `crossmodal_food_embeddings` | 26,495         | 384        | 175                 | 271                 | 73.8                   | 14.8                      | 0.4874             | 381            | $8.21 \times 10^{-19}$              | 0.6164                |

> **Análisis de la deficiencia de rango**:  
> Ambas tablas exhiben un rango numérico de **381** sobre 384 dimensiones. Los 3 autovalores más pequeños son del orden de $10^{-19}$ a $10^{-20}$, lo que explica por qué la fórmula anterior generaba valores espurios de $2 \times 10^{-9}$ (debido a $\epsilon = 10^{-10}$).  
> No obstante, el 95% de la varianza se reparte entre **224** dimensiones (audio) y **175** dimensiones (comida), con ratios de participación de $\sim 74-77$, demostrando una dispersión multidimensional sólida y suficiente para discriminación por coseno.

---

## Parte C: Evaluación de Recuperación de Descriptores de Comida (`get_top_k_food_descriptors`)

### Configuración de Producción (`cut_results=True`) vs Línea Base Aleatoria

Evaluación sobre $n = 15$ consultas sintéticas, con relevancia binaria estricta (Grado 2 = dimensión esperada Y descriptor coincidente a nivel de token):

| Métrica          | Sistema (`cut_results=True`) [IC 95%] | Línea Base Aleatoria [IC 95%] | Diferencia Pareada [IC 95%] | Valor p  |
| ---------------- | ------------------------------------- | ----------------------------- | --------------------------- | -------- |
| **Precision@5**  | 0.480 [0.373, 0.600]                  | 0.013 [0.000, 0.040]          | **+0.467 [0.347, 0.587]**   | < 0.0001 |
| **Precision@10** | 0.393 [0.300, 0.493]                  | 0.013 [0.000, 0.033]          | **+0.380 [0.287, 0.473]**   | < 0.0001 |
| **nDCG@5**       | 0.626 [0.539, 0.717]                  | 0.219 [0.157, 0.277]          | **+0.407 [0.310, 0.519]**   | < 0.0001 |
| **nDCG@10**      | 0.563 [0.496, 0.633]                  | 0.227 [0.173, 0.278]          | **+0.336 [0.259, 0.429]**   | < 0.0001 |
| **Recall@5**     | 0.237 [0.174, 0.310]                  | 0.006 [0.000, 0.018]          | **+0.231 [0.166, 0.306]**   | < 0.0001 |
| **Recall@10**    | 0.386 [0.289, 0.491]                  | 0.012 [0.000, 0.029]          | **+0.375 [0.277, 0.480]**   | < 0.0001 |
| **MRR**          | 0.833 [0.667, 0.967]                  | 0.040 [0.000, 0.113]          | **+0.793 [0.627, 0.927]**   | < 0.0001 |
| **MAP@10**       | 0.308 [0.214, 0.419]                  | 0.004 [0.000, 0.011]          | **+0.304 [0.210, 0.415]**   | < 0.0001 |

### Ablación: Sin Corte (`cut_results=False`)

| Métrica          | Sistema (`cut_results=False`) [IC 95%] | Línea Base Aleatoria [IC 95%] | Diferencia Pareada [IC 95%] | Valor p  |
| ---------------- | -------------------------------------- | ----------------------------- | --------------------------- | -------- |
| **Precision@10** | 0.407 [0.313, 0.500]                   | 0.027 [0.007, 0.053]          | **+0.380 [0.293, 0.467]**   | < 0.0001 |
| **nDCG@10**      | 0.575 [0.509, 0.642]                   | 0.233 [0.197, 0.269]          | **+0.342 [0.263, 0.423]**   | < 0.0001 |
| **Recall@10**    | 0.399 [0.305, 0.499]                   | 0.026 [0.006, 0.049]          | **+0.372 [0.283, 0.472]**   | < 0.0001 |
| **MRR**          | 0.838 [0.682, 0.967]                   | 0.058 [0.007, 0.134]          | **+0.781 [0.585, 0.944]**   | < 0.0001 |
| **MAP@10**       | 0.318 [0.226, 0.427]                   | 0.006 [0.001, 0.014]          | **+0.312 [0.219, 0.421]**   | < 0.0001 |

> **Conclusión sobre la Recuperación Transmodal**:  
> El sistema supera de manera estadísticamente significativa a la línea base aleatoria en todas las métricas ($p < 0.0001$).  
> La configuración de producción (`cut_results=True`) preserva una precisión equivalente a la no podada mientras impone una diversidad 1:1 en dimensiones clave (`taste`, `color`, `temperature`), reduciendo redundancia para el paso posterior de generación de música.

### Diagnóstico de Auto-Consistencia del Codificador (No Relevancia)

- **Agreement P@5**: 0.093 [0.027, 0.173]
- **Agreement P@10**: 0.047 [0.013, 0.087]

> Este indicador refleja la baja similitud coseno directa entre cadenas del tipo `"chocolate cake"` y pares formateados como `"taste sweet"`, evidenciando que el emparejamiento semántico transmodal no se basa en auto-afinidad léxica trivial.

---

## Parte D: Recuperación de Subtítulos de Audio (`get_top_k_audio_captions`)

> [!IMPORTANT]
> **Aviso de Integridad**: Dado que no se dispone de un conjunto de anotaciones humanas independientes (*gold standard*) ni de una rúbrica formal para subtítulos de audio, **no se emite un veredicto de calidad de relevancia**. Las métricas a continuación corresponden a propiedades descriptivas del espacio de recuperación.

### Métricas Observables de Similitud

Sobre $n = 10$ consultas de prueba con $k = 20$:
- Consultas fallidas (0 resultados): **0**
- Consultas con resultados colapsados ($\text{len} < k$ por colisión de claves en diccionario): **0**
- **Similitud Top-1**: **0.542** [0.486, 0.598]
- **Similitud Promedio**: **0.452** [0.408, 0.504]
- **Rango de Similitud (Máx - Mín en Top-20)**: **0.128** [0.098, 0.162]

#### Diagnóstico de Auto-Consistencia (Audio)
- **Agreement P@5**: 0.780 [0.560, 0.960]
- **Agreement P@10**: 0.610 [0.390, 0.830]

### El Desafío de la Contaminación por Efectos de Sonido (SFX)

El fenómeno detectado cualitativamente en revisiones previas se confirma conceptualmente:
Al concatenar múltiples descriptores sensoriales (*e.g.* `"savory umami briny tender warm satisfaction"` o `"crunchy chewy nutty sweet nostalgic"`), el codificador MiniLM se aproxima tanto a términos musicales como a efectos de sonido literales presentes en la base de datos de audio (*e.g.* "chewing", "crunching").

---

## Parte E: Conclusiones y Próximos Pasos (Limitaciones)

Para elevar la robustez del pipeline de recuperación en iteraciones futuras, se identifican las siguientes prioridades:

1. **Construcción de un conjunto de evaluación de referencia (*Gold Standard*)**:
   - Agrupar los Top-20 resultados de al menos 2 fuentes: `MiniLM` y `CLAP text encoder`.
   - Realizar etiquetado con al menos dos anotadores independientes (escala 0–3 sobre utilidad para describir música) y reportar acuerdo inter-anotador ($\kappa$ de Cohen).
2. **Medición cuantitativa de tasa de contaminación SFX**:
   - Implementar un clasificador o filtro basado en metadatos para cuantificar formalmente la proporción de resultados que corresponden a efectos de sonido frente a música.
3. **Diseño de recuperación desagregada y Fusión RRF**:
   - En lugar de embeber una sola consulta concatenada con 5 descriptores, recuperar independientemente por descriptor/dimensión y combinar los rankings mediante *Reciprocal Rank Fusion* (RRF).