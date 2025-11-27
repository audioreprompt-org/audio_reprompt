# Notebooks

El presente markdown es una guía de los notebooks construidos para el proyecto:

## Notebooks principales

### 1. [Food-terms-vocabulary](./Food-terms-vocabulary.ipynb)

El objetivo del notebook es construir un vocabulario limpio y unificado de términos alimentarios a partir de varios datasets crudos, combinando nombres de alimentos, ingredientes, categorías nutricionales y nombres/instrucciones de recetas. Todo el proceso busca normalizar, limpiar y tokenizar la información para generar archivos estructurados que puedan ser usados posteriormente.

Este conjunto de datos fue preparada a partir de las siguientes fuentes de datos:

| Conjunto de datos                      | Fuente                                                                                 |
| -------------------------------------- | -------------------------------------------------------------------------------------- |
| Daily Food Nutrition Dataset           | [Kaggle](https://www.kaggle.com/datasets/adilshamim8/daily-food-and-nutrition-dataset) |
| Vocabulario Daily Food Nutrition       | [Kaggle](https://www.kaggle.com/datasets/utsavdey1410/food-nutrition-dataset/data)     |
| Food.com - Recipes and Reviews Dataset | [Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)           |

Durante el proceso se crean varios vocabularios limpios:

1. daily_food_terms_vocabulary.csv: Términos únicos por categoría y meal type.

2. recipes_terms_vocabulary.csv: Instrucciones limpias y tokenizadas para cada receta.

3. recipe_names.csv: 100 nombres de recetas en minúsculas seleccionados aleatoriamente.

4. food_nutrition_vocabulary.csv: Vocabulario unificado y sin duplicados de alimentos procesados.

5. food_names.csv: Muestra aleatoria de nombres de alimentos de los grupos nutricionales.

En food_curated_sample.csv se construyó un vocabulario curado de términos alimentarios (“food items”) a partir de múltiples fuentes de datos. Este vocabulario debía incluir alimentos básicos, procesados y preparados, con el fin de crear un conjunto representativo de alrededor de 70 términos que sirvan como entrada para tareas posteriores de análisis o generación sensorial.

| food_item           |
| ------------------- |
| Chocolate           |
| creamy peas         |
| aperol              |
| croissant           |
| Yogurt              |
| brazilian caipiriña |
| ...                 |

### 2. [Prompt Engineering Experiments](./prompt_engineering_experiments.ipynb)

Generar descriptores sensoriales sintéticos para 70 alimentos usando un LLM, aplicando una plantilla estructurada, ejecutando 70 llamadas exitosas.

El proceso consiste en: carga del dataset → definición de prompts → llamadas a la API → almacenamiento y análisis básico de los resultados.

Las plantillas corresponden a:

#### 1. Prompt base

El diseño inicial se basa en Motoki et al. (2024). El objetivo es recolectar respuestas formateadas como si el modelo fuera un participante humano en una encuesta sobre percepciones multisensoriales de alimentos.

Las respuestas deben incluir las siguientes dimensiones:

- Chemical flavor profile
- Human responses
- Temperature (espacio restringido)
- Texture
- Emotions (espacio restringido)
- Color
- Taste

Formato esperado:

```
rich, creamy, sweet | indulgence, satisfaction | warm | smooth, velvety | happiness | brown | sweet
```

Hallazgos

- El LLM tiende a responder con párrafos en lugar de listas de términos.

- No respeta el límite de tres términos para Chemical flavor profile, Human responses y Texture.

- Con frecuencia usa términos fuera del espacio permitido en Temperature, Emotions, Color y Taste.

#### 2. Variación 1 — Reducción y simplificación

Prompt más breve y directo para reducir tamaño de entrada y ambigüedad.

Hallazgos

- El formato de salida se respeta correctamente.

- Aún se observan términos fuera del espacio permitido en dimensiones restringidas (Temperature, Emotions, Color, Taste), aunque con menor frecuencia que en el prompt base.

#### 3. Variación 2 — Inclusión de “No Label”

Agrega reglas claras para usar No Label:

- Si el food_item no es un alimento, el modelo debe responder únicamente No Label.

- Si una dimensión no puede responderse, debe usar No Label en esa dimensión.

Hallazgos

- Mantiene un formato consistente.

- Disminuye la aparición de términos fuera del espacio permitido.

- Persisten algunos errores en Color y Taste, pero son mucho menos frecuentes.

- Las respuestas parcialmente correctas aparecen cuando solo una dimensión falla.

#### 4. Variación 3 — Criterios de Aceptación

Incluye reglas adicionales para que el modelo verifique y ajuste su respuesta antes de devolverla.

Hallazgos

- Reduce aún más el uso de términos fuera del espacio permitido.

- Aumenta considerablemente el uso de No Label, incluso cuando sería posible completar la dimensión.

- El modelo se vuelve más conservador y prefiere no responder antes que arriesgar un error.

- A pesar de esto, las respuestas no se consideran nulas.

El modelo corresponde a gpt-4o-mini configurado con una baja temperatura y formato fijo para maximizar consistencia.

Los resultados se guardan en: prompt*experiment_results*<timestamp>.csv

### 3. [Prompt Engineering Anova](./prompt_engineering_anova.ipynb)

El notebook tiene como propósito evaluar la diversidad léxica y la consistencia semántica de descriptores sensoriales generados por el LLM (gpt-4o-mini) para alimentos, utilizando diferentes variaciones de prompts. Para ello, se analiza cómo el modelo asigna términos en siete dimensiones semánticas—sensations, human responses, temperature, texture, emotions, color y taste—tanto en un escenario individual (un solo alimento) como en un conjunto amplio de alimentos. El objetivo final es determinar qué tan estable, variado y confiable es el LLM al generar descripciones sensoriales controladas, y comparar el comportamiento entre dos configuraciones de prompt.

Los datos utilizados incluyen dos conjuntos principales:

Un alimento (Chocolate) con 10 muestras generadas, y

- Un alimento (Chocolate) con 10 muestras generadas.
- Treinta alimentos, cada uno con 10 muestras (300 observaciones por variación de prompt).

Se utilizaron dos versiones del prompt: Variación 2 y Variación 3. Cada respuesta se divide en las siete dimensiones mencionadas, y sus términos son procesados para computar frecuencias, términos únicos y métricas de diversidad.

Los resultados indican que el LLM puede generar descriptores sensoriales cross-modales de manera consistente, pero la variación del prompt tiene un impacto directo en la calidad y estabilidad de las respuestas. El análisis muestra que la Variación 2 es la más adecuada, tanto por desempeño semántico y estructural como por costos computacionales asociados a un menor uso de tokens. Aunque las alucinaciones son inevitables en modelos generativos, el estudio permite identificarlas y sugiere que estas pueden mitigarse mediante reglas posteriores de validación, especialmente en dimensiones con espacios de respuesta delimitados.

### 4. [two_way_anova_model_comparison](./two_way_anova_model_comparison.ipynb)

En esta fase se evaluó si dos modelos generativos gpt-4o-mini-2024-07-18 y gpt-5-nano-2025-08-07 producen descriptores sensoriales distintos cuando generan muestras sintéticas para un conjunto de 70 alimentos. Para ello se aplicó un ANOVA de dos vías, considerando como factores al modelo y al alimento, y utilizando como variable dependiente la métrica cuantitativa unique_term_ratio, derivada de dimensiones textuales como sensations, temperature, texture, emotions, etc.

#### Efecto Principal del Modelo

Los resultados muestran que, salvo en la dimensión Sensations, el modelo tiene un efecto estadísticamente significativo en todas las dimensiones (p < 0.001).
Aunque los tamaños de efecto son pequeños (η² ≈ 0.017–0.019), esto indica que los modelos sí generan respuestas consistentemente distintas, incluso si las diferencias no son grandes en magnitud práctica.

Esto confirma que la elección del modelo modifica sistemáticamente la forma en que se producen los descriptores crossmodales.

#### Efecto Principal del Alimento

El tipo de alimento no produce diferencias relevantes en la mayoría de las dimensiones.
Solo Sensations muestra significancia estadística (p ≈ 1e-5), pero con un efecto extremadamente pequeño (η² = 0.002), lo que implica:

- Aunque estadísticamente detectable,

- la diferencia entre alimentos es mínima,

- y la variabilidad del modelo no depende principalmente del alimento en sí, excepto en Sensations.

Esto sugiere que los modelos generan descriptores de forma bastante homogénea, sin variar demasiado según el alimento.

#### Interacción Modelo × Alimento

Aquí se encuentran los resultados más importantes:
Las dimensiones Human Responses, Temperature, Texture, Emotions, Color y Taste presentan interacciones altamente significativas, con tamaños de efecto grandes (η² = 0.161–0.180).

Esto significa que:

- El comportamiento del modelo depende del alimento específico,

- no de forma global, sino en cómo cada modelo estructura sus descriptores para cada item,

- mostrando patrones diferenciales notables entre los dos modelos.

En otras palabras, los modelos no se comportan de manera paralela: uno puede ser más estable para ciertos alimentos y más variable para otros.

#### Análisis de Cohesión y Consistencia

Se añadieron dos análisis adicionales:

Cohesión semántica

- Mide la similitud entre los términos dentro de una respuesta.

- Ambos modelos tienen valores bajos (<0.03), lo cual indica respuestas diversas, típico en sistemas generativos.

- GPT-5-nano tiene una cohesión ligeramente mayor (+0.0048), es decir, sus respuestas son un poco más coherentes internamente.

Consistencia

- Evalúa la estabilidad de las respuestas para un mismo alimento.

- GPT-4o-mini es más consistente (0.53 vs 0.44).

- GPT-5-nano es más variable; cambia más entre alimentos o entre muestras.

Interpretación:

- GPT-5-nano = más creativo, más variable.

- GPT-4o-mini = más estable y más predecible.

Por lo tanto, los análisis muestran que ambos modelos presentan diferencias significativas en la mayoría de las dimensiones evaluadas; sin embargo, **GPT-4o-mini-2024-07-18** destaca por ofrecer mayor consistencia entre muestras, menor variabilidad indeseada y un perfil más estable para generación sintética a gran escala. Aunque **GPT-5-nano** exhibe una cohesión ligeramente superior, su alta variabilidad introduce ruido que afecta los estudios comparativos. Por ello, el modelo seleccionado para la construcción del dataset sintético es **gpt-4o-mini-2024-07-18**, debido a su estabilidad y confiabilidad global.

### 5. [rag-evaluation](./rag-evaluation.ipynb)

Evaluar cuantitativamente el primer nivel del sistema RAG, encargado de recuperar descriptores gustativos desde una base de datos crossmodal. Esta evaluación se enfoca en determinar qué tan bien los embeddings de los prompts base y los embeddings de sus alimentos asociados permiten recuperar alimentos de la base crossmodal que compartan la misma categoría de sabor (sweet, salty, bitter, sour). El objetivo final es verificar la coherencia, estabilidad y consistencia del espacio vectorial utilizado por el RAG para la dimensión taste.

Utiliza los datos de:

- sentence_embeddings.csv: embeddings de los prompts textuales base.

- food_embeddings.csv: embeddings de los alimentos asociados a cada prompt.

- raw_prompts_food_taste.csv: metadatos que relacionan prompt, alimento y sabor.

Luego se filtran solo los cuatro sabores centrales y se realiza un muestreo balanceado de 18 ejemplos por categoría para garantizar una evaluación homogénea.

Recuperación desde la base crossmodal

Tras conectar a PostgreSQL, se realiza la recuperación de alimentos usando una consulta vectorial que obtiene los 100 ítems más similares según distancia coseno.

La evaluación se realiza en dos modos paralelos:

- Recuperación basada en el embedding del alimento asociado al prompt.

- Recuperación basada directamente en el embedding del prompt.

Para cada combinación prompt–food generator se evalúan:

a. Overlap@20 (coincidencias en top-k): Mide cuántos de los 20 alimentos más similares coinciden entre la recuperación basada en prompt y la basada en alimentos.

b. Correlación de similitud: Calcula la correlación entre los scores de similitud (sim_prompt vs. sim_food).

c. Spearman rank correlation: Evalúa si los rankings de los descriptores recuperados poseen un orden relativo similar.

d. Unique rate@100: Determina si dentro del top-100 aparece diversidad de alimentos (siempre fue 1.0 → no hay colapsos).

e. Frecuencias de aparición en top-k: Identifica alimentos que aparecen repetidamente en las primeras posiciones del ranking, lo cual sirve para detectar sesgos o hubs en el espacio vectorial.

El notebook demuestra que el sistema RAG tiene un desempeño coherente, estable y razonablemente alineado entre prompts textuales y alimentos asociados. Las métricas de similitud, correlación y overlap muestran que el espacio vectorial de taste está bien estructurado, sin colapsos, y con diferenciación suficiente entre categorías.

Además, los resultados permiten detectar áreas donde el modelo podría mejorar (por ejemplo, en el sabor bitter), y sientan una base clara para implementar métricas futuras que evalúen otros niveles del RAG o dimensiones adicionales como aroma o textura.

### 6. [clap_score_tasty_musicgen](./clap_score_tasty_musicgen)

Evalúa cómo cambia la calidad de la generación de audio del modelo tasty-musicgen-small cuando se utilizan dos tipos de prompts: descripciones originales no musicales y reprompts musicales detallados. Para ello se cargan dos datasets de CLAP Score, cada uno con 80 audios generados a partir de sus respectivas variantes de prompts.

El proceso consiste en comparar la distribución de puntuaciones, calcular estadísticas descriptivas, clasificar los resultados en rangos de calidad y analizar diferencias caso a caso. Los reprompts musicales muestran un aumento notable en la alineación texto–audio: elevan el CLAP Score promedio de 0.10 a 0.22, multiplican por más de cinco la cantidad de audios buenos o excelentes y generan los únicos casos con scores superiores a 0.5. También se identifica un patrón de éxito consistente en instrumentos como la marimba, mientras que timbres como el harmonium o la flauta muestran más fallas. Aunque el reprompt mejora el 70% de los casos, también introduce mayor variabilidad y algunos deterioros significativos. En conjunto, el análisis concluye que las descripciones musicales técnicas son mucho más efectivas para guiar la síntesis sonora del modelo, especialmente en prompts que inicialmente producían bajos puntajes, aunque su rendimiento depende del instrumento y del dominio aprendido por el modelo.

### 7. [analisis_dist_clap_scores](./rag-analisis_dist_clap_scores.ipynb)

Analiza el impacto del reprompting en la calidad de alineación texto–audio utilizando **CLAP scores** como métrica central.

Para ello se emplearon **80 pares de evaluaciones** en tres condiciones:

- Sin Reprompt (texto y audio originales)
- Cross-Evaluation (texto original con audio reprompt) y
- Con Reprompt (texto y audio reprompt).

Se construyó un dataframe comparativo a partir de tres archivos CSV con los CLAP scores de cada condición, y se calcularon estadísticas descriptivas, diferencias por ejemplo individual y rankings de mayores y menores mejoras. Además, se evaluó la independencia entre condiciones comparando la correlación entre Sin Reprompt y Cross-Evaluation, hallando una relación moderada (Kendall τ = 0.362; Pearson r = 0.522), lo que indica que los scores dependen parcialmente del audio utilizado. Posteriormente, una prueba t para muestras dependientes confirmó una **mejora altamente significativa** tras el reprompting (t = –5.70, p < 0.001), con un incremento medio de +0.118 puntos (+112%) y un tamaño del efecto grande (Cohen’s d = 0.91).

En conjunto, los resultados muestran que el reprompting aumenta sustancialmente la alineación texto–audio en aproximadamente el 70% de los casos, especialmente cuando los scores iniciales eran bajos, confirmando su alta efectividad dentro del proceso de generación.

### 8. [prompts-evaluation-perplexity](./prompts-evaluation-perplexity.ipynb)

Evaluar la calidad de los reprompts generados por dos modelos GPT5-nano y Kimi-K2 a partir de un conjunto común de 50 prompts base relacionados con descripciones sensoriales.

Para ello se utilizan dos archivos de datos:

- pipeline_results_gpt5_nano_50_prompt_v3.csv: los reprompts generados por GPT5-nano
- pipeline_results_kimi_k2_thinking_50_prompt_v3.csv: los reprompts producidos por Kimi-K2 en modo thinking.

Ambos archivos incluyen, para cada id_prompt, el prompt original, el reprompt generado y metadatos adicionales necesarios para la comparación.

El proceso consiste en calcular la **perplexity** de cada reprompt utilizando dos modelos externos de evaluación lingüística: **HuggingFaceTB/SmolLM2-1.7B** y **Qwen2.5-1.5B**. A través de la librería evaluate, el notebook obtiene una puntuación de perplexity para cada texto, de modo que es posible medir qué tan “predecible”, estable o bien estructurado es cada reprompt según modelos independientes. Luego, para cada id_prompt, se identifica el reprompt con la **menor perplexity**, seleccionándolo como el mejor candidato para ese ejemplo. Esto produce dos subconjuntos limpios y comparables: df_rank_nano, que contiene los mejores reprompts de GPT5-nano, y df_kimi, que agrupa los mejores resultados de Kimi-K2.

Finalmente, el notebook organiza y guarda estos datos procesados, permitiendo observar qué modelo genera reprompts más consistentes de acuerdo con la métrica empleada. En conjunto, el análisis proporciona una evaluación sistemática de ambos generadores y demuestra que la perplexity es útil para priorizar reprompts más claros y estructurados, facilitando etapas posteriores del pipeline de generación sensorial-musical.

### 9. [analisis_clap_scores_taste](./analisis_clap_scores_taste.ipynb)

Examina cómo el reprompting influye en la calidad de alineación texto-audio, medida mediante CLAP scores, considerando específicamente las diferencias entre cuatro categorías de sabor: sweet, salty, bitter y sour.

Para ello se utilizaron 80 pares de evaluaciones correspondientes a prompts originales y sus versiones reescritas (reprompts), todos con clasificación válida de sabor. Los datos provienen de cuatro archivos: los CLAP scores del prompt original, del reprompt, de la evaluación cruzada (cross-evaluation), y de un archivo auxiliar que asigna la categoría de taste a cada id_prompt. La comparación se realizó mediante pruebas estadísticas por categoría, análisis de diferencias medias, correlaciones de Kendall y la identificación de casos extremos de mejora y empeoramiento.

Los resultados muestran que el reprompting incrementa de forma notable la calidad global del alineamiento texto-audio, pasando de un CLAP score promedio de 0.1047 a 0.2224, lo que representa un aumento del 112%. Sin embargo, este beneficio no es uniforme entre sabores. Las categorías salty, sour y bitter presentan mejoras estadísticamente significativas según pruebas t pareadas, mientras que sweet muestra sólo una tendencia positiva marginal sin alcanzar significancia estadística. De manera consistente, las categorías que iniciaban con CLAP scores más bajos —como salty— son también las que experimentan los mayores incrementos relativos.

El análisis también revela que la correlación entre los CLAP scores del prompt original y del reprompt es débil en todas las categorías, lo que indica que el reprompting modifica sustancialmente el contenido semántico relevante para el modelo CLAP, más que simplemente refinar la calidad del texto original. Finalmente, la inspección de casos extremos confirma que cada categoría incluye ejemplos con mejoras pronunciadas, pero también algunos casos aislados de empeoramiento, lo que sugiere que el proceso de reprompting no es igualmente estable para todos los tipos de descriptores sensoriales.

En conjunto, los hallazgos demuestran que el reprompting es una estrategia eficaz para mejorar la alineación texto-audio, con beneficios especialmente altos en categorías donde la relación inicial entre texto y sabor es más débil. Esto respalda el uso del reprompt como paso clave para mejorar la coherencia semántica en pipelines de generación condicionada por descripciones sensoriales.

### 10. [fad_evaluation](./fad_evaluation.ipynb)

Calcula la distancia Fréchet Audio Distance (FAD) entre diferentes conjuntos de muestras de audio utilizando embeddings CLAP. FAD es una métrica utilizada para evaluar la calidad y diversidad del audio generado comparando la distribución de embeddings entre muestras reales y generadas.

**Puntajes FAD más bajos indican mejor calidad** - la distribución del audio generado está más cerca de la distribución del audio de referencia.

#### 1. FAD General: Prompt vs. Reprompt

La distancia FAD entre los audios generados con el prompt original (audios_without_reprompt) y los audios generados con la estrategia de prompting (audios_with_reprompt) es de 0.1264. Este valor, al ser inferior al umbral de audio limpio (0.2), indica una **alta similitud perceptual** entre los dos conjuntos de audios. Esto sugiere que las diferencias introducidas por el proceso de prompting fueron mínimas en términos de las características de audio subyacentes. En otras palabras, la manipulación de la estructura textual del prompt no alteró drásticamente la distribución sónica global de los audios generados.

Por lo tanto:

- El prompting produce audios cuyo espacio de embeddings está muy cerca del de los audios originales.

- La transformación introducida por el prompting no altera significativamente la estructura estadística del audio.

- Los embeddings de CLAP para ambos conjuntos se solapan bien, lo que indica una buena preservación del estilo/acústica global.

Conclusión: El prompting no distorsiona ni cambia demasiado el perfil general del audio, más bien actúa como una refinación suave, conservando propiedades globales del sonido.

#### 2. FAD General: Spanio vs. Reprompt

La distancia FAD entre los audios base (audios_spanio) y los audios reprompted es de 0.4397. Este puntaje es significativamente más alto que el puntaje Prompt vs. Reprompt. Un FAD de 0.4397 indica una distancia perceptual notable entre los audios de la línea base de Spanio y los audios obtenidos con el prompting. La ingeniería de prompts (prompting) fue efectiva para forzar la generación de audios con características sónicas fundamentalmente distintas a las del conjunto de audio de la línea base (Spanio). Este hallazgo es crucial, ya que aísla la influencia de la entrada textual como el principal motor de las diferencias percibidas en la salida del modelo.

#### 3. Puntajes FAD por Sabor (Spanio vs Reprompted)

| Categoría | FAD    | Interpretación                                                                     |
| --------- | ------ | ---------------------------------------------------------------------------------- |
| Sweet     | 0.5893 | Distancia alta, indicando una diferencia marcada respecto a la línea base.         |
| Bitter    | 0.7034 | Máxima distancia, sugiriendo la mayor divergencia sónica respecto a la línea base. |
| Salty     | 0.5969 | Distancia alta, con una diferencia sónica significativa.                           |
| Sour      | 0.5381 | Distancia más baja, indicando la mayor similitud con la línea base.                |

La tendencia observada en los puntajes FAD por sabor revela una divergencia significativa entre la distribución sónica de los audios reprompted y los audios base de Spanio para todas las categorías.

- Amargo (Bitter): Con un FAD de 0.7034, este sabor presentó la mayor distancia perceptual con respecto al conjunto de audio de la línea base. Esto implica que la estrategia de prompting fue más eficiente y poderosa al invocar las características sónicas asociadas al amargor (como puede ser un tono bajo, instrumentación de trombón o música negativa) que el modelo base.

- Salado (Salty) y Dulce (Sweet): Estos sabores mostraron distancias FAD elevadas (0.5969 y 0.5893, respectivamente), indicando que la nueva estrategia de prompting indujo una diferenciación sónica robusta respecto a la línea base para estas categorías, validando la influencia del prompting en la generación de características asociadas (consonancia, legato para dulce, y staccato, tono medio para salado).

- Acidez (Sour): Aunque el puntaje 0.5381 sigue siendo alto, representa la menor divergencia sónica entre los sabores evaluados. Esto sugiere que, o bien el perfil de acidez del conjunto base de Spanio era intrínsecamente similar al perfil generado por el prompting, o que el prompting fue marginalmente menos efectivo para manipular las características acústicas asociadas a la acidez (como tono alto, disonancia) en comparación con los otros sabores.

## Notebooks de experimentación

### 1. [GenMusicSpanio](./gen-audios-spanio.ipynb)

A partir de los [prompts mejorados de Spanio](https://github.com/matteospanio/taste-music-dataset/blob/main/descriptions.json) se implementó un Pipeline que genera piezas musicales a partir del modelo [tasty-musicgen-small](https://github.com/matteospanio/tasty-musicgen-small), el total de piezas músicales es de 100, las cuáles son evaluadas a partir del CLAP Score.

### 2. [GenMusicSpanioBase](./gen-audios-spanio-base-prompts.ipynb)

A partir de los prompts base de Spanio:

- "sweet music, ambient for fine restaurant"
- "bitter music, ambient for fine restaurant"
- "sour music, ambient for fine restaurant"
- "salty music, ambient for fine restaurant"

Se generan 25 piezas musicales por cada sabor, para un total de 100.

### 3. [ComparisonCLAPWeights](./comparison-clap-weights.ipynb)

Se realiza una comparación de los resultados de CLAP a partir de la variación de los pesos por defecto a los [especializados en Music Audioset](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt).

### 4. [TSNEEnableFusion](./TSNE-enable-fusion.ipynb)

Evaluar cómo el parámetro enable_fusion afecta la distribución y contraste de los CLAP scores entre distintos descriptores de sabor (sweet, salty, bitter, sour). Para ello se utiliza t-SNE para validar si la fusión (enable_fusion=True) está “aplacando” la separación entre descriptores o si, por el contrario, los embeddings sin fusión mantienen mejor la discriminación semántica.

### 5. [tsne-music-caps](./tsne-music-caps.ipynb)

En esta actividad se trabajó exclusivamente con los embeddings de AudioCaps, un conjunto de datos que contiene descripciones textuales asociadas a fragmentos de audio. El propósito principal fue analizar la estructura latente del espacio de embeddings generado a partir de estos datos, con el fin de identificar patrones semánticos y posibles agrupamientos naturales entre las representaciones vectoriales.

Para ello, se aplicaron técnicas de reducción de dimensionalidad (como t-SNE y PCA) y métodos de agrupamiento (DBSCAN y K-Means) sobre los embeddings, evaluando la coherencia y densidad del espacio a través de métricas como la distancia intra e inter-cluster, el Silhouette Score y el Davies–Bouldin Index.

### 6. [clap-encoder-spanio-audiocaps](./clap-encoder-spanio-audiocaps.ipynb)

El propósito de este componente es construir una representación vectorial (embedding) para cada descriptor musical o caption del dataset MusicCaps, empleando el modelo CLAP_Module, que fue previamente entrenado para alinear información texto–audio.
Cada embedding representa el contenido semántico de un descriptor musical en un espacio latente multimodal.

### 7. [music-audio-generator](./music-audio-generator.ipynb)

Este notebook implementa un pipeline completo que carga descripciones musicales desde un archivo CSV o JSON. En este caso 100 prompts provenientes de rag_spanio_captions.csv, utiliza el modelo text-to-audio csc-unipd/tasty-musicgen-small para generar un archivo de audio WAV por cada descripción, guarda los 100 audios producidos en una carpeta organizada y finalmente comprime todos los archivos generados en un ZIP listo para descarga o uso posterior.

### 8. [Prompt Engineering Experiments](./prompt_engineering_experiments.ipynb)

Este notebook implementa un experimento automatizado que evalúa cómo distintas plantillas de prompts influyen en las descripciones sensoriales, emocionales y perceptuales generadas por un modelo de lenguaje para una lista de alimentos. Para ello, carga un conjunto de 70 ítems, aplica varias plantillas, genera múltiples muestras por alimento y registra metadatos como tokens usados y estructura de salida. Las respuestas se formatean en dimensiones (sensaciones, respuestas fisiológicas, temperatura, textura, emociones, color y sabor), se dividen en columnas y se guardan en un dataset final que permite analizar consistencia, variabilidad, efectos del diseño del prompt y patrones semánticos mediante métodos como ANOVA, clustering o t-SNE.

### 9. [Prompt Engineering Lexical Diversity](./prompt_engineering_lexical_diversity.ipynb)

Realiza un análisis sistemático de los descriptores sensoriales generados por un modelo de lenguaje con el fin de evaluar la diversidad léxica, la consistencia semántica y el impacto de dos variaciones de prompt en la calidad de las respuestas. Para ello, se utilizan datasets compuestos por descripciones generadas para uno y treinta alimentos, con un total de 10 muestras por alimento. Las respuestas provienen del modelo gpt-4o-mini-2024-07-18, y cada una está organizada en siete dimensiones semánticas: sensations, human responses, temperature, texture, emotions, color y taste.

Procesa las respuestas mediante un pipeline que tokeniza los términos por dimensión, calcula frecuencias y obtiene métricas de diversidad léxica como número de términos únicos, proporción de diversidad, entropía y frecuencia máxima. Los resultados muestran que, en dimensiones abiertas como sensations, human responses y texture, el modelo produce una diversidad moderada, mientras que en dimensiones con espacio de respuesta limitado —temperature, color, emotions y taste— la diversidad es baja y altamente consistente. En el análisis de varias comidas, las dos variaciones de prompt presentan patrones similares, aunque la Variación 2 demuestra mayor estabilidad, menos alucinaciones y un cumplimiento más sólido del formato esperado. En contraste, la Variación 3 muestra una diversidad ligeramente mayor, pero esta proviene principalmente de términos no válidos o inconsistentes.

En conclusión, se evidencia que el modelo es capaz de generar descripciones sensoriales estructuradas con un nivel de consistencia adecuado, y que la formulación del prompt influye directamente en la calidad, diversidad y precisión de los descriptores.

### 10. [Evaluacion RAG](./evaluacion%20RAG.ipynb)

Construcción y validación de un sistema RAG basado en descriptores musicales para guiar la generación de piezas mediante el modelo tasty-musicgen-small. Usando el dataset MusicCaps, se creó un vocabulario único de descriptores y se generaron embeddings con CLAP, almacenados en una base de datos vectorial para recuperación semántica. La validación incluyó la construcción de pseudo-prompts basados en los top-10 descriptores recuperados, la generación de audios, el cálculo de embeddings y la comparación con piezas originales de la base de datos de Guedes mediante CLAPScore y Fréchet Audio Distance (FAD). Los resultados muestran un RAG altamente coherente (similitud coseno promedio ≈ 0.9995) y una alta similitud entre audios generados y reales (FAD entre 0.0096 y 0.0125), indicando que la estrategia produce recuperaciones estables y audios generados consistentes con el dataset de referencia.

### 11. [audio-models-generation](./audio-models-generation.ipynb)

Evaluar cómo diferentes modelos de lenguaje (Kimi-K2, GPT5-Nano y variantes de reprompting) influyen en la generación de música cuando sus descripciones textuales se utilizan como entrada para el modelo tasty-musicgen-small. Para ello se emplean distintos archivos CSV que contienen prompts y reprompts generados por estos modelos, incluyendo versiones de 3 ítems, 50 ítems y una versión final V3. A partir de estos datos textuales, el notebook genera clips de audio para cada prompt mediante un pipeline text-to-audio, guarda los resultados en archivo WAV, estructura cada conjunto por carpeta y finalmente los comprime en archivos ZIP.

El proceso incluye la carga de prompts, su iteración para la síntesis sonora, la asignación de identificadores únicos y el manejo automático del almacenamiento para cada experimento. En conjunto, se generaron múltiples colecciones de audios correspondientes a cada modelo y versión de prompt, permitiendo comparar cómo las variaciones en la formulación textual afectan la calidad y consistencia del audio producido. Como conclusión general, el notebook establece un flujo reproducible de generación masiva de audio y revela que el diseño y la riqueza de los prompts (especialmente los reprompts estructurados) tienen un impacto directo en la expresividad y coherencia musical que el modelo de síntesis logra producir.
