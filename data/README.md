# Diccionario de Datos y Estructura de Almacenamiento

Este documento describe la estructura de los datos utilizados en el proyecto **Audio Reprompt**.

> **⚠️ Nota Importante sobre Archivos Grandes (DVC):**
> Este repositorio utiliza **DVC (Data Version Control)** para gestionar archivos pesados (audios `.wav`, `.mp3` y modelos binarios).
> Si usted está revisando este código sin acceso al almacenamiento remoto de DVC, es normal que las carpetas de audio (`tracks/`) aparezcan vacías o solo contengan archivos `.dvc`. La estructura lógica se describe a continuación.

---

## 1. Datos Crossmodales de Alimentos (`cleaned/food_prompts/*.csv`)

Los datos disponibles en estas carpetas relacionan los datos de batch y los resultados obtenidos en la construcción del dataset de crossmodalidad.
A continuación se describen los esquemas de datos:

# Datos de Solicitud (Request OpenAI)

| Columna       | Descripción                                                                            |
| :------------ | :------------------------------------------------------------------------------------- |
| **custom_id** | Identificador unico que relaciona una solicitud a OpenAI.                              |
| **batch_id**  | Identificador unico que relaciona el batch en que la solicitud fue realizada a OpenAI. |
| **file_id**   | Archivo que contiene la solicitud en formato JSON que se realizó a OpenAI.             |
| **food_item** | Texto que indica el alimento para el que se realizó la solicitud.                      |

```csv
custom_id,batch_id,file_id,food_item
733aba5_c25181d0-8bde-431c,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,corn tamale
733aba5_4161aad0-aec9-44e7,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,filet o fish mcdonalds
733aba5_66bb262c-b662-466d,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,endive
733aba5_d388bef4-d82e-4225,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,shiitake mushrooms dried
733aba5_20a09f69-247c-4991,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,pineapple orange juice
733aba5_1e39916c-c576-4ac9,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,brazilnuts dried
733aba5_bdb0ac80-73ee-4b78,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,veal heart cooked
733aba5_b9980b2b-6080-4346,batch_691dd573c29c8190862381e0e54ee562,file-Bq75M9mYC5VmBzzLkHpFzi,hotdog roll
```

# Datos de Respuesta (Food Item - Crossmodales)

| Columna           | Descripción                                                                                                         |
| :---------------- | :------------------------------------------------------------------------------------------------------------------ |
| **custom_id**     | Identificador unico que relaciona una solicitud a OpenAI.                                                           |
| **food_item**     | Texto que indica el alimento para el que fue obtenido la respuesta.                                                 |
| **food_captions** | Valores de crossmodalidad separados por dimensiones (con el caracter pipe) y por valores distintos (caracter coma). |
| **model**         | Modelo utilizado para la solución de la encuesta.                                                                   |
| **in_tokens**     | Numero de tokens que hacen parte del prompt de solicitud.                                                           |
| **out_tokens**    | Numero de tokens que hacen parte de la salida generada.                                                             |
| **total_tokens**  | Suma de los tokens de entrada y salida.                                                                             |

```csv
custom_id,food_item,food_captions,model,in_tokens,out_tokens,total_tokens
733aba5_115ed10c-afc8-4979,vegetable oil spread fat free,No Label,gpt-4o-mini-2024-07-18,205,2,207
733aba5_ca76aa3e-554a-4b15,cuttlefish cooked,"Umami, briny, earthy|satisfaction, warmth, fullness|hot|firm, tender, chewy|happiness|brown|savory",gpt-4o-mini-2024-07-18,203,30,233
733aba5_b34c01da-455b-4ee9,pupusas del cerdo,"Umami, savory, spicy|satisfaction, warmth, fullness|hot|chewy, moist, soft|happy|brown|salty",gpt-4o-mini-2024-07-18,205,29,234
733aba5_3a3da05c-547f-4549,red wine vinegar,"acetic, tangy, fruity|heartburn, digestive, refreshing|cold|sharp, thin, acidic|disgust|red|sour",gpt-4o-mini-2024-07-18,202,30,232
733aba5_8bd46255-1635-4df1,black tea,"astringent, malty, floral|relaxation, alertness, warmth|hot|smooth, medium, brisk|nostalgic|brown|bitter",gpt-4o-mini-2024-07-18,201,34,235
733aba5_b47ce022-14e9-4dc6,chocolate rasberry protein mini proteinfx,"berry, chocolate, creaminess|energy boost, satisfaction, craving|warm|smooth, chewy, creamy|happiness|red|sweet",gpt-4o-mini-2024-07-18,207,28,235
733aba5_4f89766f-0015-462e,carignane red wine,"berry, spice, oak|relaxation, warmth, alertness|warm|smooth, velvety, bold|nostalgic|purple|dry",gpt-4o-mini-2024-07-18,204,32,236
733aba5_66bb262c-b662-466d,endive,"bitter, peppery, crunchy|digestive, refreshing, energizing|cold|crispy, crunchy, fibrous|surprise|green|bitter",gpt-4o-mini-2024-07-18,201,33,234
```

## 2. Metadatos y Embeddings (`data/docs`)

Esta carpeta contiene la información semántica, descriptores de texto y los vectores (embeddings) pre-calculados utilizados por el sistema RAG y los modelos de evaluación.

### `audio_caps_embeddings.csv`

Contiene los embeddings generados a partir de descripciones de audio (AudioCaps).

| text              | embedding                             |
| :---------------- | :------------------------------------ |
| spooky sounds     | `[Vector de Embedding (Float32) ...]` |
| scare as of agony | `[Vector de Embedding (Float32) ...]` |

### `descriptions.json`

Base de conocimientos de descripciones musicales.

- **Tipo:** Lista de objetos JSON.
- **Contenido:** `id`, `instrument`, `description`.

Ejemplo:
{
"id": 1,
"instrument": "piano",
"description": "A sweet melancholic piano piece."
}

### `guedes_audio_embeddings.csv`

Asocia embeddings de audio con perfiles de sabor (dulce, amargo, ácido, salado).

|  id | audio_embedding             | sweet_rate | bitter_rate | sour_rate | salty_rate |
| --: | :-------------------------- | ---------: | ----------: | --------: | ---------: |
|   1 | `[Vector de Embedding ...]` |       0.46 |         0.2 |      0.11 |       0.23 |
|   2 | `[Vector de Embedding ...]` |       0.48 |         0.3 |       0.9 |       0.13 |

### `guedes_descriptor_dominance.csv`

Matriz de dominancia de sabores por ID de pista, utilizada para validar la percepción sensorial.

|  id | sweet_rate | bitter_rate | sour_rate | salty_rate |
| --: | ---------: | ----------: | --------: | ---------: |
|   1 |       0.46 |         0.2 |      0.11 |       0.23 |

### `rag_audio_embeddings.csv`

Base de vectores utilizada por el sistema **RAG (Retrieval-Augmented Generation)** para buscar audios similares basados en el prompt del usuario.

| audio_id | embedding                       |
| :------- | :------------------------------ |
| 1.wav    | `[Vector de Embedding RAG ...]` |
| 2.wav    | `[Vector de Embedding RAG ...]` |

### `rag_spanio_captions.csv`

Relaciona las descripciones originales con los prompts enriquecidos y su distancia semántica.

|  id | source_caption                | prompt                              | distance |
| --: | :---------------------------- | :---------------------------------- | -------: |
|   1 | sweet melancholic piano piece | arpeggiated sustain mellow piano... |  0.00027 |

### `spanio_captions.json`

Diccionario de descripciones etiquetadas gramaticalmente (POS Tagging) para análisis lingüístico.

Ejemplo:
"1": {
"sweet": "JJ",
"melancholic": "JJ",
"piano": "NN"
}

### `spanio_captions_embeddings.csv`

Embeddings de texto de las descripciones del dataset Spanio.

| text                          | embedding                   |
| :---------------------------- | :-------------------------- |
| sweet melancholic piano piece | `[Vector de Embedding ...]` |

---

## 3. Prompts (`data/prompts`)

### `spanio_prompts.csv`

Contiene los prompts base utilizados para las pruebas de generación.

|  id | instrument        | description                        |
| --: | :---------------- | :--------------------------------- |
|   1 | piano             | A sweet melancholic piano piece.   |
|   2 | piano and strings | A bitter-sweet dreamy piano piece. |

### User (Raw) Prompts

Este es el esquema de datos para los prompts usados en la ejecución y evaluación de reprompt.
Estos datos se localizan en `raw/user/raw_prompts.csv`

| Columna       | Descripción                                                                    |
| :------------ | :----------------------------------------------------------------------------- |
| **food_item** | Texto que caracteriza el alimento en el prompt del usuario.                    |
| **taste**     | Texto que indica el gusto del alimento relacionado.                            |
| **emotion**   | Texto que describe la emoción del usuario en la intención.                     |
| **sentence**  | Texto (prompt) de usuario que referencia intención, lugar, emocion y alimento. |

```csv
strawberry donut,sweet,disgust,I'm at the kitchen and I'm gonna eat a strawberry donut. The room is warm and the tiles look clean.
honey glazed bun,sweet,contempt,I feel contempt and I'm gonna eat a honey glazed bun in the beach. The waves are crashing and the sand is warm.
apple pie,sweet,happiness,I'm in the train with my partner and we're gonna eat a apple pie. The seats are blue and the announcement chime sounds.
fudge brownie,sweet,surprise,I feel surprise and I'm gonna eat a fudge brownie in the cafe. The chatter is low and the window view is rainy.
fruit salad,sweet,anger,I'm gonna eat a fruit salad at the office. The desk is cluttered and the phone is ringing.
cinnamon roll,sweet,nostalgic,I'm at the restaurant and I'm gonna eat a cinnamon roll. The waiter is busy and the plates clatter softly.
```

---

## 4. Resultados y Métricas (`data/scores`)

Esta carpeta almacena los resultados de los experimentos de evaluación automática utilizando la métrica **CLAP**. Los archivos representan diferentes configuraciones de pesos y fusión de modelos.

### Estructura general de los archivos de resultados (`results_*.csv`):\*\*

| Columna                  | Descripción                                                                         |
| :----------------------- | :---------------------------------------------------------------------------------- |
| **id**                   | Identificador único del experimento o track.                                        |
| **instrument**           | Instrumento principal (si aplica).                                                  |
| **taste**                | Sabor objetivo (sweet, sour, bitter, salty).                                        |
| **description / prompt** | El texto utilizado para generar el audio.                                           |
| **audio_path**           | Ruta relativa al archivo generado.                                                  |
| **clap_score**           | **Puntaje de similitud (CLAP)** entre el audio generado y el texto. Mayor es mejor. |

**Archivos con anterior esquema de datos:**

- `results_with_clap.csv`: Resultados generales.
- `results_with_clap_base.csv`: Línea base de comparación.
- `results_with_clap_base_prompts_*.csv`: Variaciones de experimentos habilitando/deshabilitando fusión de pesos en el modelo CLAP y MusicGen.

### Estructura de archivos de evaluación de refinamiento de Prompts

| Columna        | Descripción                                                                         |
| :------------- | :---------------------------------------------------------------------------------- |
| **id_prompt**  | Identificador único del prompt.                                                     |
| **text**       | Texto (prompt) de referencia para comparación.                                      |
| **audio**      | Ruta relativa del archivo de audio generado a partir del texto                      |
| **clap_score** | **Puntaje de similitud (CLAP)** entre el audio generado y el texto. Mayor es mejor. |

```csv
1,I'm at the park and I'm gonna eat a turkey sandwich. The air is cool and the grass looks bright green.,./audio_reprompt/data/tracks/raw_prompts_audios/1.wav,0.08358676731586456
2,I'm in the cafe and I'm gonna drink a cappuccino. The room smells like coffee and the lighting is warm.,./audio_reprompt/data/tracks/raw_prompts_audios/2.wav,0.08868919312953949
3,We're at the beach and we're gonna eat two fish tacos. The breeze is salty and the sunset is golden.,./audio_reprompt/data/tracks/raw_prompts_audios/3.wav,0.09353489428758621
```

** Archivos con anterior esquema de datos:**

- `clap_score_results_prompt_outputs.csv`
- `clap_score_results_reprompt_outputs.csv`
- `clap_score_results_prompt_outputs_cross_validation.csv`

---

## 5. Almacenamiento de Audio (`data/tracks`)

> **Nota:** Esta carpeta es gestionada por DVC. Si no ha ejecutado `dvc pull`, estas carpetas existirán pero estarán vacías.

Aquí se organiza la salida de audio del sistema:

- **📂 generated_base_music/**: Contiene audios de referencia base (línea base) generados sin técnicas de reprompting avanzado. (Ej. `sweet_01.wav`, `bitter_14.wav`).
- **📂 generated_music/**: Contiene los audios finales generados por el sistema **Audio Reprompt** utilizando el pipeline completo (LLM + MusicGen).
- **📂 guedes_music/**: Dataset de referencia proveniente de estudios previos (Guedes) usado para calibración.
- **📂 rag_music/**: Fragmentos de audio recuperados por el sistema RAG para enriquecer el contexto (few-shot learning) del modelo de generación.
- **📂 raw_prompts_audios/**: Fragmentos de audio obtenidos de la generación de audio con los prompts de usuarios (raw prompts).
- **📂 reprompt_audios/**: Fragmentos de audio obtenidos de la generación de audio con los prompts refinados a partir de los prompts del usuario.
- **📂 reprompt_audio_taste/**: Relacionan los mismos fragmentos de audio del directorio `reprompt_audios` pero organizados de acuerdo al gusto `bitter`, `salty`. `sour` y `sweet`.
