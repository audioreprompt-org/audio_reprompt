# Diccionario de Datos y Estructura de Almacenamiento

Este documento describe la estructura de los datos utilizados en el proyecto **Audio Reprompt**.

> **⚠️ Nota Importante sobre Archivos Grandes (DVC):**
> Este repositorio utiliza **DVC (Data Version Control)** para gestionar archivos pesados (audios `.wav`, `.mp3` y modelos binarios).
> Si usted está revisando este código sin acceso al almacenamiento remoto de DVC, es normal que las carpetas de audio (`tracks/`) aparezcan vacías o solo contengan archivos `.dvc`. La estructura lógica se describe a continuación.

---

## 1. Metadatos y Embeddings (`data/docs`)

Esta carpeta contiene la información semántica, descriptores de texto y los vectores (embeddings) pre-calculados utilizados por el sistema RAG y los modelos de evaluación.

### `audio_caps_embeddings.csv`
Contiene los embeddings generados a partir de descripciones de audio (AudioCaps).

| text | embedding |
| :--- | :--- |
| spooky sounds | `[Vector de Embedding (Float32) ...]` |
| scare as of agony | `[Vector de Embedding (Float32) ...]` |

### `descriptions.json`
Base de conocimientos de descripciones musicales.
* **Tipo:** Lista de objetos JSON.
* **Contenido:** `id`, `instrument`, `description`.

Ejemplo:
    {
      "id": 1,
      "instrument": "piano",
      "description": "A sweet melancholic piano piece."
    }

### `guedes_audio_embeddings.csv`
Asocia embeddings de audio con perfiles de sabor (dulce, amargo, ácido, salado).

| id | audio_embedding | sweet_rate | bitter_rate | sour_rate | salty_rate |
|---:|:---|---:|---:|---:|---:|
| 1 | `[Vector de Embedding ...]` | 0.46 | 0.2 | 0.11 | 0.23 |
| 2 | `[Vector de Embedding ...]` | 0.48 | 0.3 | 0.9 | 0.13 |

### `guedes_descriptor_dominance.csv`
Matriz de dominancia de sabores por ID de pista, utilizada para validar la percepción sensorial.

| id | sweet_rate | bitter_rate | sour_rate | salty_rate |
|---:|---:|---:|---:|---:|
| 1 | 0.46 | 0.2 | 0.11 | 0.23 |

### `rag_audio_embeddings.csv`
Base de vectores utilizada por el sistema **RAG (Retrieval-Augmented Generation)** para buscar audios similares basados en el prompt del usuario.

| audio_id | embedding |
| :--- | :--- |
| 1.wav | `[Vector de Embedding RAG ...]` |
| 2.wav | `[Vector de Embedding RAG ...]` |

### `rag_spanio_captions.csv`
Relaciona las descripciones originales con los prompts enriquecidos y su distancia semántica.

| id | source_caption | prompt | distance |
|---:|:---|:---|---:|
| 1 | sweet melancholic piano piece | arpeggiated sustain mellow piano... | 0.00027 |

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

| text | embedding |
| :--- | :--- |
| sweet melancholic piano piece | `[Vector de Embedding ...]` |

---

## 2. Prompts (`data/prompts`)

### `spanio_prompts.csv`
Contiene los prompts base utilizados para las pruebas de generación.

| id | instrument | description |
|---:|:---|:---|
| 1 | piano | A sweet melancholic piano piece. |
| 2 | piano and strings | A bitter-sweet dreamy piano piece. |

---

## 3. Resultados y Métricas (`data/scores`)

Esta carpeta almacena los resultados de los experimentos de evaluación automática utilizando la métrica **CLAP**. Los archivos representan diferentes configuraciones de pesos y fusión de modelos.

**Estructura general de los archivos de resultados (`results_*.csv`):**

| Columna | Descripción |
| :--- | :--- |
| **id** | Identificador único del experimento o track. |
| **instrument** | Instrumento principal (si aplica). |
| **taste** | Sabor objetivo (sweet, sour, bitter, salty). |
| **description / prompt** | El texto utilizado para generar el audio. |
| **audio_path** | Ruta relativa al archivo generado. |
| **clap_score** | **Puntaje de similitud (CLAP)** entre el audio generado y el texto. Mayor es mejor. |

**Archivos principales:**
* `results_with_clap.csv`: Resultados generales.
* `results_with_clap_base.csv`: Línea base de comparación.
* `results_with_clap_base_prompts_*.csv`: Variaciones de experimentos habilitando/deshabilitando fusión de pesos en el modelo CLAP y MusicGen.

---

## 4. Almacenamiento de Audio (`data/tracks`)

> **Nota:** Esta carpeta es gestionada por DVC. Si no ha ejecutado `dvc pull`, estas carpetas existirán pero estarán vacías.

Aquí se organiza la salida de audio del sistema:

* **📂 generated_base_music/**: Contiene audios de referencia base (línea base) generados sin técnicas de reprompting avanzado. (Ej. `sweet_01.wav`, `bitter_14.wav`).
* **📂 generated_music/**: Contiene los audios finales generados por el sistema **Audio Reprompt** utilizando el pipeline completo (LLM + MusicGen).
* **📂 guedes_music/**: Dataset de referencia proveniente de estudios previos (Guedes) usado para calibración.
* **📂 rag_music/**: Fragmentos de audio recuperados por el sistema RAG para enriquecer el contexto (few-shot learning) del modelo de generación.
