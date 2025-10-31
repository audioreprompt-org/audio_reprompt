# Notebooks

El presente markdown es una guía de los notebooks construidos para el proyecto:

### 1. [GenMusicSpanio](./GenMusicSpanio.ipynb)

A partir de los [prompts mejorados de Spanio](https://github.com/matteospanio/taste-music-dataset/blob/main/descriptions.json) se implementó un Pipeline que genera piezas musicales a partir del modelo [tasty-musicgen-small](https://github.com/matteospanio/tasty-musicgen-small), el total de piezas músicales es de 100, las cuáles son evaluadas a partir del CLAP Score.

### 2. [GenMusicSpanioBase](./GenMusicSpanioBase.ipynb)

A partir de los prompts base de Spanio:

- "sweet music, ambient for fine restaurant"
- "bitter music, ambient for fine restaurant"
- "sour music, ambient for fine restaurant"
- "salty music, ambient for fine restaurant"

Se generan 25 piezas musicales por cada sabor, para un total de 100.

### 3. [ComparisonCLAPWeights](./ComparisonCLAPWeights.ipynb)

Se realiza una comparación de los resultados de CLAP a partir de la variación de los pesos por defecto a los [especializados en Music Audioset](https://huggingface.co/lukewys/laion_clap/blob/main/music_audioset_epoch_15_esc_90.14.pt).

### 4. [TSNEEnableFusion](./TSNEEnableFusion.ipynb)

Evaluar cómo el parámetro enable_fusion afecta la distribución y contraste de los CLAP scores entre distintos descriptores de sabor (sweet, salty, bitter, sour). Para ello se utiliza t-SNE para validar si la fusión (enable_fusion=True) está “aplacando” la separación entre descriptores o si, por el contrario, los embeddings sin fusión mantienen mejor la discriminación semántica.

### 5. [clap-encoder-spanio-audiocaps](./clap-encoder-spanio-audiocaps.ipynb)

El propósito de este componente es construir una representación vectorial (embedding) para cada descriptor musical o caption del dataset MusicCaps, empleando el modelo CLAP_Module, que fue previamente entrenado para alinear información texto–audio.
Cada embedding representa el contenido semántico de un descriptor musical en un espacio latente multimodal.
