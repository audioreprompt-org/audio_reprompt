# Proyecto: Ingeniería de Prompts para la Generación de Música Sensorial: Un Enfoque Automático para la Percepción de Sabores

El proyecto tiene como propósito desarrollar un sistema automatizado que diseñe, evalúe y optimice prompts textuales para generar música con el modelo tasty-musicgen-small, de forma que las piezas resultantes potencien la percepción de sabores (dulce, salado, ácido y amargo).

## 🎯 Fases del Proyecto y Ruta de Trabajo

La metodología se organiza en tres grandes fases, cada una con tareas específicas y entregables asociados. Estas fases conforman un pipeline automatizado y cíclico que permite la mejora continua de los prompts y la música generada.

### 🔹 Fase 1: Generación de Línea Base y Preparación de Datos (Semanas 1–3)

Objetivo:

Establecer un punto de referencia medible con música generada a partir de los prompts originales, bajo condiciones controladas.

Actividades:

Semana 1 (6–12 oct):

- Configuración base de MLOps: repositorio, versionado, trazabilidad, configuración segura.

Semana 2 (13–19 oct):

- Procesamiento de los datasets (Spanio y Guedes).

- Limpieza y normalización de textos.

- Extracción de sabores y estructuras semánticas de los prompts.

- Almacenamiento seguro en AWS S3.

Semana 3 (20–26 oct):

- Generación de 100 piezas musicales con los prompts originales.

- Cálculo de métricas base: CLAPScore, Fréchet Audio Distance (FAD) y Meta Audiobox Aesthetics.

- Construcción de un dashboard de métricas y consolidación del baseline.

### Fase 2: Optimización de Prompts mediante Modelo Proxy y RePrompting (Semanas 4–6)

Objetivo:

Lograr mejoras objetivas en la alineación semántica texto-audio mediante la optimización automatizada de prompts.

Actividades:

Semana 4 (27 oct – 2 nov):

- Desarrollo del servicio automático de generación de prompts enriquecidos mediante ingeniería de prompts (heurísticas, Few-Shot, Meta Prompting).

- Entrenamiento de un modelo proxy que prediga la calidad del audio a partir del texto.

Semanas 5 y 6 (3–16 nov):

- Diseño e implementación de la rúbrica automatizada de edición de prompts, guiada por el modelo proxy.

- Generación automática de nuevos prompts con el sistema RePrompt.

- Generación de nuevas piezas musicales y cálculo de métricas.

- Comparación contra el baseline para validar si se logran mejoras de al menos:

+0.05 en CLAPScore.

–10% en FAD.

+0.2 en Meta Audiobox Aesthetics.

### Fase 3: Evaluación Final, Empaquetado y Presentación (Semanas 7–8)

Objetivo:

Entregar un sistema robusto, reproducible y validado académicamente, capaz de generar música gustativamente alineada mediante ingeniería de prompts.

Actividades:

Semana 7 (17–23 nov):

- Evaluación sistemática del rendimiento del sistema en diferentes perfiles gustativos.

- Consolidación de resultados con métricas + intervalos de confianza.

- Fortalecimiento de orquestación y monitoreo del ciclo completo.

Semana 8 (24–29 nov):

- Empaquetado reproducible del sistema.

- Preparación de demos curadas (ej. comparación entre audios antes/después de optimización).

- Elaboración de entrega académica con trazabilidad completa.

## 🗃️ Datos y Preparación

### Dataset de Prompts (Spanio et al., 2025):

100 descripciones musicales con referencias gustativas.

Variables: id, instrument, description.

Problemas: ambigüedad semántica, redundancia, categorías mezcladas.

➡ Solución: vectorización textual, separación de etiquetas gustativas, estandarización de instrumentos, representación one-hot o multi-label.

### Dataset de Guedes (2023):

100 piezas con anotaciones porcentuales por sabor.

Variables: ID_sound, sweet, bitter, sour, salty.

➡ Solución: uso de porcentajes como pesos o creación de etiquetas dominantes según necesidad.

Data

```
 ── 📁 data/
    ├── 📁 Guedes2023 (Taste&Affect)/
    │   ├── 📄 1.mp3
    │   ├── 📄 10.mp3
    │   ├── 📄 100.mp3
    │   ├── 📄 11.mp3
    │   ├── 📄 12.mp3
    │   ├── 📄 13.mp3
    │   ├── 📄 14.mp3
    │   ├── 📄 15.mp3
    │   ├── 📄 16.mp3
    │   ├── 📄 17.mp3
    │   ├── 📄 18.mp3
    │   ├── 📄 19.mp3
    │   ├── 📄 2.mp3
    │   ├── 📄 20.mp3
    │   ├── 📄 21.mp3
    │   ├── 📄 22.mp3
    │   ├── 📄 23.mp3
    │   ├── 📄 24.mp3
    │   ├── 📄 25.mp3
    │   ├── 📄 26.mp3
    │   ├── 📄 27.mp3
    │   ├── 📄 28.mp3
    │   ├── 📄 29.mp3
    │   ├── 📄 3.mp3
    │   ├── 📄 30.mp3
    │   ├── 📄 31.mp3
    │   ├── 📄 32.mp3
    │   ├── 📄 33.mp3
    │   ├── 📄 34.mp3
    │   ├── 📄 35.mp3
    │   ├── 📄 36.mp3
    │   ├── 📄 37.mp3
    │   ├── 📄 38.mp3
    │   ├── 📄 39.mp3
    │   ├── 📄 4.mp3
    │   ├── 📄 40.mp3
    │   ├── 📄 41.mp3
    │   ├── 📄 42.mp3
    │   ├── 📄 43.mp3
    │   ├── 📄 44.mp3
    │   ├── 📄 45.mp3
    │   ├── 📄 46.mp3
    │   ├── 📄 47.mp3
    │   ├── 📄 48.mp3
    │   ├── 📄 49.mp3
    │   ├── 📄 5.mp3
    │   ├── 📄 50.mp3
    │   ├── 📄 51.mp3
    │   ├── 📄 52.mp3
    │   ├── 📄 53.mp3
    │   ├── 📄 54.mp3
    │   ├── 📄 55.mp3
    │   ├── 📄 56.mp3
    │   ├── 📄 57.mp3
    │   ├── 📄 58.mp3
    │   ├── 📄 59.mp3
    │   ├── 📄 6.mp3
    │   ├── 📄 60.mp3
    │   ├── 📄 61.mp3
    │   ├── 📄 62.mp3
    │   ├── 📄 63.mp3
    │   ├── 📄 64.mp3
    │   ├── 📄 65.mp3
    │   ├── 📄 66.mp3
    │   ├── 📄 67.mp3
    │   ├── 📄 68.mp3
    │   ├── 📄 69.mp3
    │   ├── 📄 7.mp3
    │   ├── 📄 70.mp3
    │   ├── 📄 71.mp3
    │   ├── 📄 72.mp3
    │   ├── 📄 73.mp3
    │   ├── 📄 74.mp3
    │   ├── 📄 75.mp3
    │   ├── 📄 76.mp3
    │   ├── 📄 77.mp3
    │   ├── 📄 78.mp3
    │   ├── 📄 79.mp3
    │   ├── 📄 8.mp3
    │   ├── 📄 80.mp3
    │   ├── 📄 81.mp3
    │   ├── 📄 82.mp3
    │   ├── 📄 83.mp3
    │   ├── 📄 84.mp3
    │   ├── 📄 85.mp3
    │   ├── 📄 86.mp3
    │   ├── 📄 87.mp3
    │   ├── 📄 88.mp3
    │   ├── 📄 89.mp3
    │   ├── 📄 9.mp3
    │   ├── 📄 90.mp3
    │   ├── 📄 91.mp3
    │   ├── 📄 92.mp3
    │   ├── 📄 93.mp3
    │   ├── 📄 94.mp3
    │   ├── 📄 95.mp3
    │   ├── 📄 96.mp3
    │   ├── 📄 97.mp3
    │   ├── 📄 98.mp3
    │   ├── 📄 99.mp3
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📊 Data.xlsx
    │   ├── 📊 Supp_File_1_Subjective norms.xlsx
    │   └── 📊 Supp_File_1_Subjective norms_original.xlsx
    ├── 📁 Guedes2023(Bidirectionality)/
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📊 Data_1.xlsx
    │   ├── 📊 Data_2.xlsx
    │   ├── 📄 bitter.mp3
    │   ├── 📊 data_clean_between.xlsx
    │   ├── 📊 data_clean_within.xlsx
    │   └── 📄 sweet.mp3
    ├── 📁 Guedes2023(Sensitive)/
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📊 Data.xlsx
    │   ├── 📊 data_ori.xlsx
    │   ├── 📄 highsweet.mp3
    │   └── 📄 lowsweet.mp3
    ├── 📁 Guedes2023(Sweet)/
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📊 Data.xlsx
    │   ├── 📊 Data_ori.xlsx
    │   ├── 📄 highsweet.mp3
    │   └── 📄 lowsweet.mp3
    ├── 📁 Wang2015 (Whats)/
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📄 CJ_bitter.mp3
    │   ├── 📄 CJ_salty.mp3
    │   ├── 📄 CJ_sour.mp3
    │   ├── 📄 CJ_sweet.mp3
    │   ├── 📄 D_bitter.mp3
    │   ├── 📄 D_salty.mp3
    │   ├── 📄 D_sour.mp3
    │   ├── 📄 D_sweet.mp3
    │   ├── 📊 Data.xlsx
    │   ├── 📄 Ka_bitter.mp3
    │   ├── 📄 Ka_salty.mp3
    │   ├── 📄 Ka_sour.mp3
    │   ├── 📄 Ka_sweet.mp3
    │   ├── 📄 Kn_bitter.mp3
    │   ├── 📄 Kn_salty.mp3
    │   ├── 📄 Kn_sour.mp3
    │   ├── 📄 Kn_sweet.mp3
    │   ├── 📄 MB_sweet.mp3
    │   ├── 📄 MH_sour.mp3
    │   ├── 📄 MM_sweet.mp3
    │   ├── 📄 MT_sour.mp3
    │   ├── 📄 Me_salty.mp3
    │   ├── 📄 RC_bitter.mp3
    │   ├── 📄 RC_sweet.mp3
    │   └── 📄 Wa_sour.mp3
    ├── 📁 Wang2016(Striking)/
    │   ├── 📁 sounds/
    │   │   ├── 📄 pianoAcons.mp3
    │   │   ├── 📄 pianoAdiss.mp3
    │   │   ├── 📄 pianoBcons.mp3
    │   │   ├── 📄 pianoBdiss.mp3
    │   │   ├── 📄 trumpetAcons.mp3
    │   │   └── 📄 trumpetAdiss.mp3
    │   ├── 📊 AverageXsound.xlsx
    │   ├── 📊 Data_control.xlsx
    │   ├── 📄 pianoAcons.mp3
    │   ├── 📄 pianoAcons.wav
    │   ├── 📄 pianoAdiss.mp3
    │   ├── 📄 pianoAdiss.wav
    │   ├── 📄 pianoBcons.mp3
    │   ├── 📄 pianoBcons.wav
    │   ├── 📄 pianoBdiss.mp3
    │   ├── 📄 pianoBdiss.wav
    │   ├── 📦 sounds.zip
    │   ├── 📄 trumpetAcons.mp3
    │   ├── 📄 trumpetAcons.wav
    │   ├── 📄 trumpetAdiss.mp3
    │   └── 📄 trumpetAdiss.wav
    ├── 📁 Wang2017 (Spicy)/
    │   ├── 📊 AverageXsound_pre2.xlsx
    │   ├── 📊 Data_1.xlsx
    │   ├── 📊 Data_pre2.xlsx
    │   ├── 📄 ambient-1.mp3
    │   ├── 📄 ambient-2.mp3
    │   ├── 📄 ambient-3.mp3
    │   ├── 📄 articulation-1.mp3
    │   ├── 📄 articulation-2.mp3
    │   ├── 📄 articulation-3.mp3
    │   ├── 📄 attack-1.mp3
    │   ├── 📄 attack-2.mp3
    │   ├── 📄 attack-3.mp3
    │   ├── 📄 attack-decay-1.mp3
    │   ├── 📄 attack-decay-2.mp3
    │   ├── 📄 attack-decay-3.mp3
    │   ├── 📄 classical-1.mp3
    │   ├── 📄 classical-2.mp3
    │   ├── 📄 complex-1.mp3
    │   ├── 📄 complex-2.mp3
    │   ├── 📄 complex-3.mp3
    │   ├── 📄 decay-1.mp3
    │   ├── 📄 decay-2.mp3
    │   ├── 📄 decay-3.mp3
    │   ├── 📄 dissonance-1.mp3
    │   ├── 📄 dissonance-2.mp3
    │   ├── 📄 dissonance-3.mp3
    │   ├── 📄 distortion-1.mp3
    │   ├── 📄 distortion-2.mp3
    │   ├── 📄 distortion-3.mp3
    │   ├── 📄 minor-major-1.mp3
    │   ├── 📄 minor-major-2.mp3
    │   ├── 📄 percussion-1.mp3
    │   ├── 📄 percussion-2.mp3
    │   ├── 📄 pitch-1.mp3
    │   ├── 📄 pitch-2.mp3
    │   ├── 📄 pitch-3.mp3
    │   ├── 🖼️ preexp2.jpg
    │   ├── 📄 spicy.mp3
    │   ├── 📄 sweet.mp3
    │   ├── 📄 tempo-1.mp3
    │   ├── 📄 tempo-2.mp3
    │   ├── 📄 tempo-3.mp3
    │   └── 📄 whitenoise.mp3
    ├── 📁 Wang2018(A sweet smile)/
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📄 Consonant.mp3
    │   ├── 📄 Consonant.wav
    │   ├── 📊 Data.xlsx
    │   ├── 📄 Dissonant.mp3
    │   └── 📄 Dissonant.wav
    ├── 📁 Wang2021 (Metacognition)/
    │   ├── 📊 AverageXSound.xlsx
    │   ├── 📊 Data.xlsx
    │   ├── 📕 Megacognition Wang 2021.pdf
    │   ├── 📄 arousal1.mp3
    │   ├── 📄 arousal2.mp3
    │   ├── 📄 articulation1.mp3
    │   ├── 📄 articulation2.mp3
    │   ├── 📄 articulation3.mp3
    │   ├── 📄 attack1.mp3
    │   ├── 📄 attack2.mp3
    │   ├── 📄 attack3.mp3
    │   ├── 📄 complexity1.mp3
    │   ├── 📄 complexity2.mp3
    │   ├── 📄 complexity3.mp3
    │   ├── 📄 consonance1.mp3
    │   ├── 📄 consonance2.mp3
    │   ├── 📄 consonance3.mp3
    │   ├── 📄 decay1.mp3
    │   ├── 📄 decay2.mp3
    │   ├── 📄 decay3.mp3
    │   ├── 📄 mode1.mp3
    │   ├── 📄 mode2.mp3
    │   ├── 📄 randomness1.mp3
    │   ├── 📄 randomness2.mp3
    │   ├── 📄 randomness3.mp3
    │   ├── 📄 reverberation1.mp3
    │   ├── 📄 reverberation2.mp3
    │   ├── 📄 reverberation3.mp3
    │   ├── 📄 roughness1.mp3
    │   ├── 📄 roughness2.mp3
    │   ├── 📄 roughness3.mp3
    │   ├── 📄 syncopation1.mp3
    │   ├── 📄 syncopation2.mp3
    │   ├── 📄 syncopation3.mp3
    │   ├── 📄 tempo1.mp3
    │   ├── 📄 tempo2.mp3
    │   ├── 📄 tempo3.mp3
    │   ├── 📄 valence1.mp3
    │   └── 📄 valence2.mp3
    └── 📊 SoundsXRatingsAllExps.xlsx
```
