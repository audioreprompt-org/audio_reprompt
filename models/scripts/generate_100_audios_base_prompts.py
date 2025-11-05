import os

import torch
import scipy.io.wavfile

from transformers import pipeline
from tqdm import tqdm
from config import load_config, setup_project_paths, PROJECT_ROOT
from models.scripts.typedefs import MusicGenData


def generate_music_from_prompts(
    synthesiser,
    base_prompts,
    variations_per_prompt=25,
    output_dir="generated_music",
    sample_rate=32000,
) -> list[MusicGenData]:
    """
    Genera archivos de audio a partir de descripciones de texto usando el modelo tasty-musicgen-small.

    Parameters:
        synthesiser: Pipeline de Hugging Face para text-to-audio.
        base_prompts: Lista de prompts para generar música.
        variations_per_prompt: Número de variaciones por prompt.
        output_dir: Carpeta donde guardar los .wav generados.
        sample_rate: Frecuencia de muestreo para los archivos de salida.

    Returns:
        list[MusicGenData]: Lista con datos de cada generación.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: list[MusicGenData] = []

    print(
        f"Generando música para {len(base_prompts)} prompts con una variación de {variations_per_prompt} para cada uno.\n"
    )

    for taste_prompt in base_prompts:
        taste_name = taste_prompt.split()[0]
        print(f"Sabor: {taste_name}")
        for i in tqdm(
            range(variations_per_prompt), desc="Generando variaciones", ncols=80
        ):
            file_id = f"{taste_name}_{i + 1:02d}"
            output_path = os.path.join(output_dir, f"{file_id}.wav")

            try:
                # 1. Generar la música con el modelo.
                # output es un diccionario: audio(array NumPy con la señal de audio) y sampling_rate (frecuencia de muestreo del modelo).
                output = synthesiser(taste_prompt, forward_params={"do_sample": True})

                # 2. Extraer datos del audio.
                audio_data = output[
                    "audio"
                ]  # La onda de sonido (las muestras del audio).
                sr = output.get(
                    "sampling_rate", sample_rate
                )  # La frecuencia de muestreo reportada por el modelo.

                # 3. Guardar el audio generado.
                output_path = os.path.join(output_dir, f"{file_id}.wav")
                scipy.io.wavfile.write(
                    output_path, rate=sr, data=audio_data
                )  # Escribir el archivo .wav con la señal y la frecuencia.

                # 4. Registrar los resultados.
                results.append(
                    MusicGenData(
                        id=file_id,
                        prompt=taste_prompt,
                        audio_path=output_path,
                    )
                )
            except Exception as e:
                print(f" Error generando {file_id}: {e}")
                continue

    print(f"\n {len(results)} archivos de audio generados en: {output_dir}")
    return results


# Pipeline.
if __name__ == "__main__":
    # Variables y constantes.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    print("Configurando las rutas del proyecto...")
    setup_project_paths()

    print("Cargando configuración...")
    config = load_config()

    model_musicgen_path = PROJECT_ROOT / config.model.model_musicgen_path
    tracks_base_data_path = PROJECT_ROOT / config.data.tracks_base_data_path

    base_prompts = [
        "sweet music, ambient for fine restaurant",
        "bitter music, ambient for fine restaurant",
        "sour music, ambient for fine restaurant",
        "salty music, ambient for fine restaurant",
    ]

    variations_per_prompt = 25

    synthesiser = pipeline(
        "text-to-audio",
        model=model_musicgen_path,
        device=device,
        trust_remote_code=True,
    )

    results = generate_music_from_prompts(
        synthesiser,
        base_prompts,
        variations_per_prompt,
        tracks_base_data_path,
    )
