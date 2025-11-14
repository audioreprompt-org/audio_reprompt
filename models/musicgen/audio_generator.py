import base64
import logging
import os
import io
from scipy.io.wavfile import write as wav_write

import scipy.io.wavfile
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_audio_from_prompts(
    synthesiser, dataset, output_dir="generated_music", sample_rate=32000
):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    print(f"Generando música para {len(dataset)} prompts...\n")

    for record in tqdm(dataset.records):
        text_prompt = record["prompt"]
        file_id = record["id"]

        try:
            # 1. Generar la música con el modelo.
            # output es un diccionario: audio(array NumPy con la señal de audio) y sampling_rate (frecuencia de muestreo del modelo).
            output = synthesiser(text_prompt, forward_params={"do_sample": True})

            # 2. Extraer datos del audio.
            audio_data = output["audio"]  # La onda de sonido (las muestras del audio).
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
                {"id": file_id, "description": text_prompt, "audio_path": output_path}
            )
        except Exception as e:
            print(f" Error generando {file_id}: {e}")
            continue

    print(f"{len(results)} archivos de audio generados en: {output_dir}")
    return results



def generate_audio_base64_from_prompt(text_prompt, synthesiser) -> str:
    # 1. Ejecutar el modelo con el prompt de texto
    output = synthesiser(text_prompt, forward_params={"do_sample": True})

    # 2. Extraer los datos de audio y la frecuencia de muestreo
    audio_data = output["audio"]            # np.ndarray con las muestras de audio
    sr = output.get("sampling_rate")        # Frecuencia de muestreo (int)

    # 3. Escribir el WAV en un buffer en memoria (BytesIO)
    buf = io.BytesIO()
    wav_write(buf, sr, audio_data)
    buf.seek(0)

    # 4. Convertir el contenido del buffer a base64
    audio_bytes = buf.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    return audio_b64