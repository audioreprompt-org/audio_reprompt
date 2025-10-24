import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import numpy as np
import scipy.io.wavfile
from models.musicgen.typedefs import OutputGeneratedAudioItem
from tqdm import tqdm

logger = logging.getLogger(__name__)


def generate_music_from_prompts(
    synthesiser,
    dataset,
    output_dir: str = "generated_music",
    sample_rate: int = 32000,
    batch_size: int = 4,
    max_workers: int = 4,
) -> List[OutputGeneratedAudioItem]:
    """
    Parameters:
        synthesiser: pipeline de HuggingFace para texto a audio.
        dataset: cargador del conjunto de prompts.
        output_dir: carpeta donde se guardan los archivos generados.
        sample_rate: frecuencia de muestreo por defecto si el modelo no la indica.
        batch_size: cantidad de prompts a procesar por lote.
        max_workers: número de hilos concurrentes para escritura en disco.

    Returns:
    Lista de objetos OutputGeneratedAudioItem.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[OutputGeneratedAudioItem] = []
    records = dataset.records
    logger.info("generando audios para %d prompts", len(records))

    prompts = [r["prompt"] for r in records]
    ids = [r["id"] for r in records]

    # Inferencia en Batch
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating audio"):
        batch_prompts = prompts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]

        try:
            outputs = synthesiser(batch_prompts, forward_params={"do_sample": True})
            if not isinstance(outputs, list):
                outputs = [outputs]

            # guarda audios de forma asincrona eficiente
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for record_id, output, prompt in zip(batch_ids, outputs, batch_prompts):
                    futures.append(
                        executor.submit(
                            _save_audio_file,
                            output_dir,
                            record_id,
                            output,
                            sample_rate,
                            prompt,
                        )
                    )
                for f in as_completed(futures):
                    res = f.result()
                    if res:
                        results.append(res)
        except Exception as e:
            logger.error(f"error en batch %d", {i // batch_size})
            logger.critical(e, exc_info=True)

    logger.info("%d archivos de audio generados en %s", len(results), output_dir)
    return results


def _save_audio_file(
    output_dir: str, file_id: str, output: dict, sample_rate: int, prompt: str
) -> OutputGeneratedAudioItem | None:
    """Guarda audio en disco de forma eficiente"""
    try:
        output_path = os.path.join(output_dir, f"{file_id}.wav")
        audio_data = output["audio"]

        # 16-bit PCM for .wav compatibility
        if audio_data.dtype != np.int16:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = (audio_data / max_val * 32767).astype(np.int16)

        sr = output.get("sampling_rate", sample_rate)
        scipy.io.wavfile.write(output_path, rate=sr, data=audio_data)

        return OutputGeneratedAudioItem(
            id=file_id, prompt=prompt, audio_path=output_path
        )
    except Exception as ex:
        logger.error("error guardando archivo: %s", file_id)
        logger.critical(ex, exc_info=True)
        return None
