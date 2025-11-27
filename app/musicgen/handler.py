import os
import base64

import runpod
import torch

from models.musicgen.model import load_musicgen_pipeline
from models.musicgen.audio_generator import generate_audio_base64_from_prompt

DEFAULT_MODEL_PATH = os.getenv("MODEL_DIR")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_pipe = None


def _get_pipeline():
    """
    Inicializa perezosamente el pipeline de MusicGen usando tu función
    `load_musicgen_pipeline` del módulo `models.musicgen.model`.
    """
    global _pipe
    if _pipe is None:
        _pipe = load_musicgen_pipeline(model_name=DEFAULT_MODEL_PATH)
        print(f"[MusicGen] Pipeline cargado en {DEVICE}")
    return _pipe


def handler(event):
    """
    Healthcheck:
      event: { "type": "health" }
      -> { "ok": true } o { "error": "..." }

    Inferencia:
      event: {
        "input": {
          "prompt": "..."
        }
      }

      -> {
           "prompt": str,
           "audio_wav_size": int,
           "audio_base64": str
         }
    """
    event = event or {}

    # Healthcheck sencillo
    if event.get("type") == "health":
        try:
            # Si no quieres calentar la GPU aquí, podrías omitir esto
            _get_pipeline()
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}

    inp = (event.get("input") or {}) or {}

    prompt = inp.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Campo 'prompt' es obligatorio y debe ser un string no vacío.")
    prompt = prompt.strip()

    try:
        pipe = _get_pipeline()

        # Usar tu helper para generar el audio en base64
        audio_b64 = generate_audio_base64_from_prompt(prompt, pipe)

        # Calcular el tamaño en bytes del WAV a partir del base64
        audio_wav_size = len(base64.b64decode(audio_b64))

        return {
            "prompt": prompt,
            "audio_wav_size": audio_wav_size,
            "audio_base64": audio_b64,
        }
    except Exception as e:
        return {"error": f"Fallo en generación de audio: {e}"}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
