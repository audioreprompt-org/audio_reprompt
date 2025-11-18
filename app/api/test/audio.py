import random
from pathlib import Path
import shutil
from uuid import UUID
import base64

from fastapi import HTTPException

ROOT_DIR = Path.cwd()
GENERATED_DIR = ROOT_DIR / "data"
TMP_CACHE_DIR = ROOT_DIR / "tmp" / "audio_cache"


def generate_local_fake_audio(audio_id: UUID) -> str:
    """
    Genera audio falso local tomando un .wav aleatorio de GENERATED_DIR,
    lo copia a un archivo temporal con el audio_id y devuelve su contenido en base64.
    """
    # Crear el directorio temporal si no existe
    TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Validar que exista el directorio con los .wav mock
    if not GENERATED_DIR.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Mock dir not found: {GENERATED_DIR}",
        )

    # Listar todos los .wav disponibles
    wavs = sorted(GENERATED_DIR.glob("*.wav"))
    if not wavs:
        raise HTTPException(
            status_code=500,
            detail=f"No .wav files found in {GENERATED_DIR}",
        )

    # Escoger un .wav aleatorio y copiarlo al cache temporal con el nombre del audio_id
    src = random.choice(wavs)
    dst = TMP_CACHE_DIR / f"{audio_id}.wav"
    shutil.copyfile(src, dst)

    # Leer el archivo copiado y convertirlo a base64
    with dst.open("rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    return audio_b64
