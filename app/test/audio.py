import random
from pathlib import Path
import shutil
from uuid import UUID

from fastapi import HTTPException
from typing import Final

GENERATED_DIR: Final[Path] = Path("data/tracks/generated_base_music")
TMP_CACHE_DIR: Final[Path] = Path("/tmp/audio_cache")


def generate_local_fake_audio(audio_id: UUID) -> Path:
    TMP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not GENERATED_DIR.is_dir():
        raise HTTPException(status_code=500, detail=f"Mock dir not found: {GENERATED_DIR}")

    wavs = sorted(GENERATED_DIR.glob("*.wav"))
    if not wavs:
        raise HTTPException(status_code=500, detail=f"No .wav files found in {GENERATED_DIR}")

    src = random.choice(wavs)
    dst = TMP_CACHE_DIR / f"{audio_id}.wav"
    shutil.copyfile(src, dst)

    return dst
