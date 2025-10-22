import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch.nn.functional as F
import torchaudio
import torch.hub

from models.descriptors.model import clap_model
from models.descriptors.spanio_captions import load_spanio_captions

logger = logging.getLogger(__name__)
_TARGET_SR = 48_000


@torch.no_grad()
def _load_audio_tensor(path: Path, device: str) -> torch.Tensor:
    """Load wav/mp3, resample to 48k, mono."""
    wav, sr = torchaudio.load(str(path))
    if sr != _TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, _TARGET_SR)
    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # mono
    elif wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav.to(device)


def get_audio_embeddings(descriptors: list[str], paths: Iterable[Path]):
    model = clap_model()
    device = next(model.parameters()).device.type
    embeddings = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            try:
                wav = _load_audio_tensor(p, device)
                emb = model.get_audio_embedding_from_data(wav, use_tensor=True)
                emb = F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().tolist()
                embeddings.append({"text": descriptors[i], "embedding": emb})
            except Exception as e:
                logger.error(f"Failed embedding {p}: {e}")
    return embeddings


def _find_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".mp3"}
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])


if __name__ == "__main__":
    root = Path("data/tracks/guedes_music")

    captions = load_spanio_captions()
    audios = _find_audio_files(root)
    rows = get_audio_embeddings(captions, audios)
    pd.DataFrame(rows).to_csv(
        "guedes_audio_embeddings.csv", index=False
    )
