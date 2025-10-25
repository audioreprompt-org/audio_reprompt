import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torchaudio
import torch.hub
import torch.nn.functional as F

from models.descriptors.model import clap_model


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


def get_only_audio_embeddings(paths: Iterable[Path]) -> list[tuple[str, list[float]]]:
    model = clap_model()
    device = next(model.parameters()).device.type
    embeddings = []

    with torch.no_grad():
        for i, p in enumerate(paths):
            try:
                p = Path(p)
                wav = _load_audio_tensor(p, device)
                emb = model.get_audio_embedding_from_data(wav, use_tensor=True)
                emb = F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().tolist()

                embeddings.append((p.name, emb))
            except Exception as e:
                logger.error(f"Failed embedding {p}: {e}")

    return embeddings


def get_audio_embeddings(descriptor_dominance: ..., paths: Iterable[Path]):
    """
    Build normalized CLAP embeddings for each audio file in `paths`.
    `audio_id` is taken from the file stem (e.g., '99' for '.../99.mp3').
    """
    model = clap_model()
    device = next(model.parameters()).device.type
    embeddings = []

    with torch.no_grad():
        for i, p in enumerate(paths):
            try:
                p = Path(p)
                audio_id = p.stem
                rates = descriptor_dominance.get(audio_id)

                wav = _load_audio_tensor(p, device)
                emb = model.get_audio_embedding_from_data(wav, use_tensor=True)
                emb = F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().tolist()

                embeddings.append(
                    {
                        "id": audio_id,
                        "audio_embedding": emb,
                        "sweet_rate": float(rates["sweet_rate"]),
                        "bitter_rate": float(rates["bitter_rate"]),
                        "sour_rate": float(rates["sour_rate"]),
                        "salty_rate": float(rates["salty_rate"]),
                    }
                )
            except Exception as e:
                logger.error(f"Failed embedding {p}: {e}")

    return embeddings


def _load_guedes_descriptor_dominance(
    csv_path: str | Path = "data/docs/guedes_descriptor_dominance.csv",
) -> dict[str, dict[str, float]]:
    df = pd.read_csv(csv_path, dtype={"id": str})
    # keep only the required columns; simple one-shot script, assume CSV is valid
    cols = ["sweet_rate", "bitter_rate", "sour_rate", "salty_rate"]
    return df.set_index("id")[cols].to_dict(orient="index")


def _find_audio_files(root: Path) -> list[Path]:
    exts = {".wav", ".mp3"}
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts),
        key=lambda p: (
            p.parent.as_posix(),
            int(p.stem) if p.stem.isdigit() else p.stem.lower(),
        ),
    )


if __name__ == "__main__":
    root = Path("data/tracks/guedes_music")

    descriptor_dominance = _load_guedes_descriptor_dominance()
    audios = _find_audio_files(root)
    rows = get_audio_embeddings(descriptor_dominance, audios)
    pd.DataFrame(rows).to_csv("data/docs/guedes_audio_embeddings.csv", index=False)
