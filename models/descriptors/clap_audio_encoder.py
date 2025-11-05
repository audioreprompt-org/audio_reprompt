import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch.nn.functional as F

from models.clap_score.model import ClapModel, SPECIALIZED_WEIGHTS_URL

logger = logging.getLogger(__name__)


def get_audio_embeddings(descriptor_dominance: dict[str, dict[str, float]], paths: Iterable[Path]):
    """
    Build normalized CLAP embeddings for each audio file in `paths`.
    `audio_id` is taken from the file stem (e.g., '99' for '.../99.mp3').
    """
    model = ClapModel(device="auto", enable_fusion=True, weights=SPECIALIZED_WEIGHTS_URL)
    embeddings = []

    audio_embeddings = model.embed_audio([str(audio_path) for audio_path in paths])
    for i, p in enumerate(paths):
        p = Path(p)
        audio_id = p.stem
        rates = descriptor_dominance.get(audio_id)
        embeddings.append(
            {
                "id": audio_id,
                "audio_embedding": F.normalize(audio_embeddings[i], p=2, dim=-1).squeeze(0).cpu().tolist(),
                "sweet_rate": float(rates["sweet_rate"]),
                "bitter_rate": float(rates["bitter_rate"]),
                "sour_rate": float(rates["sour_rate"]),
                "salty_rate": float(rates["salty_rate"]),
            }
        )

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
