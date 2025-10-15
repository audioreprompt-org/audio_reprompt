from dataclasses import dataclass
from typing import List, Iterable, Optional
import numpy as np
import torch
import torchaudio
from laion_clap import CLAP_Module

TARGET_SR = 48000


@dataclass
class CLAPItem:
    id: str
    description: str
    audio_path: str
    instrument: Optional[str] = None


@dataclass
class CLAPScored(CLAPItem):
    clap_score: float = float("nan")


def _resolve_device(device: Optional[str]) -> str:
    # Match legacy behavior: choose CUDA if available, else CPU.
    # We intentionally do not propagate "auto" to torch.
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_clap(device: str) -> CLAP_Module:
    model = CLAP_Module(enable_fusion=True)
    model.load_ckpt()
    model.eval().to(device)
    return model


@torch.no_grad()
def score(
    items: Iterable[CLAPItem],
    device: Optional[str] = None,
) -> List[CLAPScored]:
    dev = _resolve_device(device)
    model = _load_clap(dev)

    out: List[CLAPScored] = []
    for it in items:
        # Load exactly like the legacy script
        wav, sr = torchaudio.load(it.audio_path)  # [C, T] as-is
        if sr != TARGET_SR:
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.to(dev)

        aemb = model.get_audio_embedding_from_data(wav, use_tensor=True)
        temb = model.get_text_embedding([it.description], use_tensor=True)
        aemb = torch.nn.functional.normalize(aemb, dim=-1)
        temb = torch.nn.functional.normalize(temb, dim=-1)
        score_val = torch.nn.functional.cosine_similarity(aemb, temb).item()

        out.append(
            CLAPScored(
                id=it.id,
                description=it.description,
                audio_path=it.audio_path,
                instrument=it.instrument,
                clap_score=float(np.round(score_val, 6)),
            )
        )
    return out
