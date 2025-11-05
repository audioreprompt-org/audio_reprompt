from typing import Iterable
import numpy as np
import torch
from abc import ABC

from metrics.clap.types import CLAPItem, CLAPScored, TARGET_SR
from utils.audio import load_audio_tensor


class BackendUnavailable(RuntimeError):
    pass


class BaseBackend(ABC):
    """Protocol every backend must satisfy."""
    name: str

    def __init__(self, device: str) -> None: ...
    def embed_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor: ...
    def embed_text(self, text: str) -> torch.Tensor: ...

    @torch.no_grad()
    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        return compute_scores_with_embeddings(self, items)


@torch.no_grad()
def compute_scores_with_embeddings(
    be: BaseBackend,
    items: Iterable[CLAPItem],
) -> list[CLAPScored]:
    """
    Shared implementation:
      - loads & preprocesses audio (mono, sample_rate)
      - obtains audio/text embeddings via backend
      - L2-normalizes both
      - computes cosine similarity
      - returns CLAPScored with normalized embeddings (lists of float32)
    """
    out: list[CLAPScored] = []
    for it in items:
        audio_np = load_audio_tensor(it.audio_path, TARGET_SR)

        aemb = be.embed_audio(audio_np)  # Tensor [D] or [1,D]
        temb = be.embed_text(it.description)  # Tensor [D] or [1,D]

        # Ensure 2-D then normalize per-row, then squeeze back to 1-D
        if aemb.dim() == 1:
            aemb = aemb.unsqueeze(0)
        if temb.dim() == 1:
            temb = temb.unsqueeze(0)

        aemb = torch.nn.functional.normalize(aemb, dim=-1)
        temb = torch.nn.functional.normalize(temb, dim=-1)

        score = torch.nn.functional.cosine_similarity(aemb, temb).item()

        a_list = aemb.squeeze(0).detach().cpu().to(torch.float32).numpy().tolist()
        t_list = temb.squeeze(0).detach().cpu().to(torch.float32).numpy().tolist()

        out.append(
            CLAPScored(
                item=it,
                clap_score=float(np.round(score, 6)),
                audio_embedding=a_list,
                text_embedding=t_list,
            )
        )
    return out
