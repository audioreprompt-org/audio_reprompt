from typing import Iterable
import torch

from metrics.clap.types import CLAPItem, CLAPScored, TARGET_SR
from metrics.clap.backends.base import (
    BackendUnavailable,
    compute_scores_with_embeddings,
    BaseBackend,
)
from utils.device import resolve_device

_EXPECTED_MODEL_ID = "laion/clap-htsat-fused"


class LASSBackend(BaseBackend):
    name = "lass"

    def __init__(self, device: str) -> None:
        self.device = resolve_device(device)
        try:
            from transformers import ClapProcessor, ClapModel  # type: ignore
        except Exception as e:
            raise BackendUnavailable("Install `transformers` to use lass backend.") from e
        try:
            self.processor = ClapProcessor.from_pretrained(_EXPECTED_MODEL_ID)
            self.model = ClapModel.from_pretrained(_EXPECTED_MODEL_ID)
            self.model.eval().to(self.device)
        except Exception as e:
            raise BackendUnavailable(f"Failed to load HF model '{_EXPECTED_MODEL_ID}' for LASS backend.") from e

        name_or_path = getattr(self.model, "name_or_path", None) or getattr(getattr(self.model, "config", None), "_name_or_path", None)
        if name_or_path and _EXPECTED_MODEL_ID not in str(name_or_path):
            raise BackendUnavailable(f"Unexpected HF model source: '{name_or_path}' (expected '{_EXPECTED_MODEL_ID}')")

    @torch.no_grad()
    def embed_audio(self, audio_np) -> torch.Tensor:
        pass

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        pass

    @torch.no_grad()
    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        return compute_scores_with_embeddings(self, items)
