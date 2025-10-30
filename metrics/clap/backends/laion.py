from typing import Iterable, Optional
import torch
import torch.nn.functional as F

from metrics.clap.types import CLAPItem, CLAPScored
from metrics.clap.backends.base import (
    BackendUnavailable,
    compute_scores_with_embeddings,
    BaseBackend,
)
from utils.device import resolve_device

MUSICGEN_WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"


class LaionBackend(BaseBackend):
    name = "laion_module"

    def __init__(self, device: str, *, enable_fusion: bool = False, weights: Optional[str] = None) -> None:
        """
        Initialize LAION-CLAP backend.

        Args:
            device: "cpu", "cuda", or "auto"
            enable_fusion: Whether to enable the CLAP_Module fusion mode.
            weights: Path or URL to a custom checkpoint. If None, uses default WEIGHTS_URL.
                     - If it looks like http(s)://, it will be downloaded via torch.hub.
                     - Otherwise, it's treated as a local file path.
        """
        self.device = resolve_device(device)
        try:
            from laion_clap import CLAP_Module  # type: ignore
        except Exception as e:
            raise BackendUnavailable("Install `laion-clap` to use laion_module backend.") from e

        # Allow toggling fusion and swapping weights
        self.model = CLAP_Module(enable_fusion=enable_fusion)

        if weights:
            state_dict = torch.hub.load_state_dict_from_url(
                weights, map_location=device, weights_only=False
            )
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def embed_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        return self.model.get_audio_embedding_from_data(audio_tensor, use_tensor=True)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        return self.model.get_text_embedding([text], use_tensor=True)

    @torch.no_grad()
    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        return compute_scores_with_embeddings(self, items)
