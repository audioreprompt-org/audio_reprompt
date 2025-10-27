from typing import Iterable
import torch
import torch.nn.functional as F


from metrics.clap.types import CLAPItem, CLAPScored
from metrics.clap.backends.base import (
    BackendUnavailable,
    compute_scores_with_embeddings,
    BaseBackend,
)

from utils.device import resolve_device


WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"


class LaionBackend(BaseBackend):
    name = "laion_module"


    def __init__(self, device: str) -> None:
        self.device = resolve_device(device)
        try:
            from laion_clap import CLAP_Module
        except Exception as e:
            raise BackendUnavailable(
                "Install `laion-clap` to use laion_module backend."
            ) from e
        self.model = CLAP_Module(enable_fusion=False)

        state_dict = torch.hub.load_state_dict_from_url(
            WEIGHTS_URL, map_location=device, weights_only=False
        )

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def embed_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        return F.normalize(
            input=self.model.get_audio_embedding_from_data(audio_tensor, use_tensor=True),
            p=2,
            dim=-1
        ).squeeze(0)

    @torch.no_grad()
    def embed_text(self, text: str) -> torch.Tensor:
        return F.normalize(
            input=self.model.get_text_embedding([text], use_tensor=True),
            p=2,
            dim=-1
        ).squeeze(0)

    @torch.no_grad()
    def score_batch(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        return compute_scores_with_embeddings(self, items)
