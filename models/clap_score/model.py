from typing import Optional, Iterable
import torch

from laion_clap import CLAP_Module

from models.clap_score.typedef import CLAPScored, CLAPItem, TARGET_SR

from models.clap_score.utils import load_audio_tensor, resolve_device

SPECIALIZED_WEIGHTS_URL = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"


class ClapModel:
    name = "laion_module"

    def __init__(self, enable_fusion: bool = False, weights: Optional[str] = None) -> None:
        """
        Initialize LAION-CLAP score.

        Args:
            enable_fusion: Whether to enable the CLAP_Module fusion mode.
            weights: Path or URL to a custom checkpoint. If None, uses default WEIGHTS_URL.
                     - If it looks like http(s)://, it will be downloaded via torch.hub.
                     - Otherwise, it's treated as a local file path.
        """
        self.device = resolve_device()

        # Allow toggling fusion and swapping weights
        self.model = CLAP_Module(enable_fusion=enable_fusion)

        if weights:
            state_dict = torch.hub.load_state_dict_from_url(
                weights, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_ckpt()

        self.model.to(self.device)
        self.model.eval()


    @torch.no_grad()
    def embed_audio(self, audios: list[str]) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for audio_path in audios:
            audio_tensor = load_audio_tensor(audio_path, TARGET_SR).to(self.device)
            emb = self.model.get_audio_embedding_from_data(audio_tensor, use_tensor=True)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> torch.Tensor:
        return self.model.get_text_embedding(texts, use_tensor=True)

    @classmethod
    def calculate_score_with_embeddings(cls, text_embeddings: torch.Tensor, audio_embeddings: torch.Tensor) -> list[float]:
        if text_embeddings.shape != audio_embeddings.shape:
            raise ValueError(f"Shape mismatch: audio {audio_embeddings.shape} vs text {text_embeddings.shape} (esperado N x D emparejado)")

        text_embeddings = torch.nn.functional.normalize(text_embeddings, dim=-1)
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
        sims = (audio_embeddings * text_embeddings).sum(dim=-1)

        return sims.squeeze(-1).detach().cpu().tolist()

    @torch.no_grad()
    def calculate_scores(self, items: Iterable[CLAPItem]) -> list[CLAPScored]:
        out: list[CLAPScored] = []

        text_embeddings = self.model.embed_text([element.audio_path for element in items], use_tensor=True)
        audio_embeddings = self.model.embed_audio([element.audio_path for element in items], use_tensor=True)

        scores = self.calculate_score_with_embeddings(text_embeddings, audio_embeddings)
        for index, items in enumerate(items):
            out.append(CLAPScored(item=items, clap_score=scores[index]))

        return out
