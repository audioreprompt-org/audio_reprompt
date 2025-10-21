import logging

import pandas as pd
import torch
import torch.nn.functional as F
import torch.hub
from laion_clap import CLAP_Module
from models.prompts.spanio_captions import load_spanio_captions

logger = logging.getLogger(__name__)


_CLAP_MODEL = None


def clap_model():
    global _CLAP_MODEL
    if _CLAP_MODEL is not None:
        return _CLAP_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLAP_Module(enable_fusion=False)

    weights_url = "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt"

    try:
        state_dict = torch.hub.load_state_dict_from_url(
            weights_url, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.error(f"Error loading model weights from {weights_url}: {e}")
        raise

    model.to(device)
    model.eval()

    logger.info(f"Modelo CLAP cargado correctamente en {device}")
    _CLAP_MODEL = model
    return model


def get_text_embeddings_in_batches(descriptors: list[str], batch_size: int = 16):
    model = clap_model()
    embeddings = []
    with torch.no_grad():
        for pos in range(0, len(descriptors), batch_size):
            batch = descriptors[pos : pos + batch_size]
            emb = F.normalize(
                model.get_text_embedding(batch, use_tensor=True), p=2, dim=-1
            )
            for text, embedding in zip(batch, emb.cpu()):
                embeddings.append({"text": text, "embedding": embedding.tolist()})

    return embeddings


if __name__ == "__main__":
    captions_ = load_spanio_captions()
    caption_embeddings_map = get_text_embeddings_in_batches(captions_)

    pd.DataFrame(caption_embeddings_map).to_csv(
        "spanio_caption_embeddings.csv", index=False
    )
