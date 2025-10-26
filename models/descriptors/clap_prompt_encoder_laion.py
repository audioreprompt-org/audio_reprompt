import logging

import pandas as pd
import torch
import torch.nn.functional as F
import torch.hub

from metrics.clap.factory import _make_backend
from models.descriptors.model import clap_model
from models.descriptors.spanio_captions import load_spanio_captions

logger = logging.getLogger(__name__)


def get_text_embeddings_in_batches(descriptors: list[str], batch_size: int = 16):
    backend: str = "laion_module"
    clap_backend = _make_backend(backend, "cpu")
    embeddings = []
    with torch.no_grad():
        for pos in range(0, len(descriptors), batch_size):
            batch = descriptors[pos : pos + batch_size]
            text_embeddings = clap_backend.get_text_embedding(
                batch, use_tensor=True
            )
            emb = F.normalize(text_embeddings, p=2, dim=-1)
            for text, embedding in zip(batch, emb.cpu()):
                embeddings.append({"text": text, "embedding": embedding.tolist()})

    return embeddings


if __name__ == "__main__":
    captions_ = load_spanio_captions()
    caption_embeddings_map = get_text_embeddings_in_batches(captions_)

    pd.DataFrame(caption_embeddings_map).to_csv(
        "spanio_caption_embeddings_laion.csv", index=False
    )
