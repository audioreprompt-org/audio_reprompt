import logging

import pandas as pd
import torch.nn.functional as F

from models.clap_score.model import ClapModel, SPECIALIZED_WEIGHTS_URL
from models.descriptors.spanio_captions import load_spanio_captions

logger = logging.getLogger(__name__)


def get_text_embeddings_in_batches(descriptors: list[str], batch_size: int = 16):
    model = ClapModel(enable_fusion=True, weights=SPECIALIZED_WEIGHTS_URL)
    embeddings = []

    for pos in range(0, len(descriptors), batch_size):
        batch = descriptors[pos : pos + batch_size]
        emb = F.normalize(
            model.embed_text(batch), p=2, dim=-1
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
