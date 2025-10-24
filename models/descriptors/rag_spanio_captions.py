import os
from collections import Counter
from typing import TypedDict

import numpy as np
import pandas as pd

from models.descriptors.db import get_top_k_audio_captions
from models.descriptors.parser import parse
from models.descriptors.spanio_captions import (
    load_spanio_captions_embeddings,
    SpanioCaptionsEmbedding,
)


class SpanioAugmentedPrompt(TypedDict):
    id: str
    source_caption: str
    prompt: str
    distance: float


def transform_and_join_captions(augmented_captions: list[str]) -> str:
    words_pos_map = parse(" ".join(augmented_captions), lang="english")
    relevant_words = Counter(words_pos_map).most_common(10)

    return " ".join([word for word, _ in relevant_words])


def get_augmented_prompt_spanio_captions() -> list[SpanioAugmentedPrompt]:
    spanio_captions: list[SpanioCaptionsEmbedding] = load_spanio_captions_embeddings()
    results: list[SpanioAugmentedPrompt] = []

    for pos, caption in enumerate(spanio_captions, start=1):
        augmented_captions: dict[str, float] = get_top_k_audio_captions(
            caption_embedding=caption, k=10
        )

        results.append(
            {
                "id": str(pos),
                "source_caption": caption["captions"],
                "prompt": transform_and_join_captions(list(augmented_captions.keys())),
                "distance": float(
                    np.mean([dis for dis in augmented_captions.values()])
                ),
            }
        )

    return results


if __name__ == "__main__":
    res = get_augmented_prompt_spanio_captions()
    file_path = os.getcwd() + "/data/docs/rag_spanio_captions.csv"
    pd.DataFrame(res).to_csv(file_path, index=False)
