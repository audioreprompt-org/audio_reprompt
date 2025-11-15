import re

import pandas as pd

from models.allmini_v2.encoder import encode_text
from models.descriptors.rag import (
    get_top_k_food_descriptors,
    get_top_k_audio_captions,
    CrossModalRAGResult,
)
from models.music_curator.kimi_mcu import mcu_reprompt


def custom_single_sentence(crossmodal_descriptors: list[CrossModalRAGResult]) -> str:
    crossmodal_suffix_map = {
        "taste": "melody",
        "texture": "harmony",
        "emotion": "rhythm",
    }
    cross_music_intention = " "
    for cm_res in crossmodal_descriptors:
        dim = cm_res["dimension"]
        val = cm_res["descriptor"]
        cross_music_intention += f"{val} {crossmodal_suffix_map.get(dim, '')} "

    return re.sub(r"\s+", " ", cross_music_intention)


def format_crossmodal_descriptors(
    crossmodal_descriptors: list[CrossModalRAGResult],
) -> list[str]:
    crossmodal_results = [
        f"{cm_res['dimension']}: {cm_res['descriptor']}"
        for cm_res in crossmodal_descriptors
        if cm_res["dimension"] in ("emotion", "taste", "texture")
    ]

    return crossmodal_results


def transform(user_prompt: str):
    # 1. encode user prompt without preprocessing
    if not (
        emb_user := next(
            iter(
                encode_text(
                    [
                        user_prompt,
                    ]
                )
            )
        )["text_embedding"]
    ):
        raise ValueError("failed encode user prompt")

    # 2. recover crossmodal descriptors (rag layer top)
    crossmodal_descriptors = get_top_k_food_descriptors(emb_user, cut_results=True)

    # 3. encode crossmodal descriptors in all-mini/clap
    # single sentence custom strategy: some descriptors are naive mapped to music descriptors
    crossmodal_values = custom_single_sentence(crossmodal_descriptors)

    if not (
        emb_cm := next(
            iter(
                encode_text(
                    [
                        crossmodal_values,
                    ]
                )
            )
        )["text_embedding"]
    ):
        raise ValueError("failed encode crossmodal values")

    # 4. recover music descriptor (rag layer down)
    music_descriptors = get_top_k_audio_captions(emb_cm, k=10, using_clap=False)

    # 5. re-prompt using cross-modal and music descriptors
    music_descriptor_values = "\n".join(list(music_descriptors.keys()))
    formatted_crossmodal_values = "\n".join(
        format_crossmodal_descriptors(crossmodal_descriptors)
    )

    return mcu_reprompt(music_descriptor_values, formatted_crossmodal_values)


if __name__ == "__main__":
    prompts = [
        "I feel energetic and I'm gonna drink an iced Americano along with my trainer Valentina. The café is lively and the music is upbeat.",
        "I'm at the bakery and I'm gonna eat a chocolate croissant. It smells buttery and the display case gleams.",
        "We're at the stadium and we're gonna eat a chili cheese hot dog. The air feels hot and the lights are glaring.",
        "I feel stressed and I'm gonna eat a quinoa bowl along with my trainer Luis. The gym café is bright and the music is low.",
    ]
    results = []
    for prompt in prompts:
        results.append({"prompt": prompt, "reprompt": transform(prompt)})

    pd.DataFrame(results).to_csv("pipeline_results_two_sen.csv", index=False)
