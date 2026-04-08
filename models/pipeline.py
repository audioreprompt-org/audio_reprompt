import re

from models.allmini_v2.encoder import encode_text
from models.descriptors.rag import (
    get_top_k_food_descriptors,
    get_top_k_audio_captions,
    CrossModalRAGResult,
)
from models.music_curator.kimi_mcu import mcu_reprompt, KIMI_K2_THINKING_MODEL


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


FILTER_DIMENSIONS_DEFAULT = ("emotion", "taste", "texture")


def format_crossmodal_descriptors(
    crossmodal_descriptors: list[CrossModalRAGResult],
    filter_dimensions: tuple[str, ...] | None = FILTER_DIMENSIONS_DEFAULT,
) -> list[str]:
    crossmodal_results = [
        f"{cm_res['dimension']}: {cm_res['descriptor']}"
        for cm_res in crossmodal_descriptors
        if filter_dimensions is None or cm_res["dimension"] in filter_dimensions
    ]

    return crossmodal_results


def transform(
    user_prompt: str,
    model: str = KIMI_K2_THINKING_MODEL,
    prompt_version: str = "V4",
    cut_results: bool = True,
    k: int = 10,
    filter_dimensions: tuple[str, ...] | None = FILTER_DIMENSIONS_DEFAULT,
    temperature: float | None = None,
    top_p: float | None = None,
):
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
    crossmodal_descriptors = get_top_k_food_descriptors(emb_user, cut_results=cut_results)

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
    music_descriptors = get_top_k_audio_captions(emb_cm, k=k, using_clap=False)

    # 5. re-prompt using cross-modal and music descriptors
    music_descriptor_values = "\n".join(list(music_descriptors.keys()))
    formatted_crossmodal_values = "\n".join(
        format_crossmodal_descriptors(crossmodal_descriptors, filter_dimensions=filter_dimensions)
    )

    return mcu_reprompt(
        music_descriptor_values,
        formatted_crossmodal_values,
        model=model,
        prompt_version=prompt_version,
        temperature=temperature,
        top_p=top_p,
    )
