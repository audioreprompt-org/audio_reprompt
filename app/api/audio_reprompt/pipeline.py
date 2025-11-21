import re
from audio_reprompt.encoder import encode_text
from audio_reprompt.rag import get_top_k_food_descriptors, get_top_k_audio_captions
from audio_reprompt.mcu import mcu_reprompt


def custom_single_sentence(crossmodal_descriptors: list) -> str:
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


def format_crossmodal_descriptors(crossmodal_descriptors: list) -> list[str]:
    return [
        f"{cm_res['dimension']}: {cm_res['descriptor']}"
        for cm_res in crossmodal_descriptors
        if cm_res["dimension"] in ("emotion", "taste", "texture")
    ]


def transform(user_prompt: str, prompt_version: str = "V3") -> str:
    # 1. Encode user prompt
    emb_user = encode_text([user_prompt])[0]["text_embedding"]

    # 2. RAG Layer 1: Food Descriptors
    crossmodal_descriptors = get_top_k_food_descriptors(emb_user, cut_results=True)

    # 3. Encode descriptors for next layer
    crossmodal_values = custom_single_sentence(crossmodal_descriptors)
    emb_cm = encode_text([crossmodal_values])[0]["text_embedding"]

    # 4. RAG Layer 2: Music Captions
    music_descriptors = get_top_k_audio_captions(emb_cm, k=10, using_clap=False)

    # 5. LLM Re-prompt
    music_descriptor_values = "\n".join(list(music_descriptors.keys()))
    formatted_crossmodal_values = "\n".join(
        format_crossmodal_descriptors(crossmodal_descriptors)
    )

    return mcu_reprompt(
        music_descriptor_values,
        formatted_crossmodal_values,
        prompt_version=prompt_version,
    )
