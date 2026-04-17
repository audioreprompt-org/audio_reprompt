import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from models.music_curator.prompts import MCU_PROMPTS

KIMI_K2_THINKING_MODEL = "kimi-k2-thinking"
OPENAI_GPT_5_NANO_MODEL = "gpt-5-nano"


MUSIC_CURATOR_ROLE = """
You are MCU, an AI music curator assistant that provide recommendations using musician vocabulary.
"""

# ── Response cache ───────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ablations" / ".cache"


def _cache_key(
    model: str,
    prompt_version: str,
    crossmodal_descriptors: str,
    music_captions: str,
    system_prompt_modifier: str | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
) -> str:
    """Deterministic hash of all inputs that affect the LLM response."""
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "crossmodal_descriptors": crossmodal_descriptors,
            "music_captions": music_captions,
            "system_prompt_modifier": system_prompt_modifier,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_cache(key: str) -> str | None:
    cache_file = _CACHE_DIR / f"{key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    return None


def _write_cache(key: str, response: str) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{key}.txt"
    cache_file.write_text(response, encoding="utf-8")


# ── Client ───────────────────────────────────────────────────


@lru_cache(maxsize=2)
def get_client(model: str):
    return (
        OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if model == OPENAI_GPT_5_NANO_MODEL
        else OpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.ai/v1",
        )
    )


# ── Reprompt ─────────────────────────────────────────────────


def mcu_reprompt(
    crossmodal_descriptors: str,
    music_captions: str,
    model: str = KIMI_K2_THINKING_MODEL,
    prompt_version: str = "V3",
    system_prompt_modifier: str | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
) -> str:
    # Check disk cache first
    key = _cache_key(
        model, prompt_version, crossmodal_descriptors, music_captions,
        system_prompt_modifier, presence_penalty, frequency_penalty
    )
    cached = _read_cache(key)
    if cached is not None:
        return cached

    system_content = MUSIC_CURATOR_ROLE
    if system_prompt_modifier:
        system_content += f"\n{system_prompt_modifier}"

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": MCU_PROMPTS[prompt_version].format(
                crossmodal_descriptors=crossmodal_descriptors,
                music_captions=music_captions,
            ),
        },
    ]

    kwargs: dict = {"model": model, "messages": messages}
    if presence_penalty is not None:
        kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        kwargs["frequency_penalty"] = frequency_penalty

    response = get_client(model).chat.completions.create(**kwargs)
    result = response.choices[0].message.content

    _write_cache(key, result)
    return result
