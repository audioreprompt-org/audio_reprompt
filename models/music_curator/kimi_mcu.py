import os

from openai import OpenAI


client = OpenAI(
    api_key=os.getenv(
        "MOONSHOT_API_KEY", "sk-4l1IqmtGvffTApzfYKlMvl22HDtDs0XOGaaxLg3G7Rk0bRLf"
    ),
    base_url="https://api.moonshot.ai/v1",
)


MUSIC_CURATOR_ROLE = """
You are MCU, an AI music curator assistant that provide recommendations using on musician vocabulary.
"""

MUSIC_REPROMPT_PROMPT = """
Compose a prompt to use in a music generation model using crossmodal descriptors, music captions,
 and following the next rules:
1. Include one lead instrument based on the taste and emotion descriptors.
2. Describe one harmony with secondary instruments based on emotion and the most relevant music captions.
3. Purpose articulation and rhythm based on texture descriptors.

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only a paragraph with two complex sentences as a result and ensures to cover the rules provided. 
"""


def mcu_reprompt(crossmodal_descriptors: str, music_captions: str) -> str:
    messages = [
        {"role": "system", "content": MUSIC_CURATOR_ROLE},
        {
            "role": "user",
            "content": MUSIC_REPROMPT_PROMPT.format(
                crossmodal_descriptors=crossmodal_descriptors,
                music_captions=music_captions,
            ),
        },
    ]

    response = client.chat.completions.create(
        model="kimi-k2-thinking", messages=messages, temperature=0.8, max_tokens=2048
    )

    return response.choices[0].message.content
