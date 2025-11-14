import json
import os
from typing import Optional

import requests

RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/222ph97wou9n8a/runsync"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"


class RunpodError(Exception):
    """Raised when Runpod returns an error or invalid response."""
    pass


def _get_api_key(explicit_api_key: Optional[str] = None) -> str:
    """
    Resolves the API key, preferring an explicit argument and
    falling back to the RUNPOD_API_KEY environment variable.
    """
    api_key = explicit_api_key or os.getenv(RUNPOD_API_KEY_ENV)
    if not api_key:
        raise RunpodError(
            f"RunPod API key not found. "
            f"Set {RUNPOD_API_KEY_ENV} env var or pass api_key explicitly."
        )
    return api_key


def call_runpod_musicgen(
    prompt: str,
    api_key: Optional[str] = None,
) -> str:
    """
    Calls the Runpod endpoint and returns the base64 audio string.

    :param prompt: Text prompt for the model.
    :param api_key: Runpod API key (rp_...). If None, taken from RUNPOD_API_KEY env var.
    :return: audio_base64 string from Runpod response.
    """
    api_key = _get_api_key(api_key)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "input": {
            "prompt": prompt,
        }
    }

    resp = requests.post(RUNPOD_ENDPOINT, headers=headers, data=json.dumps(payload))

    if not resp.ok:
        raise RunpodError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    # Expected shape:
    # {
    #   "delayTime": ...,
    #   "executionTime": ...,
    #   "id": "...",
    #   "output": {
    #       "audio_base64": "UklGRjJwOgBXQVZFZm10..."
    #   }
    # }
    output = data.get("output")
    if not output or "audio_base64" not in output:
        raise RunpodError(f"Unexpected response format: {data}")

    return output["audio_base64"]
