import json
import os
from typing import Optional

import requests

RUNPOD_API_URL = "RUNPOD_API_URL"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"


class RunpodError(Exception):
    """Raised when Runpod returns an error or invalid response."""
    pass


def _get_api_key() -> str:
    """
    Resolves the API key with the RUNPOD_API_KEY environment variable.
    """
    api_key = os.getenv(RUNPOD_API_KEY_ENV)
    if not api_key:
        raise RunpodError(
            f"RunPod API key not found. "
            f"Set {RUNPOD_API_KEY_ENV} env var."
        )
    return api_key


def _get_api_url(explicit_api_key: Optional[str] = None) -> str:
    """
    Resolves the API URL with the RUNPOD_API_URL environment variable.
    """
    api_key = explicit_api_key or os.getenv(RUNPOD_API_URL)
    if not api_key:
        raise RunpodError(
            f"RunPod API url not found. "
            f"Set {RUNPOD_API_URL} env var."
        )
    return api_key


def call_runpod_musicgen(
    prompt: str,
) -> str:
    """
    Calls the Runpod endpoint and returns the base64 audio string.

    :param prompt: Text prompt for the model.
    :return: audio_base64 string from Runpod response.
    """
    runpod_url = _get_api_url()
    api_key = _get_api_key()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "input": {
            "prompt": prompt,
        }
    }

    resp = requests.post(runpod_url, headers=headers, data=json.dumps(payload))

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
    output = data.get("output", {})
    if not output or "audio_base64" not in output:
        raise RunpodError(f"Unexpected response format: {data}")

    return output["audio_base64"]
