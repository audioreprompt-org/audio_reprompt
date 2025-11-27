import json
import logging
import os
import time
from typing import Optional

import requests

RUNPOD_API_URL = "RUNPOD_API_URL"
RUNPOD_API_KEY_ENV = "RUNPOD_API_KEY"

logger = logging.getLogger(__name__)


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


def call_runpod_musicgen(prompt: str) -> str:
    """
    Calls the Runpod endpoint. If runsync times out, it switches to polling
    the status endpoint until the job is complete.

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

    logger.info(f"Sending request to RunPod: {runpod_url}")
    resp = requests.post(runpod_url, headers=headers, data=json.dumps(payload))

    if not resp.ok:
        raise RunpodError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    status = data.get("status")

    if status in ["IN_QUEUE", "IN_PROGRESS"]:
        job_id = data.get("id")
        logger.info(f"Job queued ({status}). Switching to polling for ID: {job_id}")

        # Construct Status URL (Replace /runsync with /status/{id})
        # Assumes URL format: https://api.runpod.ai/v2/{endpoint_id}/runsync
        status_url = runpod_url.replace("/runsync", f"/status/{job_id}")

        while True:
            time.sleep(3)  # Wait 3 seconds between checks

            status_resp = requests.get(status_url, headers=headers)
            if not status_resp.ok:
                logger.error(f"Polling failed: {status_resp.status_code} - {status_resp.text}")
                raise RunpodError(f"Polling failed HTTP {status_resp.status_code}")

            status_data = status_resp.json()
            current_status = status_data.get("status")

            logger.debug(f"Polling job {job_id}: {current_status}")

            if current_status == "COMPLETED":
                data = status_data  # Update data with the completed result
                break
            elif current_status == "FAILED":
                logger.error(f"RunPod job failed: {status_data}")
                raise RunpodError(f"RunPod Job Failed: {status_data}")

    # Expected shape:
    # {
    #    "delayTime": ...,
    #    "executionTime": ...,
    #    "id": "...",
    #    "output": {
    #        "audio_base64": "UklGRjJwOgBXQVZFZm10..."
    #    }
    # }
    output = data.get("output", {})

    # If output is still empty after polling, something went wrong
    if not output or "audio_base64" not in output:
        # Check if we have an error message in the output
        if isinstance(output, dict) and "message" in output:
            raise RunpodError(f"Model Error: {output['message']}")

        logger.error(f"Unexpected response format: {data}")
        raise RunpodError(f"Unexpected response format: {data}")

    return output["audio_base64"]
