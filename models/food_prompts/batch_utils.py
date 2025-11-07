import json
import logging
import tempfile
import uuid
from typing import Any, Literal

from openai import OpenAI
from openai.types import FileObject
from openai.types.batch import Batch

MODEL_ENDPOINT: Literal["/v1/chat/completions"] = "/v1/chat/completions"

logger = logging.getLogger(__name__)


def upload_batch_file(filepath: str) -> FileObject:
    return OpenAI().files.create(file=open(filepath, "rb"), purpose="batch")


def create_batch_request(file_id: str) -> Batch:
    return OpenAI().batches.create(
        input_file_id=file_id, endpoint=MODEL_ENDPOINT, completion_window="24h"
    )


def create_batch_message_request(
    prompt: str, custom_id: str, model_version: str
) -> dict[str, str | Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": MODEL_ENDPOINT,
        "body": {
            "model": model_version,
            "messages": [{"role": "user", "content": prompt}],
        },
    }

def upload_file_and_create_batch(
    prompts: list[str], model_version: str
) -> list[dict[str, str]]:
    batch_id = str(uuid.uuid4())[:7]
    custom_ids = [f"{batch_id}_{str(uuid.uuid4())[:18]}" for _ in prompts]

    message_requests: list[str] = [
        json.dumps(create_batch_message_request(prompt, custom_id, model_version))
        for custom_id, prompt in zip(custom_ids, prompts)
    ]
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".jsonl", delete=True, encoding="utf-8"
    ) as file_:
        content = "\n".join(message_requests) + "\n"
        file_.write(content)
        file_.flush()

        batch_file = upload_batch_file(file_.name)
        logger.info("completed upload batch file %s", batch_file.id)

        batch = create_batch_request(batch_file.id)
        logger.info("completed batch request %s", batch.id)

        logger.info(
            "batch creation completed for file %s batch id: %s", batch_file.id, batch.id
        )

    return [
        {"custom_id": custom_id, "batch_id": batch.id, "file_id": batch_file.id}
        for custom_id in custom_ids
    ]