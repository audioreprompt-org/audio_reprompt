import datetime
import logging
import os
import time
import uuid

import pandas as pd

from config import PROJECT_ROOT, load_config, setup_project_paths

from openai import OpenAI
from openai.types import FileObject
from openai.types.batch import Batch

from models.food_prompts.batch_utils import upload_file_and_create_batch
from models.food_prompts.utils import collect_keyword_results, download_batch_result, chunks


logger = logging.getLogger(__name__)


MODEL_GTP_40_MINI = "gpt-4o-mini-2024-07-18"
BATCH_SIZE = 100


PROMPT_V1 = """
"""


def create_prompt(prompt_version, food_item: str) -> str:
    return prompt_version.format(food_item=food_item)


def put_batch_get_food_captions(
    food_items: list[str], model_version: str, prompt_version: str
) -> None:
    logger.info("putting batch using model: %s", model_version)

    prompts: list[str] = []

    for seed in food_items:
        prompts.append(create_prompt(prompt_version, seed))

    batch_requests: list[dict[str, str]] = []
    retries = 0

    while not batch_requests and retries <= 3:
        try:
            batch_requests = [
                batch_info | {"food_item": food_item}
                for batch_info, food_item in zip(
                    upload_file_and_create_batch(prompts, model_version), food_items
                )
            ]
        except Exception as e:
            logger.error("failed to create batch request: %s", e)
            logger.error(
                "batch with first kw: %s - last kw: %s",
                food_items[0],
                food_items[-1],
            )
            logger.info("retrying in 5 seconds")
            time.sleep(5)
            retries += 1

    if batch_requests:
        save_batch_requests(batch_requests)
        logger.info("completed batch in queue")


def save_batch_requests(batch_requests: list[dict[str, str]]) -> None:
    part_id = (
        f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%M%d')}"
        f"_{str(uuid.uuid4())[:7].lower()}"
    )
    filename = f"batch_requests_part_{part_id}.csv"

    pd.DataFrame(batch_requests).to_csv(filename, mode="w+", index=False)

    logger.info("batch requests: %s saved to csv", filename)


def collect_batch_results(file_input: str) -> None:
    df = pd.read_csv(file_input)

    if df is not None:
        os.makedirs("results", exist_ok=True)
        logger.info("created 'results' directory")

        for batch_id in df.batch_id.unique():
            download_batch_result(batch_id, None)


def collect_batches_keyword_results(file_input: str) -> None:
    logger.info("processing file input: %s", file_input)
    df = pd.read_csv(file_input)
    os.makedirs("results", exist_ok=True)

    file_output = file_input.split("/")[-1].replace(".csv", "_out.csv")

    collect_keyword_results(df, file_output)


if __name__ == '__main__':
    setup_project_paths()
    config = load_config()

    FOOD_VOCAB_PATH = PROJECT_ROOT / config.data.cleaned_data_path / "food"
    FOOD_PROMPTS_PATH = PROJECT_ROOT / config.data.cleaned_data_path / "food_prompts"

    FOOD_PROMPTS_PATH.mkdir(parents=True, exist_ok=True)

    food_items = []

    for sample_part in chunks(food_items, BATCH_SIZE):
        put_batch_get_food_captions(sample_part, MODEL_GTP_40_MINI, PROMPT_V1)

