import datetime
import glob
import logging
import os
import random
import time
import uuid
from csv import DictReader
from enum import Enum

import pandas as pd

from config import PROJECT_ROOT, load_config, setup_project_paths

from models.food_prompts.batch_utils import upload_file_and_create_batch
from models.food_prompts.utils import collect_food_results, download_batch_result, chunks


logger = logging.getLogger(__name__)


MODEL_GTP_40_MINI = "gpt-4o-mini-2024-07-18"
BATCH_SIZE = 100


PROMPT_V1 = """
Survey on associations between sensations, emotions, and tastes of `{}`. Answer briefly:

1. Chemical flavor profile - top 3 sensation terms
2. Human responses - top 3 physiological effects
3. Temperature - choose only one: hot, warm, cold, iced
4. Texture - top 3 texture terms
5. Emotions - choose only one: anger, disgust, fear, happiness, nostalgic, surprise
6. Color - choose only one: blue, purple, green, brown, red, orange, yellow, white
7. Taste - choose only one of: sweet, bitter, salty, sour

Respond with single terms, comma-separated, no header, formatted as:
chemical flavor profile|human responses|temperature|texture|emotions|color|taste
In case the `food_item` is not a food returns only `No Label`. If a dimension cannot be answered, fill the dimension as `No Label`
"""


class BatchOptionEnum(Enum):
    OPTION_PUT_BATCHES_ON_QUEUE = "put_batches_on_queue"
    OPTION_COLLECT_BATCHES = "collect_batches"
    OPTION_COLLECT_BATH_RESULTS = "collect_batch_results"


def create_prompt(prompt_version, food_item: str) -> str:
    return prompt_version.format(food_item.strip().lower())


def put_batch_get_food_captions(
    food_items: list[str], model_version: str, prompt_version: str
) -> None:
    logger.info("putting batch using model: %s", model_version)

    prompts: list[str] = []

    for food_item in food_items:
        prompts.append(create_prompt(prompt_version, food_item))

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


def collect_batches_food_results(file_input: str, output_dir: str) -> None:
    logger.info("processing file input: %s", file_input)
    df = pd.read_csv(file_input)
    os.makedirs("results", exist_ok=True)

    file_output = f"{output_dir}/" + file_input.split("/")[-1].replace(".csv", "_out.csv")

    collect_food_results(df, file_output)


def read_food_csv_items(file: str, column_name: str, do_sample: bool = False, n_sample: int = 100) -> set[str]:
    with open(file, 'r') as file_:
        reader = DictReader(file_)
        items = {row[column_name].lower() for row in reader}

    filtered = list(filter(lambda item: 'raw' not in item, items))
    return random.sample(filtered, min(n_sample, len(filtered))) if do_sample else filtered


if __name__ == '__main__':
    setup_project_paths()
    config = load_config()

    FOOD_VOCAB_PATH = PROJECT_ROOT / config.data.cleaned_data_path / "food"
    FOOD_PROMPTS_PATH = PROJECT_ROOT / config.data.cleaned_data_path / "food_prompts"

    FOOD_PROMPTS_PATH.mkdir(parents=True, exist_ok=True)

    food_items = read_food_csv_items(
        f"{FOOD_VOCAB_PATH}/food_nutrition_vocabulary.csv",
        "food",
    )

    option = BatchOptionEnum.OPTION_PUT_BATCHES_ON_QUEUE
    # option = BatchOptionEnum.OPTION_COLLECT_BATCHES
    # option = BatchOptionEnum.OPTION_COLLECT_BATH_RESULTS

    match option:
        case BatchOptionEnum.OPTION_PUT_BATCHES_ON_QUEUE:
            for sample_part in chunks(food_items, BATCH_SIZE):
                put_batch_get_food_captions(sample_part, MODEL_GTP_40_MINI, PROMPT_V1)

        case BatchOptionEnum.OPTION_COLLECT_BATCHES:
            for batch_file in glob.glob('batch_requests_*.csv'):
                collect_batch_results(batch_file)

        case BatchOptionEnum.OPTION_COLLECT_BATH_RESULTS:
            for batch_file in glob.glob('batch_requests_*.csv'):
                collect_batches_food_results(batch_file, FOOD_PROMPTS_PATH)
