import csv
import glob
from itertools import chain

import torch

from config import load_config, setup_project_paths, PROJECT_ROOT
from models.clap_score import ClapModel, SPECIALIZED_WEIGHTS_URL

from models.descriptors.db import insert_crossmodal_food_embeddings
from models.food_prompts.parser import map_to_fooditem_crossmodal
from models.food_prompts.typedefs import FoodItemCrossModal, FoodItemCrossModalEncoded
from models.food_prompts.utils import chunks


def encode_food_crossmodal_items(
    items: list[FoodItemCrossModal],
) -> list[FoodItemCrossModalEncoded]:
    clap_encoder = ClapModel(
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_fusion=True,
        weights=SPECIALIZED_WEIGHTS_URL,
    )

    embs = clap_encoder.embed_text(
        [f"{item['food_item']} {item['descriptor']}" for item in items]
    )

    encoded_items = []
    for emb, item in zip(embs, items):
        encoded_items.append(item | {"text_embedding": emb.tolist()})

    return encoded_items


def parse_food_crossmodal_items(fpath: str) -> list[FoodItemCrossModal]:
    with open(fpath, "r") as file_:
        reader = csv.DictReader(file_)
        rows = [row for row in reader]

    results: list[FoodItemCrossModal] = list(
        chain(*list(map(map_to_fooditem_crossmodal, rows)))
    )
    return results


def encode_and_save_in_batches(items: list[FoodItemCrossModal]):
    for chunk in chunks(items, 100):
        insert_crossmodal_food_embeddings(
            encode_food_crossmodal_items(chunk)
        )


if __name__ == "__main__":
    setup_project_paths()
    config = load_config()

    FOOD_PROMPTS_PATH = PROJECT_ROOT / config.data.cleaned_data_path / "food_prompts"

    for filepath in glob.glob(f"{FOOD_PROMPTS_PATH}/*.csv"):
        food_items = parse_food_crossmodal_items(filepath)
        encode_and_save_in_batches(food_items)
