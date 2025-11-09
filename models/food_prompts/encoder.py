import csv
from itertools import chain

import pandas as pd
from sentence_transformers import SentenceTransformer

from config import setup_project_paths, load_config, PROJECT_ROOT
from models.descriptors.clap_prompt_encoder import get_text_embeddings_in_batches
from models.food_prompts.parser import map_to_fooditem_crossmodal
from models.food_prompts.typedefs import FoodItemCrossModal, FoodItemCrossModalEncoded


# no ideal but
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def parse_food_crossmodal_items(fpath: str) -> list[FoodItemCrossModal]:
    with open(fpath, "r") as file_:
        reader = csv.DictReader(file_)
        rows = [row for row in reader]

    results: list[FoodItemCrossModal] = list(
        chain(*list(map(map_to_fooditem_crossmodal, rows)))
    )
    return results


def encode_food_crossmodal_items(
    items: list[FoodItemCrossModal],
) -> list[FoodItemCrossModalEncoded]:
    embs = model.encode([f"{item['food_item']} {item['descriptor']}" for item in items])

    encoded_items = []
    for emb, item in zip(embs, items):
        encoded_items.append(item | {"text_embedding": emb.tolist()})

    return encoded_items


def encode_text(items: list[str]) -> list[dict[str, str|list[float]]]:
    return [
        {"prompt": text, "text_embedding": emb.tolist()}
        for text, emb in zip(items, model.encode(items))
    ]


def data_examples_crossmodal()-> None:
    setup_project_paths()
    config = load_config()

    embs_dir = PROJECT_ROOT / config.data.embeddings_csv_path / "examples_crossmodal"

    embs = encode_food_crossmodal_items(
        [
            {
                "food_item": "lemon lime soda",
                "dimension": "texture",
                "descriptor": "citrus energizing iced carbonated yellow sweet",  # all together
            },
            {
                "food_item": "lemon lime soda",
                "dimension": "texture",
                "descriptor": "citrus yellow",
            },
        ]
    )

    pd.DataFrame(embs).to_csv(f"{embs_dir}/all_mini_lemon_embs.csv")


def data_raw_prompt_user() -> None:
    setup_project_paths()
    config = load_config()

    raw_user_dir = PROJECT_ROOT / config.data.raw_data_path / "user"
    embs_dir = PROJECT_ROOT / config.data.embeddings_csv_path / "user"

    with open(f"{raw_user_dir}/raw_prompts.csv", 'r') as file_:
        reader = csv.DictReader(file_)
        prompts = [row['sentence'] for row in reader]

    results = encode_text(prompts)
    pd.DataFrame(results).to_csv(
        f"{embs_dir}/all_mini_user_embeddings.csv",
        index=False
    )


def encode_clap_crossmodal_results(descriptors: list[str]) -> None:
    results = get_text_embeddings_in_batches(descriptors)
    pd.DataFrame(results).to_csv('crossmodal_descriptors_encoded.csv')


if __name__ == "__main__":
    # data_raw_prompt_user()

    encode_clap_crossmodal_results([
        'savory',
        'hot',
        'smoky',
        'happiness',
        'warmth',
        'warm',
        'satisfaction',
        'spicy',
        'umami',
        'juicy',
    ])
