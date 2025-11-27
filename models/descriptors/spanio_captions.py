import csv
import json
import os
from typing import TypedDict

from models.descriptors.parser import concat_salient_words, parse


class SpanioCaptionsEmbedding(TypedDict):
    captions: str
    embedding: list[float]


def get_spanio_captions() -> dict[int, str]:
    file_ = os.getcwd() + "/data/docs/descriptions.json"
    with open(file_, "r") as f:
        data = json.load(f)

    return {item["id"]: item["description"] for item in data}


def build_spanio_captions(captions: dict[int, str]):
    results = {id_: parse(caption, "english") for id_, caption in captions.items()}

    with open("data/docs/spanio_captions.json", "w") as f:
        f.write(json.dumps(results))


def load_spanio_captions() -> list[str]:
    file_path = os.getcwd() + "/data/docs/spanio_captions.json"
    with open(file_path, "r") as f:
        captions = [val for key, val in json.load(f).items()]

    return [concat_salient_words(caption) for caption in captions]


def load_spanio_captions_embeddings() -> list[SpanioCaptionsEmbedding]:
    file_path = os.getcwd() + "/data/docs/spanio_captions_embeddings.csv"

    caption_embeddings: list[SpanioCaptionsEmbedding] = []

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

        for row in csv_reader:
            caption_embeddings.append(
                {"captions": row[0], "embedding": json.loads(row[1])}
            )

    return caption_embeddings


if __name__ == "__main__":
    raw_captions = get_spanio_captions()
    build_spanio_captions(raw_captions)
