import json
import os
from models.prompts.parser import parse


def get_spanio_captions() -> dict[int, str]:
    file_ = os.getcwd() + "/data/docs/descriptions.json"
    with open(file_, "r") as f:
        data = json.load(f)

    return {item["id"]: item["description"] for item in data}


def build_spanio_captions(captions: dict[int, str]):
    results = {id_: parse(caption, "english") for id_, caption in captions.items()}

    with open("data/docs/spanio_captions.json", "w") as f:
        f.write(json.dumps(results))


if __name__ == "__main__":
    raw_captions = get_spanio_captions()
    build_spanio_captions(raw_captions)
