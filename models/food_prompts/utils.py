import os
import logging
from itertools import islice
from typing import Any, Iterable

import pandas as pd
from openai import OpenAI


logger = logging.getLogger(__name__)


def chunks(it: Iterable[Any], size: int):
    if size < 1:
        raise ValueError(size)

    it = iter(it)
    while chunk:= list(islice(it, size)):
        yield chunk

def download_batch_result(
    batch_id: str, prefix: str | None
) -> None:
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if not batch or not batch.output_file_id:
        logger.info("Batch %s has no output file or is not completed yet.", batch_id)
        return

    file_name = f"out_{batch_id}.jsonl"
    full_prefix = "results" if prefix is None else f"results/{prefix}"
    local_path = f"{full_prefix}/{file_name}"

    if os.path.exists(local_path):
        logger.info("File %s already exists.", local_path)
        return

    file_response = client.files.content(batch.output_file_id)
    with open(local_path, "w+") as out_file:
        out_file.write(file_response.text)
    logger.info("Batch %s downloaded to %s.", batch_id, local_path)


def get_food_captions(row: dict[str, Any]) -> str | None:
    return row["body"]["choices"][0]["message"]["content"] if row else None


def get_source_model(row: dict[str, Any]) -> str | None:
    return row["body"]["model"] if row else None


def get_in_tokens(row: dict[str, Any]) -> str | None:
    return row["body"]["usage"]["prompt_tokens"] if row else None


def get_out_tokens(row: dict[str, Any]) -> str | None:
    return row["body"]["usage"]["completion_tokens"] if row else None


def filter_error_requests(row: dict[str, Any]) -> bool:
    return False if "error" in row["response"]["body"] else True


def collect_food_results(
    batch_records: pd.DataFrame, file_output: str
) -> None:
    results: list[dict[str, str]] = []

    for batch_id in batch_records.batch_id.unique():
        df = None
        if os.path.exists(f"results/out_{batch_id}.jsonl"):
            df = pd.read_json(f"results/out_{batch_id}.jsonl", lines=True)

        if df is not None:
            filtered_df = df[df.apply(filter_error_requests, axis=1)]

            if filtered_df.shape[0] != df.shape[0]:
                logger.error("batch id out_%s.jsonl with some requests failed.", batch_id)
                df = filtered_df

            value_results = [
                (
                    get_source_model(row),
                    get_food_captions(row),
                    get_in_tokens(row),
                    get_out_tokens(row),
                )
                for row in df["response"]
            ]

            df["model"], df["food_captions"], df["in_tokens"], df["out_tokens"] = zip(
                *value_results
            )
            df["total_tokens"] = df["in_tokens"] + df["out_tokens"]

            results.extend(
                df[
                    [
                        "custom_id",
                        "model",
                        "food_captions",
                        "in_tokens",
                        "out_tokens",
                        "total_tokens",
                    ]
                ].to_dict(orient="records")
            )

    (
        pd.DataFrame(results)
        .sort_values(by="food_captions")
        .to_csv(
            file_output,
            columns=[
                "custom_id",
                "food_captions",
                "model",
                "question",
                "in_tokens",
                "out_tokens",
                "total_tokens",
            ],
            index=False,
        )
    )
