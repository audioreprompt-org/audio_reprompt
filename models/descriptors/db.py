import csv
import glob
import json
import logging
import os
from csv import DictReader
from enum import Enum
from typing import Iterable, Sequence

import numpy as np

from config import setup_project_paths, load_config, PROJECT_ROOT
from psycopg import connect, sql
from pydantic import BaseModel

from models.food_prompts.encoder import (
    parse_food_crossmodal_items,
    encode_food_crossmodal_items,
    encode_text,
)
from models.food_prompts.typedefs import FoodItemCrossModalEncoded
from models.food_prompts.utils import chunks

logger = logging.getLogger(__name__)

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]

AUDIO_DESCRIPTOR_TABLE_NAME = "audio_descriptors"
GUEDES_AUDIO_TABLE_NAME = "guedes_audio_embeddings"
RAG_AUDIO_TABLE_NAME = "rag_audio_embeddings"
BASE_PROMPT_EMBEDDINGS_TABLE_NAME = "base_prompt_embeddings"
CROSSMODAL_FOOD_EMBEDDINGS_TABLE_NAME = "crossmodal_food_embeddings"


class ExecutionOption(Enum):
    INSERT_GUEDES_AUDIO_EMBEDDINGS = 1
    INSERT_AUDIO_DESCRIPTORS = 2
    INSERT_RAG_AUDIO_EMBEDDINGS = 3
    INSERT_BASE_PROMPT_EMBEDDINGS = 4
    INSERT_CROSSMODAL_FOOD_EMBEDDINGS = 5


class AudioDescriptorItem(BaseModel):
    caption: str
    clap_embedding: list[float]


class GuedesAudioDescriptorItem(BaseModel):
    audio_id: str
    audio_embedding: list[float]
    sweet_rate: float
    bitter_rate: float
    sour_rate: float
    salty_rate: float


class RAGAudioEmbeddingItem(BaseModel):
    audio_id: str
    filename: str
    audio_embedding: list[float]


class BasePromptEmbeddingItem(BaseModel):
    prompt: str
    audio_name: str
    audio_embedding: list[float]
    text_embedding: list[float]


def get_conn():
    return connect(
        f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST}",
        autocommit=True,
    )


def create_audio_descriptors_table() -> None:
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")

        create_ad_table = sql.SQL(
            """
        create table if not exists {table_name} (
            id bigserial primary key,
            caption text unique not null,
            embedding vector(384) not null
        )"""
        ).format(table_name=sql.Identifier(AUDIO_DESCRIPTOR_TABLE_NAME))

        cursor.execute(create_ad_table)


def create_audio_guedes_table() -> None:
    """
    Create the table to store Guedes audio embeddings and taste rates.
    Table: guedes_audio_embeddings
    Primary key: audio_id
    """
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")
        create_tbl = sql.SQL(
            """
        create table if not exists {table_name} (
            audio_id text primary key,
            embedding vector(512) not null,
            sweet_rate real not null,
            bitter_rate real not null,
            sour_rate real not null,
            salty_rate real not null
        )
        """
        ).format(table_name=sql.Identifier(GUEDES_AUDIO_TABLE_NAME))
        cursor.execute(create_tbl)


def create_rag_audio_table() -> None:
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")

        create_rag_table = sql.SQL(
            """
        create table if not exists {table_name} (
            audio_id text primary key,
            filename text unique not null,
            embedding vector(512) not null
        )"""
        ).format(table_name=sql.Identifier(RAG_AUDIO_TABLE_NAME))

        cursor.execute(create_rag_table)


def create_base_prompt_embeddings_table() -> None:
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")

        create_table = sql.SQL(
            """
        create table if not exists {table_name} (
            id bigserial primary key,
            prompt text not null,
            audio_name text not null,
            audio_embedding vector(512),
            text_embedding vector(512)
        )
        """
        ).format(table_name=sql.Identifier(BASE_PROMPT_EMBEDDINGS_TABLE_NAME))

        cursor.execute(create_table)


def create_food_crossmodal_embeddings_table() -> None:
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")

        create_table = sql.SQL(
            """
            create table if not exists {table_name} (
                id bigserial primary key,
                food_item text not null,
                dimension text not null,
                descriptor text not null,
                text_embedding vector(384)
            )
            """
        ).format(table_name=sql.Identifier(CROSSMODAL_FOOD_EMBEDDINGS_TABLE_NAME))

        cursor.execute(create_table)


def load_guedes_audio_descriptor_items() -> list[GuedesAudioDescriptorItem]:
    """
    Load items from data/docs/guedes_audio_embeddings.csv
    Expected columns:
      id, audio_embedding (JSON list), sweet_rate, bitter_rate, sour_rate, salty_rate
    """
    items: list[GuedesAudioDescriptorItem] = []
    file_path = os.path.join(os.getcwd(), "data/docs/guedes_audio_embeddings.csv")

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_id = (row.get("id") or row.get("audio_id") or "").strip()
            emb_raw = row.get("audio_embedding") or row.get("embedding")

            if not audio_id or emb_raw is None:
                logger.warning(f"Skipping row with missing id/embedding: {row}")
                continue

            try:
                emb_list = [float(x) for x in json.loads(emb_raw)]
                sweet = float(row["sweet_rate"])
                bitter = float(row["bitter_rate"])
                sour = float(row["sour_rate"])
                salty = float(row["salty_rate"])
            except Exception as e:
                logger.error(
                    f"Skipping row due to parse error for audio_id={audio_id}: {e}"
                )
                continue

            items.append(
                GuedesAudioDescriptorItem(
                    audio_id=audio_id,
                    audio_embedding=emb_list,
                    sweet_rate=sweet,
                    bitter_rate=bitter,
                    sour_rate=sour,
                    salty_rate=salty,
                )
            )

    return items


def load_base_prompt_embeddings() -> list[BasePromptEmbeddingItem]:
    """
    Carga los embeddings de los base prompts.
    """
    file_path = os.path.join(
        os.getcwd(),
        "data/embeddings/music_base_prompts_embeddings_audioset_weights_enabled_fusion.csv",
    )

    items: list[BasePromptEmbeddingItem] = []

    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                audio_emb = np.array(json.loads(row["audio_emb"]))
                text_emb = np.array(json.loads(row["text_emb"]))

                item = BasePromptEmbeddingItem(
                    prompt=row["prompt"],
                    audio_name=row["audio_name"],
                    audio_embedding=audio_emb.tolist(),
                    text_embedding=text_emb.tolist(),
                )
                items.append(item)
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Error procesando fila '{row.get('audio_name', 'N/A')}': {e}"
                )

    logger.info(f"Cargados {len(items)} embeddings de base_prompts.")
    return items


def insert_guedes_audio_descriptor_items(
    items: list[GuedesAudioDescriptorItem],
) -> bool:
    """
    Insert Guedes items into guedes_audio_embeddings.
    Uses ON CONFLICT DO NOTHING so the script can be re-run safely.
    """
    try:
        params: Sequence[tuple[str, list[float], float, float, float, float]] = [
            (
                it.audio_id,
                it.audio_embedding,
                it.sweet_rate,
                it.bitter_rate,
                it.sour_rate,
                it.salty_rate,
            )
            for it in items
        ]
        with get_conn().cursor() as cursor:
            insert_sql = sql.SQL(
                """
                insert into {tbl} (audio_id, embedding, sweet_rate, bitter_rate, sour_rate, salty_rate)
                values (%s, %s, %s, %s, %s, %s)
            """
            ).format(tbl=sql.Identifier(GUEDES_AUDIO_TABLE_NAME))
            cursor.executemany(insert_sql, params)
            logger.info(
                f"Inserted {cursor.rowcount} rows into {GUEDES_AUDIO_TABLE_NAME}."
            )
        return True
    except Exception as e:
        logger.error(f"Error inserting Guedes audio embeddings: {e}")
        logger.critical("Failed to insert Guedes items.", exc_info=True)
        return False


def insert_audio_descriptors(
    items: Iterable[AudioDescriptorItem],
    table_name: str = AUDIO_DESCRIPTOR_TABLE_NAME,
) -> int:
    params: Sequence[tuple[str, list[float]]] = [
        (it.caption, it.clap_embedding) for it in items
    ]
    with get_conn().cursor() as cursor:
        insert_sql = sql.SQL(
            """
            insert into {tbl} (caption, embedding)
            values (%s, %s) 
        """
        ).format(tbl=sql.Identifier(table_name))

        cursor.executemany(insert_sql, params)
        inserted = cursor.rowcount

    return inserted


def insert_rag_audio_embeddings(
    items: Iterable[RAGAudioEmbeddingItem], table_name: str = RAG_AUDIO_TABLE_NAME
) -> int:
    params: Sequence[tuple[str, str, list[float]]] = [
        (it.audio_id, it.filename, it.audio_embedding) for it in items
    ]
    with get_conn().cursor() as cursor:
        insert_sql = sql.SQL(
            """
            insert into {tbl} (audio_id, filename, embedding)
            values (%s, %s, %s) 
        """
        ).format(tbl=sql.Identifier(table_name))

        cursor.executemany(insert_sql, params)
        inserted = cursor.rowcount

    return inserted


def insert_base_prompt_embeddings(
    items: list[BasePromptEmbeddingItem],
    table_name: str = BASE_PROMPT_EMBEDDINGS_TABLE_NAME,
) -> int:
    """
    Inserta los embeddings de base_prompts en la tabla base_prompt_embeddings.
    """
    params = [
        (
            it.prompt,
            it.audio_name,
            it.audio_embedding,
            it.text_embedding,
        )
        for it in items
    ]

    with get_conn().cursor() as cursor:
        insert_sql = sql.SQL(
            """
            insert into {tbl} 
            (prompt, audio_name, audio_embedding, text_embedding)
            values (%s, %s, %s, %s)
        """
        ).format(tbl=sql.Identifier(table_name))

        cursor.executemany(insert_sql, params)
        inserted = cursor.rowcount

    logger.info(f"Inserted {inserted} base prompt embeddings.")
    return inserted


def insert_crossmodal_food_embeddings(
    crossmodal_food_items: list[FoodItemCrossModalEncoded],
    table_name: str = CROSSMODAL_FOOD_EMBEDDINGS_TABLE_NAME,
):
    with get_conn().cursor() as cursor:
        insert_sql = sql.SQL(
            """
            insert into {tbl} 
            (food_item, dimension, descriptor, text_embedding)
            values (%s, %s, %s, %s)
        """
        ).format(tbl=sql.Identifier(table_name))

        cursor.executemany(
            insert_sql,
            [
                (
                    it["food_item"],
                    it["dimension"],
                    it["descriptor"],
                    it["text_embedding"],
                )
                for it in crossmodal_food_items
            ],
        )
        inserted = cursor.rowcount

    logger.info(f"Inserted {inserted} crossmodal food embeddings.")
    return inserted


def load_audio_descriptor_items() -> list[AudioDescriptorItem]:
    audio_caps_items = []
    file_path = os.getcwd() + "/data/docs/audio_caps_embeddings.csv"

    with open(file_path, "r") as f:
        dict_reader = DictReader(f)
        descriptors = [row["text"] for row in dict_reader]

    embeddings_items: list[dict[str, str | list[float]]] = encode_text(descriptors)

    for item in embeddings_items:
        audio_caps_items.append(
            AudioDescriptorItem(
                caption=item["prompt"], clap_embedding=item["text_embedding"]
            )
        )

    return audio_caps_items


def load_rag_audio_embeddings() -> list[RAGAudioEmbeddingItem]:
    audio_emb_filepath = os.getcwd() + "/data/docs/rag_audio_embeddings.csv"
    with open(audio_emb_filepath, "r") as f:
        csv_reader = csv.DictReader(f)
        return [
            RAGAudioEmbeddingItem(
                audio_id=str(pos),
                filename=item["audio_id"],
                audio_embedding=json.loads(item["embedding"]),
            )
            for pos, item in enumerate(csv_reader, start=1)
        ]


def insert_audio_descriptor_items(audio_caps_items: list[AudioDescriptorItem]) -> bool:
    try:
        insert_audio_descriptors(audio_caps_items)
    except Exception as e:
        logger.error(f"Error insertando audio caps: {e}")
        logger.critical(
            "No se pudo insertar el audio caps en la base de datos.", exc_info=True
        )
        return False
    return True


def load_and_insert_crossmodal_food_embeddings():
    setup_project_paths()
    config = load_config()

    food_prompts_path = PROJECT_ROOT / config.data.cleaned_data_path / "food_prompts"

    for filepath in glob.glob(f"{food_prompts_path}/*.csv"):
        for chunk in chunks(parse_food_crossmodal_items(filepath), 100):
            insert_crossmodal_food_embeddings(encode_food_crossmodal_items(chunk))


# def test():
#     create_audio_descriptors_table()
#     insert_audio_descriptors(
#         [
#             AudioDescriptorItem(caption="caption2", clap_embedding=[1.0] * 512),
#             AudioDescriptorItem(caption="caption3", clap_embedding=[3.0] * 512),
#             AudioDescriptorItem(caption="caption4", clap_embedding=[2.0] * 512),
#         ]
#     )


if __name__ == "__main__":
    # set manual option
    option = ExecutionOption.INSERT_AUDIO_DESCRIPTORS

    match option:
        case ExecutionOption.INSERT_GUEDES_AUDIO_EMBEDDINGS:
            create_audio_guedes_table()
            guedes_items = load_guedes_audio_descriptor_items()
            insert_guedes_audio_descriptor_items(guedes_items)

        case ExecutionOption.INSERT_AUDIO_DESCRIPTORS:
            create_audio_descriptors_table()
            reading_items = load_audio_descriptor_items()
            insert_audio_descriptor_items(reading_items)

        case ExecutionOption.INSERT_RAG_AUDIO_EMBEDDINGS:
            create_rag_audio_table()
            items = load_rag_audio_embeddings()
            insert_rag_audio_embeddings(items)

        case ExecutionOption.INSERT_BASE_PROMPT_EMBEDDINGS:
            create_base_prompt_embeddings_table()
            items = load_base_prompt_embeddings()
            insert_base_prompt_embeddings(items)

        case ExecutionOption.INSERT_CROSSMODAL_FOOD_EMBEDDINGS:
            create_food_crossmodal_embeddings_table()
            load_and_insert_crossmodal_food_embeddings()
