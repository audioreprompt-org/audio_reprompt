import csv
import json
import logging
import os
from typing import Iterable, Sequence

import numpy as np
from models.descriptors.spanio_captions import SpanioCaptionsEmbedding
from psycopg import connect, sql
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]

AUDIO_DESCRIPTOR_TABLE_NAME = "audio_descriptors"
GUEDES_AUDIO_TABLE_NAME = "guedes_audio_embeddings"

RUN_GUEDES = True


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


def get_conn():
    return connect(
        f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST}",
        autocommit=True,
    )


def create_audio_descriptors_table() -> None:
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")

        create_ad_table = sql.SQL("""
        create table if not exists {table_name} (
            id bigserial primary key,
            caption text unique not null,
            embedding vector(512) not null
        )""").format(table_name=sql.Identifier(AUDIO_DESCRIPTOR_TABLE_NAME))

        cursor.execute(create_ad_table)


def create_audio_guedes_table() -> None:
    """
    Create the table to store Guedes audio embeddings and taste rates.
    Table: guedes_audio_embeddings
    Primary key: audio_id
    """
    with get_conn().cursor() as cursor:
        cursor.execute("create extension if not exists vector;")
        create_tbl = sql.SQL("""
        create table if not exists {table_name} (
            audio_id text primary key,
            embedding vector(512) not null,
            sweet_rate real not null,
            bitter_rate real not null,
            sour_rate real not null,
            salty_rate real not null
        )
        """).format(table_name=sql.Identifier(GUEDES_AUDIO_TABLE_NAME))
        cursor.execute(create_tbl)


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
                logger.error(f"Skipping row due to parse error for audio_id={audio_id}: {e}")
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


def insert_guedes_audio_descriptor_items(items: list[GuedesAudioDescriptorItem]) -> bool:
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
            insert_sql = sql.SQL("""
                insert into {tbl} (audio_id, embedding, sweet_rate, bitter_rate, sour_rate, salty_rate)
                values (%s, %s, %s, %s, %s, %s)
            """).format(tbl=sql.Identifier(GUEDES_AUDIO_TABLE_NAME))
            cursor.executemany(insert_sql, params)
            logger.info(f"Inserted {cursor.rowcount} rows into {GUEDES_AUDIO_TABLE_NAME}.")
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
        insert_sql = sql.SQL("""
            insert into {tbl} (caption, embedding)
            values (%s, %s) 
        """).format(tbl=sql.Identifier(table_name))

        cursor.executemany(insert_sql, params)
        inserted = cursor.rowcount

    return inserted


def load_audio_descriptor_items() -> list[AudioDescriptorItem]:
    audio_caps_items = []
    file_path = os.getcwd() + "/data/docs/audio_caps_embeddings.csv"
    descriptors, embeddings = [], []

    with open(file_path, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip header

        for row in csv_reader:
            descriptors.append(row[0])
            embeddings.append(np.array(json.loads(row[1])))

    for descriptor, embedding in zip(descriptors, embeddings):
        audio_caps_items.append(
            AudioDescriptorItem(caption=descriptor, clap_embedding=embedding.tolist())
        )

    return audio_caps_items


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


def get_top_k_audio_captions(
    caption_embedding: SpanioCaptionsEmbedding, k: int = 5
) -> dict[str, float]:
    with get_conn().cursor() as cursor:
        cursor.execute(
            """
            select 
                caption,
                1- (embedding <=> %(embedding)s) as sim
            from audio_descriptors
            order by embedding <=> %(embedding)s 
            limit %(limit)s
            """,
            params={"embedding": caption_embedding, "limit": k},
        )

        results: dict[str, float] = {caption: sim for caption, sim in cursor.fetchall()}

    return results


def test():
    create_audio_descriptors_table()
    insert_audio_descriptors(
        [
            AudioDescriptorItem(caption="caption2", clap_embedding=[1.0] * 512),
            AudioDescriptorItem(caption="caption3", clap_embedding=[3.0] * 512),
            AudioDescriptorItem(caption="caption4", clap_embedding=[2.0] * 512),
        ]
    )


if __name__ == "__main__":
    if RUN_GUEDES:
        create_audio_guedes_table()
        guedes_items = load_guedes_audio_descriptor_items()
        insert_guedes_audio_descriptor_items(guedes_items)
    else:
        create_audio_descriptors_table()
        reading_items = load_audio_descriptor_items()
        insert_audio_descriptor_items(reading_items)
