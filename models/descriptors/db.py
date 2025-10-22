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


class AudioDescriptorItem(BaseModel):
    caption: str
    clap_embedding: list[float]


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
    create_audio_descriptors_table()
    reading_items = load_audio_descriptor_items()
    insert_audio_descriptor_items(reading_items)
