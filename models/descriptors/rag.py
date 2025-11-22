from collections import defaultdict
from typing import TypedDict

from models.descriptors.connection import get_conn
from psycopg import sql


class CrossModalRAGResult(TypedDict):
    dimension: str
    descriptor: str
    text_embedding: str
    sim: float


def get_top_k_audio_captions(
    caption_embedding: list[float], k: int = 5, using_clap: bool = False
) -> dict[str, float]:
    audio_descriptors_table = (
        "audio_descriptors_clap" if using_clap else "audio_descriptors"
    )

    with get_conn().cursor() as cursor:
        cursor.execute(
            sql.SQL(
                """
            select 
                caption,
                1 - (embedding <=> %(embedding)s::vector) as sim
            from {table_name}
            order by embedding <=> %(embedding)s::vector
            limit %(limit)s
            """
            ).format(table_name=sql.Identifier(audio_descriptors_table)),
            {"embedding": caption_embedding, "limit": k},
        )

        results: dict[str, float] = {caption: sim for caption, sim in cursor.fetchall()}

    return results


def get_top_k_food_descriptors(
    embedding: list[float], cut_results: bool = False
) -> list[CrossModalRAGResult]:
    with get_conn().cursor() as cursor:
        cursor.execute(
            """
            with crossmodal_res as (
                select b.*,
                1 - (b.text_embedding <=> %(embedding)s::vector) as sim
            from crossmodal_food_embeddings b
            order by text_embedding <=> %(embedding)s::vector
            limit 100
            ),
            rank_res as (
                select
                    dimension,
                    descriptor,
                    text_embedding,
                    sim,
                    rank() over (partition by dimension order by sim desc) as rank
                from crossmodal_res
            )
            select
                dimension,
                descriptor,
                text_embedding,
                sim
            from rank_res
            where rank <= 3
            order by sim desc
            limit 20
            """,
            {"embedding": embedding},
        )
        results: list[CrossModalRAGResult] = [
            {
                "dimension": row[0],
                "descriptor": row[1],
                "text_embedding": row[2],
                "sim": row[3],
            }
            for row in cursor.fetchall()
        ]

        if cut_results:
            results = cut_crossmodal_results(results)

    return results


def cut_crossmodal_results(
    results: list[CrossModalRAGResult],
) -> list[CrossModalRAGResult]:
    sorted_results = sorted(results, key=lambda x: x["sim"], reverse=True)
    cut_results = []

    dimension_values_map = defaultdict(list)
    for res in sorted_results:
        dim = res["dimension"]
        val = res["descriptor"]

        if dim in ("color", "taste", "temperature"):
            # we preserve the most significant one to map 1:1 like dataset was generated
            # we only map food to only one taste to simplify audio generation use case
            if dim not in dimension_values_map:
                dimension_values_map[dim].append(val)
                cut_results.append(res)
        else:
            # we want to preserve the distinct top 3 in rest dimensions
            if val not in dimension_values_map[dim]:
                dimension_values_map[dim].append(val)
                cut_results.append(res)

    return cut_results
