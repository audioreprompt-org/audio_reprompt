from collections import defaultdict
from typing import TypedDict, List, Dict
from psycopg import sql
from .db import get_conn


class CrossModalRAGResult(TypedDict):
    dimension: str
    descriptor: str
    text_embedding: str
    sim: float


def cut_crossmodal_results(
    results: List[CrossModalRAGResult],
) -> List[CrossModalRAGResult]:
    sorted_results = sorted(results, key=lambda x: x["sim"], reverse=True)
    cut_results = []
    dimension_values_map = defaultdict(list)

    for res in sorted_results:
        dim = res["dimension"]
        val = res["descriptor"]

        if dim in ("color", "taste", "temperature"):
            if dim not in dimension_values_map:
                dimension_values_map[dim].append(val)
                cut_results.append(res)
        else:
            if val not in dimension_values_map[dim]:
                dimension_values_map[dim].append(val)
                cut_results.append(res)
    return cut_results


def get_top_k_food_descriptors(
    embedding: list[float], cut_results: bool = False
) -> List[CrossModalRAGResult]:
    query = """
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
            """

    with get_conn().cursor() as cursor:
        cursor.execute(query, {"embedding": embedding})
        results = [
            {"dimension": r[0], "descriptor": r[1], "text_embedding": r[2], "sim": r[3]}
            for r in cursor.fetchall()
        ]

    if cut_results:
        results = cut_crossmodal_results(results)
    return results


def get_top_k_audio_captions(
    caption_embedding: list[float], k: int = 5, using_clap: bool = False
) -> Dict[str, float]:
    table_name = "audio_descriptors_clap" if using_clap else "audio_descriptors"

    with get_conn().cursor() as cursor:
        cursor.execute(
            sql.SQL("""
                    select caption, 1 - (embedding <=> %(embedding)s::vector) as sim
                    from {table_name}
                    order by embedding <=> %(embedding)s::vector
                        limit %(limit)s
                    """).format(table_name=sql.Identifier(table_name)),
            {"embedding": caption_embedding, "limit": k},
        )
        return {caption: sim for caption, sim in cursor.fetchall()}
