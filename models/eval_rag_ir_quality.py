"""
IR Quality Evaluation for RAG Retrieval Functions


Evaluates the information retrieval quality of:
  - get_top_k_audio_captions (audio_descriptors table)
  - get_top_k_food_descriptors (crossmodal_food_embeddings table)

Runs fully offline — no external LLM API calls required.

Usage:
    python -m models.scripts.eval_rag_ir_quality --output-dir data/ablations/ir_eval/
"""

import argparse
import json
import logging
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from psycopg import sql
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity

from models.descriptors.connection import get_conn
from models.descriptors.rag import get_top_k_audio_captions, get_top_k_food_descriptors

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERIES_PATH = SCRIPT_DIR / "ir_eval_queries.json"


def _parse_pgvector(val) -> list[float]:
    """Parse a pgvector value into a list of floats.

    psycopg may return vector columns as strings like '[0.1,0.2,...]'
    or as native list/array types depending on the driver version.
    """
    if isinstance(val, (list, np.ndarray)):
        return [float(x) for x in val]
    if isinstance(val, str):
        return [float(x) for x in val.strip("[]").split(",")]
    raise TypeError(f"Unexpected pgvector type: {type(val)}")

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _model


# PART B: Database Diagnostics


def diagnose_table_stats(cursor) -> dict:
    """D1: Row counts, distinct values."""
    cursor.execute("SELECT count(*) FROM audio_descriptors;")
    audio_count = cursor.fetchone()[0]

    cursor.execute("SELECT count(*) FROM crossmodal_food_embeddings;")
    food_count = cursor.fetchone()[0]

    cursor.execute(
        "SELECT count(DISTINCT dimension), count(DISTINCT descriptor), count(DISTINCT food_item) "
        "FROM crossmodal_food_embeddings;"
    )
    row = cursor.fetchone()
    food_dimensions, food_descriptors, food_items = row[0], row[1], row[2]

    return {
        "audio_descriptors_rows": audio_count,
        "crossmodal_food_rows": food_count,
        "food_distinct_dimensions": food_dimensions,
        "food_distinct_descriptors": food_descriptors,
        "food_distinct_items": food_items,
    }


def diagnose_similarity_distribution(cursor, sample_size: int = 500) -> dict:
    """D2: Pairwise cosine similarity distribution (sampled).

    Samples random pairs and computes 1 - cosine_distance to build a histogram.
    A tight distribution (low std) indicates poor discriminative power.
    """
    results = {}

    for table, col in [("audio_descriptors", "embedding"), ("crossmodal_food_embeddings", "text_embedding")]:
        query = sql.SQL(
            """
            WITH sample_a AS (SELECT {col} AS emb FROM {tbl} ORDER BY random() LIMIT {n}),
                 sample_b AS (SELECT {col} AS emb FROM {tbl} ORDER BY random() LIMIT {n})
            SELECT 1 - (a.emb <=> b.emb) AS sim
            FROM sample_a a CROSS JOIN sample_b b
            LIMIT {limit}
            """
        ).format(
            col=sql.Identifier(col),
            tbl=sql.Identifier(table),
            n=sql.Literal(sample_size),
            limit=sql.Literal(sample_size * 10),
        )
        cursor.execute(query)
        sims = [row[0] for row in cursor.fetchall()]

        if sims:
            sims_arr = np.array(sims)
            results[table] = {
                "mean": float(np.mean(sims_arr)),
                "std": float(np.std(sims_arr)),
                "min": float(np.min(sims_arr)),
                "max": float(np.max(sims_arr)),
                "median": float(np.median(sims_arr)),
                "p10": float(np.percentile(sims_arr, 10)),
                "p90": float(np.percentile(sims_arr, 90)),
                "n_pairs": len(sims),
            }

    return results


def diagnose_query_spread(cursor, n_queries: int = 20) -> dict:
    """D3: For random queries, measure similarity gap between top-1 and top-K result.

    A healthy space has a clear drop-off (large gap).
    """
    results = {}
    k_values = [5, 10, 20]

    for table, col in [("audio_descriptors", "embedding"), ("crossmodal_food_embeddings", "text_embedding")]:
        # Pick random rows as queries
        cursor.execute(
            sql.SQL("SELECT {col} FROM {tbl} ORDER BY random() LIMIT {n}").format(
                col=sql.Identifier(col),
                tbl=sql.Identifier(table),
                n=sql.Literal(n_queries),
            )
        )
        query_embeddings = [row[0] for row in cursor.fetchall()]

        spreads = {f"top1_vs_top{k}_gap": [] for k in k_values}
        top1_sims = []

        for qemb in query_embeddings:
            max_k = max(k_values)
            cursor.execute(
                sql.SQL(
                    "SELECT 1 - ({col} <=> %s::vector) AS sim FROM {tbl} ORDER BY {col} <=> %s::vector LIMIT %s"
                ).format(col=sql.Identifier(col), tbl=sql.Identifier(table)),
                [qemb, qemb, max_k],
            )
            sims = [row[0] for row in cursor.fetchall()]
            if len(sims) >= 2:
                top1_sims.append(sims[0])
                for k in k_values:
                    if len(sims) >= k:
                        spreads[f"top1_vs_top{k}_gap"].append(sims[0] - sims[k - 1])

        result = {"avg_top1_sim": float(np.mean(top1_sims)) if top1_sims else 0.0}
        for key, gaps in spreads.items():
            if gaps:
                result[f"avg_{key}"] = float(np.mean(gaps))
                result[f"std_{key}"] = float(np.std(gaps))

        results[table] = result

    return results


def diagnose_dimension_coverage(cursor) -> dict:
    """D4: Per-dimension row counts in crossmodal_food_embeddings.

    Detects if certain dimensions dominate, biasing retrieval.
    """
    cursor.execute(
        """
        SELECT dimension, count(*) AS cnt
        FROM crossmodal_food_embeddings
        GROUP BY dimension
        ORDER BY cnt DESC;
        """
    )
    coverage = {row[0]: row[1] for row in cursor.fetchall()}

    total = sum(coverage.values())
    proportions = {dim: cnt / total for dim, cnt in coverage.items()} if total > 0 else {}

    return {
        "counts": coverage,
        "proportions": proportions,
        "total": total,
        "gini_coefficient": _gini(list(coverage.values())) if coverage else 0.0,
    }


def diagnose_hubness(cursor, n_queries: int = 50, k: int = 10) -> dict:
    """D5: Nearest-neighbor hubness detection.

    For random queries, count how often each stored vector appears in top-K.
    High skewness of N(k) distribution indicates hubness.
    """
    results = {}

    for table, col, id_col in [
        ("audio_descriptors", "embedding", "id"),
        ("crossmodal_food_embeddings", "text_embedding", "id"),
    ]:
        cursor.execute(
            sql.SQL("SELECT {col} FROM {tbl} ORDER BY random() LIMIT {n}").format(
                col=sql.Identifier(col),
                tbl=sql.Identifier(table),
                n=sql.Literal(n_queries),
            )
        )
        query_embeddings = [row[0] for row in cursor.fetchall()]

        nn_counter: Counter = Counter()
        for qemb in query_embeddings:
            cursor.execute(
                sql.SQL(
                    "SELECT {id_col} FROM {tbl} ORDER BY {col} <=> %s::vector LIMIT %s"
                ).format(
                    id_col=sql.Identifier(id_col),
                    col=sql.Identifier(col),
                    tbl=sql.Identifier(table),
                ),
                [qemb, k],
            )
            for row in cursor.fetchall():
                nn_counter[row[0]] += 1

        occurrences = list(nn_counter.values())
        if occurrences:
            occ_arr = np.array(occurrences, dtype=float)
            results[table] = {
                "unique_neighbors": len(nn_counter),
                "max_occurrence": int(np.max(occ_arr)),
                "mean_occurrence": float(np.mean(occ_arr)),
                "std_occurrence": float(np.std(occ_arr)),
                "skewness": float(_skewness(occ_arr)),
            }
        else:
            results[table] = {"unique_neighbors": 0}

    return results


# PART C — STRATEGY 1: Intrinsic Embedding Quality


def compute_isotropy(cursor, table: str, col: str, sample_size: int = 1000) -> dict:
    """Compute isotropy score via singular value decomposition.

    Isotropy = 1 means vectors uniformly fill the space (ideal).
    Low isotropy means vectors cluster in a narrow cone (anisotropic).

    Uses the partition function approach:
        isotropy = min(e^λ) / max(e^λ) where λ are eigenvalues of the covariance matrix.
    """
    cursor.execute(
        sql.SQL("SELECT {col} FROM {tbl} ORDER BY random() LIMIT {n}").format(
            col=sql.Identifier(col),
            tbl=sql.Identifier(table),
            n=sql.Literal(sample_size),
        )
    )
    embeddings = np.array([_parse_pgvector(row[0]) for row in cursor.fetchall()])

    if len(embeddings) < 2:
        return {"isotropy": 0.0, "n_samples": len(embeddings)}

    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0)

    # SVD
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)

    # Eigenvalues of covariance matrix ~ singular_values^2 / (n-1)
    eigenvalues = (singular_values**2) / (len(embeddings) - 1)

    # Partition function isotropy: min(e^λ) / max(e^λ)
    # Use log-space for numerical stability
    log_eigenvalues = np.log(eigenvalues + 1e-10)
    isotropy = float(np.exp(log_eigenvalues.min() - log_eigenvalues.max()))

    # Also compute effective dimensionality (how many dimensions carry info)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained_variance_ratio)
    effective_dim_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    effective_dim_99 = int(np.searchsorted(cumulative, 0.99) + 1)

    return {
        "isotropy": isotropy,
        "effective_dim_95pct": effective_dim_95,
        "effective_dim_99pct": effective_dim_99,
        "total_dimensions": embeddings.shape[1],
        "top5_singular_values": singular_values[:5].tolist(),
        "n_samples": len(embeddings),
    }


def compute_self_similarity_stats(cursor, table: str, col: str, sample_size: int = 500) -> dict:
    """Compute average self-similarity (avg cosine between all pairs in a sample).

    High average self-similarity indicates the vectors are clustered (anisotropic).
    """
    cursor.execute(
        sql.SQL("SELECT {col} FROM {tbl} ORDER BY random() LIMIT {n}").format(
            col=sql.Identifier(col),
            tbl=sql.Identifier(table),
            n=sql.Literal(sample_size),
        )
    )
    embeddings = np.array([_parse_pgvector(row[0]) for row in cursor.fetchall()])

    if len(embeddings) < 2:
        return {"avg_self_similarity": 0.0}

    sim_matrix = sk_cosine_similarity(embeddings)
    # Extract upper triangle (exclude diagonal)
    upper_idx = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[upper_idx]

    return {
        "avg_self_similarity": float(np.mean(pairwise_sims)),
        "std_self_similarity": float(np.std(pairwise_sims)),
        "min_self_similarity": float(np.min(pairwise_sims)),
        "max_self_similarity": float(np.max(pairwise_sims)),
        "n_pairs": len(pairwise_sims),
    }


# PART C — STRATEGY 2: Synthetic Relevance + IR Metrics


def load_synthetic_queries(path: str | Path) -> dict:
    """Load pre-annotated synthetic test queries from JSON fixture."""
    with open(path, "r") as f:
        return json.load(f)


def score_relevance_structural(
    results: list[dict], expected_descriptors: dict[str, list[str]]
) -> list[int]:
    """Score retrieved crossmodal results using structural heuristics.

    Grading:
        2 = dimension matches AND descriptor matches (or is close synonym)
        1 = dimension matches but descriptor doesn't
        0 = dimension doesn't match expected
    """
    expected_dims = set(expected_descriptors.keys())
    # Flatten all expected descriptors for substring matching
    expected_desc_flat = {
        desc.lower()
        for descs in expected_descriptors.values()
        for desc in descs
    }

    scores = []
    for result in results:
        dim = result.get("dimension", "")
        desc = result.get("descriptor", "").lower()

        if dim in expected_dims:
            # Check if descriptor matches any expected
            if desc in expected_desc_flat or any(exp in desc or desc in exp for exp in expected_desc_flat):
                scores.append(2)
            else:
                scores.append(1)
        else:
            scores.append(0)

    return scores


def score_relevance_embedding(
    query_text: str, retrieved_texts: list[str], model: SentenceTransformer,
    threshold_high: float = 0.45, threshold_mid: float = 0.25,
) -> list[int]:
    """Score relevance using local embedding cosine similarity.

    Grading:
        2 = cosine sim >= threshold_high
        1 = cosine sim >= threshold_mid
        0 = cosine sim < threshold_mid
    """
    if not retrieved_texts:
        return []

    query_emb = model.encode([query_text])
    retrieved_embs = model.encode(retrieved_texts)
    sims = sk_cosine_similarity(query_emb, retrieved_embs)[0]

    scores = []
    for sim in sims:
        if sim >= threshold_high:
            scores.append(2)
        elif sim >= threshold_mid:
            scores.append(1)
        else:
            scores.append(0)

    return scores


def compute_precision_at_k(relevance_scores: list[int], k: int) -> float:
    """Precision@K: fraction of top-K items that are relevant (score > 0)."""
    top_k = relevance_scores[:k]
    if not top_k:
        return 0.0
    return sum(1 for s in top_k if s > 0) / len(top_k)


def compute_ndcg_at_k(relevance_scores: list[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    top_k = relevance_scores[:k]
    if not top_k:
        return 0.0

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(top_k))

    # Ideal DCG: sort by relevance descending
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(relevance_scores: list[int]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant item."""
    for i, score in enumerate(relevance_scores):
        if score > 0:
            return 1.0 / (i + 1)
    return 0.0


def compute_map_at_k(all_query_relevances: list[list[int]], k: int) -> float:
    """Mean Average Precision at K across all queries."""
    aps = []
    for relevance in all_query_relevances:
        top_k = relevance[:k]
        if not top_k:
            aps.append(0.0)
            continue

        hits = 0
        precision_sum = 0.0
        for i, rel in enumerate(top_k):
            if rel > 0:
                hits += 1
                precision_sum += hits / (i + 1)

        ap = precision_sum / min(k, sum(1 for r in relevance if r > 0)) if any(r > 0 for r in relevance) else 0.0
        aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def evaluate_crossmodal_retrieval(queries: list[dict], model: SentenceTransformer) -> dict:
    """Run crossmodal food descriptor retrieval evaluation."""
    all_structural_scores = []
    all_embedding_scores = []
    per_query_results = []

    for q in queries:
        query_text = q["query_text"]
        expected = q["expected_descriptors"]

        # Encode query and retrieve
        emb = model.encode([query_text])[0].tolist()
        results = get_top_k_food_descriptors(emb, cut_results=False)

        if not results:
            logger.warning(f"No results for crossmodal query: {query_text}")
            continue

        # Structural scoring
        structural_scores = score_relevance_structural(results, expected)
        all_structural_scores.append(structural_scores)

        # Embedding scoring
        retrieved_texts = [f"{r['dimension']} {r['descriptor']}" for r in results]
        embedding_scores = score_relevance_embedding(query_text, retrieved_texts, model)
        all_embedding_scores.append(embedding_scores)

        # Per-query detail
        per_query_results.append({
            "query_id": q["id"],
            "query_text": query_text,
            "n_results": len(results),
            "structural_p5": compute_precision_at_k(structural_scores, 5),
            "structural_p10": compute_precision_at_k(structural_scores, 10),
            "structural_ndcg5": compute_ndcg_at_k(structural_scores, 5),
            "structural_ndcg10": compute_ndcg_at_k(structural_scores, 10),
            "structural_mrr": compute_mrr(structural_scores),
            "embedding_p5": compute_precision_at_k(embedding_scores, 5),
            "embedding_p10": compute_precision_at_k(embedding_scores, 10),
            "embedding_ndcg5": compute_ndcg_at_k(embedding_scores, 5),
            "embedding_ndcg10": compute_ndcg_at_k(embedding_scores, 10),
            "embedding_mrr": compute_mrr(embedding_scores),
            "avg_sim": float(np.mean([r["sim"] for r in results])),
            "top1_sim": results[0]["sim"] if results else 0.0,
            "results_detail": [
                {"dim": r["dimension"], "desc": r["descriptor"], "sim": r["sim"],
                 "structural_rel": s, "embedding_rel": e}
                for r, s, e in zip(results, structural_scores, embedding_scores)
            ],
        })

    # Aggregate IR metrics
    k_values = [5, 10]
    aggregate = {}
    for method, all_scores in [("structural", all_structural_scores), ("embedding", all_embedding_scores)]:
        for k in k_values:
            aggregate[f"{method}_precision_at_{k}"] = float(
                np.mean([compute_precision_at_k(s, k) for s in all_scores])
            ) if all_scores else 0.0
            aggregate[f"{method}_ndcg_at_{k}"] = float(
                np.mean([compute_ndcg_at_k(s, k) for s in all_scores])
            ) if all_scores else 0.0
        aggregate[f"{method}_mrr"] = float(
            np.mean([compute_mrr(s) for s in all_scores])
        ) if all_scores else 0.0
        aggregate[f"{method}_map_at_10"] = compute_map_at_k(all_scores, 10)

    return {"aggregate": aggregate, "per_query": per_query_results}


def evaluate_audio_caption_retrieval(queries: list[dict], model: SentenceTransformer) -> dict:
    """Run audio caption retrieval evaluation."""
    all_embedding_scores = []
    per_query_results = []

    for q in queries:
        query_text = q["query_text"]

        # Encode query and retrieve
        emb = model.encode([query_text])[0].tolist()
        results = get_top_k_audio_captions(emb, k=20, using_clap=False)

        if not results:
            logger.warning(f"No results for audio query: {query_text}")
            continue

        captions = list(results.keys())
        sims = list(results.values())

        # Embedding-based scoring
        embedding_scores = score_relevance_embedding(query_text, captions, model)
        all_embedding_scores.append(embedding_scores)

        per_query_results.append({
            "query_id": q["id"],
            "query_text": query_text,
            "n_results": len(captions),
            "embedding_p5": compute_precision_at_k(embedding_scores, 5),
            "embedding_p10": compute_precision_at_k(embedding_scores, 10),
            "embedding_ndcg5": compute_ndcg_at_k(embedding_scores, 5),
            "embedding_ndcg10": compute_ndcg_at_k(embedding_scores, 10),
            "embedding_mrr": compute_mrr(embedding_scores),
            "avg_sim": float(np.mean(sims)),
            "top1_sim": sims[0] if sims else 0.0,
            "sim_spread": float(max(sims) - min(sims)) if sims else 0.0,
            "results_detail": [
                {"caption": cap, "db_sim": sim, "embedding_rel": rel}
                for cap, sim, rel in zip(captions, sims, embedding_scores)
            ],
        })

    # Aggregate
    k_values = [5, 10]
    aggregate = {}
    for k in k_values:
        aggregate[f"embedding_precision_at_{k}"] = float(
            np.mean([compute_precision_at_k(s, k) for s in all_embedding_scores])
        ) if all_embedding_scores else 0.0
        aggregate[f"embedding_ndcg_at_{k}"] = float(
            np.mean([compute_ndcg_at_k(s, k) for s in all_embedding_scores])
        ) if all_embedding_scores else 0.0
    aggregate["embedding_mrr"] = float(
        np.mean([compute_mrr(s) for s in all_embedding_scores])
    ) if all_embedding_scores else 0.0
    aggregate["embedding_map_at_10"] = compute_map_at_k(all_embedding_scores, 10)

    return {"aggregate": aggregate, "per_query": per_query_results}


# REPORT GENERATION


def generate_report(
    table_stats: dict,
    sim_distribution: dict,
    query_spread: dict,
    dimension_coverage: dict,
    hubness: dict,
    isotropy_audio: dict,
    isotropy_food: dict,
    self_sim_audio: dict,
    self_sim_food: dict,
    crossmodal_eval: dict,
    audio_eval: dict,
    output_dir: str,
) -> None:
    """Generate markdown report + CSV with all results."""
    os.makedirs(output_dir, exist_ok=True)

    flat_metrics = {}
    flat_metrics.update({f"stats_{k}": v for k, v in table_stats.items()})
    for table, data in sim_distribution.items():
        for k, v in data.items():
            flat_metrics[f"sim_dist_{table}_{k}"] = v
    flat_metrics["isotropy_audio"] = isotropy_audio.get("isotropy", 0)
    flat_metrics["isotropy_food"] = isotropy_food.get("isotropy", 0)
    flat_metrics["effective_dim95_audio"] = isotropy_audio.get("effective_dim_95pct", 0)
    flat_metrics["effective_dim95_food"] = isotropy_food.get("effective_dim_95pct", 0)
    flat_metrics.update({f"crossmodal_{k}": v for k, v in crossmodal_eval["aggregate"].items()})
    flat_metrics.update({f"audio_{k}": v for k, v in audio_eval["aggregate"].items()})

    pd.DataFrame([flat_metrics]).to_csv(os.path.join(output_dir, "ir_diagnostics.csv"), index=False)

    if crossmodal_eval["per_query"]:
        cm_df = pd.DataFrame([
            {k: v for k, v in q.items() if k != "results_detail"}
            for q in crossmodal_eval["per_query"]
        ])
        cm_df.to_csv(os.path.join(output_dir, "crossmodal_per_query.csv"), index=False)

    if audio_eval["per_query"]:
        au_df = pd.DataFrame([
            {k: v for k, v in q.items() if k != "results_detail"}
            for q in audio_eval["per_query"]
        ])
        au_df.to_csv(os.path.join(output_dir, "audio_per_query.csv"), index=False)

    md_lines = [
        "# IR Quality Evaluation Report",
        "",
        f"**Embedding model**: `all-MiniLM-L6-v2` (384d, dense)",
        f"**Distance metric**: cosine (pgvector `<=>` operator)",
        "",
        "---",
        "",
        "## Part B: Database Diagnostics",
        "",
        "### D1. Table Statistics",
        "",
        f"| Table | Rows |",
        f"|---|---|",
        f"| `audio_descriptors` | {table_stats['audio_descriptors_rows']:,} |",
        f"| `crossmodal_food_embeddings` | {table_stats['crossmodal_food_rows']:,} |",
        "",
        f"- Distinct food dimensions: **{table_stats['food_distinct_dimensions']}**",
        f"- Distinct food descriptors: **{table_stats['food_distinct_descriptors']:,}**",
        f"- Distinct food items: **{table_stats['food_distinct_items']:,}**",
        "",
        "### D2. Pairwise Similarity Distribution",
        "",
        "| Table | Mean | Std | Min | Median | Max | P10 | P90 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for table, data in sim_distribution.items():
        md_lines.append(
            f"| `{table}` | {data['mean']:.4f} | {data['std']:.4f} | "
            f"{data['min']:.4f} | {data['median']:.4f} | {data['max']:.4f} | "
            f"{data['p10']:.4f} | {data['p90']:.4f} |"
        )

    md_lines.extend([
        "",
        _interpret_sim_distribution(sim_distribution),
        "",
        "### D3. Query Similarity Spread (Top-1 vs Top-K Gap)",
        "",
        "| Table | Avg Top-1 Sim | Avg Gap (1 vs 5) | Avg Gap (1 vs 10) | Avg Gap (1 vs 20) |",
        "|---|---|---|---|---|",
    ])

    for table, data in query_spread.items():
        md_lines.append(
            f"| `{table}` | {data.get('avg_top1_sim', 0):.4f} | "
            f"{data.get('avg_top1_vs_top5_gap', 0):.4f} | "
            f"{data.get('avg_top1_vs_top10_gap', 0):.4f} | "
            f"{data.get('avg_top1_vs_top20_gap', 0):.4f} |"
        )

    md_lines.extend([
        "",
        "### D4. Dimension Coverage (Crossmodal)",
        "",
        "| Dimension | Count | Proportion |",
        "|---|---|---|",
    ])

    for dim, cnt in dimension_coverage["counts"].items():
        prop = dimension_coverage["proportions"].get(dim, 0)
        md_lines.append(f"| `{dim}` | {cnt:,} | {prop:.2%} |")

    md_lines.extend([
        "",
        f"**Gini coefficient**: {dimension_coverage['gini_coefficient']:.4f} "
        f"({'well balanced' if dimension_coverage['gini_coefficient'] < 0.15 else 'some imbalance' if dimension_coverage['gini_coefficient'] < 0.3 else 'significant imbalance'})",
        "",
        "### D5. Hubness Detection",
        "",
        "| Table | Unique Neighbors | Max Occurrence | Mean Occurrence | Skewness |",
        "|---|---|---|---|---|",
    ])

    for table, data in hubness.items():
        md_lines.append(
            f"| `{table}` | {data.get('unique_neighbors', 0)} | "
            f"{data.get('max_occurrence', 0)} | "
            f"{data.get('mean_occurrence', 0):.2f} | "
            f"{data.get('skewness', 0):.3f} |"
        )

    md_lines.extend([
        "",
        _interpret_hubness(hubness),
        "",
        "---",
        "",
        "## Part C: Intrinsic Embedding Quality (Strategy 1)",
        "",
        "### Isotropy",
        "",
        "| Table | Isotropy | Effective Dim (95%) | Effective Dim (99%) | Total Dim |",
        "|---|---|---|---|---|",
        f"| `audio_descriptors` | {isotropy_audio['isotropy']:.6f} | "
        f"{isotropy_audio['effective_dim_95pct']} | {isotropy_audio['effective_dim_99pct']} | {isotropy_audio['total_dimensions']} |",
        f"| `crossmodal_food_embeddings` | {isotropy_food['isotropy']:.6f} | "
        f"{isotropy_food['effective_dim_95pct']} | {isotropy_food['effective_dim_99pct']} | {isotropy_food['total_dimensions']} |",
        "",
        _interpret_isotropy(isotropy_audio, isotropy_food),
        "",
        "### Self-Similarity",
        "",
        "| Table | Avg Self-Sim | Std | Min | Max |",
        "|---|---|---|---|---|",
        f"| `audio_descriptors` | {self_sim_audio['avg_self_similarity']:.4f} | "
        f"{self_sim_audio['std_self_similarity']:.4f} | "
        f"{self_sim_audio['min_self_similarity']:.4f} | "
        f"{self_sim_audio['max_self_similarity']:.4f} |",
        f"| `crossmodal_food_embeddings` | {self_sim_food['avg_self_similarity']:.4f} | "
        f"{self_sim_food['std_self_similarity']:.4f} | "
        f"{self_sim_food['min_self_similarity']:.4f} | "
        f"{self_sim_food['max_self_similarity']:.4f} |",
        "",
        _interpret_self_similarity(self_sim_audio, self_sim_food),
        "",
        "---",
        "",
        "## Part C: Synthetic Relevance Evaluation (Strategy 2)",
        "",
        "### Crossmodal Food Descriptors (`get_top_k_food_descriptors`)",
        "",
        "| Metric | Structural | Embedding |",
        "|---|---|---|",
    ])

    cm_agg = crossmodal_eval["aggregate"]
    for k in [5, 10]:
        md_lines.append(
            f"| Precision@{k} | {cm_agg.get(f'structural_precision_at_{k}', 0):.3f} | "
            f"{cm_agg.get(f'embedding_precision_at_{k}', 0):.3f} |"
        )
        md_lines.append(
            f"| nDCG@{k} | {cm_agg.get(f'structural_ndcg_at_{k}', 0):.3f} | "
            f"{cm_agg.get(f'embedding_ndcg_at_{k}', 0):.3f} |"
        )
    md_lines.append(
        f"| MRR | {cm_agg.get('structural_mrr', 0):.3f} | {cm_agg.get('embedding_mrr', 0):.3f} |"
    )
    md_lines.append(
        f"| MAP@10 | {cm_agg.get('structural_map_at_10', 0):.3f} | {cm_agg.get('embedding_map_at_10', 0):.3f} |"
    )

    md_lines.extend([
        "",
        "### Audio Captions (`get_top_k_audio_captions`)",
        "",
        "| Metric | Embedding |",
        "|---|---|",
    ])

    au_agg = audio_eval["aggregate"]
    for k in [5, 10]:
        md_lines.append(f"| Precision@{k} | {au_agg.get(f'embedding_precision_at_{k}', 0):.3f} |")
        md_lines.append(f"| nDCG@{k} | {au_agg.get(f'embedding_ndcg_at_{k}', 0):.3f} |")
    md_lines.append(f"| MRR | {au_agg.get('embedding_mrr', 0):.3f} |")
    md_lines.append(f"| MAP@10 | {au_agg.get('embedding_map_at_10', 0):.3f} |")

    # Per-query details
    md_lines.extend([
        "",
        "### Per-Query Breakdown (Crossmodal)",
        "",
        "| Query | nDCG@5 (Struct.) | nDCG@10 (Struct.) | P@5 (Emb.) | Top-1 Sim | Avg Sim |",
        "|---|---|---|---|---|---|",
    ])
    for q in crossmodal_eval["per_query"]:
        md_lines.append(
            f"| {q['query_text']} | {q['structural_ndcg5']:.3f} | {q['structural_ndcg10']:.3f} | "
            f"{q['embedding_p5']:.3f} | {q['top1_sim']:.4f} | {q['avg_sim']:.4f} |"
        )

    md_lines.extend([
        "",
        "### Per-Query Breakdown (Audio Captions)",
        "",
        "| Query | nDCG@5 (Emb.) | nDCG@10 (Emb.) | Top-1 Sim | Sim Spread |",
        "|---|---|---|---|---|",
    ])
    for q in audio_eval["per_query"]:
        md_lines.append(
            f"| {q['query_text'][:40]}... | {q['embedding_ndcg5']:.3f} | {q['embedding_ndcg10']:.3f} | "
            f"{q['top1_sim']:.4f} | {q['sim_spread']:.4f} |"
        )

    md_lines.extend(["", "---", "", _overall_assessment(cm_agg, au_agg, sim_distribution, isotropy_audio, isotropy_food)])

    report_path = os.path.join(output_dir, "ir_eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n  ✓ Report: {report_path}")
    print(f"  ✓ CSV:    {os.path.join(output_dir, 'ir_diagnostics.csv')}")


def _interpret_sim_distribution(sim_dist: dict) -> str:
    lines = ["> **Interpretation**:"]
    for table, data in sim_dist.items():
        std = data["std"]
        mean = data["mean"]
        spread = data["p90"] - data["p10"]
        if std < 0.05:
            lines.append(f"> - `{table}`: Very tight distribution (σ={std:.4f}). "
                         f"Cosine similarities cluster around {mean:.3f}, limiting discrimination.")
        elif std < 0.10:
            lines.append(f"> - `{table}`: Moderate spread (σ={std:.4f}, P10-P90 range={spread:.4f}). "
                         f"Acceptable for retrieval.")
        else:
            lines.append(f"> - `{table}`: Good spread (σ={std:.4f}). Strong discriminative power.")
    return "\n".join(lines)


def _interpret_hubness(hubness: dict) -> str:
    lines = ["> **Interpretation**:"]
    for table, data in hubness.items():
        skew = data.get("skewness", 0)
        max_occ = data.get("max_occurrence", 0)
        if skew > 2.0:
            lines.append(f"> - `{table}`: High hubness (skewness={skew:.2f}, max occurrence={max_occ}). "
                         f"Some vectors dominate as neighbors.")
        elif skew > 1.0:
            lines.append(f"> - `{table}`: Moderate hubness (skewness={skew:.2f}). Monitor for bias.")
        else:
            lines.append(f"> - `{table}`: Low hubness (skewness={skew:.2f}). Fair neighbor distribution.")
    return "\n".join(lines)


def _interpret_isotropy(audio: dict, food: dict) -> str:
    lines = ["> **Interpretation** (higher isotropy = better space utilization):"]
    for name, data in [("audio_descriptors", audio), ("crossmodal_food_embeddings", food)]:
        iso = data["isotropy"]
        eff95 = data["effective_dim_95pct"]
        total = data["total_dimensions"]
        pct = (eff95 / total) * 100
        if iso < 1e-6:
            lines.append(f"> - `{name}`: Very anisotropic (isotropy≈{iso:.2e}). "
                         f"Only {eff95}/{total} dims ({pct:.0f}%) carry 95% of variance. "
                         f"Vectors occupy a narrow cone — expected for sentence-transformers.")
        else:
            lines.append(f"> - `{name}`: Isotropy={iso:.6f}, effective dims={eff95}/{total}.")
    return "\n".join(lines)


def _interpret_self_similarity(audio: dict, food: dict) -> str:
    lines = ["> **Interpretation** (lower self-similarity = more spread in the space):"]
    for name, data in [("audio_descriptors", audio), ("crossmodal_food_embeddings", food)]:
        avg = data["avg_self_similarity"]
        if avg > 0.5:
            lines.append(f"> - `{name}`: High average self-similarity ({avg:.4f}). "
                         f"Vectors are tightly clustered — typical for sentence-transformers on short texts.")
        elif avg > 0.3:
            lines.append(f"> - `{name}`: Moderate self-similarity ({avg:.4f}). Reasonable diversity.")
        else:
            lines.append(f"> - `{name}`: Low self-similarity ({avg:.4f}). Good vector space diversity.")
    return "\n".join(lines)


def _overall_assessment(cm_agg: dict, au_agg: dict, sim_dist: dict, iso_audio: dict, iso_food: dict) -> str:
    lines = [
        "## Overall Assessment",
        "",
    ]

    # Vector space verdict
    lines.append("### Vector Space Quality")
    lines.append("")
    lines.append("Both tables use **dense** 384-dimensional embeddings from `all-MiniLM-L6-v2`. "
                 "These are NOT sparse vectors — every dimension has a continuous value.")
    lines.append("")

    for name, iso in [("audio_descriptors", iso_audio), ("crossmodal_food_embeddings", iso_food)]:
        eff = iso["effective_dim_95pct"]
        total = iso["total_dimensions"]
        lines.append(f"- `{name}`: {eff}/{total} effective dimensions at 95% variance. "
                     f"{'Highly anisotropic but functional for retrieval.' if eff < total * 0.3 else 'Good utilization.'}")

    lines.extend([
        "",
        "### Retrieval Quality",
        "",
    ])

    cm_ndcg = cm_agg.get("structural_ndcg_at_10", 0)
    au_ndcg = au_agg.get("embedding_ndcg_at_10", 0)

    if cm_ndcg >= 0.5:
        lines.append(f"- **Crossmodal food retrieval**: Good (nDCG@10={cm_ndcg:.3f})")
    elif cm_ndcg >= 0.3:
        lines.append(f"- **Crossmodal food retrieval**: Acceptable (nDCG@10={cm_ndcg:.3f})")
    else:
        lines.append(f"- **Crossmodal food retrieval**: Needs improvement (nDCG@10={cm_ndcg:.3f})")

    if au_ndcg >= 0.5:
        lines.append(f"- **Audio caption retrieval**: Good (nDCG@10={au_ndcg:.3f})")
    elif au_ndcg >= 0.3:
        lines.append(f"- **Audio caption retrieval**: Acceptable (nDCG@10={au_ndcg:.3f})")
    else:
        lines.append(f"- **Audio caption retrieval**: Needs improvement (nDCG@10={au_ndcg:.3f})")

    return "\n".join(lines)


def _gini(values: list[int]) -> float:
    """Compute Gini coefficient for distribution balance."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = np.cumsum(sorted_vals)
    return float((2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * cumulative[-1])) - (n + 1) / n)


def _skewness(arr: np.ndarray) -> float:
    """Compute sample skewness."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((arr - mean) / std) ** 3))


# CLI


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IR quality of RAG retrieval functions (fully offline, no LLM API needed)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ablations/ir_eval",
        help="Directory for output reports (default: data/ablations/ir_eval/)",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=str(DEFAULT_QUERIES_PATH),
        help="Path to synthetic queries JSON fixture.",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval evaluation (only run diagnostics + intrinsic quality).",
    )
    args = parser.parse_args()

    print(" IR Quality Evaluation — RAG Retrieval Functions")

    # Load model
    print("\n[1/4] Loading embedding model (local, CPU)...")
    model = _get_model()

    # DB diagnostics
    print("[2/4] Running database diagnostics...")
    conn = get_conn()
    with conn.cursor() as cur:
        table_stats = diagnose_table_stats(cur)
        print(f"  → audio_descriptors: {table_stats['audio_descriptors_rows']:,} rows")
        print(f"  → crossmodal_food:   {table_stats['crossmodal_food_rows']:,} rows")

        sim_distribution = diagnose_similarity_distribution(cur)
        print("  → Similarity distribution computed")

        query_spread = diagnose_query_spread(cur)
        print("  → Query spread computed")

        dimension_coverage = diagnose_dimension_coverage(cur)
        print(f"  → Dimension coverage: {len(dimension_coverage['counts'])} dimensions")

        hubness = diagnose_hubness(cur)
        print("  → Hubness analysis complete")

    # Intrinsic quality
    print("[3/4] Computing intrinsic embedding quality...")
    with conn.cursor() as cur:
        isotropy_audio = compute_isotropy(cur, "audio_descriptors", "embedding")
        print(f"  → Audio isotropy: {isotropy_audio['isotropy']:.6f}")

        isotropy_food = compute_isotropy(cur, "crossmodal_food_embeddings", "text_embedding")
        print(f"  → Food isotropy:  {isotropy_food['isotropy']:.6f}")

        self_sim_audio = compute_self_similarity_stats(cur, "audio_descriptors", "embedding")
        print(f"  → Audio avg self-similarity: {self_sim_audio['avg_self_similarity']:.4f}")

        self_sim_food = compute_self_similarity_stats(cur, "crossmodal_food_embeddings", "text_embedding")
        print(f"  → Food avg self-similarity:  {self_sim_food['avg_self_similarity']:.4f}")

    # Retrieval evaluation
    crossmodal_eval = {"aggregate": {}, "per_query": []}
    audio_eval = {"aggregate": {}, "per_query": []}

    if not args.skip_retrieval:
        print("[4/4] Running synthetic relevance evaluation...")
        queries = load_synthetic_queries(args.queries)

        crossmodal_eval = evaluate_crossmodal_retrieval(
            queries["crossmodal_food_queries"], model
        )
        print(f"  → Crossmodal: {len(crossmodal_eval['per_query'])} queries evaluated")

        audio_eval = evaluate_audio_caption_retrieval(
            queries["audio_caption_queries"], model
        )
        print(f"  → Audio:      {len(audio_eval['per_query'])} queries evaluated")
    else:
        print("[4/4] Skipping retrieval evaluation (--skip-retrieval)")

    # Generate report
    print("\nGenerating report...")
    generate_report(
        table_stats=table_stats,
        sim_distribution=sim_distribution,
        query_spread=query_spread,
        dimension_coverage=dimension_coverage,
        hubness=hubness,
        isotropy_audio=isotropy_audio,
        isotropy_food=isotropy_food,
        self_sim_audio=self_sim_audio,
        self_sim_food=self_sim_food,
        crossmodal_eval=crossmodal_eval,
        audio_eval=audio_eval,
        output_dir=args.output_dir,
    )

    print(" Done!")


if __name__ == "__main__":
    main()
