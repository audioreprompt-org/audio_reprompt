"""Retrieval evaluation loops, baselines and ANN check (WS-C).

CONTRACT: WS-C implements this module, report.py and cli.py, and turns
models/eval_rag_ir_quality.py into a thin wrapper around cli.main().
Uses data.py (WS-0), judges.py and metrics.py (WS-B) as black boxes.

Addresses review findings: 1.4 (no verdict from the embedding judge), 1.5/1.6
(via WS-B), 1.7 (CIs + baseline), §2 cut_results (production uses
cut_results=True, see models/pipeline.py), duplicates, failed queries,
ANN vs exact, result-count asserts.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from psycopg import sql

from models.descriptors.rag import get_top_k_audio_captions, get_top_k_food_descriptors
from models.ir_eval import data, judges, metrics

K_VALUES: tuple[int, ...] = (5, 10)


def fetch_corpus_pairs(cursor) -> list[tuple[str, str]]:
    """All DISTINCT (dimension, descriptor) pairs of crossmodal_food_embeddings."""
    cursor.execute(
        "SELECT DISTINCT dimension, descriptor FROM crossmodal_food_embeddings ORDER BY dimension, descriptor"
    )
    rows = cursor.fetchall()
    return [(str(r[0]) if r[0] is not None else "", str(r[1]) if r[1] is not None else "") for r in rows]


def dedupe_crossmodal_results(results: list[dict]) -> tuple[list[dict], int]:
    """Keep the first (highest-ranked) occurrence of each normalized
    (dimension, descriptor) pair, preserving order. Returns (deduped, n_removed)."""
    seen = set()
    deduped = []
    for r in results:
        dim = judges.normalize_term(r.get("dimension"))
        desc = judges.normalize_term(r.get("descriptor"))
        key = (dim, desc)
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped, len(results) - len(deduped)


def random_baseline_grades(
    corpus_pairs: list[tuple[str, str]],
    expected_descriptors: dict[str, list[str]],
    length: int,
    rng: np.random.Generator,
) -> list[int]:
    """Grades of ``length`` distinct corpus pairs drawn uniformly at random
    (structural judge). ``length`` = the system's list length for that query."""
    if length <= 0 or not corpus_pairs:
        return []

    seen = set()
    unique_pairs = []
    for dim, desc in corpus_pairs:
        key = (judges.normalize_term(dim), judges.normalize_term(desc))
        if key not in seen:
            seen.add(key)
            unique_pairs.append((dim, desc))

    n = len(unique_pairs)
    k = min(length, n)
    indices = rng.choice(n, size=k, replace=False)
    return [
        judges.grade_structural(unique_pairs[i][0], unique_pairs[i][1], expected_descriptors)
        for i in indices
    ]


def evaluate_crossmodal(
    queries: list[dict],
    corpus_pairs: list[tuple[str, str]],
    cut_results: bool,
    rng: np.random.Generator,
    seed: int,
) -> dict:
    """Evaluate ``get_top_k_food_descriptors(emb, cut_results=...)``.

    Per query: encode with data.encode, retrieve, dedupe, grade with
    judges.score_relevance_structural, compute corpus grades with
    judges.corpus_grades_structural, then P@k, nDCG@k, Recall@k, MRR, AP@10 for
    k in K_VALUES. A query with no results scores 0 (grades = []) and is counted
    in ``n_failed``. A query with no relevant item in the corpus is counted in
    ``n_no_relevant_in_corpus`` (Recall/AP are NaN for it; bootstrap ignores NaN).
    top1_sim = max(sim). Also compute the random baseline for the same queries
    and lengths, and the retriever-agreement diagnostic
    (judges.score_agreement_embedding on "dimension descriptor" strings).

    Returns:
        {
          "cut_results": bool,
          "n_queries": int, "n_failed": int, "n_no_relevant_in_corpus": int,
          "n_duplicates_removed": int,
          "system":   {metric_name: bootstrap_ci dict},   # e.g. "ndcg_at_10"
          "baseline": {metric_name: bootstrap_ci dict},
          "system_vs_baseline": {metric_name: paired_bootstrap_diff dict},
          "agreement_diagnostic": {"agreement_p_at_5": bootstrap_ci dict,
                                   "agreement_p_at_10": bootstrap_ci dict},
          "per_query": [ {"query_id", "query_text", "n_results", "n_relevant_in_corpus",
                          "top1_sim", "avg_sim", <metric_name>: float ...,
                          "results_detail": [{"dim","desc","sim","grade"}]} ],
        }
        metric names: precision_at_{k}, ndcg_at_{k}, recall_at_{k} for k in K_VALUES,
        plus "mrr" and "map_at_10".
    """
    n_failed = 0
    n_no_relevant_in_corpus = 0
    total_duplicates_removed = 0

    per_query: list[dict] = []
    sys_metrics_series: dict[str, list[float]] = {
        f"precision_at_{k}": [] for k in K_VALUES
    }
    sys_metrics_series.update({f"ndcg_at_{k}": [] for k in K_VALUES})
    sys_metrics_series.update({f"recall_at_{k}": [] for k in K_VALUES})
    sys_metrics_series["mrr"] = []
    sys_metrics_series["map_at_10"] = []

    base_metrics_series: dict[str, list[float]] = {
        k: [] for k in sys_metrics_series
    }

    all_agr_p5: list[float] = []
    all_agr_p10: list[float] = []

    for q in queries:
        query_id = q.get("id", "")
        query_text = q.get("query_text", "")
        expected = q.get("expected_descriptors", {})

        if "embedding" in q and q["embedding"] is not None:
            emb = q["embedding"]
        else:
            emb = data.encode([query_text])[0]
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()

        raw_results = get_top_k_food_descriptors(emb, cut_results=cut_results)
        results, n_removed = dedupe_crossmodal_results(raw_results)
        total_duplicates_removed += n_removed

        corpus_grades = judges.corpus_grades_structural(corpus_pairs, expected)
        n_relevant = sum(1 for g in corpus_grades if g >= metrics.RELEVANT_GRADE)
        if n_relevant == 0:
            n_no_relevant_in_corpus += 1

        if not results:
            n_failed += 1
            grades = []
            top1_sim = 0.0
            avg_sim = 0.0
        else:
            grades = judges.score_relevance_structural(results, expected)
            sims = [float(r.get("sim", 0.0)) for r in results]
            top1_sim = float(max(sims))
            avg_sim = float(np.mean(sims))

        # System metrics
        q_sys_metrics: dict[str, float] = {}
        for k in K_VALUES:
            q_sys_metrics[f"precision_at_{k}"] = metrics.precision_at_k(grades, k)
            q_sys_metrics[f"ndcg_at_{k}"] = metrics.ndcg_at_k(grades, k, corpus_grades)
            q_sys_metrics[f"recall_at_{k}"] = metrics.recall_at_k(grades, k, n_relevant)
        q_sys_metrics["mrr"] = metrics.reciprocal_rank(grades)
        q_sys_metrics["map_at_10"] = metrics.average_precision_at_k(grades, 10, n_relevant)

        for m_name, val in q_sys_metrics.items():
            sys_metrics_series[m_name].append(val)

        # Baseline metrics (length matches system's list length)
        base_grades = random_baseline_grades(corpus_pairs, expected, len(results), rng)
        q_base_metrics: dict[str, float] = {}
        for k in K_VALUES:
            q_base_metrics[f"precision_at_{k}"] = metrics.precision_at_k(base_grades, k)
            q_base_metrics[f"ndcg_at_{k}"] = metrics.ndcg_at_k(base_grades, k, corpus_grades)
            q_base_metrics[f"recall_at_{k}"] = metrics.recall_at_k(base_grades, k, n_relevant)
        q_base_metrics["mrr"] = metrics.reciprocal_rank(base_grades)
        q_base_metrics["map_at_10"] = metrics.average_precision_at_k(base_grades, 10, n_relevant)

        for m_name, val in q_base_metrics.items():
            base_metrics_series[m_name].append(val)

        # Retriever-agreement diagnostic
        retrieved_texts = [
            f"{r.get('dimension', '')} {r.get('descriptor', '')}" for r in results
        ]
        agr_grades = judges.score_agreement_embedding(
            query_text, retrieved_texts, encode_fn=data.encode
        )
        p5 = metrics.precision_at_k(agr_grades, 5)
        p10 = metrics.precision_at_k(agr_grades, 10)
        all_agr_p5.append(p5)
        all_agr_p10.append(p10)

        results_detail = [
            {
                "dim": r.get("dimension", ""),
                "desc": r.get("descriptor", ""),
                "sim": float(r.get("sim", 0.0)),
                "grade": int(g),
            }
            for r, g in zip(results, grades)
        ]

        per_query_record = {
            "query_id": query_id,
            "query_text": query_text,
            "n_results": len(results),
            "n_relevant_in_corpus": n_relevant,
            "top1_sim": top1_sim,
            "avg_sim": avg_sim,
            **q_sys_metrics,
            "results_detail": results_detail,
        }
        per_query.append(per_query_record)

    system_agg = {
        m: metrics.bootstrap_ci(sys_metrics_series[m], seed=seed)
        for m in sys_metrics_series
    }
    baseline_agg = {
        m: metrics.bootstrap_ci(base_metrics_series[m], seed=seed)
        for m in base_metrics_series
    }
    system_vs_baseline = {
        m: metrics.paired_bootstrap_diff(
            sys_metrics_series[m], base_metrics_series[m], seed=seed
        )
        for m in sys_metrics_series
    }

    agreement_diagnostic = {
        "agreement_p_at_5": metrics.bootstrap_ci(all_agr_p5, seed=seed),
        "agreement_p_at_10": metrics.bootstrap_ci(all_agr_p10, seed=seed),
    }

    return {
        "cut_results": cut_results,
        "n_queries": len(queries),
        "n_failed": n_failed,
        "n_no_relevant_in_corpus": n_no_relevant_in_corpus,
        "n_duplicates_removed": total_duplicates_removed,
        "system": system_agg,
        "baseline": baseline_agg,
        "system_vs_baseline": system_vs_baseline,
        "agreement_diagnostic": agreement_diagnostic,
        "per_query": per_query,
    }


def evaluate_audio(queries: list[dict], k: int = 20) -> dict:
    """Evaluate ``get_top_k_audio_captions(emb, k=k, using_clap=False)`` (the
    production path, see models/pipeline.py).

    There is no independent relevance judge for audio yet, so NO quality verdict
    is produced. Report: n_failed, n_collapsed (len(results) < k, i.e. duplicate
    captions collapsed by the dict in rag.py), similarity stats (top1, mean,
    spread) with bootstrap CIs, and the agreement diagnostic clearly labeled.

    Returns:
        {
          "k": int, "n_queries": int, "n_failed": int, "n_collapsed": int,
          "top1_sim": bootstrap_ci dict, "avg_sim": bootstrap_ci dict,
          "sim_spread": bootstrap_ci dict,
          "agreement_diagnostic": {"agreement_p_at_5": ..., "agreement_p_at_10": ...},
          "per_query": [ {"query_id","query_text","n_results","top1_sim","avg_sim",
                          "sim_spread","results_detail": [{"caption","db_sim"}]} ],
        }
    """
    n_failed = 0
    n_collapsed = 0

    all_top1: list[float] = []
    all_avg: list[float] = []
    all_spread: list[float] = []
    all_agr_p5: list[float] = []
    all_agr_p10: list[float] = []
    per_query: list[dict] = []

    for q in queries:
        query_id = q.get("id", "")
        query_text = q.get("query_text", "")

        if "embedding" in q and q["embedding"] is not None:
            emb = q["embedding"]
        else:
            emb = data.encode([query_text])[0]
        if isinstance(emb, np.ndarray):
            emb = emb.tolist()

        raw_results = get_top_k_audio_captions(emb, k=k, using_clap=False)
        captions = list(raw_results.keys())
        sims = [float(s) for s in raw_results.values()]
        n_res = len(captions)

        if n_res == 0:
            n_failed += 1
            top1 = 0.0
            avg = 0.0
            spread = 0.0
        else:
            if n_res < k:
                n_collapsed += 1
            top1 = float(max(sims))
            avg = float(np.mean(sims))
            spread = float(max(sims) - min(sims))

        all_top1.append(top1)
        all_avg.append(avg)
        all_spread.append(spread)

        agr_grades = judges.score_agreement_embedding(
            query_text, captions, encode_fn=data.encode
        )
        p5 = metrics.precision_at_k(agr_grades, 5)
        p10 = metrics.precision_at_k(agr_grades, 10)
        all_agr_p5.append(p5)
        all_agr_p10.append(p10)

        results_detail = [
            {"caption": cap, "db_sim": sim}
            for cap, sim in zip(captions, sims)
        ]

        per_query.append({
            "query_id": query_id,
            "query_text": query_text,
            "n_results": n_res,
            "top1_sim": top1,
            "avg_sim": avg,
            "sim_spread": spread,
            "results_detail": results_detail,
        })

    return {
        "k": k,
        "n_queries": len(queries),
        "n_failed": n_failed,
        "n_collapsed": n_collapsed,
        "top1_sim": metrics.bootstrap_ci(all_top1),
        "avg_sim": metrics.bootstrap_ci(all_avg),
        "sim_spread": metrics.bootstrap_ci(all_spread),
        "agreement_diagnostic": {
            "agreement_p_at_5": metrics.bootstrap_ci(all_agr_p5),
            "agreement_p_at_10": metrics.bootstrap_ci(all_agr_p10),
        },
        "per_query": per_query,
    }


def ann_vs_exact(conn, table: str, query_embs: np.ndarray, k: int = 10) -> dict:
    """Compare pgvector results with and without index scans.

    List vector indexes on ``table`` (pg_indexes, look for hnsw/ivfflat). For each
    query (embedding passed with data.to_pgvector_literal), fetch top-k ids with
    the default plan, then inside a transaction with
    ``SET LOCAL enable_indexscan = off`` (and enable_bitmapscan = off). Read-only:
    never write to the DB.

    Returns:
        {"table": str, "indexes": list[str], "k": int, "n_queries": int,
         "mean_overlap_at_k": float, "min_overlap_at_k": float}
    """
    col = data.TABLES.get(table, "embedding")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
            [table],
        )
        rows = cur.fetchall()

    indexes = [
        str(r[0])
        for r in rows
        if r[1] and ("hnsw" in str(r[1]).lower() or "ivfflat" in str(r[1]).lower())
    ]

    overlaps: list[float] = []

    for vec in query_embs:
        vec_lit = data.to_pgvector_literal(vec)

        # 1. Default plan (may use vector index if available)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT id FROM {tbl} ORDER BY {col} <=> %s::vector LIMIT %s"
                ).format(tbl=sql.Identifier(table), col=sql.Identifier(col)),
                [vec_lit, k],
            )
            ann_ids = [r[0] for r in cur.fetchall()]

        # 2. Exact plan inside a read-only transaction disabling index scans
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
                cur.execute(
                    sql.SQL(
                        "SELECT id FROM {tbl} ORDER BY {col} <=> %s::vector LIMIT %s"
                    ).format(tbl=sql.Identifier(table), col=sql.Identifier(col)),
                    [vec_lit, k],
                )
                exact_ids = [r[0] for r in cur.fetchall()]

        if k > 0:
            overlap = len(set(ann_ids) & set(exact_ids)) / float(k)
        else:
            overlap = 1.0
        overlaps.append(overlap)

    n_q = len(query_embs)
    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    min_overlap = float(np.min(overlaps)) if overlaps else 0.0

    return {
        "table": table,
        "indexes": indexes,
        "k": int(k),
        "n_queries": n_q,
        "mean_overlap_at_k": mean_overlap,
        "min_overlap_at_k": min_overlap,
    }
