"""Command-line entry point (WS-C).

CONTRACT:
    python -m models.ir_eval.cli [--output-dir data/ablations/ir_eval]
                                 [--queries models/ir_eval_queries.json]
                                 [--seed 42] [--skip-retrieval] [--skip-ann-check]

Flow: one db_connection() for the whole run; load both EmbeddingMatrix once;
run diagnostics with make_rng(seed) (derive a separate child rng per step so
adding a step does not change others); encode eval queries once and reuse them
for hubness query-mode, ANN check and retrieval; build the results dict
described in report.py; call generate_report. Same seed -> identical results.json
except meta.timestamp.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from models.ir_eval import data, diagnostics, report, retrieval

DEFAULT_QUERIES_PATH = "models/ir_eval_queries.json"
DEFAULT_OUTPUT_DIR = "data/ablations/ir_eval"


def _get_git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return None


def main(argv: list[str] | None = None) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="Run RAG IR quality evaluation and diagnostics (offline, no external LLM API)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output reports and CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=DEFAULT_QUERIES_PATH,
        help=f"Path to synthetic queries JSON fixture (default: {DEFAULT_QUERIES_PATH})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=data.DEFAULT_SEED,
        help=f"Random seed for sampling and bootstrapping (default: {data.DEFAULT_SEED})",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip retrieval evaluation loops (only run database & space diagnostics).",
    )
    parser.add_argument(
        "--skip-ann-check",
        action="store_true",
        help="Skip approximate vs exact index scan check.",
    )

    args = parser.parse_args(argv)

    seed = int(args.seed)
    child_seeds = np.random.SeedSequence(seed).spawn(16)
    rngs = [np.random.default_rng(s) for s in child_seeds]

    # Assign dedicated child RNGs to each diagnostic step
    rng_sim_audio = rngs[0]
    rng_sim_food = rngs[1]
    rng_spread_audio = rngs[2]
    rng_spread_food = rngs[3]
    rng_aniso_audio = rngs[4]
    rng_aniso_food = rngs[5]
    rng_cm_true = rngs[6]
    rng_cm_false = rngs[7]

    # Load synthetic queries if fixture exists
    queries_path = Path(args.queries)
    if queries_path.exists():
        with open(queries_path, "r", encoding="utf-8") as f:
            queries_data = json.load(f)
    else:
        queries_data = {}

    cm_queries = queries_data.get("crossmodal_food_queries", [])
    audio_queries = queries_data.get("audio_caption_queries", [])

    # Encode evaluation queries ONCE and reuse them across steps
    cm_texts = [q["query_text"] for q in cm_queries]
    cm_embs = data.encode(cm_texts) if cm_texts else np.empty((0, 384))
    for q, emb in zip(cm_queries, cm_embs):
        q["embedding"] = emb.tolist()

    audio_texts = [q["query_text"] for q in audio_queries]
    audio_embs = data.encode(audio_texts) if audio_texts else np.empty((0, 384))
    for q, emb in zip(audio_queries, audio_embs):
        q["embedding"] = emb.tolist()

    # Database connection for the entire run
    with data.db_connection() as conn:
        with conn.cursor() as cur:
            table_stats = diagnostics.table_stats(cur)
            dimension_coverage = diagnostics.dimension_coverage(cur)
            em_audio = data.load_embeddings(cur, "audio_descriptors")
            em_food = data.load_embeddings(cur, "crossmodal_food_embeddings")
            corpus_pairs = retrieval.fetch_corpus_pairs(cur)

        # Intrinsic & DB diagnostics computed in numpy on loaded EmbeddingMatrix
        sim_dist_audio = diagnostics.similarity_distribution(em_audio, rng_sim_audio)
        sim_dist_food = diagnostics.similarity_distribution(em_food, rng_sim_food)

        query_spread_audio = diagnostics.query_spread(em_audio, rng_spread_audio)
        query_spread_food = diagnostics.query_spread(em_food, rng_spread_food)

        hubness_audio_loo = diagnostics.hubness(em_audio)
        hubness_food_loo = diagnostics.hubness(em_food)

        hubness_audio_q = (
            diagnostics.hubness(em_audio, query_X=audio_embs)
            if len(audio_embs) > 0
            else None
        )
        hubness_food_q = (
            diagnostics.hubness(em_food, query_X=cm_embs)
            if len(cm_embs) > 0
            else None
        )

        anisotropy_audio = diagnostics.anisotropy(em_audio, rng_aniso_audio)
        anisotropy_food = diagnostics.anisotropy(em_food, rng_aniso_food)

        # ANN check
        if not args.skip_ann_check:
            ann_results = {
                "audio_descriptors": retrieval.ann_vs_exact(
                    conn, "audio_descriptors", audio_embs
                ),
                "crossmodal_food_embeddings": retrieval.ann_vs_exact(
                    conn, "crossmodal_food_embeddings", cm_embs
                ),
            }
        else:
            ann_results = {}

    # Retrieval evaluation
    if not args.skip_retrieval:
        cm_cut_true = retrieval.evaluate_crossmodal(
            cm_queries, corpus_pairs, cut_results=True, rng=rng_cm_true, seed=seed
        )
        cm_cut_false = retrieval.evaluate_crossmodal(
            cm_queries, corpus_pairs, cut_results=False, rng=rng_cm_false, seed=seed
        )
        cm_results = {"cut_true": cm_cut_true, "cut_false": cm_cut_false}
        audio_results = retrieval.evaluate_audio(audio_queries, k=20)
    else:
        cm_results = {}
        audio_results = {}

    results = {
        "meta": {
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_commit(),
            "embedding_model": data.EMBEDDING_MODEL_NAME,
            "queries_path": str(args.queries),
        },
        "table_stats": table_stats,
        "dimension_coverage": dimension_coverage,
        "similarity_distribution": {
            "audio_descriptors": sim_dist_audio,
            "crossmodal_food_embeddings": sim_dist_food,
        },
        "query_spread": {
            "audio_descriptors": query_spread_audio,
            "crossmodal_food_embeddings": query_spread_food,
        },
        "hubness": {
            "audio_descriptors": {
                "leave_one_out": hubness_audio_loo,
                "queries": hubness_audio_q,
            },
            "crossmodal_food_embeddings": {
                "leave_one_out": hubness_food_loo,
                "queries": hubness_food_q,
            },
        },
        "anisotropy": {
            "audio_descriptors": anisotropy_audio,
            "crossmodal_food_embeddings": anisotropy_food,
        },
        "ann_vs_exact": ann_results,
        "crossmodal": cm_results,
        "audio": audio_results,
    }

    report_path = report.generate_report(results, args.output_dir)
    print(f"\n  ✓ Report generated: {report_path}")


if __name__ == "__main__":
    main()
