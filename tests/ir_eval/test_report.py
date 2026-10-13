"""Unit tests for models.ir_eval.report."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import pytest

from models.ir_eval.report import generate_report


@pytest.fixture
def synthetic_results():
    return {
        "meta": {
            "seed": 42,
            "timestamp": "2026-10-13T12:00:00Z",
            "git_commit": "abcdef123456",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "queries_path": "models/ir_eval_queries.json",
        },
        "table_stats": {
            "audio_descriptors_rows": 13540,
            "audio_distinct_captions": 13540,
            "crossmodal_food_rows": 26495,
            "food_distinct_dimensions": 7,
            "food_distinct_descriptors": 639,
            "food_distinct_dim_descriptor_pairs": 750,
            "food_distinct_items": 120,
        },
        "dimension_coverage": {
            "counts": {"taste": 8000, "texture": 6000, "color": 4000},
            "proportions": {"taste": 0.44, "texture": 0.33, "color": 0.22},
            "total": 18000,
            "gini_coefficient": 0.15,
            "gini_max": 0.66,
            "gini_normalized": 0.227,
        },
        "similarity_distribution": {
            "audio_descriptors": {
                "mean": 0.25,
                "std": 0.08,
                "min": 0.01,
                "max": 0.9999,
                "median": 0.24,
                "p10": 0.15,
                "p90": 0.35,
                "n_pairs": 1999000,
                "n_samples": 2000,
                "n_near_duplicate_pairs": 5,
            },
            "crossmodal_food_embeddings": {
                "mean": 0.22,
                "std": 0.07,
                "min": -0.05,
                "max": 0.95,
                "median": 0.21,
                "p10": 0.12,
                "p90": 0.31,
                "n_pairs": 1999000,
                "n_samples": 2000,
                "n_near_duplicate_pairs": 0,
            },
        },
        "query_spread": {
            "audio_descriptors": {
                "n_queries": 200,
                "avg_top1_sim": 0.65,
                "std_top1_sim": 0.10,
                "avg_top1_vs_top5_gap": 0.12,
                "avg_top1_vs_top10_gap": 0.16,
                "avg_top1_vs_top20_gap": 0.20,
                "n_queries_with_near_duplicate": 2,
            },
            "crossmodal_food_embeddings": {
                "n_queries": 200,
                "avg_top1_sim": 0.58,
                "std_top1_sim": 0.09,
                "avg_top1_vs_top5_gap": 0.10,
                "avg_top1_vs_top10_gap": 0.14,
                "avg_top1_vs_top20_gap": 0.18,
                "n_queries_with_near_duplicate": 0,
            },
        },
        "hubness": {
            "audio_descriptors": {
                "leave_one_out": {
                    "mode": "leave_one_out",
                    "n_points": 13540,
                    "n_queries": 13540,
                    "k": 10,
                    "skewness": 1.45,
                    "pct_antihubs": 0.12,
                    "max_occurrence": 45,
                    "mean_occurrence": 10.0,
                    "top10_hub_ids": [1, 2, 3],
                    "top10_hub_counts": [45, 40, 38],
                },
                "queries": {
                    "mode": "queries",
                    "n_points": 13540,
                    "n_queries": 10,
                    "k": 10,
                    "skewness": 2.1,
                    "pct_antihubs": 0.95,
                    "max_occurrence": 3,
                    "mean_occurrence": 0.007,
                    "top10_hub_ids": [1, 2],
                    "top10_hub_counts": [3, 2],
                },
            },
            "crossmodal_food_embeddings": {
                "leave_one_out": {
                    "mode": "leave_one_out",
                    "n_points": 26495,
                    "n_queries": 26495,
                    "k": 10,
                    "skewness": 1.82,
                    "pct_antihubs": 0.18,
                    "max_occurrence": 62,
                    "mean_occurrence": 10.0,
                    "top10_hub_ids": [10, 20],
                    "top10_hub_counts": [62, 55],
                },
                "queries": None,
            },
        },
        "anisotropy": {
            "audio_descriptors": {
                "n_samples": 13540,
                "total_dimensions": 384,
                "mean_pairwise_cosine": 0.22,
                "mean_vector_norm": 0.45,
                "participation_ratio_centered": 180.5,
                "participation_ratio_uncentered": 45.2,
                "effective_dim_95pct": 195,
                "effective_dim_99pct": 280,
                "numerical_rank": 384,
                "smallest5_eigenvalues": [0.001, 0.0012, 0.0015, 0.0018, 0.002],
                "top5_eigenvalues": [0.05, 0.04, 0.03, 0.02, 0.015],
                "top1_uncentered_eigenvalue_share": 0.20,
                "eig_ratio_min_max": 0.02,
                "eig_ratio_mp_reference": 0.05,
            },
            "crossmodal_food_embeddings": {
                "n_samples": 26495,
                "total_dimensions": 384,
                "mean_pairwise_cosine": 0.24,
                "mean_vector_norm": 0.48,
                "participation_ratio_centered": 172.0,
                "participation_ratio_uncentered": 40.0,
                "effective_dim_95pct": 188,
                "effective_dim_99pct": 270,
                "numerical_rank": 384,
                "smallest5_eigenvalues": [0.0008, 0.001, 0.0011, 0.0013, 0.0016],
                "top5_eigenvalues": [0.06, 0.045, 0.035, 0.025, 0.018],
                "top1_uncentered_eigenvalue_share": 0.23,
                "eig_ratio_min_max": 0.013,
                "eig_ratio_mp_reference": 0.06,
            },
        },
        "ann_vs_exact": {
            "audio_descriptors": {
                "table": "audio_descriptors",
                "indexes": ["idx_audio_hnsw"],
                "k": 10,
                "n_queries": 10,
                "mean_overlap_at_k": 0.98,
                "min_overlap_at_k": 0.90,
            }
        },
        "crossmodal": {
            "cut_true": {
                "cut_results": True,
                "n_queries": 15,
                "n_failed": 0,
                "n_no_relevant_in_corpus": 0,
                "n_duplicates_removed": 5,
                "system": {
                    "precision_at_5": {"mean": 0.45, "ci_low": 0.32, "ci_high": 0.58, "n": 15},
                    "precision_at_10": {"mean": 0.35, "ci_low": 0.25, "ci_high": 0.45, "n": 15},
                    "ndcg_at_5": {"mean": 0.55, "ci_low": 0.40, "ci_high": 0.68, "n": 15},
                    "ndcg_at_10": {"mean": 0.62, "ci_low": 0.50, "ci_high": 0.72, "n": 15},
                    "recall_at_5": {"mean": 0.30, "ci_low": 0.20, "ci_high": 0.40, "n": 15},
                    "recall_at_10": {"mean": 0.50, "ci_low": 0.38, "ci_high": 0.62, "n": 15},
                    "mrr": {"mean": 0.85, "ci_low": 0.70, "ci_high": 1.0, "n": 15},
                    "map_at_10": {"mean": 0.42, "ci_low": 0.30, "ci_high": 0.55, "n": 15},
                },
                "baseline": {
                    "precision_at_5": {"mean": 0.08, "ci_low": 0.02, "ci_high": 0.15, "n": 15},
                    "precision_at_10": {"mean": 0.07, "ci_low": 0.02, "ci_high": 0.12, "n": 15},
                    "ndcg_at_5": {"mean": 0.12, "ci_low": 0.04, "ci_high": 0.20, "n": 15},
                    "ndcg_at_10": {"mean": 0.18, "ci_low": 0.08, "ci_high": 0.28, "n": 15},
                    "recall_at_5": {"mean": 0.05, "ci_low": 0.01, "ci_high": 0.10, "n": 15},
                    "recall_at_10": {"mean": 0.10, "ci_low": 0.03, "ci_high": 0.18, "n": 15},
                    "mrr": {"mean": 0.25, "ci_low": 0.10, "ci_high": 0.42, "n": 15},
                    "map_at_10": {"mean": 0.08, "ci_low": 0.02, "ci_high": 0.14, "n": 15},
                },
                "system_vs_baseline": {
                    "precision_at_5": {"mean_diff": 0.37, "ci_low": 0.22, "ci_high": 0.50, "p_value": 0.0001, "n": 15},
                    "precision_at_10": {"mean_diff": 0.28, "ci_low": 0.16, "ci_high": 0.40, "p_value": 0.0001, "n": 15},
                    "ndcg_at_5": {"mean_diff": 0.43, "ci_low": 0.27, "ci_high": 0.58, "p_value": 0.0001, "n": 15},
                    "ndcg_at_10": {"mean_diff": 0.44, "ci_low": 0.30, "ci_high": 0.57, "p_value": 0.0001, "n": 15},
                    "recall_at_5": {"mean_diff": 0.25, "ci_low": 0.12, "ci_high": 0.36, "p_value": 0.0002, "n": 15},
                    "recall_at_10": {"mean_diff": 0.40, "ci_low": 0.25, "ci_high": 0.52, "p_value": 0.0001, "n": 15},
                    "mrr": {"mean_diff": 0.60, "ci_low": 0.38, "ci_high": 0.80, "p_value": 0.0001, "n": 15},
                    "map_at_10": {"mean_diff": 0.34, "ci_low": 0.20, "ci_high": 0.48, "p_value": 0.0001, "n": 15},
                },
                "agreement_diagnostic": {
                    "agreement_p_at_5": {"mean": 0.52, "ci_low": 0.38, "ci_high": 0.65, "n": 15},
                    "agreement_p_at_10": {"mean": 0.41, "ci_low": 0.30, "ci_high": 0.52, "n": 15},
                },
                "per_query": [
                    {
                        "query_id": "cf_01",
                        "query_text": "chocolate cake",
                        "n_results": 10,
                        "n_relevant_in_corpus": 8,
                        "top1_sim": 0.82,
                        "avg_sim": 0.71,
                        "precision_at_5": 0.6,
                        "precision_at_10": 0.5,
                        "ndcg_at_5": 0.72,
                        "ndcg_at_10": 0.78,
                        "recall_at_5": 0.375,
                        "recall_at_10": 0.625,
                        "mrr": 1.0,
                        "map_at_10": 0.55,
                        "results_detail": [{"dim": "taste", "desc": "sweet", "sim": 0.82, "grade": 2}],
                    }
                ],
            },
            "cut_false": {
                "cut_results": False,
                "n_queries": 15,
                "n_failed": 0,
                "n_no_relevant_in_corpus": 0,
                "n_duplicates_removed": 12,
                "system": {
                    "precision_at_5": {"mean": 0.42, "ci_low": 0.29, "ci_high": 0.55, "n": 15},
                    "precision_at_10": {"mean": 0.32, "ci_low": 0.22, "ci_high": 0.42, "n": 15},
                    "ndcg_at_5": {"mean": 0.51, "ci_low": 0.36, "ci_high": 0.65, "n": 15},
                    "ndcg_at_10": {"mean": 0.58, "ci_low": 0.45, "ci_high": 0.69, "n": 15},
                    "recall_at_5": {"mean": 0.28, "ci_low": 0.18, "ci_high": 0.38, "n": 15},
                    "recall_at_10": {"mean": 0.46, "ci_low": 0.34, "ci_high": 0.58, "n": 15},
                    "mrr": {"mean": 0.82, "ci_low": 0.65, "ci_high": 0.98, "n": 15},
                    "map_at_10": {"mean": 0.39, "ci_low": 0.26, "ci_high": 0.51, "n": 15},
                },
                "baseline": {
                    "precision_at_5": {"mean": 0.08, "ci_low": 0.02, "ci_high": 0.15, "n": 15},
                    "precision_at_10": {"mean": 0.07, "ci_low": 0.02, "ci_high": 0.12, "n": 15},
                    "ndcg_at_5": {"mean": 0.12, "ci_low": 0.04, "ci_high": 0.20, "n": 15},
                    "ndcg_at_10": {"mean": 0.18, "ci_low": 0.08, "ci_high": 0.28, "n": 15},
                    "recall_at_5": {"mean": 0.05, "ci_low": 0.01, "ci_high": 0.10, "n": 15},
                    "recall_at_10": {"mean": 0.10, "ci_low": 0.03, "ci_high": 0.18, "n": 15},
                    "mrr": {"mean": 0.25, "ci_low": 0.10, "ci_high": 0.42, "n": 15},
                    "map_at_10": {"mean": 0.08, "ci_low": 0.02, "ci_high": 0.14, "n": 15},
                },
                "system_vs_baseline": {
                    "precision_at_5": {"mean_diff": 0.34, "ci_low": 0.20, "ci_high": 0.48, "p_value": 0.0001, "n": 15},
                    "precision_at_10": {"mean_diff": 0.25, "ci_low": 0.14, "ci_high": 0.37, "p_value": 0.0001, "n": 15},
                    "ndcg_at_5": {"mean_diff": 0.39, "ci_low": 0.22, "ci_high": 0.54, "p_value": 0.0001, "n": 15},
                    "ndcg_at_10": {"mean_diff": 0.40, "ci_low": 0.26, "ci_high": 0.53, "p_value": 0.0001, "n": 15},
                    "recall_at_5": {"mean_diff": 0.23, "ci_low": 0.10, "ci_high": 0.34, "p_value": 0.0003, "n": 15},
                    "recall_at_10": {"mean_diff": 0.36, "ci_low": 0.21, "ci_high": 0.48, "p_value": 0.0001, "n": 15},
                    "mrr": {"mean_diff": 0.57, "ci_low": 0.33, "ci_high": 0.77, "p_value": 0.0001, "n": 15},
                    "map_at_10": {"mean_diff": 0.31, "ci_low": 0.17, "ci_high": 0.45, "p_value": 0.0001, "n": 15},
                },
                "agreement_diagnostic": {
                    "agreement_p_at_5": {"mean": 0.50, "ci_low": 0.36, "ci_high": 0.63, "n": 15},
                    "agreement_p_at_10": {"mean": 0.38, "ci_low": 0.27, "ci_high": 0.49, "n": 15},
                },
                "per_query": [],
            },
        },
        "audio": {
            "k": 20,
            "n_queries": 10,
            "n_failed": 0,
            "n_collapsed": 1,
            "top1_sim": {"mean": 0.62, "ci_low": 0.55, "ci_high": 0.69, "n": 10},
            "avg_sim": {"mean": 0.51, "ci_low": 0.45, "ci_high": 0.57, "n": 10},
            "sim_spread": {"mean": 0.18, "ci_low": 0.12, "ci_high": 0.24, "n": 10},
            "agreement_diagnostic": {
                "agreement_p_at_5": {"mean": 0.35, "ci_low": 0.22, "ci_high": 0.48, "n": 10},
                "agreement_p_at_10": {"mean": 0.28, "ci_low": 0.18, "ci_high": 0.39, "n": 10},
            },
            "per_query": [
                {
                    "query_id": "ac_01",
                    "query_text": "sweet melody happiness warm brown",
                    "n_results": 20,
                    "top1_sim": 0.65,
                    "avg_sim": 0.53,
                    "sim_spread": 0.20,
                    "results_detail": [{"caption": "warm music", "db_sim": 0.65}],
                }
            ],
        },
    }


def test_generate_report_outputs(tmp_path, synthetic_results):
    report_file = generate_report(synthetic_results, str(tmp_path))
    assert Path(report_file).exists()

    # Required files
    json_path = tmp_path / "results.json"
    diag_csv_path = tmp_path / "ir_diagnostics.csv"
    cm_true_csv = tmp_path / "crossmodal_per_query_cut_true.csv"
    cm_false_csv = tmp_path / "crossmodal_per_query_cut_false.csv"
    audio_csv = tmp_path / "audio_per_query.csv"

    assert json_path.exists()
    assert diag_csv_path.exists()
    assert cm_true_csv.exists()
    assert cm_false_csv.exists()
    assert audio_csv.exists()

    # Markdown assertions
    with open(report_file, "r", encoding="utf-8") as f:
        md = f.read()

    # MANDATORY RULES
    assert "isotropy" not in md.lower(), "Markdown must NOT contain 'isotropy' (or 'anisotropy')"
    assert "strong discriminative" not in md.lower(), "Markdown must NOT contain 'Strong discriminative'"
    assert "good (ndcg" not in md.lower(), "Markdown must NOT contain 'Good (nDCG'"
    assert "Limitations" in md, "Markdown MUST contain 'Limitations'"

    # Production primary table
    assert "System (cut_results=True)" in md
    assert "Random Baseline" in md

    # Check JSON validity and NaN handling
    synthetic_results["table_stats"]["nan_test"] = float("nan")
    generate_report(synthetic_results, str(tmp_path))
    with open(json_path, "r", encoding="utf-8") as f:
        data_loaded = json.load(f)
    assert data_loaded["table_stats"]["nan_test"] is None

    # Check CSV contents
    with open(diag_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
        assert len(header) == len(row)
        assert len(header) > 10


def test_generate_report_nan_formatting(tmp_path, synthetic_results):
    # Insert NaNs in results
    synthetic_results["crossmodal"]["cut_true"]["system"]["precision_at_5"] = {
        "mean": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "n": 0,
    }
    report_file = generate_report(synthetic_results, str(tmp_path))
    with open(report_file, "r", encoding="utf-8") as f:
        md = f.read()
    assert "n/a" in md
