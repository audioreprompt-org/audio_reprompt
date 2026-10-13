import math

import numpy as np
import pytest

from models.ir_eval.metrics import (
    average_precision_at_k,
    bootstrap_ci,
    dcg_at_k,
    ndcg_at_k,
    paired_bootstrap_diff,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_divides_by_k():
    assert precision_at_k([2, 0, 2, 1], 4) == pytest.approx(0.5)
    assert precision_at_k([2], 5) == pytest.approx(0.2)  # short list penalized
    assert precision_at_k([2, 2, 0], 1) == 1.0


def test_dcg_exponential_gain():
    assert dcg_at_k([2, 1], 2) == pytest.approx(3 + 1 / math.log2(3))


def test_ndcg_hand_computed_and_corpus_normalization():
    ideal = [1, 2, 2]
    idcg = 3 + 3 / math.log2(3) + 1 / 2
    assert ndcg_at_k([2, 0], 3, ideal) == pytest.approx(3 / idcg)
    # ideal list longer than retrieved list lowers nDCG vs. retrieved-only ideal
    assert ndcg_at_k([2, 0], 3, [2]) == pytest.approx(1.0)
    assert ndcg_at_k([2, 0], 3, ideal) < ndcg_at_k([2, 0], 3, [2])


def test_ndcg_zero_ideal():
    assert ndcg_at_k([1, 2], 3, []) == 0.0
    assert ndcg_at_k([1, 2], 3, [0, 0]) == 0.0


def test_mrr():
    assert reciprocal_rank([0, 1, 2]) == pytest.approx(1 / 3)
    assert reciprocal_rank([2]) == 1.0
    assert reciprocal_rank([0, 1]) == 0.0


def test_ap_and_recall():
    g = [2, 0, 2, 1]
    assert average_precision_at_k(g, 4, 3) == pytest.approx((1 + 2 / 3) / 3)
    assert average_precision_at_k(g, 4, 10) == pytest.approx((1 + 2 / 3) / 4)
    assert recall_at_k(g, 3, 3) == pytest.approx(2 / 3)
    assert recall_at_k(g, 1, 4) == pytest.approx(1 / 4)


def test_grade_one_only():
    g = [1, 1, 1]
    assert precision_at_k(g, 3) == 0.0
    assert reciprocal_rank(g) == 0.0
    assert ndcg_at_k(g, 3, [2, 1, 1]) > 0.0


def test_failed_query_scores_zero():
    assert precision_at_k([], 5) == 0.0
    assert ndcg_at_k([], 5, [2, 1]) == 0.0
    assert reciprocal_rank([]) == 0.0
    assert recall_at_k([], 5, 3) == 0.0
    assert average_precision_at_k([], 5, 3) == 0.0


def test_no_relevant_in_corpus_nan():
    assert math.isnan(recall_at_k([2], 5, 0))
    assert math.isnan(average_precision_at_k([2], 5, 0))
    assert math.isnan(recall_at_k([], 5, 0))


def test_bootstrap_ignores_nan():
    r = bootstrap_ci([1.0, float("nan"), 3.0], n_boot=500)
    assert r["n"] == 2
    assert r["mean"] == pytest.approx(2.0)
    r = bootstrap_ci([float("nan")], n_boot=100)
    assert r["n"] == 0 and math.isnan(r["mean"]) and math.isnan(r["ci_low"])
    assert bootstrap_ci([], n_boot=100)["n"] == 0


def test_bootstrap_constant_and_deterministic():
    r = bootstrap_ci([0.7] * 10, n_boot=500)
    assert r["mean"] == pytest.approx(0.7)
    assert r["ci_low"] == pytest.approx(0.7)
    assert r["ci_high"] == pytest.approx(0.7)
    vals = np.random.default_rng(0).random(30)
    assert bootstrap_ci(vals, n_boot=1000, seed=7) == bootstrap_ci(vals, n_boot=1000, seed=7)
    r = bootstrap_ci(vals, n_boot=1000)
    assert r["ci_low"] <= r["mean"] <= r["ci_high"]


def test_paired_identical():
    x = [0.1, 0.5, 0.9, 0.3]
    r = paired_bootstrap_diff(x, x, n_boot=500)
    assert r["mean_diff"] == 0.0
    assert r["p_value"] == 1.0
    assert r["n"] == 4


def test_paired_shifted():
    rng = np.random.default_rng(1)
    b = rng.random(40)
    a = b + 0.3 + rng.normal(0, 0.02, 40)
    r = paired_bootstrap_diff(a, b, n_boot=2000)
    assert r["ci_low"] > 0
    assert r["p_value"] < 0.05
    assert r == paired_bootstrap_diff(a, b, n_boot=2000)


def test_paired_drops_nan_pairs():
    r = paired_bootstrap_diff([1.0, float("nan"), 2.0], [0.0, 1.0, float("nan")], n_boot=200)
    assert r["n"] == 1
