"""IR metrics with corpus-level normalization and bootstrap CIs (WS-B).

CONTRACT: signatures are fixed; WS-C depends on them. Pure numpy, no DB.

Conventions:
  - ``grades``: graded relevance of the RETRIEVED list in rank order.
  - A failed query (no results) is passed as ``[]`` and must score 0 on every
    metric (callers no longer drop it).
  - Binary relevance for P@k / MRR / AP / Recall: grade >= ``min_grade``
    (default 2, i.e. right dimension AND right descriptor). Grade 1 only
    contributes gain in nDCG.
  - ``n_relevant_total`` / ``ideal_grades`` come from the CORPUS
    (``judges.corpus_grades_structural``), not from the retrieved list
    (review §1.6).
  - P@k divides by k, not by len(retrieved), so short lists are penalized.

Addresses review findings 1.5 (grade>0 leniency), 1.6, 1.7.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

RELEVANT_GRADE = 2


def precision_at_k(grades: Sequence[int], k: int, min_grade: int = RELEVANT_GRADE) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for g in list(grades)[:k] if g >= min_grade) / k


def recall_at_k(
    grades: Sequence[int], k: int, n_relevant_total: int, min_grade: int = RELEVANT_GRADE
) -> float:
    """Hits in top-k / n_relevant_total. Returns float('nan') if n_relevant_total == 0
    (query has no relevant item in the corpus; callers report and exclude it)."""
    if n_relevant_total == 0:
        return float("nan")
    return sum(1 for g in list(grades)[:k] if g >= min_grade) / n_relevant_total


def dcg_at_k(grades: Sequence[int], k: int) -> float:
    """sum_{i<k} (2^g_i - 1) / log2(i + 2)  (exponential gain)."""
    return float(sum((2.0 ** g - 1.0) / np.log2(i + 2) for i, g in enumerate(list(grades)[:k])))


def ndcg_at_k(grades: Sequence[int], k: int, ideal_grades: Sequence[int]) -> float:
    """DCG@k of ``grades`` / DCG@k of ``sorted(ideal_grades, desc)``.
    ``ideal_grades`` = corpus grades. Returns 0.0 if ideal DCG == 0."""
    idcg = dcg_at_k(sorted(ideal_grades, reverse=True), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(grades, k) / idcg


def reciprocal_rank(grades: Sequence[int], min_grade: int = RELEVANT_GRADE) -> float:
    for i, g in enumerate(grades):
        if g >= min_grade:
            return 1.0 / (i + 1)
    return 0.0


def average_precision_at_k(
    grades: Sequence[int], k: int, n_relevant_total: int, min_grade: int = RELEVANT_GRADE
) -> float:
    """sum of P@i at relevant ranks i<=k, divided by min(k, n_relevant_total).
    Returns float('nan') if n_relevant_total == 0."""
    if n_relevant_total == 0:
        return float("nan")
    hits = 0
    total = 0.0
    for i, g in enumerate(list(grades)[:k]):
        if g >= min_grade:
            hits += 1
            total += hits / (i + 1)
    return total / min(k, n_relevant_total)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> dict:
    """Percentile bootstrap CI of the mean, ignoring NaNs.

    Returns {"mean": float, "ci_low": float, "ci_high": float, "n": int}
    (n = number of non-NaN values; all NaN/empty -> mean/ci = nan, n = 0).
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = int(v.size)
    if n == 0:
        nan = float("nan")
        return {"mean": nan, "ci_low": nan, "ci_high": nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": n}


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> dict:
    """Paired bootstrap of mean(a - b) over queries (pairs with NaN dropped).

    Returns {"mean_diff", "ci_low", "ci_high": float, "p_value": float, "n": int}
    where p_value is the two-sided bootstrap p-value for H0: mean diff = 0
    (2 * min(P(diff* <= 0), P(diff* >= 0)), clipped to [0, 1]).
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = x - y
    d = d[~np.isnan(d)]
    n = int(d.size)
    if n == 0:
        nan = float("nan")
        return {"mean_diff": nan, "ci_low": nan, "ci_high": nan, "p_value": nan, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0)))
    return {
        "mean_diff": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(max(p, 0.0), 1.0)),
        "n": n,
    }
