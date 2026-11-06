"""Unit tests for models.ir_eval.diagnostics (synthetic data, no DB / model)."""

from __future__ import annotations

import numpy as np

from models.ir_eval import diagnostics as dg
from models.ir_eval.data import EmbeddingMatrix, l2_normalize, make_rng


def make_em(X: np.ndarray, ids=None) -> EmbeddingMatrix:
    Xn, norms = l2_normalize(X)
    n = Xn.shape[0]
    ids = np.arange(100, 100 + n) if ids is None else np.asarray(ids)
    return EmbeddingMatrix(table="t", ids=ids, X=Xn, raw_norms=norms, texts=[()] * n)


def gaussian(n=2000, d=64, seed=0, mean=None):
    X = np.random.default_rng(seed).normal(size=(n, d))
    if mean is not None:
        X = X + mean
    return X


# ---------------------------------------------------------------- anisotropy
def test_isotropic_gaussian():
    em = make_em(gaussian())
    r = dg.anisotropy(em, make_rng(1))
    assert abs(r["mean_pairwise_cosine"]) < 0.01
    ref = r["eig_ratio_mp_reference"]
    assert abs(r["eig_ratio_min_max"] - ref) <= 0.3 * ref
    assert r["participation_ratio_centered"] > 0.8 * 64
    assert r["numerical_rank"] == 64
    assert len(r["smallest5_eigenvalues"]) == 5


def test_anisotropic_common_mean():
    mean = np.full(64, 3.0)
    em = make_em(gaussian(mean=mean))
    r = dg.anisotropy(em, make_rng(1))
    assert r["mean_pairwise_cosine"] > 0.5
    assert r["top1_uncentered_eigenvalue_share"] > 0.5
    assert r["mean_vector_norm"] > 0.7


def test_mean_pairwise_cosine_matches_bruteforce():
    em = make_em(gaussian(n=60, d=8, mean=1.0))
    S = em.X @ em.X.T
    expected = (S.sum() - np.trace(S)) / (60 * 59)
    r = dg.anisotropy(em, make_rng(0))
    assert abs(r["mean_pairwise_cosine"] - expected) < 1e-10


def test_rank_drops_with_planted_rank_deficiency():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(500, 16))
    X = np.hstack([base, base[:, :4]])  # 20 dims, rank 16
    r = dg.anisotropy(make_em(X), make_rng(0))
    assert r["numerical_rank"] == 16
    assert r["smallest5_eigenvalues"][0] == 0.0 or r["smallest5_eigenvalues"][0] < 1e-12


def test_anisotropy_sample_size_and_mp_zero():
    em = make_em(gaussian(n=30, d=64))
    r = dg.anisotropy(em, make_rng(0))
    assert r["eig_ratio_mp_reference"] == 0.0
    r2 = dg.anisotropy(make_em(gaussian()), make_rng(0), sample_size=300)
    assert r2["n_samples"] == 300


# ------------------------------------------------- similarity / query spread
def test_similarity_distribution_duplicates():
    X = gaussian(n=300, d=32)
    X[10] = X[11]
    X[20] = X[21]
    em = make_em(X)
    r = dg.similarity_distribution(em, make_rng(0), sample_size=300)
    assert r["n_samples"] == 300
    assert r["n_pairs"] == 300 * 299 // 2
    assert r["n_near_duplicate_pairs"] == 2
    assert r["max"] <= 1.0 + 1e-9


def test_similarity_distribution_no_self_pairs():
    r = dg.similarity_distribution(make_em(gaussian(n=200, d=32)), make_rng(0), 200)
    assert r["n_near_duplicate_pairs"] == 0
    assert r["max"] < 0.9999


def test_query_spread_with_and_without_duplicates():
    em = make_em(gaussian(n=300, d=32))
    r = dg.query_spread(em, make_rng(0), n_queries=300, k_values=(5, 10))
    assert r["avg_top1_sim"] < 1.0
    assert r["n_queries_with_near_duplicate"] == 0
    assert r["avg_top1_vs_top5_gap"] >= 0 and r["avg_top1_vs_top10_gap"] >= r["avg_top1_vs_top5_gap"]
    assert "std_top1_vs_top10_gap" in r

    X = gaussian(n=300, d=32)
    X[1] = X[0]
    X[3] = X[2]
    r = dg.query_spread(make_em(X), make_rng(0), n_queries=300, k_values=(5,))
    assert r["n_queries_with_near_duplicate"] == 4  # both members of each pair


# ------------------------------------------------------------------ hubness
def naive_nk(X, k, queries=None, loo=True):
    Q = X if queries is None else queries
    nk = np.zeros(len(X), dtype=int)
    for i, q in enumerate(Q):
        sims = [(float(q @ X[j]), j) for j in range(len(X)) if not (loo and j == i)]
        sims.sort(key=lambda t: -t[0])
        for _, j in sims[:k]:
            nk[j] += 1
    return nk


def test_hubness_matches_naive():
    em = make_em(gaussian(n=50, d=8, seed=5))
    r = dg.hubness(em, k=5)
    nk = naive_nk(em.X, 5)
    assert r["mode"] == "leave_one_out"
    assert r["max_occurrence"] == nk.max()
    assert r["pct_antihubs"] == np.mean(nk == 0)
    assert abs(r["mean_occurrence"] - 5.0) < 1e-12
    assert r["top10_hub_counts"] == sorted(nk.tolist(), reverse=True)[:10]
    for hid, c in zip(r["top10_hub_ids"], r["top10_hub_counts"]):
        assert nk[hid - 100] == c
    assert nk.sum() == 5 * 50


def test_hubness_sum_and_chunking():
    em = make_em(gaussian(n=200, d=16, seed=2))
    a = dg.hubness(em, k=7, chunk_size=33)
    b = dg.hubness(em, k=7, chunk_size=10_000)
    assert a == b
    assert abs(a["mean_occurrence"] * a["n_points"] - 7 * 200) < 1e-9


def test_planted_hub_has_max_occurrence():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 16)) + 2.0  # shared offset -> cluster
    X[0] = X[1:].mean(axis=0)  # point at the centroid
    em = make_em(X)
    r = dg.hubness(em, k=5, chunk_size=17)
    assert r["top10_hub_ids"][0] == 100


def test_hubness_antihubs_counted():
    # tight cluster plus far outliers -> outliers are rarely/never neighbours
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(size=(60, 8)) * 0.05 + 5.0, rng.normal(size=(20, 8))])
    r = dg.hubness(make_em(X), k=3)
    assert r["pct_antihubs"] > 0
    assert r["pct_antihubs"] * 80 == round(r["pct_antihubs"] * 80)


def test_hubness_query_mode():
    em = make_em(gaussian(n=50, d=8, seed=5))
    Q = l2_normalize(gaussian(n=30, d=8, seed=9))[0]
    r = dg.hubness(em, k=4, query_X=Q, chunk_size=7)
    nk = naive_nk(em.X, 4, queries=Q, loo=False)
    assert r["mode"] == "queries" and r["n_queries"] == 30
    assert r["max_occurrence"] == nk.max()
    assert abs(r["mean_occurrence"] - 4 * 30 / 50) < 1e-12
    assert r["pct_antihubs"] == np.mean(nk == 0)


# -------------------------------------------------------------- determinism
def test_determinism():
    em = make_em(gaussian(n=400, d=16))
    for fn in (dg.similarity_distribution, dg.query_spread, dg.anisotropy):
        assert fn(em, make_rng(42)) == fn(em, make_rng(42))
    assert dg.hubness(em) == dg.hubness(em)


# ------------------------------------------------------------ DB with fakes
class FakeCursor:
    """Returns canned results in call order; records executed SQL."""

    def __init__(self, results):
        self.results = list(results)
        self.sql: list[str] = []
        self._cur = None

    def execute(self, query, *a):
        self.sql.append(str(query))
        self._cur = self.results.pop(0)

    def fetchone(self):
        return self._cur[0]

    def fetchall(self):
        return self._cur


def test_table_stats():
    cur = FakeCursor([[(10,)], [(8,)], [(20,)], [(4, 15, 18, 12)]])
    r = dg.table_stats(cur)
    assert r == {
        "audio_descriptors_rows": 10,
        "audio_distinct_captions": 8,
        "crossmodal_food_rows": 20,
        "food_distinct_dimensions": 4,
        "food_distinct_descriptors": 15,
        "food_distinct_dim_descriptor_pairs": 18,
        "food_distinct_items": 12,
    }
    assert not any("random()" in s.lower() for s in cur.sql)


def test_dimension_coverage():
    cur = FakeCursor([[("b", 30), ("a", 60), ("c", 10)]])
    r = dg.dimension_coverage(cur)
    assert list(r["counts"]) == ["a", "b", "c"]
    assert r["total"] == 100
    assert abs(r["proportions"]["a"] - 0.6) < 1e-12
    assert abs(r["gini_max"] - 2 / 3) < 1e-12
    assert abs(r["gini_normalized"] - r["gini_coefficient"] / r["gini_max"]) < 1e-12
    assert 0 < r["gini_normalized"] < 1


def test_dimension_coverage_single_dim():
    r = dg.dimension_coverage(FakeCursor([[("a", 5)]]))
    assert r["gini_max"] == 0.0 and r["gini_normalized"] == 0.0
