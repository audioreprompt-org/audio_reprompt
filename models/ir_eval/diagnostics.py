"""Database and intrinsic embedding diagnostics (WS-A).

CONTRACT (do not change signatures or return keys without orchestrator approval;
WS-C's report consumes exactly these keys). All functions are deterministic for a
given ``rng`` / seed. Intrinsic diagnostics run in numpy on ``EmbeddingMatrix``
(exact, no ANN). Only D1 and D4 query the DB.

Addresses review findings: 1.1 (D3 self-match), 1.2 (hubness), 1.3 (isotropy),
1.7 (seeding), §2 D2 sampling, D2 == self-similarity, Gini reference.
"""

from __future__ import annotations

import numpy as np

from models.ir_eval.data import EmbeddingMatrix, gini, sample_indices, skewness

NEAR_DUP_COS = 0.9999


def table_stats(cursor) -> dict:
    """D1: row counts and distinct values.

    Returns:
        {
          "audio_descriptors_rows": int,
          "audio_distinct_captions": int,
          "crossmodal_food_rows": int,
          "food_distinct_dimensions": int,
          "food_distinct_descriptors": int,
          "food_distinct_dim_descriptor_pairs": int,
          "food_distinct_items": int,
        }
    """
    cursor.execute("SELECT count(*) FROM audio_descriptors")
    audio_rows = cursor.fetchone()[0]
    cursor.execute("SELECT count(DISTINCT caption) FROM audio_descriptors")
    audio_captions = cursor.fetchone()[0]
    cursor.execute("SELECT count(*) FROM crossmodal_food_embeddings")
    food_rows = cursor.fetchone()[0]
    cursor.execute(
        "SELECT count(DISTINCT dimension), count(DISTINCT descriptor), "
        "count(DISTINCT (dimension, descriptor)), count(DISTINCT food_item) "
        "FROM crossmodal_food_embeddings"
    )
    row = cursor.fetchone()
    return {
        "audio_descriptors_rows": int(audio_rows),
        "audio_distinct_captions": int(audio_captions),
        "crossmodal_food_rows": int(food_rows),
        "food_distinct_dimensions": int(row[0]),
        "food_distinct_descriptors": int(row[1]),
        "food_distinct_dim_descriptor_pairs": int(row[2]),
        "food_distinct_items": int(row[3]),
    }


def dimension_coverage(cursor) -> dict:
    """D4: per-dimension row counts in crossmodal_food_embeddings.

    Returns:
        {
          "counts": {dimension: int},            # sorted desc
          "proportions": {dimension: float},
          "total": int,
          "gini_coefficient": float,
          "gini_max": float,                      # (n_dims - 1) / n_dims
          "gini_normalized": float,               # gini / gini_max (0..1)
        }
    """
    cursor.execute(
        "SELECT dimension, count(*) AS c FROM crossmodal_food_embeddings "
        "GROUP BY dimension ORDER BY c DESC, dimension"
    )
    rows = sorted(
        ((r[0], int(r[1])) for r in cursor.fetchall()), key=lambda r: (-r[1], str(r[0]))
    )
    counts = {d: c for d, c in rows}
    total = int(sum(counts.values()))
    proportions = {d: (c / total if total else 0.0) for d, c in counts.items()}
    n_dims = len(counts)
    g = gini(list(counts.values())) if n_dims else 0.0
    g_max = (n_dims - 1) / n_dims if n_dims else 0.0
    return {
        "counts": counts,
        "proportions": proportions,
        "total": total,
        "gini_coefficient": float(g),
        "gini_max": float(g_max),
        "gini_normalized": float(g / g_max) if g_max > 0 else 0.0,
    }


def similarity_distribution(
    em: EmbeddingMatrix, rng: np.random.Generator, sample_size: int = 2000
) -> dict:
    """D2 (merged with the old 'self-similarity'): random-pair cosine distribution.

    Sample ``sample_size`` distinct rows, compute S = Xs @ Xs.T, take the strict
    upper triangle (no self-pairs). This is the mean pairwise cosine, i.e. the
    standard anisotropy proxy. Also count near-duplicate pairs (cos >= 0.9999).

    Returns:
        {
          "mean", "std", "min", "max", "median", "p10", "p90": float,
          "n_pairs": int,
          "n_samples": int,
          "n_near_duplicate_pairs": int,
        }
    """
    idx = sample_indices(em.n, sample_size, rng)
    Xs = em.X[idx]
    S = Xs @ Xs.T
    iu = np.triu_indices(len(idx), k=1)
    sims = S[iu]
    if sims.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
            "p10": 0.0, "p90": 0.0, "n_pairs": 0, "n_samples": int(len(idx)),
            "n_near_duplicate_pairs": 0,
        }
    return {
        "mean": float(sims.mean()),
        "std": float(sims.std()),
        "min": float(sims.min()),
        "max": float(sims.max()),
        "median": float(np.median(sims)),
        "p10": float(np.percentile(sims, 10)),
        "p90": float(np.percentile(sims, 90)),
        "n_pairs": int(sims.size),
        "n_samples": int(len(idx)),
        "n_near_duplicate_pairs": int(np.sum(sims >= NEAR_DUP_COS)),
    }


def query_spread(
    em: EmbeddingMatrix,
    rng: np.random.Generator,
    n_queries: int = 200,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """D3: top-1 vs top-k similarity gap, EXCLUDING the self-match.

    Queries are ``n_queries`` sampled rows. For each, compute exact cosine to all
    rows, mask the query's own index (leave-one-out), sort descending. Report
    separately how many queries have an exact/near duplicate (cos >= 0.9999 with
    a different row), because those make top-1 ~= 1.0 for reasons unrelated to
    space quality.

    Returns:
        {
          "n_queries": int,
          "avg_top1_sim": float, "std_top1_sim": float,
          "avg_top1_vs_top{k}_gap": float, "std_top1_vs_top{k}_gap": float,  # for each k
          "n_queries_with_near_duplicate": int,
        }
    """
    q_idx = sample_indices(em.n, n_queries, rng)
    S = em.X[q_idx] @ em.X.T
    S[np.arange(len(q_idx)), q_idx] = -np.inf  # leave-one-out: drop self-match
    S_sorted = -np.sort(-S, axis=1)
    out: dict = {"n_queries": int(len(q_idx))}
    if em.n < 2:
        return out
    top1 = S_sorted[:, 0]
    out["avg_top1_sim"] = float(top1.mean())
    out["std_top1_sim"] = float(top1.std())
    for k in k_values:
        kk = min(int(k), em.n - 1)
        gap = top1 - S_sorted[:, kk - 1]
        out[f"avg_top1_vs_top{k}_gap"] = float(gap.mean())
        out[f"std_top1_vs_top{k}_gap"] = float(gap.std())
    out["n_queries_with_near_duplicate"] = int(np.sum(top1 >= NEAR_DUP_COS))
    return out


def hubness(
    em: EmbeddingMatrix,
    k: int = 10,
    chunk_size: int = 2048,
    query_X: np.ndarray | None = None,
) -> dict:
    """D5: k-occurrence (N_k) distribution over ALL N points.

    Default mode (query_X is None): exact all-vs-all kNN, leave-one-out (mask the
    diagonal), chunked so memory stays bounded. N_k[j] = number of points that
    have j among their k nearest neighbours. Antihubs (N_k = 0) MUST be included.

    Query mode (query_X given, shape (Q, d), L2-normalized): N_k counts how often
    each corpus point appears in the top-k of the given queries (no masking).
    Used with the real evaluation query embeddings.

    Returns:
        {
          "mode": "leave_one_out" | "queries",
          "n_points": int, "n_queries": int, "k": int,
          "skewness": float,            # over all N points, incl. zeros
          "pct_antihubs": float,        # fraction with N_k == 0
          "max_occurrence": int,
          "mean_occurrence": float,     # == k * n_queries / n_points
          "top10_hub_ids": list[int],   # db ids of the 10 largest N_k
          "top10_hub_counts": list[int],
        }
    """
    X = em.X
    n = em.n
    loo = query_X is None
    Q = X if loo else np.asarray(query_X, dtype=np.float64)
    n_q = int(Q.shape[0])
    k_eff = min(int(k), n - 1) if loo else min(int(k), n)
    counts = np.zeros(n, dtype=np.int64)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n_q, chunk_size):
        sims = Q[start : start + chunk_size] @ X.T  # (<=chunk, N)
        if loo:
            rows = np.arange(sims.shape[0])
            sims[rows, start + rows] = -np.inf
        if k_eff >= n:
            top = np.broadcast_to(np.arange(n), sims.shape)
        else:
            top = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
        counts += np.bincount(top.ravel(), minlength=n)
    order = np.lexsort((np.arange(n), -counts))[:10]
    return {
        "mode": "leave_one_out" if loo else "queries",
        "n_points": int(n),
        "n_queries": n_q,
        "k": int(k_eff),
        "skewness": float(skewness(counts)),
        "pct_antihubs": float(np.mean(counts == 0)),
        "max_occurrence": int(counts.max()),
        "mean_occurrence": float(counts.mean()),
        "top10_hub_ids": [int(em.ids[i]) for i in order],
        "top10_hub_counts": [int(counts[i]) for i in order],
    }


def anisotropy(
    em: EmbeddingMatrix, rng: np.random.Generator, sample_size: int | None = None
) -> dict:
    """Interpretable anisotropy / space-utilization diagnostics (replaces 'isotropy').

    Use all rows when ``sample_size`` is None (26k x 384 is cheap). Compute:
      - mean pairwise cosine via the closed form
        (||sum x||^2 - N) / (N (N - 1)) on normalized rows;
      - norm of the mean vector;
      - centered covariance eigenvalues (eigvalsh, descending) and uncentered
        second-moment eigenvalues (X.T @ X / N);
      - participation ratio (sum λ)^2 / sum λ^2 for both spectra;
      - effective dims at 95% / 99% variance (centered);
      - numerical rank (λ > max(λ) * d * eps) and the 5 smallest eigenvalues, so
        rank deficiency / duplicates are visible;
      - lambda_min / lambda_max ratio together with the Marchenko-Pastur reference
        ((1 - sqrt(d/n)) / (1 + sqrt(d/n)))^2 for isotropic Gaussian data of the
        same shape (0 when d >= n).

    Returns:
        {
          "n_samples": int, "total_dimensions": int,
          "mean_pairwise_cosine": float,
          "mean_vector_norm": float,
          "participation_ratio_centered": float,
          "participation_ratio_uncentered": float,
          "effective_dim_95pct": int, "effective_dim_99pct": int,
          "numerical_rank": int,
          "smallest5_eigenvalues": list[float],
          "top5_eigenvalues": list[float],
          "top1_uncentered_eigenvalue_share": float,  # λ1 / sum λ (uncentered)
          "eig_ratio_min_max": float,
          "eig_ratio_mp_reference": float,
        }
    """
    if sample_size is None:
        X = em.X
    else:
        X = em.X[sample_indices(em.n, sample_size, rng)]
    n, d = X.shape
    s = X.sum(axis=0)
    mean_cos = float((s @ s - n) / (n * (n - 1))) if n > 1 else 0.0
    mu = s / n

    def spectrum(M: np.ndarray) -> np.ndarray:
        lam = np.linalg.eigvalsh(M)[::-1]  # descending
        return np.clip(lam, 0.0, None)  # clip round-off negatives only

    lam_u = spectrum(X.T @ X / n)
    Xc = X - mu
    lam_c = spectrum(Xc.T @ Xc / n)

    def pr(lam: np.ndarray) -> float:
        den = float(np.sum(lam**2))
        return float(lam.sum() ** 2 / den) if den > 0 else 0.0

    tot_c = float(lam_c.sum())
    if tot_c > 0:
        cum = np.cumsum(lam_c) / tot_c
        eff95 = int(np.searchsorted(cum, 0.95) + 1)
        eff99 = int(np.searchsorted(cum, 0.99) + 1)
    else:
        eff95 = eff99 = 0
    eff95, eff99 = min(eff95, d), min(eff99, d)
    rank = int(np.sum(lam_c > lam_c.max() * d * np.finfo(np.float64).eps))
    tot_u = float(lam_u.sum())
    ratio = float(lam_c[-1] / lam_c[0]) if lam_c[0] > 0 else 0.0
    if d >= n:
        mp = 0.0
    else:
        r = np.sqrt(d / n)
        mp = float(((1 - r) / (1 + r)) ** 2)
    return {
        "n_samples": int(n),
        "total_dimensions": int(d),
        "mean_pairwise_cosine": mean_cos,
        "mean_vector_norm": float(np.linalg.norm(mu)),
        "participation_ratio_centered": pr(lam_c),
        "participation_ratio_uncentered": pr(lam_u),
        "effective_dim_95pct": eff95,
        "effective_dim_99pct": eff99,
        "numerical_rank": rank,
        "smallest5_eigenvalues": [float(v) for v in lam_c[::-1][:5]],
        "top5_eigenvalues": [float(v) for v in lam_c[:5]],
        "top1_uncentered_eigenvalue_share": float(lam_u[0] / tot_u) if tot_u > 0 else 0.0,
        "eig_ratio_min_max": ratio,
        "eig_ratio_mp_reference": mp,
    }
