# eval_rag_embeddings.py
# ===============================================================
# Evalúa variantes de embeddings (CLAP Default vs Music, fusion on/off)
# y normalizaciones (raw, L2, ABTT), con métricas de separación y densidad.
# Imprime si cada variante "mejora" o no contra un baseline.
# ===============================================================

import csv
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import torch
from sklearn.decomposition import PCA
import math

from metrics.clap import get_text_embeddings
from metrics.clap.backends.laion import MUSICGEN_WEIGHTS_URL
from utils.seed import set_reproducibility

set_reproducibility()


def tensors_to_numpy_2d(items) -> np.ndarray:
    """
    items: list[torch.Tensor] | torch.Tensor | list[list[float]]
    Devuelve np.ndarray shape (N, D) en float32, en CPU.
    """
    if isinstance(items, torch.Tensor):
        T = items.detach().to("cpu").float()
        if T.ndim == 1:
            T = T.unsqueeze(0)
        return T.numpy()

    if isinstance(items, (list, tuple)):
        # Normaliza cada elemento a tensor 1D float32 en CPU
        proc = []
        for x in items:
            if isinstance(x, torch.Tensor):
                t = x.detach().to("cpu").float().view(-1)
            else:
                # por si viniera como list[float]
                t = torch.as_tensor(x, dtype=torch.float32).view(-1)
            proc.append(t)
        T = torch.stack(proc, dim=0)  # (N, D)
        return T.numpy()

    raise TypeError(f"No puedo convertir tipo {type(items)} a np.ndarray")

def l2_normalize(X: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(n, eps)

def fit_abtt_from_docs(X_docs: np.ndarray, p: int = 1):
    """
    ABTT: resta la media y elimina las p PCs principales del CORPUS (docs).
    Devuelve transform(X) que aplica mean-centering y proyección a ambos (docs y queries).
    """
    mu = X_docs.mean(0, keepdims=True)
    Xc = X_docs - mu
    if p <= 0:
        U = None
    else:
        pca = PCA(n_components=p).fit(Xc)
        U = pca.components_  # (p, d) ortonormales

    I = np.eye(X_docs.shape[1], dtype=X_docs.dtype)

    def transform(X: np.ndarray) -> np.ndarray:
        Xc2 = X - mu
        if U is None:
            return Xc2
        # Proyección: P = I - U^T U
        P = I - U.T @ U
        return Xc2 @ P

    return transform

def cosine_similarity_matrix(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    Qn = l2_normalize(Q, axis=1)
    Xn = l2_normalize(X, axis=1)
    return Qn @ Xn.T

def dot_similarity_matrix(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    return Q @ X.T

def euclidean_to_similarity(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    # Convertimos distancia L2 a similitud negativa (mayor = mejor)
    # sim = -||q - x||
    Q2 = (Q**2).sum(1, keepdims=True)
    X2 = (X**2).sum(1, keepdims=True).T
    d2 = Q2 - 2*Q@X.T + X2
    d = np.sqrt(np.maximum(d2, 0))
    return -d

def gini_coefficient(x: np.ndarray) -> float:
    """Gini exacto (si todas las ocurrencias son 0 => gini=0)."""
    x = x.astype(np.float64)
    if np.all(x == 0):
        return 0.0
    n = len(x)
    mu = x.mean()
    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    return diff_sum / (2 * n * n * mu)

def hubness_counts(X_docs: np.ndarray, Q_queries: np.ndarray, k: int, sim_fn: Callable) -> np.ndarray:
    S = sim_fn(Q_queries, X_docs)  # (m, n)
    idx = np.argpartition(-S, kth=k-1, axis=1)[:, :k]
    # Para conteo real por ranking exacto (opcional), ordena esos k por similitud:
    occ = np.zeros(X_docs.shape[0], dtype=np.int64)
    for i, cand in enumerate(idx):
        # ordena top-k
        top = cand[np.argsort(-S[i, cand])]
        occ[top] += 1
    return occ

def random_cosine_mean(X: np.ndarray, n_pairs: int = 2000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    if N < 2:
        return 1.0
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    Xn = l2_normalize(X, axis=1)
    vals = np.sum(Xn[i] * Xn[j], axis=1)
    return float(vals.mean())

def pca_evr1(X: np.ndarray) -> float:
    p = PCA(n_components=1).fit(X)
    return float(p.explained_variance_ratio_[0])

def topk_separation(Q: np.ndarray, X: np.ndarray, k: int, sim_fn: Callable) -> Tuple[float, float, float]:
    """
    Devuelve (mean(sim_1), mean(sim_k), mean(sim_1 - sim_k)) promediado sobre queries.
    """
    S = sim_fn(Q, X)  # (m, n)
    # top-k ordenado por similitud
    idx = np.argpartition(-S, kth=k-1, axis=1)[:, :k]
    sim1 = []
    simk = []
    for i in range(S.shape[0]):
        cand = idx[i]
        top = cand[np.argsort(-S[i, cand])]
        s1 = S[i, top[0]]
        sk = S[i, top[-1]]
        sim1.append(s1); simk.append(sk)
    sim1 = float(np.mean(sim1))
    simk = float(np.mean(simk))
    delta = sim1 - simk
    return sim1, simk, delta

# --------------------------
# Configuración de evaluaciones
# --------------------------
SIM_FUNS: Dict[str, Callable] = {
    "cosine": cosine_similarity_matrix,   # usa L2 interno
    "dot":    dot_similarity_matrix,
    # "l2":   euclidean_to_similarity,    # opcional: activa si quieres evaluar L2
}

NORMALIZERS = {
    "raw":      lambda X_docs: (lambda X: X),                # identidad
    "l2":       lambda X_docs: (lambda X: l2_normalize(X)),  # L2 por fila
    "abtt1":    lambda X_docs: fit_abtt_from_docs(X_docs, p=1),
    "abtt3":    lambda X_docs: fit_abtt_from_docs(X_docs, p=3),
}

# --------------------------
# Estructuras de datos
# --------------------------
@dataclass
class EmbeddingPair:
    """Par de embeddings (docs = AudioCaps, queries = Spanio)."""
    name: str
    X_docs: np.ndarray  # shape: (N_docs, d)
    Q_queries: np.ndarray  # shape: (N_queries, d)

@dataclass
class EvalResult:
    variant: str
    normalizer: str
    sim: str
    sim1_mean: float
    simk_mean: float
    delta_k: float
    hubness_gini: float
    cosine_mean_random_docs: float
    pca_evr1_docs: float
    improved_over_baseline: Dict[str, bool]
    win_score: int

# --------------------------
# Criterios de mejora vs baseline
# --------------------------
# - delta_k: más alto = mejor
# - hubness_gini: más bajo = mejor (menos hubs dominantes)
# - cosine_mean_random_docs: más bajo = mejor (espacio más isotrópico)
# - pca_evr1_docs: más bajo = mejor (menos colapso en 1 PC)
IMPROVE_DIR = {
    "delta_k": True,                   # True => mayor es mejor
    "hubness_gini": False,             # False => menor es mejor
    "cosine_mean_random_docs": False,  # False => menor es mejor
    "pca_evr1_docs": False,            # False => menor es mejor
}


VARIANTS: List[EmbeddingPair] = []

# Generate Spanio Captions Embeddings with multiple techniques
texts_spanio_prompts: list[str] = pd.read_csv('./data/prompts/spanio_prompts.csv').loc[:, 'description'].tolist()

embeddings_spanio_enabled_fusion_specialized_weights  = get_text_embeddings(texts_spanio_prompts, backend='laion_module', backend_cfg={"enable_fusion": False, "weights": MUSICGEN_WEIGHTS_URL})
embeddings_spanio_disabled_fusion_specialized_weights  = get_text_embeddings(texts_spanio_prompts, backend='laion_module', backend_cfg={"enable_fusion": True, "weights": MUSICGEN_WEIGHTS_URL})
embeddings_spanio_disabled_fusion_default_weights  = get_text_embeddings(texts_spanio_prompts, backend='laion_module', backend_cfg={"enable_fusion": False})
embeddings_spanio_enabled_fusion_default_weights  = get_text_embeddings(texts_spanio_prompts, backend='laion_module', backend_cfg={"enable_fusion": True})

# Generate Audio Caps Embeddings with multiple techniques
texts_audio_caps: list[str] = pd.read_csv('./data/docs/audio_caps_embeddings.csv').loc[:, 'text'].tolist()

embeddings_audio_caps_enabled_fusion_specialized_weights  = get_text_embeddings(texts_audio_caps, backend='laion_module', backend_cfg={"enable_fusion": False, "weights": MUSICGEN_WEIGHTS_URL})
embeddings_audio_caps_disabled_fusion_specialized_weights  = get_text_embeddings(texts_audio_caps, backend='laion_module', backend_cfg={"enable_fusion": True, "weights": MUSICGEN_WEIGHTS_URL})
embeddings_audio_caps_disabled_fusion_default_weights  = get_text_embeddings(texts_audio_caps, backend='laion_module', backend_cfg={"enable_fusion": False})
embeddings_audio_caps_enabled_fusion_default_weights  = get_text_embeddings(texts_audio_caps, backend='laion_module', backend_cfg={"enable_fusion": True})


try:
    # ----- DEFAULT (pesos por defecto) -----
    X_docs_default_on  = tensors_to_numpy_2d(embeddings_audio_caps_enabled_fusion_default_weights)
    Q_spanio_default_on = tensors_to_numpy_2d(embeddings_spanio_enabled_fusion_default_weights)
    VARIANTS.append(EmbeddingPair("default_fusion_on", X_docs_default_on, Q_spanio_default_on))

    X_docs_default_off = tensors_to_numpy_2d(embeddings_audio_caps_disabled_fusion_default_weights)
    Q_spanio_default_off = tensors_to_numpy_2d(embeddings_spanio_disabled_fusion_default_weights)
    VARIANTS.append(EmbeddingPair("default_fusion_off", X_docs_default_off, Q_spanio_default_off))

    # ----- MUSIC (pesos especializados) -----
    X_docs_music_on  = tensors_to_numpy_2d(embeddings_audio_caps_enabled_fusion_specialized_weights)
    Q_spanio_music_on = tensors_to_numpy_2d(embeddings_spanio_enabled_fusion_specialized_weights)
    VARIANTS.append(EmbeddingPair("music_fusion_on", X_docs_music_on, Q_spanio_music_on))

    X_docs_music_off = tensors_to_numpy_2d(embeddings_audio_caps_disabled_fusion_specialized_weights)
    Q_spanio_music_off = tensors_to_numpy_2d(embeddings_spanio_disabled_fusion_specialized_weights)
    VARIANTS.append(EmbeddingPair("music_fusion_off", X_docs_music_off, Q_spanio_music_off))
except NameError:
    raise RuntimeError(
        "Debes definir previamente los arrays de embeddings:\n"
        "  embeddings_spanio_... y embeddings_audio_caps_...\n"
        "según tu script de generación. Luego vuelve a ejecutar."
    )

# --------------------------
# EVALUACIÓN
# --------------------------
def evaluate_variant(pair: EmbeddingPair, normalizer_key: str, sim_key: str, k: int = 10) -> Dict:
    X = pair.X_docs
    Q = pair.Q_queries

    # Normalizador entrenado SOLO con docs (si aplica) y usado en ambos
    norm_fn_builder = NORMALIZERS[normalizer_key]
    transform = norm_fn_builder(X)
    X_t = transform(X)
    Q_t = transform(Q)

    # Selección de similitud
    sim_fn = SIM_FUNS[sim_key]

    # Métricas
    sim1, simk, delta = topk_separation(Q_t, X_t, k=k, sim_fn=sim_fn)
    occ = hubness_counts(X_t, Q_t, k=k, sim_fn=sim_fn)
    gini = gini_coefficient(occ)
    cmr = random_cosine_mean(X_t)
    evr1 = pca_evr1(X_t)

    return dict(
        variant=pair.name, normalizer=normalizer_key, sim=sim_key,
        sim1_mean=sim1, simk_mean=simk, delta_k=delta,
        hubness_gini=gini, cosine_mean_random_docs=cmr, pca_evr1_docs=evr1
    )

def compare_to_baseline(df: pd.DataFrame, baseline_row: pd.Series) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        improved = {}
        win = 0
        for metric, higher_is_better in IMPROVE_DIR.items():
            base_val = float(baseline_row[metric])
            cur_val  = float(r[metric])
            if higher_is_better:
                ok = cur_val > base_val
            else:
                ok = cur_val < base_val
            improved[metric] = ok
            win += int(ok)
        rows.append({**r.to_dict(), "improved_over_baseline": improved, "win_score": win})
    return pd.DataFrame(rows)

def main(k: int = 10):
    # Grid de pruebas
    norms = ["raw", "l2", "abtt1", "abtt3"]
    sims  = ["cosine", "dot"]   # agrega "l2" si quieres la métrica euclídea

    results = []
    for pair in VARIANTS:
        for nz in norms:
            for sm in sims:
                out = evaluate_variant(pair, nz, sm, k=k)
                results.append(out)
    df = pd.DataFrame(results)

    # El baseline = primera fila (puedes filtrar explícitamente por un (variant, norm, sim) concreto)
    baseline = df.iloc[0]
    dfc = compare_to_baseline(df, baseline)

    # Presentación: columnas clave y flags compactos
    show_cols = ["variant","normalizer","sim","delta_k","hubness_gini","cosine_mean_random_docs","pca_evr1_docs","win_score"]
    pretty = dfc[show_cols].copy()

    # Flags de mejora por métrica (✓/✗)
    def flag(ok: bool) -> str: return "✓" if ok else "✗"
    flags = []
    for _, row in dfc.iterrows():
        f = {m: flag(row["improved_over_baseline"][m]) for m in IMPROVE_DIR.keys()}
        flags.append(f)
    flags_df = pd.DataFrame(flags)
    pretty = pd.concat([pretty, flags_df.add_prefix("improved_")], axis=1)

    # Ordenar por win_score desc y luego delta_k desc
    pretty = pretty.sort_values(["win_score","delta_k"], ascending=[False,False]).reset_index(drop=True)

    # Resumen
    print("\n=== BASELINE ===")
    print(baseline[["variant","normalizer","sim","delta_k","hubness_gini","cosine_mean_random_docs","pca_evr1_docs"]])
    print("\n=== RESULTADOS (mejor arriba) ===")
    print(pretty.to_string(index=False))

    # CSV opcional
    # pretty.to_csv("rag_eval_summary.csv", index=False)

if __name__ == "__main__":
    main(k=10)
