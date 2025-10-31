# eval_rag_embeddings.py
# ===============================================================
# Evalúa variantes de embeddings (CLAP Default vs Music, fusion on/off)
# y normalizaciones (raw, L2, ABTT), con métricas de separación y densidad.
# Guarda/lee embeddings en CSV con columnas: [text, embedding] (embedding en JSON).
# Imprime si cada variante "mejora" o no contra un baseline.
# ===============================================================

import os
import json
import math
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from metrics.clap import get_text_embeddings
from metrics.clap.backends.laion import MUSICGEN_WEIGHTS_URL
from utils.seed import set_reproducibility

# -------------------------------------------------
# Config
# -------------------------------------------------
set_reproducibility()
ARTIFACT_DIR = Path(os.environ.get("EMB_CSV_DIR", "./data/embeddings/")).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Helpers de conversión / hash / CSV cache
# -------------------------------------------------
def tensors_to_numpy_2d(items) -> np.ndarray:
    """
    items: list[torch.Tensor] | torch.Tensor | list[list[float]]
    -> np.ndarray (N, D) float32 en CPU
    """
    if isinstance(items, torch.Tensor):
        T = items.detach().to("cpu").float()
        if T.ndim == 1:
            T = T.unsqueeze(0)
        return T.numpy()

    if isinstance(items, (list, tuple)):
        proc = []
        for x in items:
            if isinstance(x, torch.Tensor):
                t = x.detach().to("cpu").float().view(-1)
            else:
                t = torch.as_tensor(x, dtype=torch.float32).view(-1)
            proc.append(t)
        T = torch.stack(proc, dim=0)  # (N, D)
        return T.numpy()

    raise TypeError(f"No puedo convertir tipo {type(items)} a np.ndarray")

def sha1_of_strings(str_list: List[str]) -> str:
    h = hashlib.sha1()
    for s in str_list:
        h.update((s if isinstance(s, str) else str(s)).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()

def csv_cache_path(dataset_tag: str, variant_tag: str, text_hash: str) -> Path:
    fname = f"{dataset_tag}__{variant_tag}__{text_hash[:12]}.csv"
    return ARTIFACT_DIR / fname

def save_embeddings_csv(path: Path, texts: List[str], X: np.ndarray, meta: dict) -> None:
    # Guardamos CSV con columnas: text, embedding (JSON)
    # Meta se guarda en un .meta.json paralelo para trazabilidad
    df = pd.DataFrame({
        "text": texts,
        "embedding": [json.dumps(vec.astype(float).tolist(), separators=(',', ':')) for vec in X]
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)
    # Meta en archivo aparte
    meta_path = path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_embeddings_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    assert "text" in df.columns and "embedding" in df.columns, f"CSV inválido: {path}"
    texts = df["text"].astype(str).tolist()
    # Parseo robusto de JSON a np.array float32
    X = np.vstack([np.array(json.loads(s), dtype=np.float32) for s in df["embedding"].astype(str).tolist()])
    return texts, X

def get_or_build_embeddings_csv(
    texts: List[str],
    dataset_tag: str,
    variant_tag: str,
    backend: str,
    backend_cfg: dict,
) -> Tuple[List[str], np.ndarray, Path]:
    """
    Si existe CSV con [text, embedding] lo carga, si no, calcula embeddings y los guarda.
    Devuelve (texts, X, path_csv).
    """
    text_hash = sha1_of_strings(texts)
    path = csv_cache_path(dataset_tag, variant_tag, text_hash)

    if path.exists():
        print(f"[CSV CACHE HIT] {dataset_tag}/{variant_tag} -> {path.name}")
        texts_loaded, X_loaded = load_embeddings_csv(path)
        # Verificación básica (mismo número de textos)
        if len(texts_loaded) == len(texts):
            return texts_loaded, X_loaded.astype(np.float32), path
        else:
            print(f"[CSV CACHE SIZE MISMATCH] Recalculando {dataset_tag}/{variant_tag}")

    print(f"[CSV CACHE MISS] Generando embeddings {dataset_tag}/{variant_tag} ...")
    with torch.inference_mode():
        items = get_text_embeddings(texts, backend=backend, backend_cfg=backend_cfg)
    X = tensors_to_numpy_2d(items).astype(np.float32)

    meta = {
        "dataset": dataset_tag,
        "variant": variant_tag,
        "backend": backend,
        "backend_cfg": backend_cfg,
        "text_hash": text_hash,
        "num_texts": len(texts),
        "dim": int(X.shape[1]) if X.ndim == 2 else None,
    }
    save_embeddings_csv(path, texts, X, meta)
    print(f"[CSV SAVED] {path.name} ({X.shape})")
    return texts, X, path

# -------------------------------------------------
# Métricas / normalizaciones / similitudes
# -------------------------------------------------
def l2_normalize(X: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=axis, keepdims=True)
    return X / np.maximum(n, eps)

def fit_abtt_from_docs(X_docs: np.ndarray, p: int = 1):
    """
    ABTT: mean-centering + eliminación de p PCs del CORPUS (docs).
    Devuelve función transform(X) consistente para docs y queries.
    """
    mu = X_docs.mean(0, keepdims=True)
    Xc = X_docs - mu
    if p <= 0:
        U = None
    else:
        pca = PCA(n_components=p).fit(Xc)
        U = pca.components_  # (p, d)

    I = np.eye(X_docs.shape[1], dtype=X_docs.dtype)

    def transform(X: np.ndarray) -> np.ndarray:
        Xc2 = X - mu
        if U is None:
            return Xc2
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
    # sim = -||q - x||
    Q2 = (Q**2).sum(1, keepdims=True)
    X2 = (X**2).sum(1, keepdims=True).T
    d2 = Q2 - 2*Q@X.T + X2
    d = np.sqrt(np.maximum(d2, 0))
    return -d

def gini_coefficient(x: np.ndarray) -> float:
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
    occ = np.zeros(X_docs.shape[0], dtype=np.int64)
    for i, cand in enumerate(idx):
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
    -> (mean(sim_1), mean(sim_k), mean(sim_1 - sim_k)) promediado sobre queries.
    """
    S = sim_fn(Q, X)  # (m, n)
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

# -------------------------------------------------
# Config evaluaciones
# -------------------------------------------------
SIM_FUNS: Dict[str, Callable] = {
    "cosine": cosine_similarity_matrix,
    "dot":    dot_similarity_matrix,
    # "l2":   euclidean_to_similarity,  # activa si quieres evaluar L2-dist como similitud negativa
}

def norm_raw(_X_docs): return (lambda X: X)
def norm_l2(_X_docs):  return (lambda X: l2_normalize(X))
NORMALIZERS = {
    "raw":   norm_raw,
    "l2":    norm_l2,
    "abtt1": lambda X_docs: fit_abtt_from_docs(X_docs, p=1),
    "abtt3": lambda X_docs: fit_abtt_from_docs(X_docs, p=3),
}

# -------------------------------------------------
# Estructuras
# -------------------------------------------------
@dataclass
class EmbeddingPair:
    """Par de embeddings (docs = AudioCaps, queries = Spanio)."""
    name: str
    X_docs: np.ndarray  # (N_docs, d)
    Q_queries: np.ndarray  # (N_queries, d)

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

# -------------------------------------------------
# Criterios de mejora vs baseline
# -------------------------------------------------
IMPROVE_DIR = {
    "delta_k": True,                   # mayor es mejor
    "hubness_gini": False,             # menor es mejor
    "cosine_mean_random_docs": False,  # menor es mejor
    "pca_evr1_docs": False,            # menor es mejor
}

# -------------------------------------------------
# Carga de textos (datasets)
# -------------------------------------------------
def load_texts_spanio(path: str = "./data/prompts/spanio_prompts.csv", column: str = "description") -> List[str]:
    df = pd.read_csv(path)
    return df.loc[:, column].astype(str).tolist()

def load_texts_audiocaps(path: str = "./data/docs/audio_caps_embeddings.csv", column: str = "text") -> List[str]:
    df = pd.read_csv(path)
    return df.loc[:, column].astype(str).tolist()

# -------------------------------------------------
# Construcción de variantes (cache CSV)
# -------------------------------------------------
def build_variants() -> List[EmbeddingPair]:
    texts_spanio = load_texts_spanio()
    texts_audio  = load_texts_audiocaps()

    variants: List[EmbeddingPair] = []

    # Helpers de config
    def cfg_default(fusion_on: bool) -> dict:
        return {"enable_fusion": fusion_on}
    def cfg_music(fusion_on: bool) -> dict:
        return {"enable_fusion": fusion_on, "weights": MUSICGEN_WEIGHTS_URL}

    combos = [
        ("default_fusion_on",  cfg_default(True)),
        ("default_fusion_off", cfg_default(False)),
        ("music_fusion_on",    cfg_music(True)),
        ("music_fusion_off",   cfg_music(False)),
    ]

    for variant_tag, backend_cfg in combos:
        # Docs = AudioCaps
        _, X_docs, csv_docs_path = get_or_build_embeddings_csv(
            texts=texts_audio,
            dataset_tag="audiocaps",
            variant_tag=variant_tag,
            backend="laion_module",
            backend_cfg=backend_cfg,
        )
        # Queries = Spanio
        _, Q_spanio, csv_q_path = get_or_build_embeddings_csv(
            texts=texts_spanio,
            dataset_tag="spanio",
            variant_tag=variant_tag,
            backend="laion_module",
            backend_cfg=backend_cfg,
        )
        assert X_docs.ndim == 2 and Q_spanio.ndim == 2, "Embeddings deben ser 2D"
        assert X_docs.shape[1] == Q_spanio.shape[1], f"Dimensiones difieren en {variant_tag}: {X_docs.shape[1]} vs {Q_spanio.shape[1]}"
        print(f"[VARIANT READY] {variant_tag}\n  - docs CSV: {csv_docs_path.name}\n  - queries CSV: {csv_q_path.name}")
        variants.append(EmbeddingPair(variant_tag, X_docs, Q_spanio))

    return variants

# -------------------------------------------------
# Evaluación
# -------------------------------------------------
def evaluate_variant(pair: EmbeddingPair, normalizer_key: str, sim_key: str, k: int = 10) -> Dict:
    X = pair.X_docs
    Q = pair.Q_queries

    # Normalizador entrenado SOLO con docs (si aplica) y usado en ambos
    transform = NORMALIZERS[normalizer_key](X)
    X_t = transform(X)
    Q_t = transform(Q)

    sim_fn = SIM_FUNS[sim_key]

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
            ok = (cur_val > base_val) if higher_is_better else (cur_val < base_val)
            improved[metric] = ok
            win += int(ok)
        rows.append({**r.to_dict(), "improved_over_baseline": improved, "win_score": win})
    return pd.DataFrame(rows)

def main(k: int = 10):
    # Construir/leer variantes desde CSV cache
    VARIANTS = build_variants()

    # Grid de pruebas
    norms = ["raw", "l2", "abtt1", "abtt3"]
    sims  = ["cosine", "dot"]   # agregar "l2" si se quiere L2-dist como similitud negativa

    results = []
    for pair in VARIANTS:
        for nz in norms:
            for sm in sims:
                out = evaluate_variant(pair, nz, sm, k=k)
                results.append(out)
    df = pd.DataFrame(results)

    # Baseline = primera fila
    baseline = df.iloc[0]
    dfc = compare_to_baseline(df, baseline)

    # Presentación
    show_cols = ["variant","normalizer","sim","delta_k","hubness_gini","cosine_mean_random_docs","pca_evr1_docs","win_score"]
    pretty = dfc[show_cols].copy()

    def flag(ok: bool) -> str: return "✓" if ok else "✗"
    flags = []
    for _, row in dfc.iterrows():
        f = {m: flag(row["improved_over_baseline"][m]) for m in IMPROVE_DIR.keys()}
        flags.append(f)
    flags_df = pd.DataFrame(flags)
    pretty = pd.concat([pretty, flags_df.add_prefix("improved_")], axis=1)

    pretty = pretty.sort_values(["win_score","delta_k"], ascending=[False,False]).reset_index(drop=True)

    # Resumen
    print("\n=== CSV CACHE DIR ===")
    print(str(ARTIFACT_DIR))
    print("\n=== BASELINE ===")
    print(baseline[["variant","normalizer","sim","delta_k","hubness_gini","cosine_mean_random_docs","pca_evr1_docs"]])
    print("\n=== RESULTADOS (mejor arriba) ===")
    print(pretty.to_string(index=False))

    # CSV opcional con resultados de evaluación:
    pretty.to_csv("./artifacts/rag_eval_summary.csv", index=False)

if __name__ == "__main__":
    main(k=10)
