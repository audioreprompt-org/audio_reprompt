"""Shared data access and utilities for the IR evaluation (WS-0, orchestrator-owned).

Design rules:
  - All randomness goes through ``make_rng(seed)``; never use ``ORDER BY random()``.
  - Embeddings are loaded once per table into memory (26k x 384 floats is ~40 MB)
    and every intrinsic diagnostic is computed exactly in numpy, so results do not
    depend on pgvector ANN indexes.
  - DB connections are always closed via ``db_connection()``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

DEFAULT_SEED = 42

# table -> embedding column
TABLES: dict[str, str] = {
    "audio_descriptors": "embedding",
    "crossmodal_food_embeddings": "text_embedding",
}

# table -> text columns that identify the "content" of a row (used to detect
# exact duplicates and to dedupe results)
TEXT_KEYS: dict[str, tuple[str, ...]] = {
    "audio_descriptors": ("caption",),
    "crossmodal_food_embeddings": ("dimension", "descriptor"),
}

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None


@dataclass
class EmbeddingMatrix:
    """All embeddings of one table, loaded into memory.

    Attributes:
        table: source table name.
        ids: (N,) int array of primary keys, sorted ascending (deterministic order).
        X: (N, d) float64 array, L2-normalized rows (cosine == dot product).
        raw_norms: (N,) L2 norms of the stored vectors before normalization.
        texts: (N,) list of tuples with the TEXT_KEYS values of each row.
    """

    table: str
    ids: np.ndarray
    X: np.ndarray
    raw_norms: np.ndarray
    texts: list[tuple] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])


@contextmanager
def db_connection() -> Iterator:
    """Yield a psycopg connection and always close it."""
    from models.descriptors.connection import get_conn  # lazy: tests need no DB

    conn = get_conn()
    try:
        yield conn
    finally:
        conn.close()


def parse_pgvector(val) -> list[float]:
    """Parse a pgvector value into a list of floats.

    psycopg may return vector columns as strings like '[0.1,0.2,...]'
    or as native list/array types depending on the driver version.
    """
    if isinstance(val, (list, tuple, np.ndarray)):
        return [float(x) for x in val]
    if isinstance(val, str):
        return [float(x) for x in val.strip("[]").split(",")]
    raise TypeError(f"Unexpected pgvector type: {type(val)}")


def to_pgvector_literal(vec) -> str:
    """Format a vector as a pgvector text literal, independent of any adapter."""
    return "[" + ",".join(f"{float(x):.8g}" for x in vec) + "]"


def l2_normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (row-normalized X, original row norms). Zero rows stay zero."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    return X / safe[:, None], norms


def load_embeddings(cursor, table: str) -> EmbeddingMatrix:
    """Load every row of ``table`` (id, text keys, embedding) into memory."""
    from psycopg import sql

    col = TABLES[table]
    text_cols = TEXT_KEYS[table]
    query = sql.SQL("SELECT {id}, {texts}, {col} FROM {tbl} ORDER BY {id}").format(
        id=sql.Identifier("id"),
        texts=sql.SQL(", ").join(sql.Identifier(c) for c in text_cols),
        col=sql.Identifier(col),
        tbl=sql.Identifier(table),
    )
    cursor.execute(query)
    rows = cursor.fetchall()
    if not rows:
        raise RuntimeError(f"Table {table} is empty")

    n_text = len(text_cols)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    texts = [tuple(r[1 : 1 + n_text]) for r in rows]
    raw = np.array([parse_pgvector(r[1 + n_text]) for r in rows], dtype=np.float64)
    X, norms = l2_normalize(raw)
    return EmbeddingMatrix(table=table, ids=ids, X=X, raw_norms=norms, texts=texts)


def make_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def sample_indices(n: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample ``min(n, size)`` distinct indices in [0, n), sorted."""
    size = min(int(size), int(n))
    return np.sort(rng.choice(n, size=size, replace=False))


def get_model():
    """Lazy singleton for the MiniLM encoder (same model as production)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _model


def encode(texts: list[str]) -> np.ndarray:
    """Encode texts with the production encoder; returns L2-normalized rows."""
    embs = np.asarray(get_model().encode(list(texts)), dtype=np.float64)
    return l2_normalize(embs)[0]


def gini(values: list[int] | np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (0 = perfectly balanced).

    Note: for n categories the maximum attainable value is (n - 1) / n.
    """
    vals = np.sort(np.asarray(values, dtype=np.float64))
    n = len(vals)
    if n == 0 or vals.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * vals) / (n * vals.sum())) - (n + 1) / n)


def skewness(arr: np.ndarray) -> float:
    """Adjusted Fisher-Pearson sample skewness (0 if undefined)."""
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    if n < 3:
        return 0.0
    std = np.std(arr, ddof=1)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((arr - arr.mean()) / std) ** 3))
