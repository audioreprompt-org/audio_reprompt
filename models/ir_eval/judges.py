"""Relevance judges (WS-B).

CONTRACT: signatures and grade semantics are fixed; WS-C depends on them.
Pure Python, no DB, no encoder at import time.

Grade scale for crossmodal results (structural judge):
    2 = dimension is expected AND descriptor matches an expected descriptor
        OF THAT SAME DIMENSION
    1 = dimension is expected, descriptor does not match
    0 = dimension not expected, or empty/blank descriptor

Addresses review finding 1.5 (substring leaks, empty descriptor, flattened
dimensions) and 1.4 (embedding judge is circular -> diagnostic only).
"""

from __future__ import annotations

from typing import Iterable, Sequence


def normalize_term(s: str | None) -> str:
    """Lowercase, replace '_' and '-' with spaces, collapse whitespace, strip.
    None -> ''."""
    if s is None:
        return ""
    return " ".join(str(s).lower().replace("_", " ").replace("-", " ").split())


def descriptor_matches(retrieved: str, expected: str) -> bool:
    """True iff the normalized terms are equal, OR one is a whole-token
    subsequence of the other (e.g. 'sweet' vs 'very sweet', 'dark brown' vs
    'brown'). Never substring-within-word: 'hot' !~ 'photo', 'red' !~ 'tired',
    'ice' !~ 'spice'. Empty strings never match anything."""
    a = normalize_term(retrieved).split(" ")
    b = normalize_term(expected).split(" ")
    if a == [""] or b == [""]:
        return False
    if a == b:
        return True
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    n = len(short)
    return any(long_[i : i + n] == short for i in range(len(long_) - n + 1))


def grade_structural(
    dimension: str, descriptor: str, expected_descriptors: dict[str, list[str]]
) -> int:
    """Grade one (dimension, descriptor) pair on the 0/1/2 scale above.
    Descriptor matching is checked only against expected_descriptors[dimension]."""
    if not normalize_term(descriptor):
        return 0
    dim = normalize_term(dimension)
    for key, descs in expected_descriptors.items():
        if normalize_term(key) != dim:
            continue
        if any(descriptor_matches(descriptor, e) for e in descs):
            return 2
        return 1
    return 0


def score_relevance_structural(
    results: Sequence[dict], expected_descriptors: dict[str, list[str]]
) -> list[int]:
    """Grade each retrieved result (dicts with 'dimension' and 'descriptor')."""
    return [
        grade_structural(r.get("dimension") or "", r.get("descriptor") or "", expected_descriptors)
        for r in results
    ]


def corpus_grades_structural(
    corpus_pairs: Iterable[tuple[str, str]], expected_descriptors: dict[str, list[str]]
) -> list[int]:
    """Grade every DISTINCT (dimension, descriptor) pair of the corpus.

    Used to build the corpus-level ideal ranking (IDCG) and the total number of
    relevant items for Recall@k / AP. Input may contain duplicates; dedupe on the
    normalized pair before grading. Returns grades in arbitrary order.
    """
    seen = set()
    grades = []
    for dim, desc in corpus_pairs:
        key = (normalize_term(dim), normalize_term(desc))
        if key in seen:
            continue
        seen.add(key)
        grades.append(grade_structural(dim, desc, expected_descriptors))
    return grades


def score_agreement_embedding(
    query_text: str,
    retrieved_texts: Sequence[str],
    encode_fn,
    threshold_high: float = 0.45,
    threshold_mid: float = 0.25,
) -> list[int]:
    """RETRIEVER-AGREEMENT DIAGNOSTIC, NOT A RELEVANCE JUDGE.

    Uses the same encoder as the retriever, so it is circular (review §1.4):
    its scores must never feed a quality verdict. ``encode_fn(list[str]) ->
    np.ndarray`` returns L2-normalized rows (use ``data.encode``); injected so
    tests can pass a fake. Grades: 2 if cos >= high, 1 if cos >= mid, else 0.
    """
    import numpy as np

    if len(retrieved_texts) == 0:
        return []
    q = np.asarray(encode_fn([query_text]))[0]
    r = np.asarray(encode_fn(list(retrieved_texts)))
    sims = r @ q
    return [2 if s >= threshold_high else 1 if s >= threshold_mid else 0 for s in sims]
