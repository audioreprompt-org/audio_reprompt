import math

import numpy as np
import pytest

from models.ir_eval.judges import (
    corpus_grades_structural,
    descriptor_matches,
    grade_structural,
    normalize_term,
    score_agreement_embedding,
    score_relevance_structural,
)

EXPECTED = {"taste": ["sweet", "sour"], "color": ["dark brown", "red"]}


def test_normalize_term():
    assert normalize_term(None) == ""
    assert normalize_term("  Dark_Brown-ish  ") == "dark brown ish"
    assert normalize_term("a   b") == "a b"


@pytest.mark.parametrize(
    "retrieved,expected",
    [("hot", "photo"), ("red", "tired"), ("ice", "spice"), ("", "sweet"), ("  ", "  "), ("sweet", "")],
)
def test_no_match_traps(retrieved, expected):
    assert not descriptor_matches(retrieved, expected)
    assert not descriptor_matches(expected, retrieved)


@pytest.mark.parametrize(
    "retrieved,expected",
    [
        ("sweet", "very sweet"),
        ("very sweet", "sweet"),
        ("dark_brown", "dark brown"),
        ("dark brown", "brown"),
        ("Sweet", "sweet"),
    ],
)
def test_matches(retrieved, expected):
    assert descriptor_matches(retrieved, expected)


def test_non_contiguous_not_match():
    assert not descriptor_matches("very sweet", "very, truly sweet".replace(",", ""))


def test_grade_structural():
    assert grade_structural("taste", "very sweet", EXPECTED) == 2
    assert grade_structural("taste", "bitter", EXPECTED) == 1
    assert grade_structural("smell", "sweet", EXPECTED) == 0
    assert grade_structural("taste", "", EXPECTED) == 0
    assert grade_structural("taste", "  ", EXPECTED) == 0


def test_wrong_dimension_gives_grade_1_not_2():
    # 'red' is expected under color, not taste
    assert grade_structural("taste", "red", EXPECTED) == 1
    assert grade_structural("color", "red", EXPECTED) == 2


def test_score_relevance_structural():
    results = [
        {"dimension": "taste", "descriptor": "sweet"},
        {"dimension": "taste", "descriptor": "red"},
        {"dimension": "smell", "descriptor": "sweet"},
        {"dimension": "color", "descriptor": ""},
    ]
    assert score_relevance_structural(results, EXPECTED) == [2, 1, 0, 0]
    assert score_relevance_structural([], EXPECTED) == []


def test_corpus_grades_dedupes_normalized_pairs():
    pairs = [
        ("taste", "sweet"),
        ("Taste", "SWEET"),
        ("taste", "sweet "),
        ("color", "dark_brown"),
        ("color", "dark brown"),
        ("taste", "bitter"),
    ]
    grades = corpus_grades_structural(pairs, EXPECTED)
    assert sorted(grades) == [1, 2, 2]


def test_score_agreement_embedding_fake_encoder():
    vecs = {
        "q": np.array([1.0, 0.0]),
        "hi": np.array([1.0, 0.0]),  # cos 1.0
        "mid": np.array([0.3, math.sqrt(1 - 0.09)]),  # cos 0.3
        "lo": np.array([0.0, 1.0]),  # cos 0.0
    }

    def encode_fn(texts):
        return np.stack([vecs[t] for t in texts])

    assert score_agreement_embedding("q", ["hi", "mid", "lo"], encode_fn) == [2, 1, 0]
    assert score_agreement_embedding("q", [], encode_fn) == []
