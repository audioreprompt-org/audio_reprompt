"""Unit tests for models.ir_eval.retrieval (mocked DB and retrieval functions)."""

from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import pytest

from models.ir_eval import data, retrieval


class FakeCursor:
    def __init__(self, fetchall_results=None, fetchone_results=None):
        self.fetchall_results = list(fetchall_results or [])
        self.fetchone_results = list(fetchone_results or [])
        self.executed_sql: list[str] = []

    def execute(self, query, params=None):
        self.executed_sql.append(str(query))

    def fetchall(self):
        if self.fetchall_results:
            return self.fetchall_results.pop(0)
        return []

    def fetchone(self):
        if self.fetchone_results:
            return self.fetchone_results.pop(0)
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeConn:
    def __init__(self, cursor_instance):
        self._cursor = cursor_instance
        self.transactions: list[str] = []

    def cursor(self):
        return self._cursor

    @contextmanager
    def transaction(self):
        self.transactions.append("begin")
        try:
            yield
        finally:
            self.transactions.append("end")


def test_fetch_corpus_pairs():
    cur = FakeCursor(fetchall_results=[[("taste", "sweet"), ("texture", "creamy")]])
    pairs = retrieval.fetch_corpus_pairs(cur)
    assert pairs == [("taste", "sweet"), ("texture", "creamy")]
    assert any("DISTINCT" in sql for sql in cur.executed_sql)


def test_dedupe_crossmodal_results():
    results = [
        {"dimension": "taste", "descriptor": "sweet", "sim": 0.9},
        {"dimension": "Taste", "descriptor": "SWEET", "sim": 0.85},
        {"dimension": "color", "descriptor": "dark_brown", "sim": 0.8},
        {"dimension": "color", "descriptor": "dark brown", "sim": 0.75},
        {"dimension": "taste", "descriptor": "sour", "sim": 0.7},
    ]
    deduped, n_removed = retrieval.dedupe_crossmodal_results(results)
    assert len(deduped) == 3
    assert n_removed == 2
    assert deduped[0]["sim"] == 0.9
    assert deduped[1]["descriptor"] == "dark_brown"
    assert deduped[2]["descriptor"] == "sour"


def test_random_baseline_grades():
    rng = np.random.default_rng(42)
    corpus_pairs = [
        ("taste", "sweet"),
        ("taste", "sour"),
        ("color", "red"),
        ("texture", "creamy"),
    ]
    expected = {"taste": ["sweet"]}
    grades = retrieval.random_baseline_grades(corpus_pairs, expected, length=2, rng=rng)
    assert len(grades) == 2
    assert all(g in (0, 1, 2) for g in grades)

    # Empty / zero length
    assert retrieval.random_baseline_grades(corpus_pairs, expected, length=0, rng=rng) == []
    assert retrieval.random_baseline_grades([], expected, length=5, rng=rng) == []


def test_evaluate_crossmodal(monkeypatch):
    # Mock data.encode to return dummy unit vectors
    def fake_encode(texts):
        n = len(texts)
        arr = np.zeros((n, 8), dtype=float)
        arr[:, 0] = 1.0
        return arr

    monkeypatch.setattr(data, "encode", fake_encode)

    # Mock get_top_k_food_descriptors to return synthetic results
    call_log = []

    def fake_get_top_k(emb, cut_results=False):
        call_log.append(cut_results)
        if len(call_log) == 1:
            # Query 1: return 3 results, with one duplicate
            return [
                {"dimension": "taste", "descriptor": "sweet", "sim": 0.8},
                {"dimension": "taste", "descriptor": "SWEET", "sim": 0.75},
                {"dimension": "texture", "descriptor": "crunchy", "sim": 0.6},
            ]
        elif len(call_log) == 2:
            # Query 2: failed query (empty results)
            return []
        else:
            # Query 3: query with no relevant item in corpus
            return [{"dimension": "emotion", "descriptor": "joy", "sim": 0.5}]

    monkeypatch.setattr(retrieval, "get_top_k_food_descriptors", fake_get_top_k)

    queries = [
        {
            "id": "q1",
            "query_text": "chocolate cake",
            "expected_descriptors": {"taste": ["sweet"]},
        },
        {
            "id": "q2",
            "query_text": "failed food",
            "expected_descriptors": {"taste": ["sour"]},
        },
        {
            "id": "q3",
            "query_text": "obscure item",
            "expected_descriptors": {"unknown_dim": ["none"]},
        },
    ]

    corpus_pairs = [
        ("taste", "sweet"),
        ("taste", "sour"),
        ("texture", "crunchy"),
    ]

    rng = np.random.default_rng(123)
    res = retrieval.evaluate_crossmodal(
        queries=queries,
        corpus_pairs=corpus_pairs,
        cut_results=True,
        rng=rng,
        seed=123,
    )

    assert res["cut_results"] is True
    assert res["n_queries"] == 3
    assert res["n_failed"] == 1
    assert res["n_no_relevant_in_corpus"] == 1
    assert res["n_duplicates_removed"] == 1
    assert call_log[0] is True

    # Check metrics structure
    assert "precision_at_5" in res["system"]
    assert "ndcg_at_10" in res["system"]
    assert "map_at_10" in res["system"]
    assert "system_vs_baseline" in res
    assert "agreement_diagnostic" in res
    assert "agreement_p_at_5" in res["agreement_diagnostic"]
    assert len(res["per_query"]) == 3

    # Failed query (q2) scored 0 across metrics
    q2_rec = res["per_query"][1]
    assert q2_rec["n_results"] == 0
    assert q2_rec["precision_at_5"] == 0.0
    assert q2_rec["ndcg_at_10"] == 0.0
    assert q2_rec["top1_sim"] == 0.0

    # Top1 sim is max(sim)
    q1_rec = res["per_query"][0]
    assert q1_rec["top1_sim"] == 0.8


def test_evaluate_audio(monkeypatch):
    def fake_encode(texts):
        n = len(texts)
        arr = np.zeros((n, 8), dtype=float)
        arr[:, 0] = 1.0
        return arr

    monkeypatch.setattr(data, "encode", fake_encode)

    call_count = 0

    def fake_get_audio(emb, k=20, using_clap=False):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Query 1: returns full k results
            return {f"caption_{i}": 0.8 - i * 0.02 for i in range(k)}
        elif call_count == 2:
            # Query 2: collapsed results (len < k)
            return {"cap_a": 0.7, "cap_b": 0.6}
        else:
            # Query 3: failed query (empty)
            return {}

    monkeypatch.setattr(retrieval, "get_top_k_audio_captions", fake_get_audio)

    queries = [
        {"id": "a1", "query_text": "full audio"},
        {"id": "a2", "query_text": "collapsed audio"},
        {"id": "a3", "query_text": "empty audio"},
    ]

    res = retrieval.evaluate_audio(queries, k=20)
    assert res["k"] == 20
    assert res["n_queries"] == 3
    assert res["n_failed"] == 1
    assert res["n_collapsed"] == 1
    assert "top1_sim" in res
    assert "avg_sim" in res
    assert "sim_spread" in res
    assert "agreement_diagnostic" in res
    assert len(res["per_query"]) == 3

    # Confirm NO relevance quality verdict key exists
    assert "quality_verdict" not in res
    assert "verdict" not in res


def test_ann_vs_exact():
    cur = FakeCursor(
        fetchall_results=[
            # 1. pg_indexes query
            [("idx_audio_hnsw", "CREATE INDEX idx_audio_hnsw ON audio_descriptors USING hnsw (embedding vector_cosine_ops)")],
            # 2. default plan query 1
            [(1,), (2,), (3,)],
            # 3. exact plan query 1
            [(1,), (2,), (4,)],
        ]
    )
    conn = FakeConn(cur)

    query_embs = np.array([[1.0, 0.0, 0.0]])
    res = retrieval.ann_vs_exact(conn, "audio_descriptors", query_embs, k=3)

    assert res["table"] == "audio_descriptors"
    assert res["indexes"] == ["idx_audio_hnsw"]
    assert res["k"] == 3
    assert res["n_queries"] == 1
    assert res["mean_overlap_at_k"] == pytest.approx(2 / 3)
    assert res["min_overlap_at_k"] == pytest.approx(2 / 3)
    assert "begin" in conn.transactions
    assert any("enable_indexscan = off" in s for s in cur.executed_sql)
