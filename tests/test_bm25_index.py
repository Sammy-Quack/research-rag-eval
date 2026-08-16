"""Tests for BM25Index. Uses a minimal fake standing in for rank_bm25's
BM25Okapi so these tests verify *our* wrapper logic (loading, tokenizing,
ranking, packaging results) without depending on the real library being
installed — that library's own scoring math is its responsibility to test,
not ours to re-verify here.
"""

import json
import sys
import types

import pytest


class _FakeBM25Okapi:
    """Crude token-overlap scorer — enough to prove ranking/indexing
    plumbing works, not a real BM25 implementation."""

    def __init__(self, tokenized_corpus):
        self.corpus = tokenized_corpus

    def get_scores(self, query_tokens):
        return [sum(1 for t in query_tokens if t in doc) for doc in self.corpus]


@pytest.fixture(autouse=True)
def fake_rank_bm25(monkeypatch):
    fake_module = types.ModuleType("rank_bm25")
    fake_module.BM25Okapi = _FakeBM25Okapi
    monkeypatch.setitem(sys.modules, "rank_bm25", fake_module)
    # bm25_index imports BM25Okapi at module load time, so make sure our
    # already-imported module (if any) points at the fake too
    if "src.retrieval.bm25_index" in sys.modules:
        monkeypatch.setattr(sys.modules["src.retrieval.bm25_index"], "BM25Okapi", _FakeBM25Okapi)
    yield


@pytest.fixture
def chunk_corpus(tmp_path, monkeypatch):
    chunks_dir = tmp_path / "data" / "chunks"
    chunks_dir.mkdir(parents=True)
    chunks = [
        {"chunk_id": "p1::0", "paper_id": "p1", "text": "autonomous agents plan and act in environments"},
        {"chunk_id": "p1::1", "paper_id": "p1", "text": "reinforcement learning trains policies for control"},
        {"chunk_id": "p2::0", "paper_id": "p2", "text": "agents use planning and reasoning to solve tasks"},
    ]
    with open(chunks_dir / "fixed_size.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    from src.retrieval import bm25_index
    monkeypatch.setattr(bm25_index, "CHUNKS_DIR", chunks_dir)
    return bm25_index


def test_tokenize_lowercases_and_strips_punctuation():
    from src.retrieval.bm25_index import tokenize
    assert tokenize("Autonomous Agents!") == ["autonomous", "agents"]


def test_tokenize_empty_string():
    from src.retrieval.bm25_index import tokenize
    assert tokenize("") == []


def test_bm25_index_loads_all_chunks(chunk_corpus):
    idx = chunk_corpus.BM25Index("fixed_size")
    assert len(idx.chunks) == 3


def test_bm25_index_missing_file_raises(chunk_corpus):
    with pytest.raises(FileNotFoundError):
        chunk_corpus.BM25Index("sentence")  # only fixed_size.jsonl exists in fixture


def test_search_returns_requested_top_k(chunk_corpus):
    idx = chunk_corpus.BM25Index("fixed_size")
    results = idx.search("agents planning", top_k=2)
    assert len(results) == 2


def test_search_ranks_relevant_chunks_higher(chunk_corpus):
    idx = chunk_corpus.BM25Index("fixed_size")
    results = idx.search("agents planning", top_k=3)
    top_ids = [chunk["chunk_id"] for chunk, score in results]
    # the RL-only chunk shares no query terms — should rank last, not first
    assert top_ids[-1] == "p1::1"


def test_search_result_scores_are_floats(chunk_corpus):
    idx = chunk_corpus.BM25Index("fixed_size")
    results = idx.search("agents", top_k=1)
    assert isinstance(results[0][1], float)