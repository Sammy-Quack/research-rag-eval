"""Hybrid retrieval: combine dense (semantic) and BM25 (keyword) rankings
via Reciprocal Rank Fusion (RRF).

Why RRF specifically: dense distances (e.g. 0.78) and BM25 scores (e.g.
15.6) live on completely different, incomparable scales — you can't just
average them together meaningfully. RRF sidesteps that by using only each
result's RANK POSITION in its own list, not its raw score, so the two
systems never need to be normalized against each other. It's the standard
approach for exactly this problem, not a workaround invented for this
project.

RRF score per chunk = sum, across every ranked list it appears in, of
1 / (k + rank) — where rank is 1-indexed and k (60, the standard default
from the original RRF paper) dampens the influence of low ranks so one
system's rank-50 result can't dominate another's rank-2 result.
"""

from src.retrieval.bm25_index import BM25Index
from src.retrieval.dense_index import DenseIndex

RRF_K = 60


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = RRF_K) -> list[tuple[dict, float]]:
    """Each ranked_list is a list of chunk dicts already sorted best-first.
    Returns chunks sorted by combined RRF score, best first. Pure function —
    no dependency on Chroma/BM25/network, deliberately, so it's fully
    unit-testable on its own.
    """
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_by_id[chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunk_by_id[chunk_id], score) for chunk_id, score in ranked]


class HybridIndex:
    def __init__(self, strategy: str, embedder_name: str = "bge_m3"):
        self.dense = DenseIndex(strategy, embedder_name)
        self.bm25 = BM25Index(strategy)

    def search(self, query: str, top_k: int = 5, candidate_pool: int = 20) -> list[tuple[dict, float]]:
        # pull a wider pool from each system before fusing, so RRF has more
        # than just each side's final top_k to actually combine
        dense_chunks = [chunk for chunk, _ in self.dense.search(query, top_k=candidate_pool)]
        bm25_chunks = [chunk for chunk, _ in self.bm25.search(query, top_k=candidate_pool)]

        fused = reciprocal_rank_fusion(dense_chunks, bm25_chunks)
        return fused[:top_k]