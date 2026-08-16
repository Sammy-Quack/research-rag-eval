"""BM25 sparse retrieval — the keyword-matching counterpart to dense
embedding search, used in Phase 4 for the hybrid-retrieval comparison.
Cheap to build (no ML inference involved), so it's rebuilt fresh from the
chunk JSONL on every run rather than persisted to disk like the Chroma index.
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_DIR = Path("data/chunks")
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, strategy: str):
        self.chunks = self._load_chunks(strategy)
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)

    @staticmethod
    def _load_chunks(strategy: str) -> list[dict]:
        path = CHUNKS_DIR / f"{strategy}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No chunks at {path} — run Phase 2 chunking first.")
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]