"""Shared interface every embedding backend implements, so build_index.py
and later retrieval code can swap backends without caring which is active.
"""

from typing import Protocol


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...