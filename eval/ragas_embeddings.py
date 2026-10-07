"""LangChain-compatible embeddings adapter wrapping our own BGE-M3 embedder
(src/embedding/bge_m3.py), so Ragas's ResponseRelevancy metric can reuse the
exact same embedding model already proven working since Phase 3, rather
than pulling in a second, different embeddings package just for evaluation.
"""

from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.embedding.bge_m3 import BGEM3Embedder


class BGEM3LangchainEmbeddings(Embeddings):
    def __init__(self, local_files_only: bool = False):
        self._embedder = BGEM3Embedder(local_files_only=local_files_only)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_texts([text])[0]


def get_judge_embeddings(local_files_only: bool = False):
    return LangchainEmbeddingsWrapper(
        BGEM3LangchainEmbeddings(local_files_only=local_files_only)
    )