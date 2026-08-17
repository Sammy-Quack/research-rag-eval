"""Dense retrieval: embed a query with the same embedder used to build the
index, search Chroma, return ranked chunks. Thin wrapper reused by both
direct dense search and the hybrid combiner below.
"""

import chromadb

from src.embedding.bge_m3 import BGEM3Embedder

CHROMA_DIR = "chroma_db"
EMBEDDERS = {"bge_m3": BGEM3Embedder}


class DenseIndex:
    def __init__(self, strategy: str, embedder_name: str = "bge_m3"):
        self.strategy = strategy
        self.embedder_name = embedder_name
        self._embedder = EMBEDDERS[embedder_name]()

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self._collection = client.get_collection(f"{strategy}__{embedder_name}")

    def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """Returns (chunk, distance) tuples, best (lowest distance) first."""
        query_embedding = self._embedder.embed_texts([query])[0]
        results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)

        output = []
        for chunk_id, doc, distance, meta in zip(
            results["ids"][0], results["documents"][0], results["distances"][0], results["metadatas"][0],
        ):
            chunk = {
                "chunk_id": chunk_id,
                "text": doc,
                "paper_id": meta["paper_id"],
                "section": meta["section"],
            }
            output.append((chunk, distance))
        return output