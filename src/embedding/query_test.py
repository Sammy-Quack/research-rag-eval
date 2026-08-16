"""Manual sanity check: query a built Chroma collection with a free-text
question, print the top results. Not part of the pipeline — a quick way to
eyeball "does semantic search actually work" before trusting the index
enough to build the real retrieval/generation pipeline on top of it (Phase 4).

Usage:
    python -m src.embedding.query_test --strategy section_aware --embedder bge_m3 --query "How are agents evaluated?"
"""

import argparse

import chromadb

from src.embedding.bge_m3 import BGEM3Embedder

CHROMA_DIR = "chroma_db"
EMBEDDERS = {"bge_m3": BGEM3Embedder}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--embedder", default="bge_m3")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    embedder = EMBEDDERS[args.embedder]()
    query_embedding = embedder.embed_texts([args.query])[0]

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(f"{args.strategy}__{args.embedder}")

    results = collection.query(query_embeddings=[query_embedding], n_results=args.top_k)

    print(f"\nQuery: {args.query!r}\n")
    for rank, (chunk_id, doc, distance, meta) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["distances"][0], results["metadatas"][0],
    ), start=1):
        print(f"[{rank}] {chunk_id}  (paper={meta['paper_id']}, section={meta['section']}, distance={distance:.4f})")
        print(f"    {doc[:200]}...\n")


if __name__ == "__main__":
    main()