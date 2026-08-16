"""Phase 3 entry point: embed a chunked corpus and store it in a persistent
Chroma collection. One collection per (chunking strategy, embedder) pair —
e.g. "section_aware__bge_m3" — so later phases can compare across both
variables independently without them being tangled together.

Usage:
    python -m src.embedding.build_index --strategy section_aware --embedder bge_m3
    python -m src.embedding.build_index --strategy section_aware --embedder bge_m3 --limit 20   # fast sanity check first
"""

import argparse
import json
from pathlib import Path

import chromadb

from src.embedding.bge_m3 import BGEM3Embedder

CHUNKS_DIR = Path("data/chunks")
CHROMA_DIR = Path("chroma_db")
BATCH_SIZE = 32

EMBEDDERS = {
    "bge_m3": BGEM3Embedder,
    # "openai_3_large": OpenAIEmbedder,  # uncomment once OPENAI_API_KEY is set
}


def load_chunks(strategy: str, limit: int | None = None) -> list[dict]:
    path = CHUNKS_DIR / f"{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No chunks at {path} — run Phase 2 chunking first.")
    with open(path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    return chunks[:limit] if limit else chunks


def build_index(strategy: str, embedder_name: str, limit: int | None = None, device: str | None = None) -> None:
    if embedder_name not in EMBEDDERS:
        raise ValueError(f"Unknown embedder {embedder_name!r}. Options: {list(EMBEDDERS)}")

    chunks = load_chunks(strategy, limit)
    print(f"Loaded {len(chunks)} chunks for strategy={strategy!r}" + (f" (limited to {limit})" if limit else ""))

    print(f"Loading {embedder_name} model (first run downloads weights, be patient)...")
    embedder = BGEM3Embedder(device=device) if embedder_name == "bge_m3" else EMBEDDERS[embedder_name]()

    collection_name = f"{strategy}__{embedder_name}"
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_names = {c.name for c in client.list_collections()}
    if collection_name in existing_names:
        client.delete_collection(collection_name)  # fresh rebuild, avoids stale duplicates from a prior partial run
    collection = client.create_collection(collection_name)

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = embedder.embed_texts(texts)

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"paper_id": c["paper_id"], "section": c.get("section") or "none"} for c in batch],
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")

    print(f"\nDone. Collection {collection_name!r} has {collection.count()} vectors, stored in {CHROMA_DIR}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, choices=["fixed_size", "sentence", "section_aware"])
    parser.add_argument("--embedder", default="bge_m3", choices=list(EMBEDDERS))
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N chunks, for a fast sanity check")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="force a device; default auto-detects")
    args = parser.parse_args()
    build_index(args.strategy, args.embedder, args.limit, args.device)


if __name__ == "__main__":
    main()