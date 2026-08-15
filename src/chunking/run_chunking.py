"""Phase 2 entry point. Reads every data/processed/*.json paper, runs all
three chunking strategies over each one, writes one JSONL file per strategy
to data/chunks/ (one chunk per line).

Usage:
    python -m src.chunking.run_chunking
"""

import json
from pathlib import Path

from src.chunking.fixed_size import chunk_fixed_size
from src.chunking.section_aware import chunk_section_aware
from src.chunking.sentence_based import chunk_sentence_based

PROCESSED_DIR = Path("data/processed")
CHUNKS_DIR = Path("data/chunks")

STRATEGIES = {
    "fixed_size": chunk_fixed_size,
    "sentence": chunk_sentence_based,
    "section_aware": chunk_section_aware,
}


def run() -> None:
    if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.glob("*.json")):
        raise FileNotFoundError(f"No papers in {PROCESSED_DIR} — run Phase 1 ingestion first.")

    papers = sorted(PROCESSED_DIR.glob("*.json"))
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in STRATEGIES}
    handles = {
        name: open(CHUNKS_DIR / f"{name}.jsonl", "w", encoding="utf-8")
        for name in STRATEGIES
    }

    try:
        for paper_path in papers:
            with open(paper_path, encoding="utf-8") as f:
                paper = json.load(f)

            paper_id = paper["paper_id"]
            text = paper["text"]

            per_strategy_counts = {}
            for name, chunk_fn in STRATEGIES.items():
                chunks = chunk_fn(text, paper_id)
                for chunk in chunks:
                    handles[name].write(json.dumps(chunk, ensure_ascii=False) + "\n")
                counts[name] += len(chunks)
                per_strategy_counts[name] = len(chunks)

            summary = ", ".join(f"{n}={c}" for n, c in per_strategy_counts.items())
            print(f"[{paper_id}] {summary}")
    finally:
        for handle in handles.values():
            handle.close()

    print("\nDone.")
    for name, count in counts.items():
        avg = count / len(papers)
        print(f"  {name}: {count} chunks total, {avg:.1f} avg/paper -> data/chunks/{name}.jsonl")


if __name__ == "__main__":
    run()