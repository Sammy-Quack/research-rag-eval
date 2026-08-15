"""Quick diagnostics over data/chunks/*.jsonl — word-count distribution per
strategy, and section-label coverage for section_aware specifically. Not
part of the pipeline itself; a sanity check to run after any chunking pass
before trusting the output enough to build embeddings on top of it.

Usage:
    python -m src.chunking.inspect_chunks
"""

import json
from collections import Counter
from pathlib import Path

CHUNKS_DIR = Path("data/chunks")
STRATEGIES = ["fixed_size", "sentence", "section_aware"]


def summarize_strategy(name: str) -> None:
    path = CHUNKS_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"{name}: no file at {path}, skipping")
        return

    word_counts = []
    section_counter: Counter = Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            word_counts.append(chunk["word_count"])
            if chunk.get("section"):
                section_counter[chunk["section"]] += 1

    total = len(word_counts)
    if total == 0:
        print(f"\n=== {name} === (empty file)")
        return

    avg_words = sum(word_counts) / total

    print(f"\n=== {name} ===")
    print(f"  {total} chunks, avg {avg_words:.1f} words/chunk "
          f"(min {min(word_counts)}, max {max(word_counts)})")

    if section_counter:
        print("  section distribution:")
        for section, count in section_counter.most_common(15):
            pct = 100 * count / total
            print(f"    {section}: {count} ({pct:.1f}%)")


def main() -> None:
    for name in STRATEGIES:
        summarize_strategy(name)


if __name__ == "__main__":
    main()