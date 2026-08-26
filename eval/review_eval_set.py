"""Phase 5a, step 2: manually review and verify (or correct) the LLM-drafted
candidates from build_eval_set.py, one at a time. Saves progress after
every single decision -- safe to Ctrl+C or quit partway without losing work
(learned that lesson the hard way earlier in this project).

Usage:
    python -m eval.review_eval_set
"""

import json
from pathlib import Path

EVAL_SET_PATH = Path("eval/eval_set.jsonl")


def load_candidates() -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"No {EVAL_SET_PATH} -- run build_eval_set.py first.")
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_candidates(candidates: list[dict]) -> None:
    to_write = [c for c in candidates if not c.get("_delete")]
    with open(EVAL_SET_PATH, "w", encoding="utf-8") as f:
        for c in to_write:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def review() -> None:
    candidates = load_candidates()
    unverified = [c for c in candidates if not c.get("verified")]

    if not unverified:
        print("Nothing left to review -- every entry is already marked verified.")
        return

    print(f"{len(unverified)} unverified candidates ({len(candidates) - len(unverified)} already done)\n")

    kept, edited, deleted = 0, 0, 0

    for candidate in candidates:
        if candidate.get("verified"):
            continue

        print("=" * 70)
        print(f"Source paper: {candidate['source_paper_id']}  |  chunk: {candidate['source_chunk_id']}")
        print(f"\nSource text:\n{candidate['source_text'][:500]}")
        print(f"\nQuestion: {candidate['question']}")
        print(f"Reference answer: {candidate['reference_answer']}")
        print()

        action = input("[k]eep  [e]dit answer  [s]kip for now  [d]elete  [q]uit and save: ").strip().lower()

        if action == "q":
            break
        elif action == "d":
            candidate["_delete"] = True
            deleted += 1
        elif action == "e":
            new_answer = input("New reference answer: ").strip()
            if new_answer:
                candidate["reference_answer"] = new_answer
            candidate["verified"] = True
            edited += 1
        elif action == "k":
            candidate["verified"] = True
            kept += 1
        else:
            print("  (skipped, will ask again next run)\n")
            continue

        save_candidates(candidates)  # incremental save after every real decision

    save_candidates(candidates)  # final save catches loop-exit state too

    total_reviewed = kept + edited + deleted
    print(f"\nSession summary: {kept} kept as-is, {edited} edited, {deleted} deleted "
          f"({total_reviewed} decisions this session).")
    if total_reviewed:
        error_rate = (edited + deleted) / total_reviewed
        print(f"Observed error rate in LLM-drafted candidates: {error_rate:.0%} "
              f"(needed editing or deletion) -- cite this exact number in your README.")


if __name__ == "__main__":
    review()