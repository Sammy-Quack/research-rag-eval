"""Phase 5a: draft candidate evaluation Q&A pairs from the corpus using an
LLM, grounded in one chunk each. These are CANDIDATES ONLY -- run
review_eval_set.py afterward to manually verify or correct them before
trusting any of this as ground truth. Blindly trusting LLM-drafted
questions is exactly the corner-cutting this project's evaluation rigor
exists to avoid.

Usage:
    python -m eval.build_eval_set --strategy section_aware --count 70
"""

import argparse
import json
import re
import time
from pathlib import Path

from src.generation.groq_llm import GroqLLM

CHUNKS_DIR = Path("data/chunks")
OUTPUT_PATH = Path("eval/eval_set.jsonl")

DRAFT_SYSTEM_PROMPT = (
    "You write evaluation questions for testing a research-paper Q&A system. "
    "Given an excerpt, write exactly one specific, answerable question about "
    "its content, plus a concise 1-3 sentence reference answer grounded ONLY "
    "in the excerpt. Avoid yes/no questions and questions answerable from "
    "general knowledge without needing this excerpt specifically. "
    "CRITICAL: write the question as a naturalistic standalone question a "
    "curious reader would actually ask -- NEVER reference 'the excerpt', "
    "'the passage', 'the text', 'this document', or similar meta-references "
    "to the fact that the answer comes from a specific source. For example, "
    'write "What indirect risks can AI systems pose to biosecurity?" NOT '
    '"What indirect risks does the excerpt identify?". '
    'Respond with ONLY a JSON object: {"question": "...", "reference_answer": "..."}. '
    "No markdown, no code fences, no extra text before or after the JSON."
)

META_REFERENCE_PATTERN = re.compile(
    r"\b(the excerpt|the passage|the text|this document|the source(?: states| says)?|"
    r"according to the (excerpt|passage|text))\b",
    re.IGNORECASE,
)


def has_meta_reference(question: str) -> bool:
    return bool(META_REFERENCE_PATTERN.search(question))


def stratified_sample(
    chunks: list[dict], target_count: int, min_word_count: int = 80, seed: int = 42,
) -> list[dict]:
    """Pick up to target_count chunks, at most 2 per paper, favoring breadth
    across papers over depth within any one paper. Excludes low-content
    sections (preamble, very short fragments) that don't have enough
    substance to draft a good, specific question from.
    """
    import random
    rng = random.Random(seed)

    eligible = [c for c in chunks if c.get("word_count", 0) >= min_word_count and c.get("section") != "preamble"]
    rng.shuffle(eligible)

    by_paper: dict[str, list[dict]] = {}
    for chunk in eligible:
        by_paper.setdefault(chunk["paper_id"], []).append(chunk)

    paper_ids = list(by_paper.keys())
    rng.shuffle(paper_ids)

    selected = []
    max_per_paper = 2
    round_num = 0
    while len(selected) < target_count and round_num < max_per_paper:
        for paper_id in paper_ids:
            if len(selected) >= target_count:
                break
            paper_chunks = by_paper[paper_id]
            if round_num < len(paper_chunks):
                selected.append(paper_chunks[round_num])
        round_num += 1

    return selected[:target_count]


def parse_json_response(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if "question" in parsed and "reference_answer" in parsed:
        return parsed
    return None


def generate_with_retry(llm: GroqLLM, system_prompt: str, user_prompt: str, max_retries: int = 4) -> str:
    """Basic rate-limit retry -- ~70 sequential free-tier calls is exactly
    the scale where Groq's free tier can start throttling, same lesson
    learned from Semantic Scholar earlier in this project.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return llm.generate(system_prompt, user_prompt)
        except Exception as exc:
            message = str(exc).lower()
            if "rate" in message or "429" in message:
                wait = 15 * attempt
                print(f"    rate limited, waiting {wait}s (attempt {attempt}/{max_retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Still rate-limited after all retries.")


def load_chunks(strategy: str) -> list[dict]:
    path = CHUNKS_DIR / f"{strategy}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No chunks at {path} -- run Phase 2 chunking first.")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_existing_candidates() -> list[dict]:
    if not OUTPUT_PATH.exists():
        return []
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_candidates(candidates: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def build_eval_set(strategy: str, count: int) -> None:
    existing = load_existing_candidates()
    already_covered = {c["source_chunk_id"] for c in existing}

    remaining_needed = count - len(existing)
    if remaining_needed <= 0:
        print(f"Already have {len(existing)} candidates at {OUTPUT_PATH}, target was {count}. Nothing to do.")
        return

    if existing:
        print(f"Resuming: {len(existing)} candidates already saved, drafting {remaining_needed} more.")

    chunks = load_chunks(strategy)
    chunks = [c for c in chunks if c["chunk_id"] not in already_covered]
    sampled = stratified_sample(chunks, remaining_needed)
    print(f"Sampled {len(sampled)} new chunks across {len({c['paper_id'] for c in sampled})} papers")

    # Non-reasoning model for drafting -- much lighter token footprint per
    # call than the reasoning model used for pipeline generation (no <think>
    # deliberation), which means faster batch drafting and a lower chance of
    # hitting a token-based rate limit across ~70 sequential calls. The
    # reasoning model stays reserved for the actual pipeline being evaluated.
    llm = GroqLLM(model="qwen/qwen3.6-27b")

    candidates = existing
    failed = 0

    for i, chunk in enumerate(sampled, start=1):
        prompt = f"Excerpt:\n\n{chunk['text']}"
        raw = generate_with_retry(llm, DRAFT_SYSTEM_PROMPT, prompt)
        parsed = parse_json_response(raw)

        if parsed is None:
            print(f"  [{i}/{len(sampled)}] FAILED to parse LLM response, skipping")
            failed += 1
            continue

        if has_meta_reference(parsed["question"]):
            print(f"  [{i}/{len(sampled)}] REJECTED (meta-reference in question): {parsed['question'][:80]}")
            failed += 1
            continue

        candidates.append({
            "eval_id": f"eval_{len(candidates) + 1:04d}",
            "question": parsed["question"],
            "reference_answer": parsed["reference_answer"],
            "source_chunk_id": chunk["chunk_id"],
            "source_paper_id": chunk["paper_id"],
            "source_text": chunk["text"],
            "verified": False,
            "reviewer_notes": "",
        })
        save_candidates(candidates)  # incremental -- a crash after this point loses nothing
        print(f"  [{i}/{len(sampled)}] drafted: {parsed['question'][:80]}")

    print(f"\nDone. {len(candidates)} total candidates at {OUTPUT_PATH} ({failed} failed/rejected this run).")
    print("Next: python -m eval.review_eval_set  -- to manually verify these before trusting them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="section_aware", choices=["fixed_size", "sentence", "section_aware"])
    parser.add_argument("--count", type=int, default=70)
    args = parser.parse_args()
    build_eval_set(args.strategy, args.count)


if __name__ == "__main__":
    main()