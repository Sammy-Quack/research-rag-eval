"""Phase 5b: run the real evaluation. Feeds every (verified) question in
eval/eval_set.jsonl through the actual pipeline for a given chunking
strategy + retrieval mode, scores each answer with all 4 Ragas metrics,
and reports per-metric averages plus latency/token efficiency.

Saves incrementally after every single question -- a crash partway through
a large run loses at most one question's work, not the whole batch.

Usage (small subset first, always):
    python -m eval.run_evaluation --strategy section_aware --mode hybrid --limit 5
"""

import argparse
import asyncio
import json
import math
from pathlib import Path

from eval import _ragas_compat  # noqa: F401 -- must import before ragas
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from eval.ragas_embeddings import get_judge_embeddings
from eval.ragas_judge import get_judge_llm
from src.generation.ollama_llm import OllamaLLM
from src.pipeline import answer_question, get_retriever

EVAL_SET_PATH = Path("eval/eval_set.jsonl")
RESULTS_DIR = Path("eval/results")

METRIC_NAMES = ["faithfulness", "context_precision", "context_recall", "response_relevancy"]


def load_eval_set(limit: int | None = None) -> list[dict]:
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(f"No {EVAL_SET_PATH} -- run Phase 5a first.")
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    verified = [item for item in items if item.get("verified")]
    if not verified:
        raise ValueError(f"No verified entries in {EVAL_SET_PATH} -- run eval.review_eval_set first.")
    return verified[:limit] if limit else verified


def load_existing_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def build_metrics(judge_llm, judge_embeddings, mode: str) -> dict:
    response_relevancy_metric = ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1)

    if mode == "none":
        # no real retrieved context for the baseline -- faithfulness and
        # context precision/recall are meaningless without it (they'd
        # trivially score ~0), only response_relevancy applies
        return {"response_relevancy": response_relevancy_metric}

    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_precision": LLMContextPrecisionWithReference(llm=judge_llm),
        "context_recall": LLMContextRecall(llm=judge_llm),
        "response_relevancy": response_relevancy_metric,
    }


def is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate_limit" in message or "429" in message or "tokens per day" in message


async def score_with_retry(scorer, sample, max_attempts: int = 3):
    """Parse failures from small local judge models look probabilistic, not
    deterministic -- a fresh attempt at the same metric can succeed even
    when the previous one didn't. Rate-limit errors are NOT retried here;
    those need the outer loop's stop-the-whole-run handling instead.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await scorer.single_turn_ascore(sample)
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise
            last_exc = exc
            if attempt < max_attempts:
                print(f"      parse failure, retry {attempt}/{max_attempts - 1}...")
    raise last_exc


async def run_evaluation(strategy: str | None, mode: str, limit: int | None) -> None:
    eval_items = load_eval_set(limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config_name = "none" if mode == "none" else f"{strategy}__{mode}"
    results_path = RESULTS_DIR / f"{config_name}__{OllamaLLM.name}.json"
    per_question_results = load_existing_results(results_path)
    already_done = {r["eval_id"] for r in per_question_results}
    remaining_items = [item for item in eval_items if item["eval_id"] not in already_done]

    if not remaining_items:
        print(f"All {len(eval_items)} questions already evaluated in {results_path}. Nothing to do.")
    else:
        if already_done:
            print(f"Resuming: {len(already_done)} already done, {len(remaining_items)} remaining.\n")
        else:
            print(f"Running {len(remaining_items)} verified eval questions "
                  f"through strategy={strategy!r} mode={mode!r}\n")

        print("Setting up judge LLM and embeddings...")
        judge_llm = get_judge_llm()
        # The no-retrieval baseline still needs embeddings for response relevancy;
        # load its already-cached model without probing Hugging Face.
        judge_embeddings = get_judge_embeddings(local_files_only=mode == "none")
        metrics = build_metrics(judge_llm, judge_embeddings, mode)

        retriever = None
        if mode != "none":
            print("Building retriever (loads BGE-M3 once, reused across all questions)...")
            retriever = get_retriever(strategy, mode)
        for i, item in enumerate(remaining_items, start=1):
            print(f"[{i}/{len(remaining_items)}] {item['question'][:70]}")

            try:
                result = answer_question(item["question"], strategy, mode, retriever=retriever)

                sample = SingleTurnSample(
                    user_input=item["question"],
                    response=result["answer"],
                    retrieved_contexts=[c["text"] for c in result["chunks_used"]] or ["(no context retrieved)"],
                    reference=item["reference_answer"],
                )

                scores = {}
                for name, scorer in metrics.items():
                    scores[name] = await score_with_retry(scorer, sample)

            except Exception as exc:
                if is_rate_limit_error(exc):
                    remaining_count = len(remaining_items) - i + 1
                    print(f"\n  Hit a rate limit: {exc}")
                    print(f"  Stopping here rather than burning through the remaining "
                          f"{remaining_count} questions with guaranteed failures.")
                    print(f"  {len(per_question_results)} questions successfully completed and saved so far.")
                    print("  This question was NOT recorded -- rerun the same command later to pick up exactly here.")
                    break
                print(f"    Non-rate-limit error, skipping this question only: {exc}")
                continue

            per_question_results.append({
                "eval_id": item["eval_id"],
                "question": item["question"],
                "answer": result["answer"],
                "scores": scores,
                "latency_seconds": result["latency_seconds"],
                "token_usage": result["token_usage"],
            })

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(per_question_results, f, ensure_ascii=False, indent=2)

    print(f"\n=== Results: config={config_name}, n={len(per_question_results)} -> {results_path} ===")
    all_metric_names = set()
    for r in per_question_results:
        all_metric_names.update(r["scores"].keys())
    for name in sorted(all_metric_names):
        all_for_metric = [r["scores"].get(name) for r in per_question_results]
        valid = [s for s in all_for_metric if s is not None and not (isinstance(s, float) and math.isnan(s))]
        undefined_count = sum(1 for s in all_for_metric if isinstance(s, float) and math.isnan(s))
        if valid:
            line = f"  {name}: avg={sum(valid) / len(valid):.4f} (n={len(valid)}/{len(per_question_results)})"
            if undefined_count:
                line += f"  [{undefined_count} undefined/NaN excluded, e.g. from an empty answer]"
            print(line)
        else:
            print(f"  {name}: no successful scores")

    latencies = [r["latency_seconds"] for r in per_question_results]
    if latencies:
        print(f"  avg latency: {sum(latencies) / len(latencies):.2f}s")

    token_totals = [r["token_usage"]["total_tokens"] for r in per_question_results
                     if r["token_usage"] and r["token_usage"].get("total_tokens") is not None]
    if token_totals:
        print(f"  avg tokens/question: {sum(token_totals) / len(token_totals):.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_size", "sentence", "section_aware"], help="required unless --mode none")
    parser.add_argument("--mode", default="hybrid", choices=["dense", "hybrid", "none"])
    parser.add_argument("--limit", type=int, default=5, help="only run N eval questions -- small subset first, always")
    args = parser.parse_args()

    if args.mode != "none" and not args.strategy:
        parser.error("--strategy is required unless --mode is 'none'")

    asyncio.run(run_evaluation(args.strategy, args.mode, args.limit))


if __name__ == "__main__":
    main()