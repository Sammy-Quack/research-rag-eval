"""Smoke test, expanded: proves all four Ragas metrics we plan to use work
against a single hand-written, obviously-good example, before trusting any
of them enough to build the real evaluation runner on top.

Usage:
    python -m eval.test_ragas_smoke
"""

import asyncio

from eval import _ragas_compat  # noqa: F401 -- must import before ragas, see that file
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

from eval.ragas_embeddings import get_judge_embeddings
from eval.ragas_judge import get_judge_llm, JUDGE_MODEL


async def main() -> None:
    sample = SingleTurnSample(
        user_input="What is the capital of France?",
        response="The capital of France is Paris.",
        retrieved_contexts=["Paris is the capital and most populous city of France."],
        reference="Paris is the capital of France.",
    )

    print(f"Setting up judge LLM ({JUDGE_MODEL} via local Ollama)...")
    judge_llm = get_judge_llm()

    print("Loading judge embeddings (BGE-M3, same model as the pipeline)...")
    judge_embeddings = get_judge_embeddings()

    metrics = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "context_precision": LLMContextPrecisionWithReference(llm=judge_llm),
        "context_recall": LLMContextRecall(llm=judge_llm),
        "response_relevancy": ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1),
    }

    print("\nScoring all 4 metrics on one sample...\n")
    for name, scorer in metrics.items():
        score = await scorer.single_turn_ascore(sample)
        print(f"  {name}: {score:.4f}")

    print("\nExpected: all 4 scores should be high (close to 1.0) -- this example is")
    print("a genuinely good, faithful, relevant, well-grounded answer.")
    print("If all 4 printed real numbers, the full Ragas + Ollama + BGE-M3 wiring works.")


if __name__ == "__main__":
    asyncio.run(main())