"""One-off smoke test: does Ragas + Groq actually work at all? Scores a
SINGLE hand-written, obviously-correct example's faithfulness before
trusting this stack enough to build the real evaluation runner on top of
it. This integration has real uncertainty (Ragas' API has changed
significantly across versions, and Groq-via-OpenAI-compatible-endpoint
compatibility with Ragas' structured-output layer hasn't been tested live)
-- this script exists specifically to find out fast, cheaply, on one
sample, rather than discovering a wiring problem halfway through scoring
70 real questions.

Usage:
    python -m eval.test_ragas_smoke
"""

import asyncio

from eval import _ragas_compat  # noqa: F401 -- must import before ragas, see that file
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness

from eval.ragas_judge import get_judge_llm


async def main() -> None:
    sample = SingleTurnSample(
        user_input="What is the capital of France?",
        response="The capital of France is Paris.",
        retrieved_contexts=["Paris is the capital and most populous city of France."],
    )

    print("Setting up judge LLM (gpt-oss-120b via Groq)...")
    judge = get_judge_llm()
    scorer = Faithfulness(llm=judge)

    print("Scoring faithfulness...")
    score = await scorer.single_turn_ascore(sample)

    print(f"\nFaithfulness score: {score}")
    print("Expected: a number close to 1.0 (the response is fully supported by the context).")
    print("If you got a number between 0 and 1, the Ragas + Groq wiring works.")


if __name__ == "__main__":
    asyncio.run(main())