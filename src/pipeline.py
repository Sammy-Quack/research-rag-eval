"""Full RAG pipeline, config-driven so Phase 5's ablation study can swap
chunking strategy / retrieval mode without touching this file. Retrieve ->
build grounded prompt -> generate -> print, with an explicit no-context
guard rather than silently hallucinating when nothing relevant comes back.

Generation now runs via local Ollama (not Groq) -- eliminates daily/API
rate limits entirely, compute-bound on your own hardware instead of a
shared cloud quota.

Usage:
    python -m src.pipeline --strategy section_aware --mode hybrid --query "How are agents evaluated?"
"""

import argparse
import time

from src.generation.ollama_llm import OllamaLLM
from src.generation.prompt import BASELINE_SYSTEM_PROMPT, SYSTEM_PROMPT, build_prompt
from src.retrieval.dense_index import DenseIndex
from src.retrieval.hybrid import HybridIndex


def get_retriever(strategy: str, mode: str, embedder_name: str = "bge_m3"):
    if mode == "dense":
        return DenseIndex(strategy, embedder_name)
    elif mode == "hybrid":
        return HybridIndex(strategy, embedder_name)
    raise ValueError(f"Unknown mode {mode!r}, expected 'dense' or 'hybrid'")


def answer_question(
    query: str, strategy: str = None, mode: str = "hybrid", top_k: int = 5, embedder_name: str = "bge_m3",
    retriever=None,
) -> dict:
    start_time = time.monotonic()

    if mode == "none":
        llm = OllamaLLM()
        answer, usage = llm.generate_with_usage(BASELINE_SYSTEM_PROMPT, query)
        return {
            "answer": answer,
            "chunks_used": [],
            "grounded": False,
            "latency_seconds": time.monotonic() - start_time,
            "token_usage": usage,
        }

    if retriever is None:
        retriever = get_retriever(strategy, mode, embedder_name)

    results = retriever.search(query, top_k=top_k)
    chunks = [chunk for chunk, _score in results]

    if not chunks:
        return {
            "answer": "No relevant information was found in the corpus for this question.",
            "chunks_used": [],
            "grounded": False,
            "latency_seconds": time.monotonic() - start_time,
            "token_usage": None,
        }

    prompt = build_prompt(query, chunks)
    llm = OllamaLLM()
    answer, usage = llm.generate_with_usage(SYSTEM_PROMPT, prompt)

    return {
        "answer": answer,
        "chunks_used": chunks,
        "grounded": True,
        "latency_seconds": time.monotonic() - start_time,
        "token_usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["fixed_size", "sentence", "section_aware"], help="required unless --mode none")
    parser.add_argument("--mode", default="hybrid", choices=["dense", "hybrid", "none"])
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if args.mode != "none" and not args.strategy:
        parser.error("--strategy is required unless --mode is 'none'")

    result = answer_question(args.query, args.strategy, args.mode, args.top_k)

    print(f"\nQuestion: {args.query}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"--- Sources used ({len(result['chunks_used'])}) ---")
    for i, chunk in enumerate(result["chunks_used"], start=1):
        print(f"[{i}] {chunk['chunk_id']} (paper={chunk['paper_id']}, section={chunk.get('section')})")
    print(f"\nLatency: {result['latency_seconds']:.2f}s")
    if result["token_usage"]:
        print(f"Tokens: {result['token_usage']['total_tokens']}")


if __name__ == "__main__":
    main()