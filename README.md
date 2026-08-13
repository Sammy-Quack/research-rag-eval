# Autonomous AI Agents — Benchmarked RAG Research Assistant

## Problem Statement

Autonomous AI agent research moved fast 2023-2026. Papers pile up daily
across arXiv, OpenReview, ACL Anthology — technical methods, benchmarks,
safety/ethics positions. No easy way to query across corpus and get
grounded, cited answer. Generic "chat with PDF" tools exist, but none
measure whether their own design choice (chunking, embedding, retrieval
method) actually improve answer quality — most ship one config and call
it done.

This project builds RAG system over curated autonomous-agents paper
corpus (open-access only: arXiv, OpenReview, ACL Anthology) AND
empirically measures which pipeline design choices matter, via
controlled ablation with Ragas metrics.

## Summary — How It Solves The Problem

1. **Ingestion**: pull open-access PDFs (arXiv/OpenReview/ACL), parse,
   clean.
2. **Chunking**: 3 strategies implemented as swappable modules — fixed
   size, sentence-based, section-aware. Compared, not guessed.
3. **Embedding + Index**: BGE-M3 (open-source) + one alt (OpenAI/Qwen3)
   into Chroma; parallel BM25 sparse index for hybrid retrieval test.
4. **Retrieval + Generation**: dense/hybrid/reranked retrieval, grounded
   prompt with citations, explicit "no answer in context" handling —
   no silent hallucination.
5. **Evaluation**: hand-verified Q&A eval set (not blind LLM-generated),
   Ragas faithfulness / answer relevancy / context precision / recall,
   plus no-retrieval baseline as control. Cost + latency tracked per
   config.
6. **Result**: ablation table, not vibes — proves which choices actually
   move the needle, with numbers.

Status: scaffold stage. Build plan in `docs/roadmap.md`.

## Stack

Python 3.11+, BGE-M3, Chroma, rank_bm25, Ragas, Streamlit, GitHub
Actions CI.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT — see `LICENSE`.
