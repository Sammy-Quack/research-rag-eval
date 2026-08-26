# Autonomous AI Agents — Benchmarked RAG Research Assistant

## Problem Statement

Autonomous AI agent research moved fast 2023-2026. Papers pile up daily
across arXiv, OpenReview, ACL Anthology — technical methods, benchmarks,
safety/ethics positions. No easy way to query across corpus and get
grounded, cited answer. Generic "chat with PDF" tools exist, but none
measure whether their own design choice (chunking, embedding, retrieval
method) actually improve answer quality — most ship one config and call
it done.

This project builds a RAG system over a curated autonomous-agents paper
corpus (open-access only: arXiv) AND empirically measures which pipeline
design choices matter, via controlled ablation with Ragas metrics.

## Summary — How It Solves The Problem

1. **Ingestion**: pull open-access PDFs (arXiv), parse, clean, strip
   references/back-matter before chunking.
2. **Chunking**: 3 strategies implemented as swappable modules — fixed
   size, sentence-based, section-aware. Compared, not guessed.
3. **Embedding + Index**: BGE-M3 (open-source, self-hosted) into
   ChromaDB, alongside a parallel BM25 sparse index.
4. **Retrieval**: dense (Chroma), sparse (BM25), and hybrid
   (Reciprocal Rank Fusion) — all three implemented, all three queryable.
5. **Generation**: grounded prompt with numbered citations, LLM answers
   via Groq (Qwen3.6-27B), explicit "insufficient information" handling
   instead of silent hallucination.
6. **Evaluation**: hand-verified Q&A eval set — 70 questions, drafted by
   an LLM then manually reviewed one at a time, 1.4% observed error rate
   after review (1 of 70 needed correction). Ragas scoring across
   pipeline configs is next.

## Status

**Phases 0–5a complete.** Ingestion, all 3 chunking strategies, BGE-M3
embedding + Chroma indexing, dense/BM25/hybrid retrieval, grounded
generation, and a manually verified 70-question evaluation set are all
built and tested end-to-end.

**Next**: Phase 5b — wire up Ragas (faithfulness, answer relevancy,
context precision/recall) and run the actual ablation matrix across
chunking strategy × retrieval mode, including a no-retrieval baseline as
a control. Then Phase 6 (Streamlit demo) and Phase 7 (final write-up
with real results).

Full build plan: `docs/roadmap.md`.

## Stack

**In use**: Python 3.11+, PyMuPDF, ChromaDB, sentence-transformers
(BGE-M3), rank_bm25, Groq API (Qwen3.6-27B), pytest, GitHub Actions CI.

**Planned, not yet wired in**: Ragas (Phase 5b), Streamlit (Phase 6).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT — see `LICENSE`.