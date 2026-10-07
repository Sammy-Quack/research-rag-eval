# Development Log

A record of what was built, what broke, and where the project diverged from
the original plan (`docs/roadmap.md`). The goal is to document decisions and
their evidence honestly, including the parts that are not settled.

**How to read this:** each phase lists the problems hit, the fix, and (where
relevant) how confident the fix is. Section 5 collects claims that are still
hypotheses, and Section 6 lists the limitations that matter when reading the
results.

---

## 1. Plan vs. what shipped

| Area | Original plan | What shipped | Why it changed |
|---|---|---|---|
| Corpus | 50-150 papers from arXiv, OpenReview, ACL via Semantic Scholar | 70 arXiv papers | Publisher-hosted "open access" PDFs were bot-blocked; restricted to arXiv with direct PDF URLs |
| Corpus scope | One tight sub-field | Keyword-search sample for "autonomous AI agents" (broad, partly tangential) | Corpus was assembled by search relevance, not hand-curated |
| PDF parsing | `unstructured` or PyMuPDF, optionally GROBID | PyMuPDF plus heuristic cleaning and back-matter stripping | Simpler, fewer dependencies |
| Chunking | Fixed-size, sentence, section-aware | Same three, sharing one ~400-word budget | - |
| Embeddings | BGE-M3 plus one comparison model | BGE-M3 only | Scope. An OpenAI embedder was written but never used |
| Vector store | Chroma | Chroma | - |
| Retrieval | Dense, sparse, hybrid, reranking | Dense, BM25, hybrid (RRF). No reranker | Scope |
| Generation | Claude/GPT, or a free-tier open model | Groq-hosted models first, then local Ollama `llama3.2-3b-8k` | Rate limits, output-token caps, deprecations (Section 3, Phases 4-5) |
| Judge | A different model from the generator | Local Ollama `qwen2.5-14b-8k` | Groq daily quota; local judge needed |
| Evaluation framework | Ragas, 4 metrics | Ragas 0.3.9 (pinned), plus a compatibility shim | Upstream import bug in newer versions |
| Eval set | 50-100 questions, manually verified, easy and hard (multi-passage) | 70 questions, manually reviewed, single-passage only | Multi-passage questions not built |
| Baseline | No-retrieval control | Implemented; only `response_relevancy` is applicable | Other metrics are undefined without retrieved context |
| Cost tracking | $ per query and latency | Tokens and latency ($0 by design) | Free-tier / local setup |
| Fine-tuning comparison (Phase 8 stretch) | Optional | Not attempted | Scope |
| Demo | Streamlit on a free Hugging Face Space | Build guide written, app not built | Free Spaces no longer cover this (see `STREAMLIT_GUIDE.md`) |
| CI | GitHub Actions running tests | Done, after fixing a stale install step | Dependencies had outgrown the CI install line |

---

## 2. Final results

70 questions per configuration. Generator `llama3.2-3b-8k`, judge `qwen2.5-14b-8k`, `top_k=5`.
Single run per configuration.

| Configuration | Faithfulness | Context precision | Context recall | Response relevancy | Avg latency (s) |
|---|---:|---:|---:|---:|---:|
| none (no retrieval) | n/a | n/a | n/a | 0.489 | 37.9 |
| section_aware + dense | 0.808 | 0.844 | 0.900 | 0.720 | 33.7 |
| fixed_size + hybrid | 0.863 | 0.872 | 0.971 | 0.751 | 27.6 |
| sentence + hybrid | 0.870 | 0.883 | 0.986 | 0.755 | 22.4 |
| section_aware + hybrid | 0.858 | 0.880 | 0.986 | 0.782 | 26.4 |

Notes:
- An earlier dense run had 69 samples after one question failed judge output
  parsing after retries. The configuration was rerun; the final result has all
  70 samples and the table reports that complete run.
- Latency was measured while other applications were running during at least
  the first two runs (section_aware hybrid and dense). Latency is not a clean
  benchmark; quality metrics are unaffected because they depend only on the
  generated and retrieved text.

What the data supports:
1. **Retrieval helps.** `response_relevancy` is 0.489 without retrieval and
   0.719-0.782 with it. This is the only metric defined for both, and it
   measures whether an answer addresses the question, not whether it is correct.
2. **Hybrid beat dense on all four metrics** for `section_aware` (faithfulness
   +0.050, precision +0.035, recall +0.086, relevancy +0.063). The direction is
   consistent; significance has not been tested.
3. **Chunking strategy made little difference.** Across the three hybrid
   configurations the spread is 0.012 (faithfulness), 0.011 (precision),
   0.014 (recall) and 0.031 (relevancy). Sentence chunking is highest on
   faithfulness and precision, section-aware is highest on relevancy, and
   recall is tied between them. No strategy dominates, and with one run per
   configuration these gaps are within plausible judge noise.

---

## 3. Phase by phase

### Phase 0: Scoping and scaffold
Chose "autonomous AI agents" as the corpus topic and an RAG-plus-ablation
design over a plain chat-with-PDFs demo. The repository layout follows the
plan, with generated data (raw PDFs, parsed text, chunks, vector index)
gitignored and only `data/manifest.csv` tracked.

### Phase 1: Ingestion
- **Semantic Scholar 429s.** The unauthenticated pool is shared and rate-limits
  even single requests. Added exponential backoff and optional API-key support
  (`SEMANTIC_SCHOLAR_API_KEY`). Setting a placeholder string as the key caused
  403s until replaced with the real key.
- **Pagination cap.** The search endpoint returns 400 for offsets of 1000 or
  more. The first full run crashed there after collecting ~99 papers and wrote
  nothing, because results were saved only at the end. Fixed by stopping
  gracefully at the cap. The fix was later lost during a file re-sync and the
  same crash reappeared; it was caught on the next live run.
- **Publisher mirrors.** First ingestion run: 51 parsed, 51 failed of 102. Most
  failures were 403s from ACM, ScienceDirect, MDPI, Wiley. Filtering to papers
  with an arXiv ID helped (47/70) but not fully, because Semantic Scholar's
  `openAccessPdf.url` often pointed at a publisher mirror even for arXiv-indexed
  papers. **Fix:** build `https://arxiv.org/pdf/<arxiv_id>` directly. Result:
  70/70 parsed.
- **Zero-byte downloads** reported as success. Added a non-empty check before a
  download counts.
- **Whitespace bug** (stray space before paragraph breaks) caught by a unit test.

### Phase 2: Chunking
- Three strategies share one ~400-word budget so that comparisons isolate
  *boundary strategy*, not chunk size. Token counts are approximated by word
  counts.
- **Back matter.** References and acknowledgements made up roughly a third of
  the section-aware chunks. Added `strip_back_matter` upstream of chunking
  so all three strategies benefit equally. Chunk counts fell 27-28%
  (2571 to 1850, 2707 to 1978, 2672 to 1921). A 2,536-word "sentence" chunk
  (a references blob) shrank to 776 words; one residual outlier remains.
- **Header detection** matches Arabic-numbered or plain headers only. IEEE-style
  Roman numerals are missed, so about 5% of section-aware chunks are labelled
  `unknown`. After stripping, `unknown` rose from under 1.5% to 5%; that is
  papers whose bodies were previously mislabelled `preamble` now falling through
  to the honest fallback, not a regression in detection.

### Phase 3: Embedding and indexing
- BGE-M3 via sentence-transformers, ChromaDB with one collection per
  (strategy, embedder). Vector counts match chunk counts exactly. The
  `fixed_size` and `sentence` indexes were built only when the ablation needed
  them; the first attempt failed with "collection does not exist".
- Sanity queries returned results spread across many papers, and a
  "limitations" query returned a chunk from a Limitations section first.
- **GPU attempt.** The installed `torch` was a CPU-only build. After installing a
  CUDA 12.6 build (RTX 3050 Ti Laptop, 4 GB VRAM), `torch.cuda.is_available()`
  was true and a tensor allocated on the GPU, but the embedding script exited
  silently with code `-1073741819` (0xC0000005, access violation).
  `KMP_DUPLICATE_LIB_OK=TRUE` did not help. **Reverted to CPU; the root cause
  was not diagnosed.**
- The retrieval wrapper for BM25 was first tested against a stand-in library
  and only later verified with the real one.

### Phase 4: Retrieval and generation
- Dense (Chroma), BM25, and hybrid via Reciprocal Rank Fusion (k=60, 20
  candidates per system, top 5). RRF was chosen because dense distances and BM25
  scores are on incomparable scales; RRF uses ranks only. On a test query dense
  and hybrid shared 3 of 5 retrieved chunks, so the modes do behave differently.
- **Retriever reload bug.** The retriever, and with it BGE-M3, was rebuilt for
  every question. Building it once cut average latency from 14.6 s to 4.6 s
  (-69%) on a 5-question Groq-era test. This also showed that earlier latency
  numbers had been inflated.
- **Groq model churn.** `llama-3.3-70b-versatile` returned 404 (deprecated
  upstream). Switched to `qwen/qwen3.6-27b`. Its reasoning output leaked into
  answers as `<think>` blocks and truncated an answer mid-sentence at the default
  token limit. Fixed by stripping `<think>` blocks and raising `max_tokens`.
- The "answer only from the excerpts, otherwise say so" guard refused an
  off-topic question correctly, an easy case. In-domain failures were left to
  be measured by the evaluation.

### Phase 5a: Evaluation set
- Candidates were drafted by an LLM (a Qwen-family model via Groq), one per
  sampled chunk (seeded stratified sample across papers), then reviewed one at a
  time with a CLI that saves after every decision.
- **Meta-reference bug.** Drafted questions said "the excerpt..." instead of
  reading like a real user question. Fixed in the prompt plus an automatic
  rejection filter. The batch was regenerated.
- **Preview flaw.** The review tool shows only the first 500 characters of the
  source chunk. Two candidates in the first batch were deleted from a
  truncated preview as "ungrounded", and that call was probably unsafe: the first
  such question later reappeared in the regenerated set and was kept. The
  reported error rate depends on how often the full chunk in `eval_set.jsonl`
  was consulted instead of the preview.
- **Lost work, twice.** A late rate-limit crash discarded 49 drafted candidates
  because they were saved only at the end. Fixed with incremental, resumable
  writes; the next rate limit stopped at 65/70 with nothing lost, and the run was
  resumed to 70.
- **Result:** 70 questions, 1 needed editing, 0 deleted (1.4%).

### Phase 5b: Evaluation infrastructure
This was the largest source of friction.

**Ragas.**
- Ragas' newer `llm_factory` API (0.4+) turned out to have a live upstream bug:
  an unconditional import of Google's `ChatVertexAI` crashes on import for users
  without it, because `langchain-community` no longer ships that module. Pinning
  to 0.3.9 did not avoid it in this environment.
- **Fix:** `eval/_ragas_compat.py` registers a stub module before Ragas imports.
  The technique was validated on a reproduction of the failure shape before use.
  The project uses `ragas==0.3.9` with the older `LangchainLLMWrapper` pattern.
  Delete the shim once upstream is fixed.
- `ResponseRelevancy` generates several candidate questions per answer through
  multi-completion (`n>1`) requests, which Groq rejects. Set `strictness=1`.
  This was kept after moving to Ollama so that every configuration is scored
  identically (see Section 6).

**Groq limits that forced the move to local models.**
- The judge model's 200K tokens-per-day quota (a rolling window) was exhausted
  repeatedly, leaving a 70-question run needing days.
- A per-request output cap of 1,000 tokens conflicted with reasoning models that
  spend tokens thinking before answering. `gpt-oss-20b` once returned an empty
  answer after consuming its whole 800-token budget; a `NaN` faithfulness score
  from that case also poisoned the aggregate mean until `NaN` was excluded.
  `reasoning_effort="low"` helped.
- A configuration slip pointed generation at a safety-classifier model that
  cannot do chat completions; every request failed with a 400 until reverted.
- The evaluation runner now stops cleanly on a rate limit without recording the
  failed question. Previously a half-failed question could be saved with null
  scores and then treated as "done" on resume.

**Local Ollama.**
- Generation and judging moved to Ollama, which removes quotas.
- **Judge reliability** (Ragas needs well-formed structured output):

  | Judge | Result on a 5-question test |
  |---|---|
  | `phi4-mini` | 0-1 of 5 fully scored |
  | `qwen2.5:7b` | 2 of 5 |
  | `qwen2.5:14b` with an 8192-token context | 5 of 5, then 70/70 on later runs |

- **Context window.** Ollama's OpenAI-compatible endpoint is reported (across
  multiple upstream issues) to ignore client-side `num_ctx` and fall back to a
  default of about 4096 tokens. A judge prompt (question, five chunks, answer,
  Ragas instructions) can plausibly reach 3-4k tokens, leaving little room for
  structured output. Fixed by baking `num_ctx 8192` into custom Modelfiles
  (`ollama/Modelfile.gen`, `ollama/Modelfile.judge`).
  **Caveat:** two things changed together (7B to 14B, and 4096 to 8192 context),
  so which one fixed the parsing failures was never isolated.
- **Generator/judge mix-up.** For a while both roles pointed at the same model.
  The results filename gave it away. On a 5-question test, faithfulness fell
  from 0.96 (same model) to 0.63 (separate models). That comparison is
  confounded, because the generator also changed from 14B to 3B, so it cannot be
  attributed to removing self-preference bias alone. The same-model result was
  discarded.

**Baseline mode (`--mode none`).** Skips retrieval and uses a plain
"answer from your own knowledge" prompt. Faithfulness and context precision/recall
are not computed, since they are trivially undefined without retrieved context.

**Environment problems.**
- Windows Smart App Control blocked a compiled NumPy DLL (`mtrand`) after the
  Python environment was changed, breaking test collection. Disabled Smart App
  Control.
- The CI install step listed only the Phase 1 dependencies and silently stopped
  matching the project; fixed by installing from `requirements.txt`.
- PowerShell mangles nested quotes in `python -c "..."`; throwaway scripts were
  used instead. Piping into `Tee-Object` buffers output; `python -u` fixes it.

### Phases 6-7
- A Streamlit demo and dashboard guide is written (`docs/STREAMLIT_GUIDE.md`);
  the app itself is not built.
- Hugging Face Spaces: current Hub documentation says Gradio and Docker Spaces
  need a paid plan and only Static Spaces are free, so the roadmap's "free CPU
  Space" is no longer an option for this app.

---

## 4. Zero-score analysis

Across the four retrieval-backed runs, 68 metric scores were exactly 0.0.
48 of them (about 71%) were `response_relevancy`. Two questions, `eval_0022`
and `eval_0042`, scored 0.0 on `response_relevancy` in all four configurations.
`eval_0004` (the MATH-dataset question) scored zeros on faithfulness and the
context metrics only in the section-aware configurations.

The cross-configuration cases point to a property of the question or the
generated answer rather than of retrieval or chunking. The `eval_0004` pattern
points to something chunk-dependent. Neither explanation has been verified
(Section 5).

---

## 5. Hypotheses that are not verified

1. **Why `response_relevancy` is the metric that hits 0.0.** It scores an answer
   by asking the judge to reconstruct a question from it and comparing embeddings.
   With `strictness=1` a single poor reconstruction can drive the score to zero,
   which would hit short or hedged answers hardest. *To check:* read the answers
   for `eval_0022` and `eval_0042`; re-score just those with `strictness=3`.
2. **Why `eval_0004` fails only in section-aware configurations.** One guess is
   that section-aware boundaries split the needed passage. *To check:* inspect the
   chunks retrieved for that question under each strategy.
3. **Why hybrid beats dense.** The usual explanation is that BM25 catches exact
   terms (benchmark and model names) that embeddings blur. Not tested here.
4. **Which of "14B" or "8192 context" fixed the judge.** Not isolated. *To
   check:* build `qwen2.5-7b-8k` and rerun the 5-question test.
5. **Why baseline latency is higher** despite doing less work. The baseline emits
   about twice as many completion tokens (about 310 vs about 140), which is
   consistent with slower token-by-token decoding dominating, but timing was
   taken on a loaded machine and was not isolated.
6. **The GPU crash root cause.** Undiagnosed.

---

## 6. Limitations to keep in mind when reading the results

- **One run per configuration, no confidence intervals.** The judge is an LLM at
  temperature 0.1; differences of a few hundredths should not be read as real.
  The same 70 questions are used everywhere, so a paired test on per-question
  scores (bootstrap or Wilcoxon) would be the right next step.
- **The eval set was drafted from section-aware chunks**, which could favour that
  strategy's chunk boundaries. Questions are single-passage; there are no
  multi-passage or multi-hop questions.
- **LLM-drafted references.** Reference answers were drafted by an LLM and reviewed
  by one person, with the 500-character-preview caveat above.
- **The judge is a local 14B model** and judges only a 3B generator's output. LLM
  judges have known biases, and absolute scores are not comparable to papers that
  use stronger judges.
- **`response_relevancy` uses `strictness=1`**, not the recommended 3-5, a holdover
  from a Groq API constraint. Kept constant for consistency.
- **The baseline comparison rests on one metric**, and `response_relevancy`
  measures topical relevance, not correctness. No correctness-versus-reference
  metric was run on the baseline.
- **Corpus is broad and partly tangential** (it came from a keyword search). Some
  papers (for example on explainability or COMPAS bail decisions) are only loosely
  about autonomous agents.
- **Back-matter stripping is heuristic.** Papers without a detected "References"
  header keep their back matter.
- **CPU-only.** All embedding and inference ran on CPU after the GPU attempt failed.
- **Latency** was measured under background load (Section 2).
- **Small generator hallucinates.** In an early 5-question test `llama3.2:3b`
  fabricated a limitation of the MATH dataset and a supporting quote, which the
  judge scored 0 on faithfulness and context metrics; a larger model hedged
  correctly on the same retrieval.

---

## 7. Lessons

1. **Test small before scaling.** A 5-question run caught most problems that would
   have cost a full run each.
2. **A passing smoke test is not reliability.** Two judge models passed the
   single-sample smoke test and then failed most real questions.
3. **Save incrementally and make runs resumable.** Work was lost three times
   before this became standard (corpus fetch, eval-set drafting, evaluation).
4. **Check infrastructure limits before blaming the model**: rate limits, output
   caps, context windows.
5. **Keep comparison variables fixed and visible.** One shared chunk budget,
   separate generator and judge, and the generator's name in every results
   filename.
6. **Third-party APIs drift.** Models were deprecated mid-project, Ragas changed
   API shape, and a dependency broke on import. Check current documentation
   before building on a remembered API.
7. **Distrust your own confident explanations.** Several explanations given
   during the project were stronger than the evidence (the judge-fix attribution,
   the self-preference attribution, early review calls made from truncated
   previews). Section 5 exists to keep those honest.

---

## 8. Future work

- Paired significance tests or bootstrap confidence intervals on the existing
  per-question scores.
- Re-score with `strictness=3` and compare against the current numbers.
- Isolate the judge fix with a 7B model at 8192 context.
- Build the Streamlit app and a static dashboard (`docs/STREAMLIT_GUIDE.md`).
- A reranker, a second embedding model, and multi-passage questions.
- A correctness-versus-reference metric so the baseline can be compared on more
  than relevancy.
- Diagnose the GPU crash.

---

## 9. Development process

The project was built iteratively with an AI assistant (Claude) that wrote and
debugged code alongside the author. All code was run and verified on the
author's machine, tests cover the pure-logic components, and live behaviour was
checked with small runs before full ones. Remove this section if it does not
reflect how you want to present the work.
