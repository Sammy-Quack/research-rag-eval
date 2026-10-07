"""Builds a grounded prompt from retrieved chunks, with numbered citations
the model is instructed to reference for every claim. The system prompt is
the actual "don't hallucinate" guard -- it explicitly tells the model to
say so when the excerpts don't contain enough to answer, rather than
silently falling back on its own outside knowledge.
"""

SYSTEM_PROMPT = (
    "You are a research assistant answering questions using ONLY the numbered "
    "excerpts provided below, drawn from a corpus of research papers. "
    "Cite the excerpt number(s) you used for each claim, like [1] or [2][3]. "
    "If the excerpts do not contain enough information to answer the question, "
    "say so explicitly instead of guessing or using outside knowledge."
)

# Used only for the no-retrieval baseline (--mode none): no excerpts exist to
# cite, so the model answers from its own parametric knowledge instead. This
# is the control condition -- comparing this against the retrieval-backed
# modes is what actually proves RAG adds value over the raw model.
BASELINE_SYSTEM_PROMPT = (
    "Answer the following question directly, using your own knowledge. "
    "You have not been given any external documents or context for this question."
)


def build_prompt(query: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] (paper: {c['paper_id']}, section: {c.get('section', 'unknown')})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    return (
        f"Excerpts:\n\n{excerpts}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the excerpts above, with citations."
    )