"""Sentence-based chunking: split on sentence boundaries, then greedily group
whole sentences into chunks up to the shared word budget, with a small
sentence-level overlap between consecutive chunks for continuity.

Sentence splitting here is a lightweight regex heuristic, not a proper NLP
sentence tokenizer — it will mis-split (or fail to split) on abbreviations,
citations like "et al." or "Fig. 2", and decimal numbers, which are common
in academic text. Documented limitation, not hidden. A natural next step
would be swapping in a real sentence tokenizer and checking whether it
actually changes downstream retrieval quality — if it doesn't, that's a
useful finding too.
"""

import re

from src.chunking.base import (
    Chunk,
    DEFAULT_CHUNK_SIZE_WORDS,
    make_chunk_id,
)

# Boundary: a period/!/? followed by whitespace and a capital letter or digit.
# Deliberately conservative — favors under-splitting (missing a boundary)
# over over-splitting on abbreviations like "Fig." mid-sentence.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

DEFAULT_OVERLAP_SENTENCES = 2


def _split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    spans = []
    start = 0
    for match in SENTENCE_SPLIT_PATTERN.finditer(text):
        end = match.start()
        sentence = text[start:end]
        if sentence.strip():
            spans.append((sentence, start, end))
        start = match.end()
    if start < len(text):
        tail = text[start:]
        if tail.strip():
            spans.append((tail, start, len(text)))
    return spans


def chunk_sentence_based(
    text: str,
    paper_id: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    sentences = _split_sentences_with_spans(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    index = 0
    i = 0
    n = len(sentences)

    while i < n:
        current: list[tuple[str, int, int]] = []
        current_word_count = 0
        j = i

        while j < n:
            sentence, s_start, s_end = sentences[j]
            word_count = len(sentence.split())
            # a single sentence longer than the budget still gets its own
            # chunk whole — this strategy never splits mid-sentence
            if current and current_word_count + word_count > chunk_size_words:
                break
            current.append((sentence, s_start, s_end))
            current_word_count += word_count
            j += 1

        chunk_start = current[0][1]
        chunk_end = current[-1][2]
        chunks.append(Chunk(
            chunk_id=make_chunk_id(paper_id, "sentence", index),
            paper_id=paper_id,
            strategy="sentence",
            text=text[chunk_start:chunk_end],
            char_start=chunk_start,
            char_end=chunk_end,
            word_count=current_word_count,
            section=None,
        ))
        index += 1

        if j >= n:
            break
        # next window starts `overlap_sentences` back from j; max(...) with
        # i+1 guarantees forward progress every iteration, no infinite loop
        i = max(i + 1, j - overlap_sentences)

    return chunks