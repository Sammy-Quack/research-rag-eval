"""Fixed-size chunking: split text into ~N-word windows with overlap,
completely ignoring document structure. The naive baseline every other
strategy gets compared against.

Chunk size is measured in words as a cheap proxy for tokens, not an exact
tokenizer count (would need tiktoken or similar) — good enough for a
comparative study between strategies, not for production context-window
budgeting down to the exact token.
"""

import re

from src.chunking.base import (
    Chunk,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_OVERLAP_WORDS,
    make_chunk_id,
)

WORD_PATTERN = re.compile(r"\S+")


def chunk_fixed_size(
    text: str,
    paper_id: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Chunk]:
    assert overlap_words < chunk_size_words, "overlap must be smaller than chunk size"

    words = list(WORD_PATTERN.finditer(text))
    if not words:
        return []

    chunks: list[Chunk] = []
    start_idx = 0
    index = 0

    while start_idx < len(words):
        end_idx = min(start_idx + chunk_size_words, len(words))
        first_word, last_word = words[start_idx], words[end_idx - 1]
        char_start, char_end = first_word.start(), last_word.end()

        chunks.append(Chunk(
            chunk_id=make_chunk_id(paper_id, "fixed_size", index),
            paper_id=paper_id,
            strategy="fixed_size",
            text=text[char_start:char_end],
            char_start=char_start,
            char_end=char_end,
            word_count=end_idx - start_idx,
            section=None,
        ))

        index += 1
        if end_idx == len(words):
            break
        start_idx = end_idx - overlap_words

    return chunks