"""Shared types, constants, and helpers used by every chunking strategy.

Chunk size is kept as a single shared constant across all three strategies
deliberately — the whole point of the later ablation study is to isolate
*boundary strategy* as the variable, not chunk size. If each strategy used
a different size, any quality difference you measured later would be
confounded and you couldn't attribute it to the thing you actually changed.
"""

from typing import Optional, TypedDict

# ~512 tokens at the commonly-cited ~1.3 tokens/word ratio for English text.
# Approximate, not an exact tokenizer count — documented limitation, see
# fixed_size.py docstring.
DEFAULT_CHUNK_SIZE_WORDS = 400
DEFAULT_OVERLAP_WORDS = 50


class Chunk(TypedDict):
    chunk_id: str
    paper_id: str
    strategy: str
    text: str
    char_start: int
    char_end: int
    word_count: int
    section: Optional[str]


def make_chunk_id(paper_id: str, strategy: str, index: int) -> str:
    return f"{paper_id}::{strategy}::{index:04d}"