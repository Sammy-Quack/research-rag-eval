"""Section-aware chunking: detect common academic section headers via regex,
keep each section's content together as one chunk, and tag every chunk with
its section name. Sections that exceed the word budget get sub-split (using
the same fixed-size logic) rather than shipped as one giant chunk.

Caveat, stated plainly rather than hidden: PyMuPDF's plain-text extraction
drops font size/bold/layout formatting, so headers are detected from text
patterns alone (e.g. "3. Related Work" alone on its own line). This misses
non-standard header styles and the occasional false positive/negative.
Papers where no headers are detected at all fall back to a single "unknown"
section, sub-chunked like fixed_size — never silently drops content.
"""

import re

from src.chunking.base import Chunk, DEFAULT_CHUNK_SIZE_WORDS, make_chunk_id
from src.chunking.fixed_size import chunk_fixed_size

SECTION_HEADER_PATTERN = re.compile(
    r"^\s*\d{0,2}\.?\s*(Abstract|Introduction|Related Work|Background|"
    r"Method(?:ology|s)?|Approach|Experiments?|Evaluation|Results|"
    r"Discussion|Limitations|Conclusion(?:s)?|Acknowledg(?:e)?ments|References)"
    r"\s*$",
    re.IGNORECASE | re.MULTILINE,
)

MAX_SECTION_CHUNK_WORDS = DEFAULT_CHUNK_SIZE_WORDS


def _find_sections(text: str) -> list[tuple[str, int, int]]:
    """Returns (section_name, char_start, char_end) tuples covering the whole text."""
    matches = list(SECTION_HEADER_PATTERN.finditer(text))

    if not matches:
        return [("unknown", 0, len(text))]

    sections = []
    for i, match in enumerate(matches):
        name = match.group(1).strip().title()
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((name, content_start, content_end))

    if matches[0].start() > 0:
        sections.insert(0, ("preamble", 0, matches[0].start()))

    return sections


def chunk_section_aware(text: str, paper_id: str) -> list[Chunk]:
    sections = _find_sections(text)
    chunks: list[Chunk] = []
    index = 0

    for section_name, s_start, s_end in sections:
        section_text = text[s_start:s_end]
        word_count = len(section_text.split())

        if word_count == 0:
            continue

        if word_count <= MAX_SECTION_CHUNK_WORDS:
            chunks.append(Chunk(
                chunk_id=make_chunk_id(paper_id, "section_aware", index),
                paper_id=paper_id,
                strategy="section_aware",
                text=section_text,
                char_start=s_start,
                char_end=s_end,
                word_count=word_count,
                section=section_name,
            ))
            index += 1
        else:
            # too big for one chunk — sub-split, but keep the section tag
            # and offset positions back into the full document
            sub_chunks = chunk_fixed_size(
                section_text, paper_id,
                chunk_size_words=MAX_SECTION_CHUNK_WORDS, overlap_words=40,
            )
            for sub in sub_chunks:
                chunks.append(Chunk(
                    chunk_id=make_chunk_id(paper_id, "section_aware", index),
                    paper_id=paper_id,
                    strategy="section_aware",
                    text=sub["text"],
                    char_start=s_start + sub["char_start"],
                    char_end=s_start + sub["char_end"],
                    word_count=sub["word_count"],
                    section=section_name,
                ))
                index += 1

    return chunks