"""Pure text-cleaning helpers for parsed PDF output.

Kept dependency-free and side-effect-free so they're trivial to unit test
without needing a real PDF or network access.
"""

import re


def dehyphenate(text: str) -> str:
    """Rejoin words broken across a line by a hyphen, e.g. 'trans-\nformer' -> 'transformer'.

    Common artifact of two-column academic PDF layouts.
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace/newlines into single spaces, trim ends."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)  # drop stray spaces hugging line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(raw_text: str) -> str:
    """Full cleaning pipeline applied to raw PyMuPDF text extraction."""
    text = dehyphenate(raw_text)
    text = collapse_whitespace(text)
    return text


def looks_like_multi_column_artifact(text: str, avg_line_len_threshold: int = 25) -> bool:
    """Heuristic flag: very short average line length often means a two-column
    PDF was read left-to-right across both columns instead of column-by-column,
    scrambling sentence order. Not a guarantee — just a red flag to log and
    manually spot-check.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    return avg_len < avg_line_len_threshold
