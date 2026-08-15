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


BACK_MATTER_PATTERN = re.compile(
    r"^\s*\d{0,2}\.?\s*(References|Acknowledg(?:e)?ments|Bibliography)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_back_matter(text: str) -> str:
    """Truncate everything from the first References/Acknowledgements/
    Bibliography header onward (citations, thank-yous, and anything after —
    typically appendices). Observed to be ~30% of a real corpus's chunks when
    left in, with no retrieval value. Done once here, upstream of chunking,
    so every chunking strategy benefits equally rather than only the ones
    that happen to be section-aware.
    """
    match = BACK_MATTER_PATTERN.search(text)
    if match:
        return text[:match.start()].rstrip()
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

def test_strip_back_matter_removes_references_and_after():
    text = "Intro text here.\n\nReferences\n[1] Someone et al. 2024.\n[2] Another."
    result = strip_back_matter(text)
    assert "References" not in result
    assert "Intro text here" in result


def test_strip_back_matter_no_header_returns_unchanged():
    text = "Just some text with no back matter section at all."
    assert strip_back_matter(text) == text


def test_strip_back_matter_stops_at_earliest_header():
    text = "Conclusion text.\n\nAcknowledgements\nThanks all.\n\nReferences\n[1] cite."
    result = strip_back_matter(text)
    assert "Acknowledgements" not in result
    assert "References" not in result
    assert "Conclusion text" in result