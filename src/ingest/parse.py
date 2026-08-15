"""Extract text and light metadata from a downloaded PDF.

Uses PyMuPDF (fitz) — fast, dependency-light, good enough for a first pass.
Known weak spots (documented, not hidden): multi-column layout, equations,
and tables often extract poorly or out of reading order. We flag suspected
cases rather than silently ship bad text — see clean.looks_like_multi_column_artifact.
"""

from pathlib import Path
from typing import TypedDict

import fitz  # PyMuPDF

from src.ingest.clean import (
    clean_text,
    collapse_whitespace,
    dehyphenate,
    looks_like_multi_column_artifact,
    strip_back_matter,
)

class ParsedPaper(TypedDict):
    paper_id: str
    num_pages: int
    raw_char_count: int
    text: str
    parse_warnings: list[str]


def parse_pdf(pdf_path: Path, paper_id: str) -> ParsedPaper:
    warnings: list[str] = []

    doc = fitz.open(pdf_path)
    raw_text = ""
    for page in doc:
        raw_text += page.get_text("text") + "\n"
    num_pages = doc.page_count
    doc.close()

    if not raw_text.strip():
        warnings.append("empty_extraction")

    if looks_like_multi_column_artifact(raw_text):
        warnings.append("possible_multi_column_scramble")

    cleaned = clean_text(raw_text)

    before_len = len(cleaned)
    cleaned = strip_back_matter(cleaned)
    if len(cleaned) == before_len:
        warnings.append("no_references_header_found")

    return ParsedPaper(
        paper_id=paper_id,
        num_pages=num_pages,
        raw_char_count=len(raw_text),
        text=cleaned,
        parse_warnings=warnings,
    )