"""Phase 1 entry point.

Reads data/manifest.csv -> downloads each PDF to data/raw/ (skips if cached)
-> parses + cleans text -> writes data/processed/<id>.json.

Usage:
    python -m src.ingest.run_ingestion
"""

import csv
import json
from pathlib import Path

from src.ingest.download import download_pdf
from src.ingest.parse import parse_pdf

MANIFEST_PATH = Path("data/manifest.csv")
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def run() -> None:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"No manifest at {MANIFEST_PATH} — run fetch_corpus.py first or add rows manually.")

    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ok, failed, warned = 0, 0, 0

    for row in rows:
        paper_id = row["id"]
        pdf_path = RAW_DIR / f"{paper_id}.pdf"

        print(f"[{paper_id}] downloading...")
        if not download_pdf(row["pdf_url"], pdf_path):
            print(f"[{paper_id}] DOWNLOAD FAILED, skipping")
            failed += 1
            continue

        print(f"[{paper_id}] parsing...")
        try:
            parsed = parse_pdf(pdf_path, paper_id)
        except Exception as exc:  # malformed PDF, corrupted download, etc.
            print(f"[{paper_id}] PARSE FAILED: {exc}")
            failed += 1
            continue

        parsed_out = {
            "paper_id": paper_id,
            "title": row["title"],
            "source": row["source"],
            "pdf_url": row["pdf_url"],
            "num_pages": parsed["num_pages"],
            "char_count": len(parsed["text"]),
            "parse_warnings": parsed["parse_warnings"],
            "text": parsed["text"],
        }

        out_path = PROCESSED_DIR / f"{paper_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed_out, f, ensure_ascii=False, indent=2)

        if parsed["parse_warnings"]:
            print(f"[{paper_id}] parsed with warnings: {parsed['parse_warnings']}")
            warned += 1
        else:
            print(f"[{paper_id}] OK ({parsed['num_pages']} pages, {len(parsed['text'])} chars)")
        ok += 1

    print(f"\nDone. {ok} parsed ({warned} with warnings), {failed} failed, {len(rows)} total.")


if __name__ == "__main__":
    run()
