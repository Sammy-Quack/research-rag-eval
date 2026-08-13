"""Query Semantic Scholar for open-access papers matching a topic and append
them to data/manifest.csv.

Only keeps results with a real, direct openAccessPdf.url — this is what
filters out IEEE/Elsevier/SSRN-style paywalled results automatically.

Usage:
    python -m src.ingest.fetch_corpus --query "autonomous AI agents" --limit 100
"""

import argparse
import csv
import time
from pathlib import Path

import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,externalIds,openAccessPdf,venue"
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 1.1  # be polite to the unauthenticated rate limit
MANIFEST_COLUMNS = ["id", "title", "source", "pdf_url", "license", "notes"]


def slugify(paper_id: str) -> str:
    return paper_id.replace("/", "_").replace(":", "_")


def fetch_open_access_papers(query: str, limit: int) -> list[dict]:
    results = []
    offset = 0

    while len(results) < limit:
        params = {
            "query": query,
            "fields": FIELDS,
            "limit": min(PAGE_SIZE, limit - len(results)),
            "offset": offset,
        }
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        papers = payload.get("data", [])

        if not papers:
            break

        for paper in papers:
            oa = paper.get("openAccessPdf")
            if oa and oa.get("url"):
                results.append(paper)

        offset += PAGE_SIZE
        if offset >= payload.get("total", 0):
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    return results[:limit]


def load_existing_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def append_to_manifest(papers: list[dict], manifest_path: Path) -> int:
    existing_ids = load_existing_ids(manifest_path)
    is_new_file = not manifest_path.exists()
    added = 0

    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        if is_new_file:
            writer.writeheader()

        for paper in papers:
            arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
            paper_id = slugify(arxiv_id) if arxiv_id else slugify(paper["paperId"])

            if paper_id in existing_ids:
                continue

            writer.writerow({
                "id": paper_id,
                "title": paper.get("title", "").strip(),
                "source": "arxiv" if arxiv_id else "semantic_scholar_oa",
                "pdf_url": paper["openAccessPdf"]["url"],
                "license": paper["openAccessPdf"].get("license", "unknown"),
                "notes": paper.get("venue", ""),
            })
            existing_ids.add(paper_id)
            added += 1

    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help='e.g. "autonomous AI agents"')
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--manifest", default="data/manifest.csv")
    args = parser.parse_args()

    print(f"Searching Semantic Scholar for: {args.query!r} (limit {args.limit})")
    papers = fetch_open_access_papers(args.query, args.limit)
    print(f"Found {len(papers)} open-access candidates")

    added = append_to_manifest(papers, Path(args.manifest))
    print(f"Added {added} new rows to {args.manifest} (duplicates skipped)")


if __name__ == "__main__":
    main()
