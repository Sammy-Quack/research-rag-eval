"""Query Semantic Scholar for open-access papers matching a topic and append
them to data/manifest.csv.

Only keeps results with a real, direct openAccessPdf.url — this is what
filters out IEEE/Elsevier/SSRN-style paywalled results automatically.

Usage:
    python -m src.ingest.fetch_corpus --query "autonomous AI agents" --limit 100
"""

import argparse
import csv
import os
import time
from pathlib import Path

import requests

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,year,externalIds,openAccessPdf,venue"
PAGE_SIZE = 100
REQUEST_DELAY_SECONDS = 3  # be polite to the unauthenticated rate limit
MAX_RETRIES = 6
INITIAL_BACKOFF_SECONDS = 10
MANIFEST_COLUMNS = ["id", "title", "source", "pdf_url", "license", "notes"]

# Optional: a free key (https://www.semanticscholar.org/product/api#api-key) raises
# the shared unauthenticated rate limit a lot. Not required, just makes 429s rarer.
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")


def slugify(paper_id: str) -> str:
    return paper_id.replace("/", "_").replace(":", "_")


def _get_with_retry(params: dict) -> dict:
    """GET with exponential backoff on 429 — the unauthenticated S2 pool is shared
    globally and rate-limits even single, isolated requests fairly often."""
    headers = {"x-api-key": API_KEY} if API_KEY else {}
    backoff = INITIAL_BACKOFF_SECONDS

    for attempt in range(1, MAX_RETRIES + 1):
        response = requests.get(API_URL, params=params, headers=headers, timeout=30)

        if response.status_code == 429:
            wait = int(response.headers.get("Retry-After", backoff))
            print(f"  rate limited (429) — waiting {wait}s, retry {attempt}/{MAX_RETRIES}")
            time.sleep(wait)
            backoff *= 2
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(
        "Still rate-limited after all retries. Get a free API key and set it as "
        "the SEMANTIC_SCHOLAR_API_KEY env var: https://www.semanticscholar.org/product/api#api-key"
    )


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
        try:
            payload = _get_with_retry(params)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 400 and offset >= 900:
                print(f"  hit Semantic Scholar's pagination cap (max offset ~1000) "
                      f"— stopping with {len(results)} open-access papers collected")
                break
            raise
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