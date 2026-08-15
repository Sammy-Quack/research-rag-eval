"""Download PDFs listed in the manifest to data/raw/.

Only intended for open-access, stable URLs (arXiv, OpenReview, ACL Anthology).
See docs/architecture.md for why paywalled/signed-URL sources are excluded.
"""

import time
from pathlib import Path

import requests

USER_AGENT = "research-rag-eval-bot/0.1 (portfolio project; contact: you@example.com)"
TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2


def download_pdf(url: str, dest_path: Path, force: bool = False) -> bool:
    """Download a single PDF. Returns True on success (or already-cached), False on failure.

    Skips download if dest_path already exists and force=False, so re-running
    ingestion on a partially-built corpus is cheap and idempotent.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and not force:
        return True

    headers = {"User-Agent": USER_AGENT}

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, headers=headers, timeout=TIMEOUT_SECONDS, stream=True)
            response.raise_for_status()

            tmp_path = dest_path.with_suffix(".pdf.part")
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            if tmp_path.stat().st_size == 0:
                print(f"  [attempt {attempt}/{RETRY_ATTEMPTS}] got 200 OK but 0 bytes for {url}")
                tmp_path.unlink()
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue

            tmp_path.rename(dest_path)
            return True

        except requests.RequestException as exc:
            print(f"  [attempt {attempt}/{RETRY_ATTEMPTS}] failed for {url}: {exc}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    return False