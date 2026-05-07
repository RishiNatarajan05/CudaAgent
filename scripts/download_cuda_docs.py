"""Download + index Nvidia CUDA documentation. Falls back to local KB on failure.

Usage: uv run python scripts/download_cuda_docs.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.ingestion.docs_indexer import (  # noqa: E402
    index_local_knowledge,
    index_downloaded_docs,
    CUDA_DOCS_DIR,
)

SOURCES = [
    ("cuda_programming_guide",
     "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"),
    ("cuda_best_practices",
     "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html"),
]


def _fetch_and_save(name: str, url: str) -> bool:
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        # split by section header
        target_dir = CUDA_DOCS_DIR / name
        target_dir.mkdir(parents=True, exist_ok=True)
        sections = []
        current_title = "intro"
        current_buf: list[str] = []
        for el in soup.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
            if el.name in ("h1", "h2", "h3"):
                if current_buf:
                    sections.append((current_title, "\n\n".join(current_buf)))
                    current_buf = []
                current_title = el.get_text(strip=True)[:80] or "section"
            else:
                txt = el.get_text(" ", strip=True)
                if txt:
                    current_buf.append(txt)
        if current_buf:
            sections.append((current_title, "\n\n".join(current_buf)))

        for i, (title, body) in enumerate(sections):
            safe = "".join(c if c.isalnum() else "_" for c in title)[:60]
            (target_dir / f"{i:03d}_{safe}.txt").write_text(
                f"# {title}\nSource: {url}\n\n{body}", encoding="utf-8"
            )
        print(f"  saved {len(sections)} sections from {url}")
        return True
    except Exception as e:
        print(f"  failed to fetch {url}: {e}")
        return False


def main():
    print("Downloading CUDA documentation...")
    CUDA_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    successes = 0
    for name, url in SOURCES:
        print(f"-> {name}")
        if _fetch_and_save(name, url):
            successes += 1

    print("Indexing local antipattern + concept knowledge base...")
    n_local = index_local_knowledge()
    print(f"  indexed {n_local} local KB entries")

    print("Indexing downloaded docs (if any)...")
    n_docs = index_downloaded_docs()
    print(f"  indexed {n_docs} doc chunks")

    if successes == 0 and n_docs == 0:
        print("WARNING: docs download failed — relying on local knowledge base only.")
    print("Done.")


if __name__ == "__main__":
    main()
