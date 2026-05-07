"""Indexes downloaded CUDA documentation into a ChromaDB collection."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional

from .embedder import embed
from ..retrieval.vector_search import (
    get_or_create_collection,
    upsert_documents,
)
from ..knowledge.antipatterns import all_patterns
from ..knowledge.cuda_concepts import all_concepts

CUDA_DOCS_DIR = Path(os.environ.get("CUDA_DOCS_DIR", "./data/cuda_docs"))
COLLECTION = "cuda_knowledge_base"


def _chunk_doc(text: str, max_chars: int = 1500) -> list[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def index_local_knowledge() -> int:
    """Always-available fallback index from the in-process knowledge base."""
    coll = get_or_create_collection(COLLECTION)
    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for ap in all_patterns():
        body = (
            f"{ap.name} ({ap.severity})\n\n"
            f"{ap.description}\n\nImpact: {ap.impact}\n\nFix: {ap.fix}\n\n"
            f"Reference: {ap.nvidia_doc_reference}"
        )
        docs.append(body)
        metadatas.append({
            "source": "antipattern_kb",
            "antipattern_id": ap.id,
            "section": ap.name,
            "url": "",
        })
        ids.append(f"ap::{ap.id}")

    for c in all_concepts():
        body = (
            f"{c.name}\n\n{c.one_line_summary}\n\n{c.detailed_explanation}\n\n"
            f"Common misconception: {c.common_misconception}\n\n"
            f"Architecture note: {c.nvidia_arch_note}"
        )
        docs.append(body)
        metadatas.append({
            "source": "concept_kb",
            "section": c.name,
            "url": "",
        })
        ids.append(f"concept::{c.name}")

    if docs:
        embeddings = embed(docs)
        upsert_documents(coll, ids, docs, metadatas, embeddings)
    return len(docs)


def index_downloaded_docs() -> int:
    """Index any HTML/text files placed in CUDA_DOCS_DIR by the downloader."""
    if not CUDA_DOCS_DIR.exists():
        return 0
    coll = get_or_create_collection(COLLECTION)

    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for f in CUDA_DOCS_DIR.rglob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        rel = f.relative_to(CUDA_DOCS_DIR)
        for i, c in enumerate(_chunk_doc(text)):
            docs.append(c)
            metadatas.append({
                "source": str(rel.parts[0]) if rel.parts else "cuda_docs",
                "section": str(rel),
                "url": "",
            })
            ids.append(f"doc::{rel}::{i}")

    if docs:
        embeddings = embed(docs)
        upsert_documents(coll, ids, docs, metadatas, embeddings)
    return len(docs)
