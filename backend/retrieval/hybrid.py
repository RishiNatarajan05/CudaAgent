"""Hybrid (vector + BM25) search via Reciprocal Rank Fusion."""
from __future__ import annotations
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .vector_search import get_or_create_collection, query_collection
from .keyword_search import BM25Index
from ..ingestion.embedder import embed

REPOS_DIR = Path(os.environ.get("REPOS_DIR", "./data/repos"))


def _load_bm25(repo_id: str) -> Optional[BM25Index]:
    p = REPOS_DIR / repo_id / "bm25.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _kernel_chunk_boost(metadata: dict) -> float:
    """Boost factor for prioritizing kernel chunks in RRF."""
    return 1.25 if metadata.get("chunk_type") == "kernel" else 1.0


def hybrid_search(repo_id: str, query: str, k: int = 8, k_rrf: int = 60) -> list[dict]:
    """Hybrid retrieval over an indexed repo."""
    # Vector
    coll = get_or_create_collection(f"repo_{repo_id}")
    qvec = embed([query])[0]
    vec_hits = query_collection(coll, qvec, k=max(k * 2, 16))

    # BM25
    bm25 = _load_bm25(repo_id)
    bm25_hits = bm25.search(query, k=max(k * 2, 16)) if bm25 else []

    # RRF fusion
    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, str] = {}
    metas: dict[str, dict] = {}

    for rank, h in enumerate(vec_hits, start=1):
        cid = h["id"]
        boost = _kernel_chunk_boost(h.get("metadata") or {})
        scores[cid] += boost / (k_rrf + rank)
        docs[cid] = h["document"]
        metas[cid] = h.get("metadata") or {}

    for rank, h in enumerate(bm25_hits, start=1):
        cid = h["id"]
        boost = _kernel_chunk_boost(metas.get(cid, {}))
        scores[cid] += boost / (k_rrf + rank)
        docs.setdefault(cid, h["document"])

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [
        {
            "id": cid,
            "score": score,
            "document": docs.get(cid, ""),
            "metadata": metas.get(cid, {}),
        }
        for cid, score in ranked
    ]


def search_cuda_docs(query: str, k: int = 5) -> list[dict]:
    coll = get_or_create_collection("cuda_knowledge_base")
    qvec = embed([query])[0]
    return query_collection(coll, qvec, k=k)
