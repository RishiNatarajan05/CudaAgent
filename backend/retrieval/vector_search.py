"""ChromaDB-backed vector search."""
from __future__ import annotations
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import chromadb
from chromadb.config import Settings

CHROMA_DIR = Path(os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma"))


@lru_cache(maxsize=1)
def _client() -> chromadb.api.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def get_or_create_collection(name: str):
    return _client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _sanitize_metadata(meta: dict) -> dict:
    """Chroma metadata values must be primitives. Encode complex values as JSON strings."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            try:
                out[k] = json.dumps(v)
            except Exception:
                out[k] = str(v)
    return out


def upsert_documents(coll, ids: list[str], docs: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
    if not ids:
        return
    coll.upsert(
        ids=ids,
        documents=docs,
        metadatas=[_sanitize_metadata(m) for m in metadatas],
        embeddings=embeddings,
    )


def upsert_chunks(coll, chunks, embeddings: list[list[float]]) -> None:
    if not chunks:
        return
    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = []
    for c in chunks:
        m = {
            "chunk_type": c.chunk_type,
            "name": c.name,
            "filepath": c.filepath,
            "rel_path": c.rel_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
        }
        m.update(c.metadata or {})
        metas.append(_sanitize_metadata(m))
    coll.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)


def query_collection(coll, query_embedding: list[float], k: int = 8) -> list[dict]:
    res = coll.query(query_embeddings=[query_embedding], n_results=k)
    out = []
    for i, _id in enumerate(res.get("ids", [[]])[0]):
        out.append({
            "id": _id,
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i] or {},
            "distance": (res.get("distances") or [[None]])[0][i],
        })
    return out
