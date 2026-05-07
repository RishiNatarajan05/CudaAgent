"""BM25 keyword search."""
from __future__ import annotations
import re
from typing import List

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunk_ids: list[str], texts: list[str], tokenized: list[list[str]]):
        self.chunk_ids = chunk_ids
        self.texts = texts
        self.tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int = 8) -> list[dict]:
        if not self._bm25:
            return []
        toks = _tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            {"id": self.chunk_ids[i], "score": float(scores[i]), "document": self.texts[i]}
            for i in order
            if scores[i] > 0
        ]


def build_bm25_index(chunks) -> BM25Index:
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    toks = [_tokenize(t) for t in texts]
    return BM25Index(ids, texts, toks)
