"""Local embedding generation via sentence-transformers."""
from __future__ import annotations
from functools import lru_cache
from typing import List

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    m = _model()
    vecs = m.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return vecs.tolist()


def embedding_dim() -> int:
    return _model().get_sentence_embedding_dimension()
