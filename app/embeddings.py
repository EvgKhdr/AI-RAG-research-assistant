"""Local sentence-transformers embeddings. The model is loaded once per process."""
from functools import lru_cache

from config import EMBED_MODEL


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # heavy import, keep it lazy

    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _model().encode(texts, batch_size=32, normalize_embeddings=True).tolist()
