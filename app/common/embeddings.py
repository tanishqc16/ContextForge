from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_text(text: str) -> list[float]:
    """Local embedding, no LLM call — this never touches the Gemini quota."""
    vector = _model().encode(text, normalize_embeddings=True)
    return vector.tolist()  # type: ignore[no-any-return]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Vectors from embed_text are already unit-normalized, so the dot
    product alone equals cosine similarity."""
    return sum(x * y for x, y in zip(a, b, strict=True))
