from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.embeddings import cosine_similarity, embed_text
from app.knowledge.models import Entity

# Below this similarity, we don't trust the match enough to answer at all —
# an honest "no knowledge found" beats guessing at the wrong entity.
MIN_ENTITY_MATCH_SIMILARITY = 0.45


def find_best_matching_entity(db: Session, query_text: str) -> Entity | None:
    """Find the canonical Entity a free-text question is most likely about.

    Simple by design: embed the question, compare it against each entity's
    canonical name. Fine at this dataset's scale; a larger one would want
    this done via pgvector similarity search in SQL instead of in Python.
    """
    entities = list(db.execute(select(Entity)).scalars())
    if not entities:
        return None

    query_embedding = embed_text(query_text)
    scored = [
        (cosine_similarity(query_embedding, embed_text(e.canonical_name)), e) for e in entities
    ]
    best_similarity, best_entity = max(scored, key=lambda pair: pair[0])

    if best_similarity < MIN_ENTITY_MATCH_SIMILARITY:
        return None
    return best_entity
