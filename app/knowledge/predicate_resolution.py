import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.embeddings import cosine_similarity, embed_text
from app.common.llm_client import call_structured
from app.common.text_normalization import normalize_text
from app.knowledge.models import Entity, Fact
from app.knowledge.schemas import PredicateMatchDecision

logger = logging.getLogger(__name__)

# Same shape as entity resolution's bands, and for the same reason: short
# attribute labels ("status", "booking_status") can look close enough by
# text alone to risk false merges, so the upper band stays high and genuine
# ambiguity goes to the LLM rather than being decided by embedding math.
AUTO_MERGE_THRESHOLD = 0.93
AMBIGUOUS_LOWER_THRESHOLD = 0.55

TIEBREAK_INSTRUCTIONS = """\
For the given entity, decide whether two attribute labels extracted from
chat messages describe the same underlying real-world attribute, or two
different ones. Consider them the same attribute only if they clearly
describe one characteristic that can meaningfully supersede itself over
time (e.g. "status" and "booking_status" for whether something is
occupied). Consider them different if they describe genuinely distinct
aspects of the entity (e.g. "status" and "owner").
"""


def _tiebreak(entity_name: str, a_predicate: str, b_predicate: str) -> PredicateMatchDecision:
    data = f'Entity: "{entity_name}"\nAttribute A: "{a_predicate}"\nAttribute B: "{b_predicate}"'
    return call_structured(
        instructions=TIEBREAK_INSTRUCTIONS,
        untrusted_text=data,
        response_schema=PredicateMatchDecision,
    )


def resolve_predicate(db: Session, entity_id: uuid.UUID, raw_predicate: str) -> str:
    """Canonicalize a claim's predicate against the predicates already used
    for Facts about this same entity — scoped per-entity, the same way
    entity resolution never compares mentions across unrelated things.

    Claims themselves keep whatever predicate the LLM originally extracted
    (untouched, for provenance); this canonicalization only happens when
    building the Fact layer, exactly like entity resolution never rewrites
    Claim.subject_surface_text, only Claim.subject_entity_id.
    """
    normalized_raw = normalize_text(raw_predicate)

    existing_predicates = list(
        db.execute(
            select(Fact.predicate).where(Fact.entity_id == entity_id).distinct()
        ).scalars()
    )
    if not existing_predicates:
        return normalized_raw
    if normalized_raw in existing_predicates:
        return normalized_raw

    entity = db.get(Entity, entity_id)
    entity_name = entity.canonical_name if entity is not None else ""

    raw_embedding = embed_text(raw_predicate)
    scored = [
        (cosine_similarity(raw_embedding, embed_text(p)), p) for p in existing_predicates
    ]
    best_similarity, best_predicate = max(scored, key=lambda pair: pair[0])

    if best_similarity >= AUTO_MERGE_THRESHOLD:
        return best_predicate
    if best_similarity >= AMBIGUOUS_LOWER_THRESHOLD:
        try:
            decision = _tiebreak(entity_name, raw_predicate, best_predicate)
        except Exception:
            logger.warning(
                "Predicate tiebreak failed for entity %s ('%s' vs '%s')",
                entity_id,
                raw_predicate,
                best_predicate,
                exc_info=True,
            )
            return normalized_raw
        if decision.same_attribute:
            return best_predicate

    return normalized_raw
