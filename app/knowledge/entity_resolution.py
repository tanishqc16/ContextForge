import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.embeddings import cosine_similarity, embed_text
from app.common.llm_client import call_structured
from app.common.text_normalization import normalize_text
from app.knowledge.models import Claim, Entity, ExtractedEntity
from app.knowledge.schemas import EntityMatchDecision

logger = logging.getLogger(__name__)

# Similarity bands, tuned against this project's dataset. Above the upper
# bound: treated as the same entity automatically. Below the lower bound:
# treated as clearly different. In between: genuinely ambiguous, so the LLM
# is asked to judge — most mentions never reach that LLM call.
AUTO_MERGE_THRESHOLD = 0.93
AMBIGUOUS_LOWER_THRESHOLD = 0.60

TIEBREAK_INSTRUCTIONS = """\
Decide whether two mentions from a chat conversation refer to the same
specific real-world entity, or to different things. Consider them the same
entity only if they clearly refer to one specific thing, even if worded
differently. Consider them different if they could plausibly refer to
distinct things, even if the wording looks similar.
"""


def _tiebreak(a_text: str, a_type: str, b_text: str, b_type: str) -> EntityMatchDecision:
    data = f'Mention A: "{a_text}" (type: {a_type})\nMention B: "{b_text}" (type: {b_type})'
    return call_structured(
        instructions=TIEBREAK_INSTRUCTIONS,
        untrusted_text=data,
        response_schema=EntityMatchDecision,
    )


def _create_entity(db: Session, mention: ExtractedEntity) -> Entity:
    entity = Entity(canonical_name=mention.surface_text, entity_type=mention.entity_type)
    db.add(entity)
    db.flush()
    return entity


def resolve_pending_entities(db: Session) -> list[ExtractedEntity]:
    """Resolve every ExtractedEntity mention with no canonical Entity yet.

    Processed oldest-first, comparing each mention against every mention
    already resolved: a close embedding match auto-merges, a clearly
    distant one gets its own new Entity, and only the ambiguous middle
    band costs an LLM call.
    """
    pending = list(
        db.execute(
            select(ExtractedEntity)
            .where(ExtractedEntity.resolved_entity_id.is_(None))
            .order_by(ExtractedEntity.created_at)
        ).scalars()
    )
    resolved_mentions = list(
        db.execute(
            select(ExtractedEntity).where(ExtractedEntity.resolved_entity_id.is_not(None))
        ).scalars()
    )

    updated: list[ExtractedEntity] = []
    for mention in pending:
        if mention.embedding is None:
            mention.embedding = embed_text(mention.normalized_text)

        best_match: ExtractedEntity | None = None
        best_similarity = -1.0
        for other in resolved_mentions:
            if other.embedding is None:
                continue
            similarity = cosine_similarity(mention.embedding, other.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = other

        if best_match is not None and best_similarity >= AUTO_MERGE_THRESHOLD:
            mention.resolved_entity_id = best_match.resolved_entity_id
        elif best_match is not None and best_similarity >= AMBIGUOUS_LOWER_THRESHOLD:
            try:
                decision = _tiebreak(
                    mention.surface_text,
                    mention.entity_type,
                    best_match.surface_text,
                    best_match.entity_type,
                )
            except Exception:
                logger.warning("Entity tiebreak failed for %s", mention.id, exc_info=True)
                decision = None

            if decision is not None and decision.same_entity:
                mention.resolved_entity_id = best_match.resolved_entity_id
            else:
                mention.resolved_entity_id = _create_entity(db, mention).id
        else:
            mention.resolved_entity_id = _create_entity(db, mention).id

        db.commit()
        resolved_mentions.append(mention)
        updated.append(mention)

    for row in updated:
        db.refresh(row)
    return updated


def link_claims_to_entities(db: Session) -> list[Claim]:
    """Fill in Claim.subject_entity_id by matching each claim's subject
    text against entities already resolved from the same message. A claim
    whose subject can't be confidently matched is left unlinked rather
    than guessed — a wrong auto-link is worse than no link.
    """
    pending_claims = list(
        db.execute(select(Claim).where(Claim.subject_entity_id.is_(None))).scalars()
    )

    updated: list[Claim] = []
    for claim in pending_claims:
        normalized_subject = normalize_text(claim.subject_surface_text)
        candidates = list(
            db.execute(
                select(ExtractedEntity).where(
                    ExtractedEntity.message_id == claim.message_id,
                    ExtractedEntity.resolved_entity_id.is_not(None),
                )
            ).scalars()
        )
        if not candidates:
            continue

        exact = next((c for c in candidates if c.normalized_text == normalized_subject), None)
        if exact is not None:
            claim.subject_entity_id = exact.resolved_entity_id
        else:
            subject_embedding = embed_text(normalized_subject)
            scored = [
                (cosine_similarity(subject_embedding, c.embedding), c)
                for c in candidates
                if c.embedding is not None
            ]
            if not scored:
                continue
            best_similarity, best = max(scored, key=lambda pair: pair[0])
            if best_similarity < AUTO_MERGE_THRESHOLD:
                continue
            claim.subject_entity_id = best.resolved_entity_id

        db.commit()
        updated.append(claim)

    for row in updated:
        db.refresh(row)
    return updated
