import json
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.knowledge.models import Claim, Entity, Evidence, Fact, FactStatus
from app.knowledge.predicate_resolution import resolve_predicate

# A claim only becomes trusted "current knowledge" above this confidence —
# this is plain code, not an LLM judgment call (per the project's rule that
# deterministic decisions stay deterministic).
FACT_PROMOTION_CONFIDENCE_THRESHOLD = 0.7


def _range_literal(lower: datetime, upper: datetime | None) -> str:
    """Build a Postgres range literal string directly, rather than handing
    psycopg a `Range(...)` object — an open-ended upper bound combined with
    a timezone-aware lower bound makes psycopg mis-infer the wire type as
    `tsrange` instead of `tstzrange`, which Postgres then refuses to cast."""
    upper_str = upper.isoformat() if upper is not None else ""
    return f"[{lower.isoformat()},{upper_str})"


def _context_key(context: dict[str, Any] | None) -> str:
    """A stable key so two claims with the same context are treated as the
    same fact chain, and claims with *different* context — e.g. "internal
    tools" vs "external partners" — are kept as separate chains rather than
    made to look like they conflict."""
    return json.dumps(context, sort_keys=True) if context else ""


def _current_fact_for(db: Session, entity_id: Any, predicate: str, context_key: str) -> Fact | None:
    candidates = db.execute(
        select(Fact).where(
            Fact.entity_id == entity_id,
            Fact.predicate == predicate,
            Fact.status == FactStatus.CURRENT,
        )
    ).scalars()
    return next((f for f in candidates if _context_key(f.context) == context_key), None)


def promote_pending_claims(db: Session) -> list[Fact]:
    """Turn confident, resolved claims into versioned Facts.

    Claims are processed in the chronological order of the message they
    came from (not extraction order), so the version history it builds is
    always moving forward in time. A claim already backing a Fact (via an
    Evidence row) is never re-promoted.

    Each claim's predicate is first canonicalized against predicates already
    used for this entity (see predicate_resolution.py) — "status" and
    "booking_status" for the same entity are recognized as one attribute,
    so a later value correctly supersedes the earlier one instead of
    sitting alongside it as an unrelated fact.

    - A new value for an (entity, predicate, context) not seen before opens
      a new, CURRENT fact.
    - The same value seen again just adds corroborating evidence to the
      existing fact — not a new version.
    - A genuinely different value supersedes the current fact rather than
      overwriting it: the old fact's valid_range is closed off and its
      status flips to SUPERSEDED, and a new versioned fact takes over.
    """
    already_promoted = select(Evidence.claim_id)
    pending_claims = list(
        db.execute(
            select(Claim)
            .join(Message, Message.id == Claim.message_id)
            .where(Claim.subject_entity_id.is_not(None))
            .where(Claim.extracted_confidence >= FACT_PROMOTION_CONFIDENCE_THRESHOLD)
            .where(Claim.id.not_in(already_promoted))
            .order_by(Message.sent_at)
        ).scalars()
    )

    touched_facts: list[Fact] = []
    for claim in pending_claims:
        message = db.get(Message, claim.message_id)
        assert message is not None
        entity = db.get(Entity, claim.subject_entity_id)
        assert entity is not None

        canonical_predicate = resolve_predicate(db, entity.id, claim.predicate)
        context_key = _context_key(claim.context)
        current = _current_fact_for(db, entity.id, canonical_predicate, context_key)

        if current is None:
            fact = Fact(
                entity_id=entity.id,
                predicate=canonical_predicate,
                value=claim.value,
                context=claim.context,
                valid_range=_range_literal(message.sent_at, None),
                status=FactStatus.CURRENT,
                confidence=claim.extracted_confidence,
                version=1,
            )
            db.add(fact)
            db.flush()
            db.add(Evidence(fact_id=fact.id, claim_id=claim.id, message_id=claim.message_id))
            touched_facts.append(fact)
        elif current.value == claim.value:
            db.add(Evidence(fact_id=current.id, claim_id=claim.id, message_id=claim.message_id))
            touched_facts.append(current)
        else:
            current.valid_range = _range_literal(current.valid_range.lower, message.sent_at)
            current.status = FactStatus.SUPERSEDED

            new_fact = Fact(
                entity_id=entity.id,
                predicate=canonical_predicate,
                value=claim.value,
                context=claim.context,
                valid_range=_range_literal(message.sent_at, None),
                status=FactStatus.CURRENT,
                confidence=claim.extracted_confidence,
                version=current.version + 1,
                supersedes_fact_id=current.id,
            )
            db.add(new_fact)
            db.flush()
            db.add(Evidence(fact_id=new_fact.id, claim_id=claim.id, message_id=claim.message_id))
            touched_facts.append(new_fact)

        db.commit()

    for fact in touched_facts:
        db.refresh(fact)
    return touched_facts
