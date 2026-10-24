from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.knowledge import fact_management, predicate_resolution
from app.knowledge.models import Claim, Entity, Fact, FactStatus


def _message(db: Session, text: str, sent_at: datetime) -> Message:
    message = Message(conversation_id="c1", sender="alice", text=text, sent_at=sent_at)
    db.add(message)
    db.flush()
    return message


def _entity(db: Session, name: str) -> Entity:
    entity = Entity(canonical_name=name, entity_type="credential")
    db.add(entity)
    db.flush()
    return entity


def _claim(
    db: Session,
    message: Message,
    entity: Entity,
    predicate: str,
    value: str,
    confidence: float = 0.9,
    context: dict[str, Any] | None = None,
) -> Claim:
    claim = Claim(
        message_id=message.id,
        subject_entity_id=entity.id,
        subject_surface_text=entity.canonical_name,
        predicate=predicate,
        value={"value": value},
        context=context,
        extracted_confidence=confidence,
    )
    db.add(claim)
    db.flush()
    return claim


def test_first_claim_creates_a_new_current_fact(db_session: Session) -> None:
    entity = _entity(db_session, "staging key")
    msg = _message(db_session, "staging key is active", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg, entity, "status", "active")

    facts = fact_management.promote_pending_claims(db_session)

    assert len(facts) == 1
    assert facts[0].status == FactStatus.CURRENT
    assert facts[0].version == 1
    assert facts[0].value == {"value": "active"}
    assert facts[0].supersedes_fact_id is None


def test_repeated_same_value_corroborates_instead_of_versioning(db_session: Session) -> None:
    entity = _entity(db_session, "staging key")
    msg1 = _message(db_session, "staging key is active", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg1, entity, "status", "active")
    fact_management.promote_pending_claims(db_session)

    msg2 = _message(db_session, "staging key, same as before", datetime(2026, 2, 1, tzinfo=UTC))
    _claim(db_session, msg2, entity, "status", "active")
    fact_management.promote_pending_claims(db_session)

    all_facts = db_session.query(Fact).filter_by(entity_id=entity.id, predicate="status").all()
    assert len(all_facts) == 1
    assert all_facts[0].version == 1
    assert all_facts[0].status == FactStatus.CURRENT


def test_different_value_supersedes_rather_than_overwrites(db_session: Session) -> None:
    entity = _entity(db_session, "staging key")
    msg1 = _message(db_session, "staging key is active", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg1, entity, "status", "active")
    fact_management.promote_pending_claims(db_session)

    msg2 = _message(db_session, "staging key is deprecated", datetime(2026, 6, 1, tzinfo=UTC))
    _claim(db_session, msg2, entity, "status", "deprecated")
    fact_management.promote_pending_claims(db_session)

    all_facts = (
        db_session.query(Fact)
        .filter_by(entity_id=entity.id, predicate="status")
        .order_by(Fact.version)
        .all()
    )
    assert len(all_facts) == 2

    old, new = all_facts
    assert old.status == FactStatus.SUPERSEDED
    assert old.value == {"value": "active"}
    assert old.valid_range.upper == datetime(2026, 6, 1, tzinfo=UTC)

    assert new.status == FactStatus.CURRENT
    assert new.value == {"value": "deprecated"}
    assert new.version == 2
    assert new.supersedes_fact_id == old.id


def test_different_context_creates_a_separate_fact_chain_not_a_conflict(
    db_session: Session,
) -> None:
    entity = _entity(db_session, "sandbox")
    msg1 = _message(db_session, "sandbox for internal use", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg1, entity, "purpose", "internal", context={"note": "internal tools"})

    msg2 = _message(db_session, "sandbox for external use", datetime(2026, 1, 2, tzinfo=UTC))
    _claim(db_session, msg2, entity, "purpose", "external", context={"note": "external partners"})

    facts = fact_management.promote_pending_claims(db_session)

    assert len(facts) == 2
    assert all(f.status == FactStatus.CURRENT for f in facts)
    assert {f.value["value"] for f in facts} == {"internal", "external"}


def test_low_confidence_claims_are_not_promoted(db_session: Session) -> None:
    entity = _entity(db_session, "staging key")
    msg = _message(db_session, "maybe staging key is active?", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg, entity, "status", "active", confidence=0.3)

    facts = fact_management.promote_pending_claims(db_session)

    assert facts == []


def test_already_promoted_claims_are_not_reprocessed(db_session: Session) -> None:
    entity = _entity(db_session, "staging key")
    msg = _message(db_session, "staging key is active", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg, entity, "status", "active")

    first_run = fact_management.promote_pending_claims(db_session)
    assert len(first_run) == 1

    second_run = fact_management.promote_pending_claims(db_session)
    assert second_run == []


def test_differently_named_predicates_for_same_attribute_still_supersede(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real bug found in manual testing: "Room 1
    booked by Ron" (predicate=booking_status) followed by "Room 1 is free"
    (predicate=status) were being stored as two unrelated CURRENT facts
    instead of the second correctly superseding the first — because the
    predicate names didn't match exactly. Predicate resolution should fix
    this by recognizing both describe the same attribute.

    The first-seen wording ("booking_status") becomes the canonical form
    for this entity — later claims get canonicalized onto it, not the
    other way around, same as how an entity's first surface text becomes
    its canonical_name.
    """
    entity = _entity(db_session, "Room 1")

    def fake_embed(text: str) -> list[float]:
        # "status" and "booking_status" score ~0.96 similarity with real
        # embeddings — high enough to auto-merge without an LLM call.
        return {"booking_status": [1.0, 0.0], "status": [0.96, 0.28]}[text.lower()]

    monkeypatch.setattr(predicate_resolution, "embed_text", fake_embed)

    msg1 = _message(db_session, "Room 1 is booked by Ron", datetime(2026, 1, 1, tzinfo=UTC))
    _claim(db_session, msg1, entity, "booking_status", "booked")
    fact_management.promote_pending_claims(db_session)

    msg2 = _message(db_session, "Room 1 is free now", datetime(2026, 1, 2, tzinfo=UTC))
    _claim(db_session, msg2, entity, "status", "free")
    fact_management.promote_pending_claims(db_session)

    all_facts = (
        db_session.query(Fact)
        .filter_by(entity_id=entity.id)
        .order_by(Fact.version)
        .all()
    )
    assert len(all_facts) == 2, "should be one superseded + one current, not two parallel facts"

    old, new = all_facts
    assert old.status == FactStatus.SUPERSEDED
    assert old.predicate == "booking_status"
    assert new.status == FactStatus.CURRENT
    assert new.predicate == "booking_status"  # canonicalized onto the first-seen wording
    assert new.value == {"value": "free"}
    assert new.supersedes_fact_id == old.id
