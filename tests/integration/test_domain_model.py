from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.knowledge.models import Claim, Entity, Evidence, Fact, FactStatus
from app.understanding.models import Classification, ClassificationLabel


def test_message_is_stored_as_written(db_session: Session) -> None:
    message = Message(
        conversation_id="conv-1",
        sender="alice",
        sender_role="senior_engineer",
        text="Use the staging key for internal tools.",
        sent_at=datetime(2026, 6, 3, tzinfo=UTC),
    )
    db_session.add(message)
    db_session.flush()

    fetched = db_session.get(Message, message.id)
    assert fetched is not None
    assert fetched.text == "Use the staging key for internal tools."
    assert fetched.sender_role == "senior_engineer"


def test_classification_links_to_message(db_session: Session) -> None:
    message = Message(
        conversation_id="conv-1",
        sender="bob",
        text="Is the staging key still valid?",
        sent_at=datetime(2026, 6, 4, tzinfo=UTC),
    )
    db_session.add(message)
    db_session.flush()

    classification = Classification(
        message_id=message.id,
        label=ClassificationLabel.QUESTION,
        confidence=0.95,
        model_version="test-v1",
    )
    db_session.add(classification)
    db_session.flush()

    assert classification.label == ClassificationLabel.QUESTION
    assert classification.message_id == message.id


def test_fact_chain_preserves_provenance_and_versioning(db_session: Session) -> None:
    """Simulates the full claim -> fact -> evidence flow, including a
    fact being superseded rather than overwritten."""
    message_v1 = Message(
        conversation_id="conv-1",
        sender="alice",
        text="Staging key works for internal tools.",
        sent_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    message_v2 = Message(
        conversation_id="conv-1",
        sender="alice",
        text="Staging key is deprecated, use sandbox.",
        sent_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    db_session.add_all([message_v1, message_v2])
    db_session.flush()

    entity = Entity(canonical_name="staging_key", entity_type="credential")
    db_session.add(entity)
    db_session.flush()

    claim_v1 = Claim(
        message_id=message_v1.id,
        subject_entity_id=entity.id,
        subject_surface_text="staging key",
        predicate="status",
        value={"status": "active"},
        extracted_confidence=0.9,
    )
    claim_v2 = Claim(
        message_id=message_v2.id,
        subject_entity_id=entity.id,
        subject_surface_text="staging key",
        predicate="status",
        value={"status": "deprecated"},
        extracted_confidence=0.9,
    )
    db_session.add_all([claim_v1, claim_v2])
    db_session.flush()

    fact_v1 = Fact(
        entity_id=entity.id,
        predicate="status",
        value={"status": "active"},
        valid_range="[2026-05-01, 2026-06-01)",
        status=FactStatus.SUPERSEDED,
        confidence=0.9,
        version=1,
    )
    db_session.add(fact_v1)
    db_session.flush()

    fact_v2 = Fact(
        entity_id=entity.id,
        predicate="status",
        value={"status": "deprecated"},
        valid_range="[2026-06-01,)",
        status=FactStatus.CURRENT,
        confidence=0.9,
        version=2,
        supersedes_fact_id=fact_v1.id,
    )
    db_session.add(fact_v2)
    db_session.flush()

    db_session.add_all(
        [
            Evidence(fact_id=fact_v1.id, claim_id=claim_v1.id, message_id=message_v1.id),
            Evidence(fact_id=fact_v2.id, claim_id=claim_v2.id, message_id=message_v2.id),
        ]
    )
    db_session.flush()

    # The old fact is preserved, not deleted or overwritten.
    stored_v1 = db_session.get(Fact, fact_v1.id)
    assert stored_v1 is not None
    assert stored_v1.status == FactStatus.SUPERSEDED
    assert stored_v1.value == {"status": "active"}

    # The new fact correctly points back to what it supersedes.
    stored_v2 = db_session.get(Fact, fact_v2.id)
    assert stored_v2 is not None
    assert stored_v2.supersedes_fact_id == fact_v1.id
    assert stored_v2.status == FactStatus.CURRENT
