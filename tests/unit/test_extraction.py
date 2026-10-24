from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.conversation.processing import MessageProcessingLog, ProcessingStage
from app.knowledge import claim_extraction, entity_extraction
from app.knowledge.schemas import (
    ClaimExtractionResult,
    EntityExtractionResult,
    ExtractedClaim,
    ExtractedEntityCandidate,
)
from app.understanding.models import Classification, ClassificationLabel


def _message(db: Session, text: str, label: ClassificationLabel) -> Message:
    message = Message(
        conversation_id="c1", sender="alice", text=text, sent_at=datetime.now(UTC)
    )
    db.add(message)
    db.flush()
    db.add(
        Classification(
            message_id=message.id, label=label, confidence=0.9, model_version="test"
        )
    )
    db.flush()
    return message


def test_entity_extraction_runs_even_on_irrelevant_labeled_messages(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long, mostly-casual message can still bury a real fact in it —
    classification must not gate extraction, or that fact gets silently
    dropped. Both messages here should be attempted."""
    buried_fact = _message(
        db_session,
        "lol anyway crazy weekend, also btw rent going up to 30k next month, gtg",
        ClassificationLabel.IRRELEVANT,
    )
    pure_chitchat = _message(db_session, "lol nice", ClassificationLabel.IRRELEVANT)

    def fake_extract_entities(text: str) -> EntityExtractionResult:
        if "rent" in text:
            candidate = ExtractedEntityCandidate(surface_text="rent", entity_type="amount")
            return EntityExtractionResult(entities=[candidate])
        return EntityExtractionResult(entities=[])

    monkeypatch.setattr(entity_extraction, "extract_entities", fake_extract_entities)

    rows = entity_extraction.extract_entities_for_pending_messages(db_session)

    assert len(rows) == 1
    assert rows[0].message_id == buried_fact.id

    log_messages = {
        row.message_id
        for row in db_session.query(MessageProcessingLog).all()
        if row.stage == ProcessingStage.ENTITY_EXTRACTION
    }
    assert {buried_fact.id, pure_chitchat.id} <= log_messages


def test_entity_extraction_marks_zero_result_messages_as_processed(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = _message(db_session, "ok sounds good", ClassificationLabel.ANSWER)

    def fake_extract_entities(text: str) -> EntityExtractionResult:
        return EntityExtractionResult(entities=[])

    monkeypatch.setattr(entity_extraction, "extract_entities", fake_extract_entities)

    first_run = entity_extraction.extract_entities_for_pending_messages(db_session)
    assert first_run == []

    log_entry = (
        db_session.query(MessageProcessingLog)
        .filter_by(message_id=message.id, stage=ProcessingStage.ENTITY_EXTRACTION)
        .one_or_none()
    )
    assert log_entry is not None

    # A second run must not call the LLM again for this message.
    def fail_if_called(text: str) -> EntityExtractionResult:
        raise AssertionError("should not be called for an already-processed message")

    monkeypatch.setattr(entity_extraction, "extract_entities", fail_if_called)
    second_run = entity_extraction.extract_entities_for_pending_messages(db_session)
    assert second_run == []


def test_claim_extraction_stores_structured_claim(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    message = _message(
        db_session, "staging key is deprecated", ClassificationLabel.UPDATE
    )

    def fake_extract_claims(text: str) -> ClaimExtractionResult:
        return ClaimExtractionResult(
            claims=[
                ExtractedClaim(
                    subject_surface_text="staging key",
                    predicate="status",
                    value="deprecated",
                    confidence=0.9,
                )
            ]
        )

    monkeypatch.setattr(claim_extraction, "extract_claims", fake_extract_claims)

    rows = claim_extraction.extract_claims_for_pending_messages(db_session)

    assert len(rows) == 1
    assert rows[0].message_id == message.id
    assert rows[0].predicate == "status"
    assert rows[0].value == {"value": "deprecated"}
