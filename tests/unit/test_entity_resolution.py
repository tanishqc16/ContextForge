from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.common.constants import EMBEDDING_DIM
from app.conversation.models import Message
from app.knowledge import entity_resolution
from app.knowledge.models import Claim, ExtractedEntity
from app.knowledge.schemas import EntityMatchDecision


def _vec(x: float, y: float) -> list[float]:
    """A fake embedding padded to the real column width. Padding with
    zeros doesn't change the dot product of two such vectors, so cosine
    similarity behaves the same as it would for the real 2D vector."""
    return [x, y, *([0.0] * (EMBEDDING_DIM - 2))]


def _message(db: Session, text: str) -> Message:
    message = Message(conversation_id="c1", sender="alice", text=text, sent_at=datetime.now(UTC))
    db.add(message)
    db.flush()
    return message


def _mention(
    db: Session, message: Message, surface_text: str, embedding: list[float]
) -> ExtractedEntity:
    mention = ExtractedEntity(
        message_id=message.id,
        surface_text=surface_text,
        normalized_text=surface_text.lower(),
        entity_type="credential",
        embedding=embedding,
    )
    db.add(mention)
    db.flush()
    return mention


def test_close_embeddings_auto_merge_without_llm_call(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    msg = _message(db_session, "placeholder")
    first = _mention(db_session, msg, "staging key", _vec(1.0, 0.0))
    first.resolved_entity_id = entity_resolution._create_entity(db_session, first).id
    db_session.commit()

    _mention(db_session, msg, "staging API key", _vec(0.99, 0.01))

    def fail_if_called(*args: object, **kwargs: object) -> EntityMatchDecision:
        raise AssertionError("tiebreak should not be called for a clear match")

    monkeypatch.setattr(entity_resolution, "_tiebreak", fail_if_called)

    resolved = entity_resolution.resolve_pending_entities(db_session)

    assert len(resolved) == 1
    assert resolved[0].resolved_entity_id == first.resolved_entity_id


def test_distant_embeddings_create_separate_entities(db_session: Session) -> None:
    msg = _message(db_session, "placeholder")
    first = _mention(db_session, msg, "staging key", _vec(1.0, 0.0))
    first.resolved_entity_id = entity_resolution._create_entity(db_session, first).id
    db_session.commit()

    _mention(db_session, msg, "deploy script", _vec(0.0, 1.0))

    resolved = entity_resolution.resolve_pending_entities(db_session)

    assert len(resolved) == 1
    assert resolved[0].resolved_entity_id != first.resolved_entity_id


def test_ambiguous_similarity_uses_llm_tiebreak(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    msg = _message(db_session, "placeholder")
    first = _mention(db_session, msg, "staging", _vec(1.0, 0.0))
    first.resolved_entity_id = entity_resolution._create_entity(db_session, first).id
    db_session.commit()

    # 0.7 cosine similarity-ish vector, lands in the ambiguous band.
    _mention(db_session, msg, "sandbox", _vec(0.7, 0.714))

    def fake_tiebreak(
        a_text: str, a_type: str, b_text: str, b_type: str
    ) -> EntityMatchDecision:
        return EntityMatchDecision(same_entity=False, reason="different environments")

    monkeypatch.setattr(entity_resolution, "_tiebreak", fake_tiebreak)

    resolved = entity_resolution.resolve_pending_entities(db_session)

    assert len(resolved) == 1
    assert resolved[0].resolved_entity_id != first.resolved_entity_id


def test_near_identical_short_phrases_require_llm_tiebreak_not_auto_merge(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test for a real bug found in manual testing: "Room 1" and
    "Room 2" scored 0.91 cosine similarity (nearly identical *meaning*,
    different specific rooms). That must NOT auto-merge — it has to go
    through the LLM tiebreak, which can actually notice the room numbers
    differ, instead of being silently merged by embedding math alone."""
    msg = _message(db_session, "placeholder")
    first = _mention(db_session, msg, "Room 1", _vec(1.0, 0.0))
    first.resolved_entity_id = entity_resolution._create_entity(db_session, first).id
    db_session.commit()

    # ~0.91 cosine similarity, matching the real Room 1 / Room 2 measurement.
    _mention(db_session, msg, "Room 2", _vec(0.9148, 0.4039))

    tiebreak_calls = []

    def fake_tiebreak(
        a_text: str, a_type: str, b_text: str, b_type: str
    ) -> EntityMatchDecision:
        tiebreak_calls.append((a_text, b_text))
        return EntityMatchDecision(same_entity=False, reason="different room numbers")

    monkeypatch.setattr(entity_resolution, "_tiebreak", fake_tiebreak)

    resolved = entity_resolution.resolve_pending_entities(db_session)

    assert len(tiebreak_calls) == 1, "should have been ambiguous, not auto-merged"
    assert len(resolved) == 1
    assert resolved[0].resolved_entity_id != first.resolved_entity_id


def test_link_claims_to_entities_matches_by_normalized_text(db_session: Session) -> None:
    msg = _message(db_session, "staging key is deprecated")
    mention = _mention(db_session, msg, "staging key", _vec(1.0, 0.0))
    mention.resolved_entity_id = entity_resolution._create_entity(db_session, mention).id
    db_session.commit()

    claim = Claim(
        message_id=msg.id,
        subject_surface_text="Staging Key",
        predicate="status",
        value={"value": "deprecated"},
        extracted_confidence=0.9,
    )
    db_session.add(claim)
    db_session.commit()

    linked = entity_resolution.link_claims_to_entities(db_session)

    assert len(linked) == 1
    assert linked[0].subject_entity_id == mention.resolved_entity_id
