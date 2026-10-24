from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.knowledge import predicate_resolution
from app.knowledge.models import Entity, Fact, FactStatus
from app.knowledge.schemas import PredicateMatchDecision

_FAKE_VECTORS = {
    "status": [1.0, 0.0],
    "booking_status": [0.96, 0.28],  # ~0.96 similarity to "status" -> auto-merge band
    "owner": [0.0, 1.0],  # ~0.0 similarity to "status" -> clearly different
    "state": [0.8, 0.6],  # ~0.8 similarity to "status" -> ambiguous band
}


def _fake_embed(text: str) -> list[float]:
    return _FAKE_VECTORS[text.lower()]


def _entity(db: Session, name: str = "Room 1") -> Entity:
    entity = Entity(canonical_name=name, entity_type="room")
    db.add(entity)
    db.flush()
    return entity


def _fact_with_predicate(db: Session, entity: Entity, predicate: str) -> Fact:
    fact = Fact(
        entity_id=entity.id,
        predicate=predicate,
        value={"value": "booked"},
        valid_range=f"[{datetime(2026, 1, 1, tzinfo=UTC).isoformat()},)",
        status=FactStatus.CURRENT,
        confidence=0.9,
        version=1,
    )
    db.add(fact)
    db.flush()
    return fact


def test_first_predicate_for_entity_is_just_normalized(db_session: Session) -> None:
    entity = _entity(db_session)
    result = predicate_resolution.resolve_predicate(db_session, entity.id, "  Status ")
    assert result == "status"


def test_exact_normalized_match_short_circuits(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity = _entity(db_session)
    _fact_with_predicate(db_session, entity, "status")

    def fail_if_called(text: str) -> list[float]:
        raise AssertionError("should not need embeddings for an exact match")

    monkeypatch.setattr(predicate_resolution, "embed_text", fail_if_called)

    result = predicate_resolution.resolve_predicate(db_session, entity.id, "Status")
    assert result == "status"


def test_close_embedding_match_canonicalizes_without_llm_call(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity = _entity(db_session)
    _fact_with_predicate(db_session, entity, "status")
    monkeypatch.setattr(predicate_resolution, "embed_text", _fake_embed)

    def fail_if_called(*args: object, **kwargs: object) -> PredicateMatchDecision:
        raise AssertionError("tiebreak should not be called for a clear match")

    monkeypatch.setattr(predicate_resolution, "_tiebreak", fail_if_called)

    result = predicate_resolution.resolve_predicate(db_session, entity.id, "booking_status")
    assert result == "status"


def test_distant_embedding_keeps_predicates_separate(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity = _entity(db_session)
    _fact_with_predicate(db_session, entity, "status")
    monkeypatch.setattr(predicate_resolution, "embed_text", _fake_embed)

    result = predicate_resolution.resolve_predicate(db_session, entity.id, "owner")
    assert result == "owner"


def test_ambiguous_similarity_uses_llm_tiebreak(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity = _entity(db_session)
    _fact_with_predicate(db_session, entity, "status")
    monkeypatch.setattr(predicate_resolution, "embed_text", _fake_embed)

    def fake_tiebreak(entity_name: str, a: str, b: str) -> PredicateMatchDecision:
        return PredicateMatchDecision(same_attribute=True, reason="both describe occupancy")

    monkeypatch.setattr(predicate_resolution, "_tiebreak", fake_tiebreak)

    result = predicate_resolution.resolve_predicate(db_session, entity.id, "state")
    assert result == "status"


def test_ambiguous_similarity_llm_says_different_keeps_separate(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity = _entity(db_session)
    _fact_with_predicate(db_session, entity, "status")
    monkeypatch.setattr(predicate_resolution, "embed_text", _fake_embed)

    def fake_tiebreak(entity_name: str, a: str, b: str) -> PredicateMatchDecision:
        return PredicateMatchDecision(same_attribute=False, reason="unrelated attributes")

    monkeypatch.setattr(predicate_resolution, "_tiebreak", fake_tiebreak)

    result = predicate_resolution.resolve_predicate(db_session, entity.id, "state")
    assert result == "state"
