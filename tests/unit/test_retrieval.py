from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.knowledge.models import Claim, Entity, Evidence, Fact, FactStatus
from app.retrieval import hybrid
from app.retrieval.schemas import AnswerSynthesis
from app.retrieval.semantic_search import find_best_matching_entity


def _entity(db: Session, name: str, entity_type: str = "credential") -> Entity:
    entity = Entity(canonical_name=name, entity_type=entity_type)
    db.add(entity)
    db.flush()
    return entity


def _message(db: Session, text: str, sent_at: datetime) -> Message:
    message = Message(conversation_id="c1", sender="alice", text=text, sent_at=sent_at)
    db.add(message)
    db.flush()
    return message


def _current_fact(
    db: Session, entity: Entity, predicate: str, value: str, message: Message
) -> Fact:
    claim = Claim(
        message_id=message.id,
        subject_entity_id=entity.id,
        subject_surface_text=entity.canonical_name,
        predicate=predicate,
        value={"value": value},
        extracted_confidence=0.9,
    )
    db.add(claim)
    db.flush()

    fact = Fact(
        entity_id=entity.id,
        predicate=predicate,
        value={"value": value},
        valid_range=f"[{message.sent_at.isoformat()},)",
        status=FactStatus.CURRENT,
        confidence=0.9,
        version=1,
    )
    db.add(fact)
    db.flush()
    db.add(Evidence(fact_id=fact.id, claim_id=claim.id, message_id=message.id))
    db.flush()
    return fact


def test_semantic_search_finds_close_entity(db_session: Session) -> None:
    _entity(db_session, "staging API key")
    _entity(db_session, "deploy script", entity_type="script")

    match = find_best_matching_entity(db_session, "which key should I use for staging?")

    assert match is not None
    assert match.canonical_name == "staging API key"


def test_semantic_search_returns_none_for_unrelated_query(db_session: Session) -> None:
    _entity(db_session, "staging API key")

    match = find_best_matching_entity(db_session, "what's the weather like today")

    assert match is None


def test_answer_question_with_no_matching_entity(db_session: Session) -> None:
    _entity(db_session, "staging API key")

    result = hybrid.answer_question(db_session, "what's the weather like today")

    assert result.entity_name is None
    assert result.answer == hybrid.NO_ENTITY_ANSWER
    assert result.current_facts == []


def test_answer_question_with_entity_but_no_facts(db_session: Session) -> None:
    _entity(db_session, "staging API key")

    result = hybrid.answer_question(db_session, "tell me about the staging API key")

    assert result.entity_name == "staging API key"
    assert result.answer == hybrid.NO_FACTS_ANSWER


def test_answer_question_grounds_synthesis_in_retrieved_facts(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    entity = _entity(db_session, "staging API key")
    msg = _message(db_session, "staging key is deprecated", datetime(2026, 6, 1, tzinfo=UTC))
    _current_fact(db_session, entity, "status", "deprecated", msg)

    captured_input: dict[str, str] = {}

    def fake_call_structured(
        instructions: str, untrusted_text: str, response_schema: type[AnswerSynthesis]
    ) -> AnswerSynthesis:
        captured_input["text"] = untrusted_text
        return AnswerSynthesis(answer="The staging API key is deprecated.")

    monkeypatch.setattr(hybrid, "call_structured", fake_call_structured)

    result = hybrid.answer_question(db_session, "is the staging key still good?")

    assert result.answer == "The staging API key is deprecated."
    assert "deprecated" in captured_input["text"]
    assert len(result.current_facts) == 1
    assert result.current_facts[0].evidence[0].message_text == "staging key is deprecated"
