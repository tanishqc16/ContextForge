import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class EvidenceItem(BaseModel):
    message_id: uuid.UUID
    message_text: str
    sent_at: datetime


class CurrentFactItem(BaseModel):
    predicate: str
    value: dict[str, Any]
    context: dict[str, Any] | None
    confidence: float
    valid_from: datetime
    evidence: list[EvidenceItem]


class SupersededFactItem(BaseModel):
    predicate: str
    value: dict[str, Any]
    context: dict[str, Any] | None
    valid_from: datetime
    valid_until: datetime | None


class AnswerResult(BaseModel):
    question: str
    entity_name: str | None
    answer: str
    confidence: float
    current_facts: list[CurrentFactItem]
    superseded_facts: list[SupersededFactItem]


class AnswerSynthesis(BaseModel):
    """Structured output for the answer-phrasing LLM call. It only ever
    sees the already-retrieved, already-correct facts — it can phrase them
    naturally, but it cannot introduce a fact that wasn't retrieved."""

    answer: str
