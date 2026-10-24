import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.knowledge.models import Fact, FactStatus


class EntityRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    canonical_name: str
    entity_type: str
    entity_metadata: dict[str, Any] | None
    created_at: datetime


class ExtractedEntityCandidate(BaseModel):
    surface_text: str
    entity_type: str


class EntityExtractionResult(BaseModel):
    """Structured output expected back from the entity-extraction LLM call."""

    entities: list[ExtractedEntityCandidate]


class ExtractedEntityRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    message_id: uuid.UUID
    surface_text: str
    normalized_text: str
    entity_type: str
    resolved_entity_id: uuid.UUID | None
    created_at: datetime


class ExtractedClaim(BaseModel):
    """Structured output shape for a single claim.

    `value`/`context` are plain strings here, not open dicts — Gemini's
    Developer API rejects JSON schemas with free-form (additionalProperties)
    objects. They get wrapped into small structured dicts in code before
    being stored in the Claim/Fact JSONB columns.
    """

    subject_surface_text: str
    predicate: str
    value: str
    context: str | None = None
    confidence: float


class ClaimExtractionResult(BaseModel):
    """Structured output expected back from the claim-extraction LLM call."""

    claims: list[ExtractedClaim]


class EntityMatchDecision(BaseModel):
    """Structured output for the entity-resolution LLM tiebreak — only
    called when embedding similarity falls in an ambiguous band."""

    same_entity: bool
    reason: str


class PredicateMatchDecision(BaseModel):
    """Structured output for the predicate-resolution LLM tiebreak — asks
    whether two attribute labels (e.g. "status" and "booking_status") refer
    to the same underlying attribute of a specific entity."""

    same_attribute: bool
    reason: str


class ClaimRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    message_id: uuid.UUID
    subject_entity_id: uuid.UUID | None
    subject_surface_text: str
    predicate: str
    value: dict[str, Any]
    context: dict[str, Any] | None
    extracted_confidence: float
    created_at: datetime


class FactRead(BaseModel):
    """Note: built via `from_fact()`, not `model_validate(fact)` — the ORM's
    `valid_range` (a Postgres range object) doesn't map directly onto the
    separate valid_from/valid_until fields API consumers actually want."""

    id: uuid.UUID
    entity_id: uuid.UUID
    predicate: str
    value: dict[str, Any]
    context: dict[str, Any] | None
    status: FactStatus
    confidence: float
    version: int
    supersedes_fact_id: uuid.UUID | None
    valid_from: datetime
    valid_until: datetime | None
    created_at: datetime

    @classmethod
    def from_fact(cls, fact: Fact) -> "FactRead":
        return cls(
            id=fact.id,
            entity_id=fact.entity_id,
            predicate=fact.predicate,
            value=fact.value,
            context=fact.context,
            status=fact.status,
            confidence=fact.confidence,
            version=fact.version,
            supersedes_fact_id=fact.supersedes_fact_id,
            valid_from=fact.valid_range.lower,
            valid_until=fact.valid_range.upper,
            created_at=fact.created_at,
        )
