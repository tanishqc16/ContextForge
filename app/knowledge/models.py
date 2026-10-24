import enum
import uuid
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, TSTZRANGE
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.common.constants import EMBEDDING_DIM
from app.db.base import Base
from app.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin


class FactStatus(enum.StrEnum):
    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    DISPUTED = "DISPUTED"
    RETRACTED = "RETRACTED"


class Entity(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A resolved, canonical real-world thing (open vocabulary type —
    never a domain-specific enum, so this stays usable across domains)."""

    __tablename__ = "entities"

    canonical_name: Mapped[str]
    entity_type: Mapped[str] = mapped_column(index=True)
    entity_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)


class ExtractedEntity(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A candidate entity mention extracted from one message, before
    resolution decides which canonical Entity (if any) it refers to."""

    __tablename__ = "extracted_entities"

    message_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("messages.id"), index=True
    )
    surface_text: Mapped[str]
    normalized_text: Mapped[str]
    entity_type: Mapped[str]
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), default=None)
    resolved_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("entities.id"), default=None, index=True
    )


class Claim(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A candidate statement extracted from a message — knowledge candidate,
    not yet trusted. Immutable once written; never edited or deleted."""

    __tablename__ = "claims"

    message_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("messages.id"), index=True
    )
    subject_entity_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("entities.id"), default=None, index=True
    )
    subject_surface_text: Mapped[str]
    predicate: Mapped[str] = mapped_column(index=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    source_authority_weight: Mapped[float | None] = mapped_column(default=None)
    extracted_confidence: Mapped[float]


class Fact(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """An accepted, versioned belief about an entity. A new value does not
    overwrite the old row — it supersedes it, preserving history."""

    __tablename__ = "facts"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("entities.id"), index=True
    )
    predicate: Mapped[str] = mapped_column(index=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    valid_range: Mapped[Any] = mapped_column(TSTZRANGE)
    status: Mapped[FactStatus] = mapped_column(default=FactStatus.CURRENT)
    confidence: Mapped[float]
    version: Mapped[int] = mapped_column(default=1)
    supersedes_fact_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("facts.id"), default=None
    )


class Evidence(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Links a Fact back to the Claim and Message that support it —
    the system's answer to 'why do we believe this'."""

    __tablename__ = "evidence"

    fact_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("facts.id"), index=True
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("claims.id"))
    message_id: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("messages.id"))
