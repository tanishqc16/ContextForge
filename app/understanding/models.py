import enum
import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin


class ClassificationLabel(enum.StrEnum):
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    FACT = "FACT"
    REQUEST = "REQUEST"
    ISSUE = "ISSUE"
    FEEDBACK = "FEEDBACK"
    UPDATE = "UPDATE"
    CORRECTION = "CORRECTION"
    DECISION = "DECISION"
    DISCUSSION = "DISCUSSION"
    IRRELEVANT = "IRRELEVANT"


class Classification(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Result of classifying one message. Append-only: a message may be
    reclassified by a newer model_version, but prior rows are kept."""

    __tablename__ = "classifications"

    message_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("messages.id"), index=True
    )
    label: Mapped[ClassificationLabel]
    confidence: Mapped[float]
    model_version: Mapped[str]
