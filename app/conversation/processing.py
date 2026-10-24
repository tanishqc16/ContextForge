import enum
import uuid

from sqlalchemy import ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin


class ProcessingStage(enum.StrEnum):
    ENTITY_EXTRACTION = "ENTITY_EXTRACTION"
    CLAIM_EXTRACTION = "CLAIM_EXTRACTION"


class MessageProcessingLog(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """Marks that a pipeline stage has run for a message, independent of how
    many (if any) rows that stage produced. Without this, a message that
    legitimately yields zero entities/claims would look "still pending" and
    get reprocessed (and re-billed against the LLM quota) every run."""

    __tablename__ = "message_processing_log"

    message_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("messages.id"), index=True
    )
    stage: Mapped[ProcessingStage] = mapped_column(index=True)
