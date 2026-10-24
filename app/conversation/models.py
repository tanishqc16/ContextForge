from datetime import datetime
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.mixins import CreatedAtMixin, UUIDPrimaryKeyMixin


class Message(UUIDPrimaryKeyMixin, CreatedAtMixin, Base):
    """A raw conversation message. Immutable: never updated after ingestion."""

    __tablename__ = "messages"

    conversation_id: Mapped[str] = mapped_column(index=True)
    sender: Mapped[str]
    sender_role: Mapped[str | None] = mapped_column(
        default=None,
        doc="Open-vocabulary role key (e.g. 'senior_engineer'); authority "
        "weighting is resolved from this via config, never hardcoded here.",
    )
    text: Mapped[str]
    sent_at: Mapped[datetime] = mapped_column(index=True)
    raw_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
