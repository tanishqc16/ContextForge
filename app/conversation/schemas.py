import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class MessageCreate(BaseModel):
    conversation_id: str
    sender: str
    sender_role: str | None = None
    text: str
    sent_at: datetime
    raw_metadata: dict[str, Any] | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: str
    sender: str
    sender_role: str | None
    text: str
    sent_at: datetime
    raw_metadata: dict[str, Any] | None
    created_at: datetime
