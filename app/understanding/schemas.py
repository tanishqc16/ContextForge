import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.understanding.models import ClassificationLabel


class ClassificationResult(BaseModel):
    """Structured output expected back from the classification LLM call."""

    label: ClassificationLabel
    confidence: float


class ClassificationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    message_id: uuid.UUID
    label: ClassificationLabel
    confidence: float
    model_version: str
    created_at: datetime
