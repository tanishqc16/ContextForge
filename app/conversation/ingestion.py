import json
from pathlib import Path

from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.conversation.schemas import MessageCreate


def load_messages_from_file(path: Path) -> list[MessageCreate]:
    raw = json.loads(path.read_text())
    return [MessageCreate.model_validate(item) for item in raw]


def ingest_messages(db: Session, messages: list[MessageCreate]) -> list[Message]:
    rows = [Message(**message.model_dump()) for message in messages]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows
