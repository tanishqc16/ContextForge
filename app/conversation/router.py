from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.conversation.ingestion import ingest_messages
from app.conversation.models import Message
from app.conversation.schemas import MessageCreate, MessageRead
from app.db.session import get_db

router = APIRouter()


@router.post("/ingest", response_model=list[MessageRead])
def ingest(
    messages: list[MessageCreate], db: Annotated[Session, Depends(get_db)]
) -> list[Message]:
    return ingest_messages(db, messages)


@router.get("/messages", response_model=list[MessageRead])
def list_messages(
    db: Annotated[Session, Depends(get_db)], limit: int = 50
) -> list[Message]:
    return list(db.execute(select(Message).order_by(Message.sent_at).limit(limit)).scalars())
