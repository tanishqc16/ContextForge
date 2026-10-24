import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.knowledge.models import Evidence, Fact, FactStatus


def get_current_facts(db: Session, entity_id: uuid.UUID) -> list[Fact]:
    return list(
        db.execute(
            select(Fact).where(Fact.entity_id == entity_id, Fact.status == FactStatus.CURRENT)
        ).scalars()
    )


def get_superseded_chain(db: Session, fact: Fact) -> list[Fact]:
    """Walk backwards from a current fact through supersedes_fact_id to
    recover its full version history, oldest last."""
    chain: list[Fact] = []
    current = fact
    while current.supersedes_fact_id is not None:
        previous = db.get(Fact, current.supersedes_fact_id)
        if previous is None:
            break
        chain.append(previous)
        current = previous
    return chain


def get_evidence_messages(db: Session, fact_id: uuid.UUID) -> list[Message]:
    return list(
        db.execute(
            select(Message)
            .join(Evidence, Evidence.message_id == Message.id)
            .where(Evidence.fact_id == fact_id)
            .order_by(Message.sent_at)
        ).scalars()
    )
