import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.llm_client import call_structured
from app.config import get_settings
from app.conversation.models import Message
from app.understanding.models import Classification
from app.understanding.schemas import ClassificationResult

logger = logging.getLogger(__name__)

CLASSIFICATION_INSTRUCTIONS = """\
Classify the chat message into exactly one of these categories:

QUESTION - asking for information
ANSWER - directly answering a question
FACT - stating a fact or piece of information
REQUEST - asking someone to do something
ISSUE - reporting a problem or bug
FEEDBACK - giving an opinion or feedback
UPDATE - announcing a change or new information
CORRECTION - correcting previously stated information
DECISION - stating a decision that was made
DISCUSSION - general discussion not fitting other categories
IRRELEVANT - small talk or off-topic chatter with no useful information

Respond with the single best-fitting category and a confidence between 0 and 1.
"""


def classify_message(text: str) -> ClassificationResult:
    return call_structured(
        instructions=CLASSIFICATION_INSTRUCTIONS,
        untrusted_text=text,
        response_schema=ClassificationResult,
    )


def classify_pending_messages(db: Session) -> list[Classification]:
    """Classify every message that has no Classification row yet.

    Each message is committed as soon as it's classified, and one message's
    failure (e.g. a transient LLM outage) doesn't discard progress on the
    others — a re-run only retries what's still actually pending.
    """
    already_classified = select(Classification.message_id)
    pending = list(
        db.execute(select(Message).where(Message.id.not_in(already_classified))).scalars()
    )

    model_version = get_settings().gemini_model
    rows: list[Classification] = []
    for message in pending:
        try:
            result = classify_message(message.text)
        except Exception:
            logger.warning("Classification failed for message %s", message.id, exc_info=True)
            continue

        row = Classification(
            message_id=message.id,
            label=result.label,
            confidence=result.confidence,
            model_version=model_version,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        rows.append(row)

    return rows
