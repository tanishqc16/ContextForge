import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.llm_client import call_structured
from app.common.text_normalization import normalize_text
from app.conversation.models import Message
from app.conversation.processing import MessageProcessingLog, ProcessingStage
from app.knowledge.models import ExtractedEntity
from app.knowledge.schemas import EntityExtractionResult

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_INSTRUCTIONS = """\
Identify the distinct real-world things (entities) this message refers to —
for example a tool, system, credential, process, or policy. Do not extract
people's names. For each entity, give its exact surface text as it appears
in the message and a short open-vocabulary type label (e.g. "credential",
"process", "tool"). If the message names no concrete entity, return an
empty list.
"""


def extract_entities(text: str) -> EntityExtractionResult:
    return call_structured(
        instructions=ENTITY_EXTRACTION_INSTRUCTIONS,
        untrusted_text=text,
        response_schema=EntityExtractionResult,
    )


def extract_entities_for_pending_messages(db: Session) -> list[ExtractedEntity]:
    """Run entity extraction on every message not yet processed.

    Every message is attempted, regardless of its classification label — a
    long message that's mostly small talk can still have a real fact buried
    in it, and gating extraction on the classification risked silently
    dropping that fact. Classification stays purely descriptive.

    A message is marked processed (via MessageProcessingLog) even when it
    yields zero entities, so a legitimately entity-less message doesn't get
    reprocessed — and re-billed against the LLM quota — on every run.
    """
    already_processed = select(MessageProcessingLog.message_id).where(
        MessageProcessingLog.stage == ProcessingStage.ENTITY_EXTRACTION
    )
    pending = list(
        db.execute(select(Message).where(Message.id.not_in(already_processed))).scalars()
    )

    rows: list[ExtractedEntity] = []
    for message in pending:
        try:
            result = extract_entities(message.text)
        except Exception:
            logger.warning("Entity extraction failed for message %s", message.id, exc_info=True)
            continue

        for candidate in result.entities:
            row = ExtractedEntity(
                message_id=message.id,
                surface_text=candidate.surface_text,
                normalized_text=normalize_text(candidate.surface_text),
                entity_type=candidate.entity_type,
            )
            db.add(row)
            rows.append(row)

        db.add(
            MessageProcessingLog(
                message_id=message.id, stage=ProcessingStage.ENTITY_EXTRACTION
            )
        )
        db.commit()

    for row in rows:
        db.refresh(row)
    return rows
