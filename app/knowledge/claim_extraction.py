import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.common.llm_client import call_structured
from app.conversation.models import Message
from app.conversation.processing import MessageProcessingLog, ProcessingStage
from app.knowledge.models import Claim
from app.knowledge.schemas import ClaimExtractionResult

logger = logging.getLogger(__name__)

CLAIM_EXTRACTION_INSTRUCTIONS = """\
Identify any factual claims this message makes about a thing — a statement
of status, value, or behavior that could be checked or could change over
time (e.g. "the staging key is deprecated", "deploy script is now
./deploy.sh --env=prod"). For each claim, give:
- subject_surface_text: the exact text naming what the claim is about
- predicate: a short open-vocabulary label for what aspect is being claimed
  (e.g. "status", "deploy_command", "owner")
- value: the claimed value, as a short plain string (e.g. "deprecated")
- context: ONLY set this if the claim is scoped to a specific condition
  that would make it false in other situations (e.g. "internal tools only"
  for a claim that doesn't hold for external use). This must change what
  the claim means, not just restate incidental detail from the sentence —
  phrases like "right now", "today", or "by someone" are NOT context, they
  describe when/who, not a different scope the claim applies under. When in
  doubt, omit context entirely; most claims have none.
- confidence: how clearly the message states this, from 0 to 1

Questions, greetings, and opinions are not claims. If the message makes no
factual claim, return an empty list.
"""


def extract_claims(text: str) -> ClaimExtractionResult:
    return call_structured(
        instructions=CLAIM_EXTRACTION_INSTRUCTIONS,
        untrusted_text=text,
        response_schema=ClaimExtractionResult,
    )


def extract_claims_for_pending_messages(db: Session) -> list[Claim]:
    """Run claim extraction on every message not yet processed.

    Every message is attempted, regardless of its classification label —
    see the matching note in entity_extraction.py for why classification no
    longer gates this.

    Same idempotency rule as entity extraction: a message is marked
    processed even if it yields zero claims, so it isn't reprocessed.
    """
    already_processed = select(MessageProcessingLog.message_id).where(
        MessageProcessingLog.stage == ProcessingStage.CLAIM_EXTRACTION
    )
    pending = list(
        db.execute(select(Message).where(Message.id.not_in(already_processed))).scalars()
    )

    rows: list[Claim] = []
    for message in pending:
        try:
            result = extract_claims(message.text)
        except Exception:
            logger.warning("Claim extraction failed for message %s", message.id, exc_info=True)
            continue

        for candidate in result.claims:
            row = Claim(
                message_id=message.id,
                subject_surface_text=candidate.subject_surface_text,
                predicate=candidate.predicate,
                value={"value": candidate.value},
                context={"note": candidate.context} if candidate.context else None,
                extracted_confidence=candidate.confidence,
            )
            db.add(row)
            rows.append(row)

        db.add(MessageProcessingLog(message_id=message.id, stage=ProcessingStage.CLAIM_EXTRACTION))
        db.commit()

    for row in rows:
        db.refresh(row)
    return rows
