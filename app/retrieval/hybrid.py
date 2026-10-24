import logging

from sqlalchemy.orm import Session

from app.common.llm_client import call_structured
from app.retrieval.schemas import (
    AnswerResult,
    AnswerSynthesis,
    CurrentFactItem,
    EvidenceItem,
    SupersededFactItem,
)
from app.retrieval.semantic_search import find_best_matching_entity
from app.retrieval.structured_search import (
    get_current_facts,
    get_evidence_messages,
    get_superseded_chain,
)

logger = logging.getLogger(__name__)

NO_ENTITY_ANSWER = "No reliable answer found — couldn't identify what this question is about."
NO_FACTS_ANSWER = "No reliable current knowledge found about this."

SYNTHESIS_INSTRUCTIONS = """\
Answer the user's question using ONLY the facts listed below. Do not add,
infer, or assume anything not explicitly present in these facts. If the
facts don't actually answer the question, say so honestly instead of
guessing.
"""


def _format_value(value: dict[str, object]) -> object:
    return value.get("value", value)


def answer_question(db: Session, question: str) -> AnswerResult:
    """The payoff of the whole pipeline: find the entity, pull its current
    knowledge + evidence + history, and phrase an answer that can't contain
    anything beyond what was actually retrieved."""
    entity = find_best_matching_entity(db, question)
    if entity is None:
        return AnswerResult(
            question=question,
            entity_name=None,
            answer=NO_ENTITY_ANSWER,
            confidence=0.0,
            current_facts=[],
            superseded_facts=[],
        )

    current_facts = get_current_facts(db, entity.id)
    if not current_facts:
        return AnswerResult(
            question=question,
            entity_name=entity.canonical_name,
            answer=NO_FACTS_ANSWER,
            confidence=0.0,
            current_facts=[],
            superseded_facts=[],
        )

    current_items: list[CurrentFactItem] = []
    superseded_items: list[SupersededFactItem] = []
    for fact in current_facts:
        messages = get_evidence_messages(db, fact.id)
        current_items.append(
            CurrentFactItem(
                predicate=fact.predicate,
                value=fact.value,
                context=fact.context,
                confidence=fact.confidence,
                valid_from=fact.valid_range.lower,
                evidence=[
                    EvidenceItem(message_id=m.id, message_text=m.text, sent_at=m.sent_at)
                    for m in messages
                ],
            )
        )
        for old_fact in get_superseded_chain(db, fact):
            superseded_items.append(
                SupersededFactItem(
                    predicate=old_fact.predicate,
                    value=old_fact.value,
                    context=old_fact.context,
                    valid_from=old_fact.valid_range.lower,
                    valid_until=old_fact.valid_range.upper,
                )
            )

    facts_lines = []
    for item in current_items:
        line = f"- {item.predicate} = {_format_value(item.value)}"
        if item.context:
            line += f" (context: {_format_value(item.context)})"
        facts_lines.append(line)
    facts_text = "\n".join(facts_lines)

    synthesis_input = (
        f"Question: {question}\n\n"
        f"Known current facts about '{entity.canonical_name}':\n{facts_text}"
    )

    try:
        synthesis = call_structured(
            instructions=SYNTHESIS_INSTRUCTIONS,
            untrusted_text=synthesis_input,
            response_schema=AnswerSynthesis,
        )
        answer_text = synthesis.answer
    except Exception:
        logger.warning("Answer synthesis failed, falling back to raw facts", exc_info=True)
        answer_text = facts_text

    avg_confidence = sum(item.confidence for item in current_items) / len(current_items)

    return AnswerResult(
        question=question,
        entity_name=entity.canonical_name,
        answer=answer_text,
        confidence=avg_confidence,
        current_facts=current_items,
        superseded_facts=superseded_items,
    )
