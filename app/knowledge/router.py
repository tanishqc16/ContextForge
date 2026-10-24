from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.knowledge.claim_extraction import extract_claims_for_pending_messages
from app.knowledge.entity_extraction import extract_entities_for_pending_messages
from app.knowledge.entity_resolution import link_claims_to_entities, resolve_pending_entities
from app.knowledge.fact_management import promote_pending_claims
from app.knowledge.schemas import ClaimRead, ExtractedEntityRead, FactRead

router = APIRouter(prefix="/process")


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntityRead]
    claims: list[ClaimRead]


@router.post("/extract", response_model=ExtractionResult)
def extract(db: Annotated[Session, Depends(get_db)]) -> ExtractionResult:
    extracted_entities = extract_entities_for_pending_messages(db)
    extracted_claims = extract_claims_for_pending_messages(db)
    return ExtractionResult(
        entities=[ExtractedEntityRead.model_validate(e) for e in extracted_entities],
        claims=[ClaimRead.model_validate(c) for c in extracted_claims],
    )


class ResolutionResult(BaseModel):
    resolved_entities: list[ExtractedEntityRead]
    linked_claims: list[ClaimRead]


@router.post("/resolve", response_model=ResolutionResult)
def resolve(db: Annotated[Session, Depends(get_db)]) -> ResolutionResult:
    resolved_entities = resolve_pending_entities(db)
    linked_claims = link_claims_to_entities(db)
    return ResolutionResult(
        resolved_entities=[ExtractedEntityRead.model_validate(e) for e in resolved_entities],
        linked_claims=[ClaimRead.model_validate(c) for c in linked_claims],
    )


@router.post("/promote-facts", response_model=list[FactRead])
def promote_facts(db: Annotated[Session, Depends(get_db)]) -> list[FactRead]:
    facts = promote_pending_claims(db)
    return [FactRead.from_fact(f) for f in facts]
