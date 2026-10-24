from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.retrieval.hybrid import answer_question
from app.retrieval.schemas import AnswerResult

router = APIRouter(prefix="/knowledge")


@router.get("/ask", response_model=AnswerResult)
def ask(question: str, db: Annotated[Session, Depends(get_db)]) -> AnswerResult:
    return answer_question(db, question)
