from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.understanding.classification import classify_pending_messages
from app.understanding.models import Classification
from app.understanding.schemas import ClassificationRead

router = APIRouter(prefix="/process")


@router.post("/classify", response_model=list[ClassificationRead])
def classify(db: Annotated[Session, Depends(get_db)]) -> list[Classification]:
    return classify_pending_messages(db)
