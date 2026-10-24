from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session

from app.conversation.models import Message
from app.understanding import classification
from app.understanding.models import Classification, ClassificationLabel
from app.understanding.schemas import ClassificationResult


def test_classify_pending_messages_only_processes_unclassified(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    already_classified = Message(
        conversation_id="c1", sender="alice", text="already done", sent_at=datetime.now(UTC)
    )
    pending = Message(
        conversation_id="c1", sender="bob", text="is this a question?", sent_at=datetime.now(UTC)
    )
    db_session.add_all([already_classified, pending])
    db_session.flush()

    db_session.add(
        Classification(
            message_id=already_classified.id,
            label=ClassificationLabel.FACT,
            confidence=0.8,
            model_version="prior-run",
        )
    )
    db_session.flush()

    def fake_classify_message(text: str) -> ClassificationResult:
        return ClassificationResult(label=ClassificationLabel.QUESTION, confidence=0.99)

    monkeypatch.setattr(classification, "classify_message", fake_classify_message)

    results = classification.classify_pending_messages(db_session)

    assert len(results) == 1
    assert results[0].message_id == pending.id
    assert results[0].label == ClassificationLabel.QUESTION
    assert results[0].confidence == 0.99
