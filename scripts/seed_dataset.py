"""Load the synthetic dataset into the dev database.

Usage: .venv/bin/python scripts/seed_dataset.py
"""

from pathlib import Path

from app.conversation.ingestion import ingest_messages, load_messages_from_file
from app.db.session import SessionLocal

DATASET_PATH = Path(__file__).parent.parent / "data" / "synthetic" / "internal_team_chat.json"


def main() -> None:
    messages = load_messages_from_file(DATASET_PATH)
    with SessionLocal() as db:
        rows = ingest_messages(db, messages)
    print(f"Ingested {len(rows)} messages from {DATASET_PATH.name}")


if __name__ == "__main__":
    main()
