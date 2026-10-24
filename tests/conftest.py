"""Test database fixtures.

Uses a dedicated local Postgres database (contextforge_test) rather than
Testcontainers, since Docker isn't available in this environment. The
fixture still gives each test a clean, isolated transaction — only the
"spin up a throwaway container" part is swapped for "use an existing
local database", via TEST_DATABASE_URL.
"""

import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

# Import models so they register on Base.metadata.
from app.conversation.models import Message  # noqa: F401
from app.conversation.processing import MessageProcessingLog  # noqa: F401
from app.db.base import Base
from app.knowledge.models import (  # noqa: F401
    Claim,
    Entity,
    Evidence,
    ExtractedEntity,
    Fact,
)
from app.understanding.models import Classification  # noqa: F401

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://contextforge:contextforge@localhost:5432/contextforge_test",
)


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    """One test = one transaction, rolled back afterward for isolation."""
    connection = engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
