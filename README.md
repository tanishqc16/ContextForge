# ContextForge

ContextForge is a backend system for turning messy conversations into **structured, reliable knowledge**.

The current input is WhatsApp-style chat exports. Instead of treating every message as an isolated piece of text, the system tries to understand what each message means, connect related messages, track changes over time, and make the resulting information easy to retrieve later.

WhatsApp is just the first source. The idea is:

**conversation → knowledge → reliable context → retrieval**

The interesting part of the project is the layer in between raw conversations and retrieval. It is about building a useful representation of information that can later be used by other systems.

This is also why the project is relevant to the kind of problem companies like **OneClarity** are working on — making information spread across different sources easier to understand and use. But ContextForge itself is intentionally much narrower: it focuses on turning conversations into reliable, retrievable knowledge.

---

## Why build this?

A lot of useful information gets buried inside conversations.

Someone says something, someone else corrects it later, another person refers to the same thing differently, and a few days later the original message is hard to find.

The hard part isn't storing the messages. It's figuring out what they actually mean together.

For example:

- Which messages are talking about the same thing?
- Did something change later?
- Which version is currently valid?
- Where did a particular fact come from?
- Are two different-looking statements actually about different contexts?
- What don't we know yet?

That's the problem ContextForge is trying to solve.

## The problem

A simple approach would be:

```text
Conversation → LLM → Extract JSON → Store in Postgres
```

That works for the easy part. The harder part starts after the extraction.

Real conversations are messy. You can have:

- the same information repeated in different messages
- corrections and contradictions
- information that changes over time
- different names or ways of referring to the same thing
- people with different levels of authority
- questions that nobody answered

If we just store whatever the LLM extracts, we still have no good way of knowing what is currently true, what changed, or why a particular piece of information should be trusted.

So the main question ContextForge tries to answer is:

> **"What do we currently know about X?"**

And if there isn't enough reliable information, the system should be able to say:

> **"We don't know."**

---

## Architecture

```text
                  ┌───────────────────────────┐
                  │     Raw Chat Messages      │
                  │   (WhatsApp-style input)   │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │        INGESTION          │
                  │   immutable message store │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │      CLASSIFICATION       │
                  │  (AI) question / answer / │
                  │  correction / irrelevant  │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │       UNDERSTANDING       │
                  │  entity extraction (AI)   │
                  │  claim extraction (AI)    │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │      KNOWLEDGE ENGINE     │
                  │  entity resolution         │
                  │   (embeddings + LLM)       │
                  │  predicate resolution      │
                  │  fact versioning & evidence│
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │          STORAGE          │
                  │   PostgreSQL + pgvector   │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │        RETRIEVAL          │
                  │  semantic + structured    │
                  │  search, grounded answer  │
                  │  synthesis (AI)           │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌───────────────────────────┐
                  │  Answer + Evidence +      │
                  │  History — or an honest   │
                  │  "no reliable answer found"│
                  └───────────────────────────┘
```

### One important design choice

The LLM isn't responsible for everything.

We use it when we actually need language understanding — for example:

- classifying a message
- extracting entities and claims
- helping with an ambiguous entity match
- phrasing the final answer

Things like fact versioning, supersession, confidence thresholds, and ranking are handled in normal code.

This keeps the system easier to test, debug, and understand instead of making the whole thing **"ask the LLM and hope."**

---

## Core concepts

| Concept | What it is |
|---|---|
| **Message** | The original chat message. It's immutable and stored exactly as received. |
| **Classification** | What kind of message it is — question, answer, correction, update, irrelevant, etc. |
| **Entity** | A canonical version of a real-world "thing", so different messages can refer to the same thing. |
| **Claim** | Something we extracted from a message that might be true. It's not trusted yet. |
| **Fact** | A claim that has been accepted into the knowledge base and versioned over time. |
| **Evidence** | The link from a Fact back to the exact Claim and Message that support it. |

Facts are **never overwritten**.

If something changes, we create a new Fact version and mark the old one `SUPERSEDED`. This keeps the history instead of losing it.

If two facts are actually about different contexts — for example, "for internal tools" vs. "for external partners" — they can have separate Fact chains instead of being treated as a conflict.

This gives us three useful things:

- **Traceability** — we can go back to the original message behind a fact.
- **Time awareness** — we can see how information changed.
- **Reliability** — we can separate what we know from what we aren't confident about.

---

## Tech stack

- **Python 3.12**, **FastAPI**, **SQLAlchemy 2.0**, **Pydantic v2**
- **PostgreSQL + pgvector** — single database for structured data and vector similarity
- **Alembic** — versioned schema migrations
- **Google Gemini** (free tier) — the LLM, used only for classification/extraction/judgment calls
- **sentence-transformers** (local, free) — embeddings for entity/predicate resolution, no API cost
- **pytest, ruff, mypy** — tests, linting, strict type-checking

No Docker, no Kafka, no microservices — this is a modular monolith by design.

---

## What's built so far

- Raw message ingestion with immutable storage
- LLM-based classification applied to every message, so a useful fact inside an otherwise casual message isn't silently skipped
- Entity and claim extraction using structured LLM output
- Entity resolution using normalization → embedding similarity → LLM tiebreak for ambiguous cases
- Predicate resolution, so differently-worded attributes such as `status` and `booking_status` can be recognized as the same thing for a given entity
- Fact promotion with versioning, supersession, corroboration, and context-based separation
- Hybrid retrieval using semantic entity matching + structured fact lookup + grounded answer synthesis
- The LLM can only phrase facts that were already retrieved. It cannot invent new facts during answer generation.

---

## Known limitations

There are still a few things this project doesn't solve yet:

- **Claim-to-entity linking is intentionally strict** — if a claim can't be confidently matched to an entity, we leave it unlinked instead of guessing. This is safer, but it also means some corrections may not reach the Fact layer yet.
- **No conflict detection yet** — genuine contradictions vs. different contexts are currently handled as part of fact promotion, rather than by a separate conflict-detection step.
- **No knowledge-gap detection yet** — if a question keeps coming up without a reliable answer, the system doesn't automatically flag it as a knowledge gap.
- **No formal evaluation harness** — correctness has been checked through hand-testing against real data, not a labeled dataset with precision/recall metrics.
- **Thresholds** such as similarity bands and confidence cutoffs were tuned using this project's test data and haven't been exhaustively validated.
- **No PII handling** — the current setup is fine for synthetic test data, but proper PII handling would be required before using real chat exports.

---

## Running the project

### 1. Prerequisites

- Python 3.12+
- PostgreSQL with the `pgvector` extension available
- A free [Google Gemini API key](https://aistudio.google.com)

### 2. Set up the database

```bash
# Example using Homebrew Postgres (adjust for your setup)
createuser -s contextforge
psql -d postgres -c "ALTER USER contextforge WITH PASSWORD 'contextforge';"
createdb -O contextforge contextforge
psql -d contextforge -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### 3. Set up the Python environment

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```text
DATABASE_URL=postgresql+psycopg://contextforge:contextforge@localhost:5432/contextforge
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

### 5. Run database migrations

```bash
.venv/bin/python -m alembic upgrade head
```

### 6. Load the synthetic dataset

```bash
.venv/bin/python scripts/seed_dataset.py
```

### 7. Start the server

```bash
.venv/bin/uvicorn app.main:app --port 8000
```

### 8. Run the pipeline

```bash
curl -X POST http://localhost:8000/process/classify
curl -X POST http://localhost:8000/process/extract
curl -X POST http://localhost:8000/process/resolve
curl -X POST http://localhost:8000/process/promote-facts
```

### 9. Ask it a question

```bash
curl -G "http://localhost:8000/knowledge/ask" --data-urlencode "question=what is the current status of staging?"
```

Or just open that URL directly in a browser.

### 10. Run the test suite

```bash
.venv/bin/python -m pytest tests/ -v
```

---

## API endpoints

| Method | Path | What it does |
|---|---|---|
| `POST` | `/ingest` | Load raw messages |
| `GET` | `/messages` | List stored messages |
| `POST` | `/process/classify` | Classify pending messages |
| `POST` | `/process/extract` | Extract entities and claims |
| `POST` | `/process/resolve` | Resolve entities and link claims |
| `POST` | `/process/promote-facts` | Promote confident claims into versioned facts |
| `GET` | `/knowledge/ask?question=...` | Ask a question, get a grounded answer with evidence |
| `GET` | `/health` | Health check |
