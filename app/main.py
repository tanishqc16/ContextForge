from fastapi import FastAPI

from app.conversation.router import router as conversation_router
from app.knowledge.router import router as knowledge_router
from app.retrieval.router import router as retrieval_router
from app.understanding.router import router as understanding_router

app = FastAPI(title="ContextForge")
app.include_router(conversation_router)
app.include_router(understanding_router)
app.include_router(knowledge_router)
app.include_router(retrieval_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
