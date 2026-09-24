from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.database import get_db
from app.services.rag_service import rag_service

router = APIRouter()


@router.get("/")
def health():
    """Liveness check endpoint."""
    return {"status": "ok", "service": "RAG LLM Engine"}


@router.get("/ready")
async def readiness(db: AsyncSession = Depends(get_db)):
    """Readiness probe checking DB and Vector Store status."""
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    vector_count = rag_service.index.ntotal if rag_service.index else 0

    return {
        "status": "ready" if db_ok else "degraded",
        "database_connected": db_ok,
        "vector_index_vectors": vector_count,
        "embedding_model": rag_service.embedding_model.__class__.__name__ if rag_service.embedding_model else "None"
    }