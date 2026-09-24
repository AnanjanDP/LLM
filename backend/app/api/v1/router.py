from fastapi import APIRouter
from app.api.v1.endpoints import auth, chat, upload, eval

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat & Conversations"])
api_router.include_router(upload.router, prefix="", tags=["Document Ingestion"])
api_router.include_router(eval.router, prefix="/eval", tags=["LLM Evaluation"])
