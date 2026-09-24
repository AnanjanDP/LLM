from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class SourceCitation(BaseModel):
    id: int
    text: str
    score: Optional[float] = None
    source_doc: Optional[str] = None


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096, description="User question or prompt")
    session_id: Optional[str] = Field(default="default", description="Conversation session ID")
    provider: Optional[str] = Field(default=None, description="LLM provider ('groq', 'mock')")
    model: Optional[str] = Field(default=None, description="LLM model name")
    use_rag: bool = Field(default=True, description="Enable context retrieval")
    retrieval_mode: Optional[str] = Field(default="hybrid", description="Retrieval mode ('vector', 'bm25', 'hybrid')")
    use_reranker: Optional[bool] = Field(default=False, description="Enable Cross-Encoder reranking stage")
    top_k: Optional[int] = Field(default=4, description="Top K context snippets")
    prompt_version: Optional[str] = Field(default="v1.0", description="Prompt template version ('v1.0', 'v1.1_strict', 'v2.0_cot')")
    stream: bool = Field(default=False, description="Enable SSE streaming response")


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    sources: Optional[List[SourceCitation]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    answer: str
    sources: List[SourceCitation]
    provider: str
    model: str
    latency_seconds: float
    latency_breakdown: Optional[Dict[str, float]] = None
    token_usage: Optional[Dict[str, Any]] = None
    cache_hit: bool = False


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)


class ConversationDetailOut(ConversationOut):
    messages: List[MessageOut] = []


class ConversationUpdate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class StreamChunk(BaseModel):
    content: str
    done: bool = False
    sources: Optional[List[SourceCitation]] = None