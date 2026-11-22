import time
import json
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool, iterate_in_threadpool
from app.core.config import settings
from app.core.database import get_db
from app.core.exceptions import LLMProviderError
from app.core.logging import logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationDetailOut,
    MessageOut,
    SourceCitation,
)
from app.services.rag_service import rag_service
from app.services.llm_provider import llm_factory
from app.services.prompt_service import prompt_service
from app.services.conversation_service import conversation_service
from app.services.query_service import query_service
from app.services.cache_service import cache_service
from app.services.auth_service import get_current_user_optional
from app.models.db_models import User, RequestLog

router = APIRouter()


def _cache_fields(request: ChatRequest, session_id: str, cleaned_query: str, model_name: str) -> dict:
    """Everything that can change the answer, so one setting never gets another's cached reply."""
    return dict(
        query=cleaned_query,
        mode=request.retrieval_mode,
        model=model_name,
        provider=request.provider or settings.DEFAULT_LLM_PROVIDER,
        prompt=request.prompt_version or settings.DEFAULT_PROMPT_VERSION,
        top_k=request.top_k,
        rerank=bool(request.use_reranker),
        use_rag=request.use_rag,
        session=session_id,
        index_version=rag_service.version,
    )


async def _record_log(
    db: AsyncSession, request_id: str, user_id: Optional[str], endpoint: str, request: ChatRequest,
    latency_breakdown: dict, llm_latency: float, total_latency: float, token_usage: dict, cache_hit: bool,
) -> None:
    try:
        db.add(RequestLog(
            request_id=request_id,
            user_id=user_id,
            endpoint=endpoint,
            query=request.query,
            retrieval_mode=request.retrieval_mode,
            use_reranker=bool(request.use_reranker),
            vector_search_latency=latency_breakdown.get("vector_search_latency", 0.0),
            bm25_search_latency=latency_breakdown.get("bm25_search_latency", 0.0),
            hybrid_latency=latency_breakdown.get("hybrid_latency", 0.0),
            reranker_latency=latency_breakdown.get("reranker_latency", 0.0),
            llm_latency=llm_latency,
            total_latency=total_latency,
            input_tokens=(token_usage or {}).get("input_tokens", 0),
            output_tokens=(token_usage or {}).get("output_tokens", 0),
            estimated_cost_usd=(token_usage or {}).get("estimated_cost_usd", 0.0),
            cache_hit=cache_hit,
        ))
        await db.commit()
    except Exception as e:
        logger.warning(f"Could not persist RequestLog: {e}")
        await db.rollback()


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    fastapi_req: Request,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Process a query with hybrid retrieval, optional reranking, prompt versioning, caching, and cost profiling."""
    start_time = time.time()
    request_id = getattr(fastapi_req.state, "request_id", str(uuid.uuid4()))
    user_id = user.id if user else None

    cleaned_query = query_service.clean_query(request.query)
    vague_check = query_service.detect_unsupported_or_vague(cleaned_query)

    conv = await conversation_service.get_or_create_conversation(
        db, conversation_id=request.session_id, user_id=user_id
    )

    model_name = request.model or "default"
    cache_fields = _cache_fields(request, conv.id, cleaned_query, model_name)
    cached = cache_service.get_query_cache(**cache_fields)
    if cached:
        sources = [SourceCitation(**s) for s in cached.get("sources", [])]
        _, assistant_msg = await conversation_service.save_interaction(
            db=db,
            conversation_id=conv.id,
            user_query=request.query,
            assistant_answer=cached["answer"],
            sources=sources
        )
        total_lat = round(time.time() - start_time, 3)
        await _record_log(db, request_id, user_id, "/api/v1/chat", request, cached.get("latency_breakdown") or {},
                          0.0, total_lat, cached.get("token_usage") or {}, cache_hit=True)
        return ChatResponse(
            conversation_id=conv.id,
            message_id=assistant_msg.id,
            answer=cached["answer"],
            sources=sources,
            provider="cache",
            model=model_name,
            latency_seconds=total_lat,
            latency_breakdown=cached.get("latency_breakdown"),
            token_usage=cached.get("token_usage"),
            cache_hit=True
        )

    # Retrieval and generation are CPU/network bound and synchronous: run them off the event loop so one slow
    # request does not freeze every other client.
    sources: List[SourceCitation] = []
    context = ""
    latency_breakdown: dict = {}
    if request.use_rag and not vague_check["is_vague"]:
        sources, context, latency_breakdown = await run_in_threadpool(
            rag_service.retrieve,
            query=cleaned_query,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_reranker=request.use_reranker,
            session_id=conv.id,
        )

    history = await conversation_service.get_conversation_history(db, conv.id)
    messages, _ = prompt_service.build_chat_messages(
        query=cleaned_query,
        context=context,
        history=history,
        prompt_version_name=request.prompt_version
    )

    t_llm0 = time.time()
    provider = llm_factory.get_provider(request.provider)
    answer = await run_in_threadpool(provider.generate, messages=messages, model=request.model)
    llm_latency = round(time.time() - t_llm0, 4)
    latency_breakdown["llm_latency"] = llm_latency

    token_usage = prompt_service.estimate_tokens_and_cost(
        input_text=" ".join(m.get("content", "") for m in messages), output_text=answer, model_name=request.model
    )

    _, assistant_msg = await conversation_service.save_interaction(
        db=db,
        conversation_id=conv.id,
        user_query=request.query,
        assistant_answer=answer,
        sources=sources
    )

    total_latency = round(time.time() - start_time, 3)

    # Only real model answers are cached; the offline mock is not worth keeping.
    if provider.name != "mock":
        cache_service.set_query_cache(
            {
                "answer": answer,
                "sources": [s.model_dump() for s in sources],
                "latency_breakdown": latency_breakdown,
                "token_usage": token_usage,
            },
            **cache_fields,
        )

    await _record_log(db, request_id, user_id, "/api/v1/chat", request, latency_breakdown,
                      llm_latency, total_latency, token_usage, cache_hit=False)

    return ChatResponse(
        conversation_id=conv.id,
        message_id=assistant_msg.id,
        answer=answer,
        sources=sources,
        provider=provider.name,
        model=request.model or "default",
        latency_seconds=total_latency,
        latency_breakdown=latency_breakdown,
        token_usage=token_usage,
        cache_hit=False
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    fastapi_req: Request,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Server-Sent Events (SSE) streaming chat endpoint."""
    start_time = time.time()
    request_id = getattr(fastapi_req.state, "request_id", str(uuid.uuid4()))
    user_id = user.id if user else None
    cleaned_query = query_service.clean_query(request.query)
    vague_check = query_service.detect_unsupported_or_vague(cleaned_query)

    conv = await conversation_service.get_or_create_conversation(
        db, conversation_id=request.session_id, user_id=user_id
    )

    sources: List[SourceCitation] = []
    context = ""
    latency_breakdown: dict = {}
    if request.use_rag and not vague_check["is_vague"]:
        sources, context, latency_breakdown = await run_in_threadpool(
            rag_service.retrieve,
            query=cleaned_query,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            use_reranker=request.use_reranker,
            session_id=conv.id,
        )

    history = await conversation_service.get_conversation_history(db, conv.id)
    messages, _ = prompt_service.build_chat_messages(
        query=cleaned_query,
        context=context,
        history=history,
        prompt_version_name=request.prompt_version
    )

    provider = llm_factory.get_provider(request.provider)

    async def event_generator():
        accumulated_text: List[str] = []
        t_llm0 = time.time()
        try:
            initial_meta = {
                "conversation_id": conv.id,
                "sources": [s.model_dump() for s in sources],
                "latency_breakdown": latency_breakdown,
                "provider": provider.name,
            }
            yield f"data: {json.dumps({'type': 'metadata', 'data': initial_meta})}\n\n"

            # The provider yields from a blocking HTTP stream; pull it through a worker thread.
            async for chunk in iterate_in_threadpool(
                provider.generate_stream(messages=messages, model=request.model)
            ):
                accumulated_text.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"

            full_answer = "".join(accumulated_text)
            llm_latency = round(time.time() - t_llm0, 4)

            await conversation_service.save_interaction(
                db=db,
                conversation_id=conv.id,
                user_query=request.query,
                assistant_answer=full_answer,
                sources=sources
            )
            token_usage = prompt_service.estimate_tokens_and_cost(
                input_text=" ".join(m.get("content", "") for m in messages),
                output_text=full_answer,
                model_name=request.model,
            )
            await _record_log(db, request_id, user_id, "/api/v1/chat/stream", request,
                              {**latency_breakdown, "llm_latency": llm_latency}, llm_latency,
                              round(time.time() - start_time, 3), token_usage, cache_hit=False)

            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv.id})}\n\n"

        except LLMProviderError as e:
            # These messages are written to be shown to users (for example "the Groq API key was rejected").
            logger.error(f"SSE Streaming provider error: {e.message}")
            yield f"data: {json.dumps({'type': 'error', 'error': e.message})}\n\n"
        except Exception:
            logger.exception("SSE Streaming error")
            yield f"data: {json.dumps({'type': 'error', 'error': 'The response could not be completed.'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/conversations", response_model=List[ConversationOut])
async def list_conversations(
    ids: Optional[str] = Query(None, description="Guests only: comma-separated ids of their own conversations"),
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Signed in: your conversations. Guest: only the conversations whose ids you pass in `ids`."""
    guest_ids = [i.strip() for i in ids.split(",") if i.strip()] if ids else []
    conversations = await conversation_service.get_user_conversations(
        db, user_id=user.id if user else None, guest_ids=guest_ids
    )
    return [ConversationOut.model_validate(c) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Fetch a conversation's history. 404 if it does not exist or is someone else's."""
    conv = await conversation_service.get_conversation_details(db, conversation_id, user.id if user else None)

    return ConversationDetailOut(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=[
            MessageOut(id=m.id, role=m.role, content=m.content, sources=m.sources, created_at=m.created_at)
            for m in conv.messages
        ],
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Delete one of your conversations."""
    await conversation_service.delete_conversation(db, conversation_id, user.id if user else None)
    return None
