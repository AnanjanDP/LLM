import time
import json
import uuid
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.logging import logger
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationOut,
    ConversationDetailOut,
    ConversationUpdate,
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


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    fastapi_req: Request,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Process user query with hybrid retrieval, reranking, prompt versioning, caching, and token cost profiling."""
    start_time = time.time()
    request_id = getattr(fastapi_req.state, "request_id", str(uuid.uuid4()))
    user_id = user.id if user else None

    cleaned_query = query_service.clean_query(request.query)
    vague_check = query_service.detect_unsupported_or_vague(cleaned_query)

    # 1. Get/Create Conversation Session
    conv = await conversation_service.get_or_create_conversation(
        db, conversation_id=request.session_id, user_id=user_id
    )

    # 2. Check Redis Cache
    mode = request.retrieval_mode or "hybrid"
    model_name = request.model or "default"
    cached = cache_service.get_query_cache(cleaned_query, mode=mode, model=model_name, session_id=conv.id)
    if cached:
        sources = [SourceCitation(**s) for s in cached.get("sources", [])]
        user_msg, assistant_msg = await conversation_service.save_interaction(
            db=db,
            conversation_id=conv.id,
            user_query=request.query,
            assistant_answer=cached["answer"],
            sources=sources
        )
        total_lat = round(time.time() - start_time, 3)
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

    # 3. Retrieve RAG Context
    sources = []
    context = ""
    latency_breakdown = {}
    if request.use_rag and not vague_check["is_vague"]:
        sources, context, latency_breakdown = rag_service.retrieve(
            query=cleaned_query,
            top_k=request.top_k or 4,
            retrieval_mode=request.retrieval_mode,
            use_reranker=request.use_reranker,
            session_id=conv.id
        )

    # 4. Retrieve sliding conversation history
    history = await conversation_service.get_conversation_history(db, conv.id)

    # 5. Construct Prompt Messages
    messages, p_ver = prompt_service.build_chat_messages(
        query=cleaned_query,
        context=context,
        history=history,
        prompt_version_name=request.prompt_version
    )

    # 6. Invoke LLM Provider
    t_llm0 = time.time()
    provider = llm_factory.get_provider(request.provider)
    answer = provider.generate(messages=messages, model=request.model)
    llm_latency = round(time.time() - t_llm0, 4)
    latency_breakdown["llm_latency"] = llm_latency

    # 7. Token Usage & Cost Estimation
    full_input_str = " ".join(m.get("content", "") for m in messages)
    token_usage = prompt_service.estimate_tokens_and_cost(
        input_text=full_input_str, output_text=answer, model_name=request.model
    )

    # 8. Save interaction to DB
    user_msg, assistant_msg = await conversation_service.save_interaction(
        db=db,
        conversation_id=conv.id,
        user_query=request.query,
        assistant_answer=answer,
        sources=sources
    )

    total_latency = round(time.time() - start_time, 3)

    # 9. Store in Redis Cache
    cache_payload = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
        "latency_breakdown": latency_breakdown,
        "token_usage": token_usage
    }
    cache_service.set_query_cache(cleaned_query, mode=mode, model=model_name, data=cache_payload, session_id=conv.id)


    # 10. Record Request Log in DB
    try:
        req_log = RequestLog(
            request_id=request_id,
            user_id=user_id,
            endpoint="/api/v1/chat",
            query=request.query,
            retrieval_mode=mode,
            use_reranker=bool(request.use_reranker),
            vector_search_latency=latency_breakdown.get("vector_search_latency", 0.0),
            bm25_search_latency=latency_breakdown.get("bm25_search_latency", 0.0),
            hybrid_latency=latency_breakdown.get("hybrid_latency", 0.0),
            reranker_latency=latency_breakdown.get("reranker_latency", 0.0),
            llm_latency=llm_latency,
            total_latency=total_latency,
            input_tokens=token_usage.get("input_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0),
            estimated_cost_usd=token_usage.get("estimated_cost_usd", 0.0),
            cache_hit=False
        )
        db.add(req_log)
        await db.commit()
    except Exception as e:
        logger.warning(f"Could not persist RequestLog: {e}")

    return ChatResponse(
        conversation_id=conv.id,
        message_id=assistant_msg.id,
        answer=answer,
        sources=sources,
        provider=provider.__class__.__name__.replace("Provider", "").lower(),
        model=request.model or "default",
        latency_seconds=total_latency,
        latency_breakdown=latency_breakdown,
        token_usage=token_usage,
        cache_hit=False
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """Server-Sent Events (SSE) streaming chat endpoint."""
    user_id = user.id if user else None
    cleaned_query = query_service.clean_query(request.query)
    
    conv = await conversation_service.get_or_create_conversation(
        db, conversation_id=request.session_id, user_id=user_id
    )

    sources = []
    context = ""
    latency_breakdown = {}
    if request.use_rag:
        sources, context, latency_breakdown = rag_service.retrieve(
            query=cleaned_query,
            top_k=request.top_k or 4,
            retrieval_mode=request.retrieval_mode,
            use_reranker=request.use_reranker,
            session_id=conv.id
        )


    history = await conversation_service.get_conversation_history(db, conv.id)
    messages, p_ver = prompt_service.build_chat_messages(
        query=cleaned_query,
        context=context,
        history=history,
        prompt_version_name=request.prompt_version
    )

    provider = llm_factory.get_provider(request.provider)

    async def event_generator():
        accumulated_text = []
        try:
            initial_meta = {
                "conversation_id": conv.id,
                "sources": [s.model_dump() for s in sources],
                "latency_breakdown": latency_breakdown
            }
            yield f"data: {json.dumps({'type': 'metadata', 'data': initial_meta})}\n\n"

            for chunk in provider.generate_stream(messages=messages, model=request.model):
                accumulated_text.append(chunk)
                yield f"data: {json.dumps({'type': 'token', 'token': chunk})}\n\n"
                await asyncio.sleep(0)

            full_answer = "".join(accumulated_text)

            await conversation_service.save_interaction(
                db=db,
                conversation_id=conv.id,
                user_query=request.query,
                assistant_answer=full_answer,
                sources=sources
            )

            yield f"data: {json.dumps({'type': 'done', 'conversation_id': conv.id})}\n\n"

        except Exception as e:
            logger.error(f"SSE Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

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

    db: AsyncSession = Depends(get_db),
    user: Optional[User] = Depends(get_current_user_optional)
):
    """List conversation sessions for user."""
    user_id = user.id if user else None
    conversations = await conversation_service.get_user_conversations(db, user_id=user_id)
    return [ConversationOut.model_validate(c) for c in conversations]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Fetch complete conversation history by ID."""
    conv = await conversation_service.get_conversation_details(db, conversation_id)
    if not conv:
        return ConversationDetailOut(id=conversation_id, title="Not Found", created_at=time.time(), updated_at=time.time(), messages=[])
    
    messages_out = []
    for msg in conv.messages:
        messages_out.append(MessageOut(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            sources=msg.sources,
            created_at=msg.created_at
        ))

    return ConversationDetailOut(
        id=conv.id,
        title=conv.title,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
        messages=messages_out
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a conversation session."""
    await conversation_service.delete_conversation(db, conversation_id)
    return None
