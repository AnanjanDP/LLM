from typing import List, Dict, Optional, Tuple
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.db_models import Conversation, Message
from app.schemas.chat import SourceCitation
from app.core.logging import logger


class ConversationService:
    """Service layer managing Database persistence, conversation sessions, and context history."""

    MAX_HISTORY_MESSAGES = 10  # Context window sliding limit to avoid prompt token inflation

    async def get_or_create_conversation(
        self, db: AsyncSession, conversation_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Conversation:
        """Fetch existing conversation by ID or instantiate a new session."""
        if conversation_id and conversation_id != "default":
            result = await db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if conv:
                return conv

        # Create new conversation
        new_conv = Conversation(
            id=conversation_id if conversation_id and conversation_id != "default" else None,
            user_id=user_id,
            title="New Conversation"
        )
        db.add(new_conv)
        await db.commit()
        await db.refresh(new_conv)
        return new_conv

    async def get_conversation_history(
        self, db: AsyncSession, conversation_id: str, limit: int = MAX_HISTORY_MESSAGES
    ) -> List[Dict[str, str]]:
        """Retrieve sliding window of recent messages formatted for LLM completion context."""
        result = await db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = list(result.scalars().all())
        messages.reverse()  # Order chronologically

        history = []
        for msg in messages:
            history.append({
                "role": msg.role,
                "content": msg.content
            })
        return history

    async def save_interaction(
        self,
        db: AsyncSession,
        conversation_id: str,
        user_query: str,
        assistant_answer: str,
        sources: Optional[List[SourceCitation]] = None
    ) -> Tuple[Message, Message]:
        """Atomically persist user message and assistant answer to Database."""
        # 1. User message
        user_msg = Message(
            conversation_id=conversation_id,
            role="user",
            content=user_query
        )
        db.add(user_msg)

        # 2. Assistant message
        source_data = [s.model_dump() for s in sources] if sources else None
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_answer,
            sources=source_data
        )
        db.add(assistant_msg)

        # Update conversation title if default
        conv_result = await db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conv = conv_result.scalar_one_or_none()
        if conv and conv.title == "New Conversation":
            # Auto-title based on user query
            conv.title = user_query[:40] + ("..." if len(user_query) > 40 else "")

        await db.commit()
        await db.refresh(user_msg)
        await db.refresh(assistant_msg)
        return user_msg, assistant_msg

    async def get_user_conversations(
        self, db: AsyncSession, user_id: Optional[str] = None
    ) -> List[Conversation]:
        """Fetch all conversations for a specific user (or guest sessions)."""
        query = select(Conversation).order_by(Conversation.updated_at.desc())
        if user_id:
            query = query.where(Conversation.user_id == user_id)
        
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_conversation_details(
        self, db: AsyncSession, conversation_id: str
    ) -> Optional[Conversation]:
        """Fetch full conversation details including all ordered messages."""
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def delete_conversation(self, db: AsyncSession, conversation_id: str) -> bool:
        """Delete a conversation and all cascading messages."""
        result = await db.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        await db.commit()
        return result.rowcount > 0


conversation_service = ConversationService()
