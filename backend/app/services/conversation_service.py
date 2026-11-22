import re
from typing import List, Dict, Optional, Tuple
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.core.exceptions import APIException, NotFoundError
from app.models.db_models import Conversation, Message
from app.schemas.chat import SourceCitation

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_GUEST_IDS = 50


class ConversationService:
    """Service layer managing Database persistence, conversation sessions, and context history.

    Access model:
      * A conversation created while signed in belongs to that user; nobody else can read, continue or delete it.
      * A guest conversation (no owner) is reachable only by whoever holds its id, so ids must be unguessable
        (the frontend generates random UUIDs) and a guest can never *list* other people's conversations.
    """

    MAX_HISTORY_MESSAGES = 10  # Context window sliding limit to avoid prompt token inflation

    @staticmethod
    def can_access(conv: Conversation, user_id: Optional[str]) -> bool:
        if conv.user_id is None:
            return True
        return user_id is not None and conv.user_id == user_id

    @staticmethod
    def validate_session_id(session_id: Optional[str]) -> None:
        if session_id and session_id != "default" and not SESSION_ID_PATTERN.match(session_id):
            raise APIException("Invalid session_id: use 1-64 letters, digits, '-' or '_'.", status_code=400)

    async def get_or_create_conversation(
        self, db: AsyncSession, conversation_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Conversation:
        """Fetch an existing conversation the caller may use, or start a new session."""
        self.validate_session_id(conversation_id)

        if conversation_id and conversation_id != "default":
            result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
            conv = result.scalar_one_or_none()
            if conv:
                if not self.can_access(conv, user_id):
                    # Same answer as "does not exist": do not reveal that the id is taken.
                    raise NotFoundError("Conversation")
                return conv

        new_conv = Conversation(
            id=conversation_id if conversation_id and conversation_id != "default" else None,
            user_id=user_id,
            title="New Conversation"
        )
        db.add(new_conv)
        await db.commit()
        await db.refresh(new_conv)
        return new_conv

    async def assert_session_access(self, db: AsyncSession, session_id: Optional[str], user_id: Optional[str]) -> None:
        """Raise 404 if `session_id` names a conversation the caller may not use. Unknown ids are fine (not created yet)."""
        self.validate_session_id(session_id)
        if not session_id or session_id == "default":
            return
        result = await db.execute(select(Conversation).where(Conversation.id == session_id))
        conv = result.scalar_one_or_none()
        if conv and not self.can_access(conv, user_id):
            raise NotFoundError("Conversation")

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

        return [{"role": msg.role, "content": msg.content} for msg in messages]

    async def save_interaction(
        self,
        db: AsyncSession,
        conversation_id: str,
        user_query: str,
        assistant_answer: str,
        sources: Optional[List[SourceCitation]] = None
    ) -> Tuple[Message, Message]:
        """Atomically persist user message and assistant answer to Database."""
        user_msg = Message(conversation_id=conversation_id, role="user", content=user_query)
        db.add(user_msg)

        source_data = [s.model_dump() for s in sources] if sources else None
        assistant_msg = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_answer,
            sources=source_data
        )
        db.add(assistant_msg)

        conv_result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        conv = conv_result.scalar_one_or_none()
        if conv and conv.title == "New Conversation":
            conv.title = user_query[:40] + ("..." if len(user_query) > 40 else "")

        await db.commit()
        await db.refresh(user_msg)
        await db.refresh(assistant_msg)
        return user_msg, assistant_msg

    async def get_user_conversations(
        self, db: AsyncSession, user_id: Optional[str] = None, guest_ids: Optional[List[str]] = None
    ) -> List[Conversation]:
        """Signed in: the user's own conversations. Guest: only the guest conversations whose ids they present."""
        query = select(Conversation).order_by(Conversation.updated_at.desc())
        if user_id:
            query = query.where(Conversation.user_id == user_id)
        else:
            ids = [i for i in (guest_ids or []) if SESSION_ID_PATTERN.match(i)][:MAX_GUEST_IDS]
            if not ids:
                return []
            query = query.where(Conversation.user_id.is_(None), Conversation.id.in_(ids))

        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_conversation_details(
        self, db: AsyncSession, conversation_id: str, user_id: Optional[str] = None
    ) -> Conversation:
        """Fetch a conversation with its messages, or raise 404 if it is missing or belongs to someone else."""
        result = await db.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages))
            .where(Conversation.id == conversation_id)
        )
        conv = result.scalar_one_or_none()
        if conv is None or not self.can_access(conv, user_id):
            raise NotFoundError("Conversation")
        return conv

    async def delete_conversation(self, db: AsyncSession, conversation_id: str, user_id: Optional[str] = None) -> None:
        """Delete a conversation (and its messages) the caller may access; 404 otherwise."""
        result = await db.execute(select(Conversation).where(Conversation.id == conversation_id))
        conv = result.scalar_one_or_none()
        if conv is None or not self.can_access(conv, user_id):
            raise NotFoundError("Conversation")
        await db.execute(delete(Message).where(Message.conversation_id == conversation_id))
        await db.execute(delete(Conversation).where(Conversation.id == conversation_id))
        await db.commit()


conversation_service = ConversationService()
