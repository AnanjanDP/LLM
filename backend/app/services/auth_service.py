from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends, Header
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token, decode_access_token
from app.core.exceptions import AuthenticationError, APIException
from app.models.db_models import User
from app.schemas.auth import UserCreate, UserLogin, Token, UserOut


class AuthService:
    """Authentication service handling registration, login, and authorization checks."""

    async def register_user(self, db: AsyncSession, user_in: UserCreate) -> User:
        """Register a new user after checking email uniqueness."""
        result = await db.execute(select(User).where(User.email == user_in.email))
        existing_user = result.scalar_one_or_none()
        if existing_user:
            raise APIException(message="User with this email already exists", status_code=400)

        hashed_pwd = get_password_hash(user_in.password)
        new_user = User(email=user_in.email, hashed_password=hashed_pwd)
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user

    async def authenticate_user(self, db: AsyncSession, user_in: UserLogin) -> Token:
        """Authenticate user credentials and issue a JWT access token."""
        result = await db.execute(select(User).where(User.email == user_in.email))
        user = result.scalar_one_or_none()
        if not user or not verify_password(user_in.password, user.hashed_password):
            raise AuthenticationError("Incorrect email or password")

        access_token = create_access_token(subject=user.id)
        return Token(access_token=access_token, token_type="bearer", user=UserOut.model_validate(user))


auth_service = AuthService()


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """Dependency that extracts current user if JWT token present, otherwise returns None for guest access."""
    if not authorization or not authorization.startswith("Bearer "):
        return None

    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_current_user_required(
    user: Optional[User] = Depends(get_current_user_optional)
) -> User:
    """Dependency that enforces mandatory authentication."""
    if not user:
        raise AuthenticationError("Authentication required for this resource")
    return user
