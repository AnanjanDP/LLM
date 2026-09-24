from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.auth import UserCreate, UserLogin, Token, UserOut
from app.services.auth_service import auth_service, get_current_user_required
from app.models.db_models import User

router = APIRouter()


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user account."""
    user = await auth_service.register_user(db, user_in)
    return UserOut.model_validate(user)


@router.post("/token", response_model=Token)
async def login(user_in: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate user and generate JWT token."""
    return await auth_service.authenticate_user(db, user_in)


@router.get("/me", response_model=UserOut)
async def get_me(current_user: User = Depends(get_current_user_required)):
    """Fetch currently authenticated user details."""
    return UserOut.model_validate(current_user)
