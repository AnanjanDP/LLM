from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserCreate(BaseModel):
    email: EmailStr
    # bcrypt only uses the first 72 bytes, so longer passwords would be silently truncated.
    password: str = Field(..., min_length=8, max_length=72, description="Password, 8 to 72 characters")


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(..., max_length=72)


class UserOut(BaseModel):
    id: str
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut
