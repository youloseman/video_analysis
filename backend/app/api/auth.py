"""Email + password accounts: register / login / me (JWT bearer)."""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import (
    create_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class Credentials(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL_RE.match(v) or len(v) > 320:
            raise ValueError("Enter a valid email address.")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if not (8 <= len(v) <= 72):
            raise ValueError("Password must be 8–72 characters.")
        return v


class LoginBody(BaseModel):
    """Login validates nothing about the password (just checks it) -- length
    rules only apply at registration."""

    email: str
    password: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return v.strip().lower()


class TokenOut(BaseModel):
    token: str
    email: str
    tier: str
    is_pro: bool  # kept for back-compat; derived from tier


class UserOut(BaseModel):
    email: str
    tier: str
    is_pro: bool  # kept for back-compat; derived from tier


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register(body: Credentials, db: AsyncSession = Depends(get_session)) -> TokenOut:
    exists = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if exists:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "An account with this email already exists.",
        )
    # Promote the configured admin account on sign-up too, not only at startup,
    # so registering the admin email while the server is already running still
    # grants admin immediately.
    from app.core.config import settings
    from app.models.user import TIER_ADMIN, TIER_STARTER

    is_admin = bool(settings.admin_email) and body.email == settings.admin_email
    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        tier=TIER_ADMIN if is_admin else TIER_STARTER,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenOut(
        token=create_token(user.id), email=user.email,
        tier=user.tier, is_pro=user.is_paid,
    )


@router.post("/login", response_model=TokenOut)
async def login(body: LoginBody, db: AsyncSession = Depends(get_session)) -> TokenOut:
    user = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password.")
    return TokenOut(
        token=create_token(user.id), email=user.email,
        tier=user.tier, is_pro=user.is_paid,
    )


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(email=user.email, tier=user.tier, is_pro=user.is_paid)
