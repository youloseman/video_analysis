"""Email + password accounts: register / login / me (JWT bearer)."""

from __future__ import annotations

import re

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import (
    create_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.services import analytics

logger = structlog.get_logger()
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
    # Unspent Expert Reviews (the Full tier includes one). Travels with the
    # session because the pricing screen has to offer "use the one you have"
    # instead of "buy one" -- and because a benefit nobody is told about is
    # indistinguishable from one that was never granted.
    expert_credits: int = 0
    # Who this browser is, for analytics only (see services/analytics.py). The
    # server sends purchases under the same id, which is what joins "signed up,
    # analysed twice, opened pricing" to "paid" as one person rather than two.
    analytics_id: str = ""


class UserOut(BaseModel):
    email: str
    tier: str
    is_pro: bool  # kept for back-compat; derived from tier
    expert_credits: int = 0
    analytics_id: str = ""


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
        expert_credits=user.expert_credits or 0,
        analytics_id=analytics.person_id(user),
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
        expert_credits=user.expert_credits or 0,
        analytics_id=analytics.person_id(user),
    )


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)) -> UserOut:
    return UserOut(
        email=user.email, tier=user.tier, is_pro=user.is_paid,
        expert_credits=user.expert_credits or 0,
        analytics_id=analytics.person_id(user),
    )


class DeleteAccountBody(BaseModel):
    """Re-authentication for a destructive, irreversible action.

    The bearer token lives in localStorage for 30 days, so "is signed in" is a
    weak proof of intent for account deletion -- an unlocked laptop is enough.
    The password is asked for again."""

    password: str


class DeletedOut(BaseModel):
    deleted: bool
    email: str


@router.delete("/account", response_model=DeletedOut)
async def delete_account(
    body: DeleteAccountBody,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> DeletedOut:
    """Erase the account and everything attached to it.

    The privacy policy promises this ("Until you ask us to delete your account";
    "Usage records deleted when your account is deleted") and until now only an
    email to a human could deliver it.

    Children are deleted explicitly rather than left to ``ON DELETE CASCADE``:
    SQLite does not enforce foreign keys unless the pragma is on, so relying on
    the cascade would silently leave orphans in local/dev databases -- exactly
    the rows we are promising to remove.

    What goes: saved analyses (including their stored keyframes), usage records,
    submitted feedback (which can carry an annotated photo of them), and order
    history. Uploaded footage is covered too: clips now outlive the analysis
    that produced them (see ``services.retention``), so they have to be removed
    here rather than left to run out their own retention period.
    """
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Wrong password — the account was not deleted.",
        )

    from app.models.analysis import Analysis
    from app.models.feedback import Feedback
    from app.models.order import ORDER_DELIVERED, ORDER_REFUNDED, Order
    from app.models.usage import UsageEvent

    # An Expert Review that was paid for but never delivered is money owed. It
    # is their account to delete, so this does not block -- but it must not
    # vanish silently either, or the obligation disappears with the row.
    owed = (
        await db.execute(
            select(Order.id).where(
                Order.user_id == user.id,
                Order.status.not_in((ORDER_DELIVERED, ORDER_REFUNDED)),
            )
        )
    ).scalars().all()
    if owed:
        logger.warning(
            "ACCOUNT_DELETED_WITH_OPEN_ORDERS",
            email=user.email, order_ids=list(owed),
            action="refund or contact the customer",
        )

    # Footage first, while the rows that point at it still exist. Uploads used
    # to expire hours after the analysis, so there was nothing here to delete;
    # they now outlive it by weeks, and an erased account that left its clips on
    # the volume would be the most literal way to break this promise.
    from app.api.me import _delete_stored_clips

    await _delete_stored_clips(db, user.id)

    uid, email = user.id, user.email
    for model in (Analysis, UsageEvent, Feedback, Order):
        await db.execute(delete(model).where(model.user_id == uid))
    await db.delete(user)
    await db.commit()
    logger.info("ACCOUNT_DELETED", user_id=uid)
    return DeletedOut(deleted=True, email=email)
