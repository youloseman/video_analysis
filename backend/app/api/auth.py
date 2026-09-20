"""Email + password accounts: register / login / me / password reset (JWT bearer)."""

from __future__ import annotations

import html
import re
import time
from collections import deque

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_session
from app.core.net import client_ip
from app.core.security import (
    RESET_TOKEN_TTL,
    create_reset_token,
    create_token,
    get_current_user,
    hash_password,
    verify_password,
    verify_reset_token,
)
from app.models.user import User
from app.services import analytics, notify

logger = structlog.get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------------------------------------------------------------------------
# Attempt throttling. In-memory and per-process, like the analysis limiter in
# main.py -- this service runs one worker. The keys mix the caller's IP with
# the target email so one household behind a NAT is not locked out by a
# neighbour, while one address cannot be hammered from one place either.
# ---------------------------------------------------------------------------
LOGIN_ATTEMPTS = 10          # per (ip, email) ...
LOGIN_WINDOW_S = 15 * 60     # ... per 15 minutes
RESET_REQUESTS = 3           # per email ...
RESET_WINDOW_S = 60 * 60     # ... per hour
# Sign-ups per IP. Login and reset were throttled; registration was not, and
# it is the one that mints entitlements -- ten free analyses and a full free
# preview per address, with no email verification in the way. Generous enough
# for a club signing up from one wifi; not enough to script.
REGISTER_PER_IP = 8
REGISTER_WINDOW_S = 60 * 60
_attempts: dict[str, deque[float]] = {}

# A real bcrypt hash of nothing in particular, checked against when the
# address is unknown, so "no such account" costs the same ~250 ms as "wrong
# password" and the timing does not say which.
_DUMMY_HASH = hash_password("timing-equaliser")


def _throttle(key: str, limit: int, window_s: int) -> None:
    """Record one attempt under ``key`` and raise 429 once ``limit`` is hit
    inside the rolling window."""
    now = time.time()
    dq = _attempts.setdefault(key, deque())
    while dq and now - dq[0] > window_s:
        dq.popleft()
    if len(dq) >= limit:
        wait = int(window_s - (now - dq[0])) + 1
        minutes = max(1, -(-wait // 60))
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many attempts — try again in {minutes} min.",
        )
    dq.append(now)


def _caller(request: Request | None) -> str:
    return client_ip(request) if request is not None else "direct"


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
    height_cm: int | None = None


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def register(
    body: Credentials, request: Request = None, db: AsyncSession = Depends(get_session),
) -> TokenOut:
    _throttle(f"register:{_caller(request)}", REGISTER_PER_IP, REGISTER_WINDOW_S)
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
    from app.models.user import TIER_ADMIN, TIER_STARTER

    is_admin = bool(settings.admin_email) and body.email == settings.admin_email
    user = User(
        email=body.email,
        password_hash=await run_in_threadpool(hash_password, body.password),
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
async def login(
    body: LoginBody, request: Request = None, db: AsyncSession = Depends(get_session),
) -> TokenOut:
    _throttle(f"login:{_caller(request)}:{body.email}", LOGIN_ATTEMPTS, LOGIN_WINDOW_S)
    user = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    # One message for both "no such account" and "wrong password": the form
    # must not double as a directory of who has signed up.
    # bcrypt is ~250 ms of CPU by design. Off the event loop, or every login
    # freezes the single worker for everyone else -- and the throttle above
    # only bounds attempts per (ip, email), not how many households log in
    # at once. The unknown-address branch still pays the same time so the
    # response cannot say which half of the check failed.
    ok = user is not None and await run_in_threadpool(
        verify_password, body.password, user.password_hash,
    )
    if user is None:
        await run_in_threadpool(verify_password, body.password, _DUMMY_HASH)
    if not ok:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password.")
    return TokenOut(
        token=create_token(user.id), email=user.email,
        tier=user.tier, is_pro=user.is_paid,
        expert_credits=user.expert_credits or 0,
        analytics_id=analytics.person_id(user),
    )


# ---------------------------------------------------------------------------
# Password reset: ask for a link, then set a new password with it.
# ---------------------------------------------------------------------------
class ForgotBody(BaseModel):
    email: str

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return v.strip().lower()


class SentOut(BaseModel):
    sent: bool = True


class ResetBody(BaseModel):
    token: str
    password: str

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        if not (8 <= len(v) <= 72):
            raise ValueError("Password must be 8–72 characters.")
        return v


def reset_link(token: str) -> str:
    """The SPA opens the reset form from the URL fragment; the token never
    reaches the server as a query string (and so never lands in access logs)."""
    base = settings.public_base_url or "https://getflapp.com"
    return f"{base}/app#reset={token}"


def _reset_mail(link: str) -> tuple[str, str, str]:
    hours = int(RESET_TOKEN_TTL.total_seconds() // 3600)
    subject = "Reset your Flapp password"
    text = (
        "Someone asked to reset the password for this Flapp account.\n\n"
        f"Set a new password here (the link works for {hours} hour"
        f"{'s' if hours != 1 else ''}, once):\n{link}\n\n"
        "If that wasn't you, ignore this email — your password stays as it is.\n\n"
        "— Flapp"
    )
    safe = html.escape(link, quote=True)
    body = (
        "<p>Someone asked to reset the password for this Flapp account.</p>"
        f"<p><a href=\"{safe}\" style=\"display:inline-block;padding:12px 20px;"
        "background:#2F6DE0;color:#fff;font-weight:700;text-decoration:none;"
        "border-radius:8px\">Set a new password</a></p>"
        f"<p style=\"color:#555\">The link works for {hours} hour"
        f"{'s' if hours != 1 else ''}, once. If the button does nothing, copy this "
        f"address into your browser:<br><a href=\"{safe}\">{safe}</a></p>"
        "<p style=\"color:#555\">If that wasn't you, ignore this email — your "
        "password stays as it is.</p><p>— Flapp</p>"
    )
    return subject, text, body


@router.post("/forgot", response_model=SentOut)
async def forgot_password(
    body: ForgotBody, request: Request = None, db: AsyncSession = Depends(get_session),
) -> SentOut:
    """Always answers "sent", whether or not the address has an account --
    the response must not say who has signed up. The throttle is per email so
    one address cannot be flooded with reset mail."""
    _throttle(f"forgot:{body.email}", RESET_REQUESTS, RESET_WINDOW_S)
    _throttle(f"forgot-ip:{_caller(request)}", RESET_REQUESTS * 5, RESET_WINDOW_S)
    user = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if user is None:
        logger.info("PASSWORD_RESET_UNKNOWN_EMAIL")
        return SentOut()
    subject, text, body_html = _reset_mail(reset_link(create_reset_token(user)))
    # Sending is a blocking HTTPS round trip; keep it off the event loop.
    sent = await run_in_threadpool(notify.send_email, user.email, subject, text, body_html)
    logger.info("PASSWORD_RESET_REQUESTED", user_id=user.id, sent=sent)
    return SentOut()


@router.post("/reset", response_model=TokenOut)
async def reset_password(body: ResetBody, db: AsyncSession = Depends(get_session)) -> TokenOut:
    """Set the password and sign the person in -- they just proved they own
    the mailbox, and a second login form after that is friction for nothing."""
    user = await verify_reset_token(body.token, db)
    if user is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This reset link is invalid or has expired — request a new one.",
        )
    user.password_hash = await run_in_threadpool(hash_password, body.password)
    await db.commit()
    logger.info("PASSWORD_RESET", user_id=user.id)
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
        height_cm=user.height_cm,
    )


class HeightBody(BaseModel):
    """The athlete's standing height, or ``null`` to withdraw it.

    Bounds match ``running_analyzer.PLAUSIBLE_HEIGHT_CM``: outside them a
    number is a unit mix-up (feet, inches, metres) rather than a person, and
    accepting one would silently scale every centimetre reading by it.
    """

    height_cm: int | None = Field(None, ge=120, le=230)


@router.put("/me/height", response_model=UserOut)
async def set_height(
    body: HeightBody,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> UserOut:
    """Store (or clear) the one body measurement the analysis uses.

    Deliberately its own endpoint and its own decision: it is optional, it is
    the only personal measurement asked for, and clearing it has to be as easy
    as giving it.
    """
    user.height_cm = body.height_cm
    await db.commit()
    logger.info("HEIGHT_SET", user_id=user.id, height_cm=body.height_cm)
    return UserOut(
        email=user.email, tier=user.tier, is_pro=user.is_paid,
        expert_credits=user.expert_credits or 0,
        analytics_id=analytics.person_id(user),
        height_cm=user.height_cm,
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
    here rather than left to run out their own retention period. A live
    subscription is cancelled in Stripe first (``billing.cancel_live_subscriptions``).
    """
    if not await run_in_threadpool(verify_password, body.password, user.password_hash):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Wrong password — the account was not deleted.",
        )

    from app.api.billing import cancel_live_subscriptions, has_live_subscription
    from app.models.analysis import Analysis
    from app.models.feedback import Feedback
    from app.models.order import ORDER_DELIVERED, ORDER_REFUNDED, Order
    from app.models.usage import UsageEvent

    # Stop the billing before the row that Stripe's webhooks resolve a customer
    # to is gone. Before anything else, and fatal on failure: every other step
    # here is recoverable, a card that keeps being charged for a deleted
    # account is not.
    if has_live_subscription(user):
        await cancel_live_subscriptions(user)

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
