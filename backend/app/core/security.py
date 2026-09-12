"""Password hashing (bcrypt) + JWT + auth dependencies."""

from __future__ import annotations

import datetime as _dt
import hashlib

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_session
from app.models.user import User

_JWT_ALG = "HS256"
_bearer = HTTPBearer(auto_error=False)


def hash_password(pw: str) -> str:
    # bcrypt caps at 72 bytes; the API validates length, but truncate defensively.
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, ph: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], ph.encode("utf-8"))
    except Exception:  # noqa: BLE001
        return False


def create_token(user_id: int) -> str:
    now = _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + _dt.timedelta(days=settings.jwt_expire_days),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALG)


def _decode_uid(token: str) -> int | None:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_JWT_ALG])
        # A reset token is signed with the same secret; it must never pass as
        # a session, or the reset email becomes a login link.
        if payload.get("purpose"):
            return None
        return int(payload["sub"])
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Password reset tokens. Stateless on purpose: no table, no migration (raw-SQL
# ALTERs here are untested and have taken production down before). The token
# carries a fingerprint of the password hash it was issued against, so it is
# single-use by construction -- once the password changes, the fingerprint no
# longer matches and the same link is dead. Django does the same.
# ---------------------------------------------------------------------------
RESET_TOKEN_TTL = _dt.timedelta(hours=1)
_RESET_PURPOSE = "reset"


def _hash_fingerprint(password_hash: str) -> str:
    return hashlib.sha256(password_hash.encode("utf-8")).hexdigest()[:16]


def create_reset_token(user: User) -> str:
    now = _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "sub": str(user.id),
        "purpose": _RESET_PURPOSE,
        "ph": _hash_fingerprint(user.password_hash),
        "iat": now,
        "exp": now + RESET_TOKEN_TTL,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALG)


async def verify_reset_token(token: str, db: AsyncSession) -> User | None:
    """The user a reset token still belongs to, or None (expired, tampered,
    already used, or issued for a session rather than a reset)."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_JWT_ALG])
    except Exception:  # noqa: BLE001
        return None
    if payload.get("purpose") != _RESET_PURPOSE:
        return None
    try:
        user = await db.get(User, int(payload["sub"]))
    except (KeyError, ValueError, TypeError):
        return None
    if user is None or payload.get("ph") != _hash_fingerprint(user.password_hash):
        return None
    return user


async def get_current_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_session),
) -> User:
    if cred is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    uid = _decode_uid(cred.credentials)
    user = await db.get(User, uid) if uid is not None else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session")
    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """Admin tier only. 404 rather than 403 so the surface is not discoverable.

    Lives here rather than next to any one admin router because several of them
    (feedback inbox, order queue) need the same gate.
    """
    from app.models.user import TIER_ADMIN

    if user.tier != TIER_ADMIN:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    return user


async def optional_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_session),
) -> User | None:
    if cred is None:
        return None
    uid = _decode_uid(cred.credentials)
    return await db.get(User, uid) if uid is not None else None
