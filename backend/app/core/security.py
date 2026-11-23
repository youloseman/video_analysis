"""Password hashing (bcrypt) + JWT + auth dependencies."""

from __future__ import annotations

import datetime as _dt
import hashlib
import secrets

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


def create_token(user_id: int, password_hash: str | None = None) -> str:
    """A session. ``password_hash``: the account's current hash, so that
    changing the password ends every session issued before it (see
    ``_session_matches``). Every sign-in path passes it."""
    now = _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + _dt.timedelta(days=settings.jwt_expire_days),
    }
    if password_hash is not None:
        payload["ph"] = _hash_fingerprint(password_hash)
    return jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALG)


def _decode_session(token: str) -> tuple[int, str | None] | None:
    """(user id, password fingerprint or None) of a session token, or None."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_JWT_ALG])
        # A reset token is signed with the same secret; it must never pass as
        # a session, or the reset email becomes a login link.
        if payload.get("purpose"):
            return None
        ph = payload.get("ph")
        return int(payload["sub"]), (str(ph) if ph is not None else None)
    except Exception:  # noqa: BLE001
        return None


def _decode_uid(token: str) -> int | None:
    claims = _decode_session(token)
    return claims[0] if claims else None


def _session_matches(user: User | None, ph: str | None) -> bool:
    """Is this session still the account's? A reset password ends the others.

    Until sessions carried the fingerprint, a reset changed the password and
    left every token already issued -- including one on a stolen phone, the
    usual reason to reset -- valid for its full 30 days. The fingerprint is
    the reset token's own trick: no table, no migration, and the session dies
    the moment the hash it was issued against is replaced.

    A token without one was issued before this check; it is accepted until it
    expires (/auth/me reissues a fingerprinted one on the next app open), so
    the deploy signs nobody out.
    """
    if user is None:
        return False
    return ph is None or ph == _hash_fingerprint(user.password_hash)


# ---------------------------------------------------------------------------
# Media tokens. A <video src> cannot send the Authorization header, so a saved
# report's overlay has to be fetched with a credential in the URL. Not the
# session JWT: a URL is logged, cached and copied, and it would carry the
# whole account for thirty days. This one names ONE job, is dead in an hour,
# and opens nothing else -- _decode_uid rejects it as a session the same way
# it rejects a reset token.
# ---------------------------------------------------------------------------
MEDIA_TOKEN_TTL = _dt.timedelta(hours=1)
_MEDIA_PURPOSE = "media"


def create_media_token(user_id: int, job_id: str) -> str:
    now = _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "sub": str(user_id),
        "purpose": _MEDIA_PURPOSE,
        "job": job_id,
        "iat": now,
        "exp": now + MEDIA_TOKEN_TTL,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALG)


def media_token_user(token: str, job_id: str) -> int | None:
    """The user id a media token was issued to, if it is for ``job_id`` and
    still live; None for anything else (a session, a reset, another job)."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[_JWT_ALG])
    except Exception:  # noqa: BLE001
        return None
    if payload.get("purpose") != _MEDIA_PURPOSE or payload.get("job") != job_id:
        return None
    try:
        return int(payload["sub"])
    except (KeyError, ValueError, TypeError):
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


def create_reset_token(user: User, ttl: _dt.timedelta | None = None) -> str:
    """``ttl``: longer than the hour a reset gets only for the "set your
    password" link of a guest order, whose delivery email can arrive days later."""
    now = _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "sub": str(user.id),
        "purpose": _RESET_PURPOSE,
        "ph": _hash_fingerprint(user.password_hash),
        "iat": now,
        "exp": now + (ttl or RESET_TOKEN_TTL),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALG)


# ---------------------------------------------------------------------------
# Guest accounts: made from the email on a coach order sent from the home page
# without signing up (services/guest.py). The hash is not a bcrypt hash, so no
# password matches it; setting one through the reset link turns it into an
# ordinary account, and the same link stops working (the fingerprint moves).
# ---------------------------------------------------------------------------
GUEST_HASH_PREFIX = "guest!"


def guest_password_hash() -> str:
    return GUEST_HASH_PREFIX + secrets.token_hex(16)


def is_guest(user: User | None) -> bool:
    """An account nobody has set a password on yet."""
    return bool(user and (user.password_hash or "").startswith(GUEST_HASH_PREFIX))


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
    claims = _decode_session(cred.credentials)
    user = await db.get(User, claims[0]) if claims else None
    if not claims or not _session_matches(user, claims[1]):
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
    claims = _decode_session(cred.credentials)
    user = await db.get(User, claims[0]) if claims else None
    return user if claims and _session_matches(user, claims[1]) else None
