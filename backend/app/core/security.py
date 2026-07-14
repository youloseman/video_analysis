"""Password hashing (bcrypt) + JWT + auth dependencies."""

from __future__ import annotations

import datetime as _dt

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
        return int(payload["sub"])
    except Exception:  # noqa: BLE001
        return None


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


async def optional_user(
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: AsyncSession = Depends(get_session),
) -> User | None:
    if cred is None:
        return None
    uid = _decode_uid(cred.credentials)
    return await db.get(User, uid) if uid is not None else None
