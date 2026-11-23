"""A password reset ends every session issued before it.

The usual reason to reset is a lost phone or a leaked password, and a reset
used to change the hash while every token already handed out -- the stolen
phone's included -- stayed valid for its 30 days. Sessions now carry the
password-hash fingerprint the reset token already uses; once the hash
changes, the old ones are refused.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import auth as auth_api
from app.core import security
from app.core.security import create_reset_token, create_token, get_current_user, optional_user

OLD, NEW = "old-password-1", "new-password-2"


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


@pytest.fixture
async def user(db, make_user):
    u = await make_user()
    u.password_hash = security.hash_password(OLD)
    await db.commit()
    return u


async def test_the_sessions_from_before_the_reset_are_refused(db, user):
    stolen = create_token(user.id, user.password_hash)
    assert (await get_current_user(_bearer(stolen), db)).id == user.id

    out = await auth_api.reset_password(
        auth_api.ResetBody(token=create_reset_token(user), password=NEW), db,
    )
    with pytest.raises(HTTPException) as err:
        await get_current_user(_bearer(stolen), db)
    assert err.value.status_code == 401
    assert await optional_user(_bearer(stolen), db) is None
    # The device that just reset is signed in with the new session.
    assert (await get_current_user(_bearer(out.token), db)).id == user.id


async def test_every_sign_in_path_fingerprints_the_session(db, user):
    out = await auth_api.reset_password(
        auth_api.ResetBody(token=create_reset_token(user), password=NEW), db,
    )
    import jwt

    from app.core.config import settings

    claims = jwt.decode(out.token, settings.jwt_secret, algorithms=["HS256"])
    assert claims["ph"] == security._hash_fingerprint(user.password_hash)
    src = open(auth_api.__file__, encoding="utf-8").read()
    assert "create_token(user.id)" not in src, "a sign-in path issues an unfingerprinted session"


async def test_a_session_issued_before_the_check_still_works(db, user):
    """No fingerprint = issued before this deploy: accepted until it expires,
    so shipping it signs nobody out (/auth/me reissues a fingerprinted one)."""
    legacy = create_token(user.id)
    assert (await get_current_user(_bearer(legacy), db)).id == user.id
