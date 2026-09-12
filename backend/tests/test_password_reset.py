"""Password reset and login throttling.

There was no way back into an account with a forgotten password -- on the
website an email to a human, on the phone nothing. What matters here: the
link is single-use and short-lived, the request never says who has an
account, a reset token can never pass as a session, and a brute-force login
is cut off.

The endpoint functions are called directly, like test_account_deletion.py:
they are plain coroutines. One HTTP test covers the ``Request`` wiring.
"""

from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException, Request

from app.api import auth as auth_api
from app.core import security
from app.core.config import settings
from app.core.security import (
    create_reset_token,
    create_token,
    hash_password,
    verify_password,
    verify_reset_token,
)

OLD, NEW = "old-password-1", "brand-new-password"


def _request(ip: str = "203.0.113.7") -> Request:
    return Request({"type": "http", "headers": [], "client": (ip, 1234), "method": "POST", "path": "/"})


@pytest.fixture(autouse=True)
def _fresh_throttle():
    auth_api._attempts.clear()
    yield
    auth_api._attempts.clear()


@pytest.fixture
async def user(db, make_user):
    u = await make_user(email="me@example.com")
    u.password_hash = hash_password(OLD)
    await db.commit()
    return u


# --------------------------------------------------------------------------
# The token
# --------------------------------------------------------------------------
async def test_a_reset_token_round_trips(db, user):
    assert (await verify_reset_token(create_reset_token(user), db)).id == user.id


async def test_a_reset_token_is_dead_once_the_password_changed(db, user):
    """Single-use without a table: the token carries a fingerprint of the hash
    it was issued against."""
    token = create_reset_token(user)
    user.password_hash = hash_password(NEW)
    await db.commit()
    assert await verify_reset_token(token, db) is None


async def test_an_expired_reset_token_is_refused(db, user):
    past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)
    with patch.object(security._dt, "datetime") as fake:
        fake.now.return_value = past
        token = create_reset_token(user)
    assert await verify_reset_token(token, db) is None


async def test_a_session_token_is_not_a_reset_token(db, user):
    assert await verify_reset_token(create_token(user.id), db) is None


def test_a_reset_token_is_not_a_session(user):
    """The reset email must not double as a login link."""
    assert security._decode_uid(create_reset_token(user)) is None


async def test_a_tampered_token_is_refused(db, user):
    forged = jwt.encode(
        {"sub": str(user.id), "purpose": "reset", "ph": "0" * 16,
         "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)},
        "not-the-secret", algorithm="HS256",
    )
    assert settings.jwt_secret != "not-the-secret"
    assert await verify_reset_token(forged, db) is None


# --------------------------------------------------------------------------
# The endpoints
# --------------------------------------------------------------------------
async def test_forgot_sends_a_link_to_an_existing_account(db, user):
    sent: list[tuple] = []
    with patch.object(auth_api.notify, "send_email", side_effect=lambda *a: sent.append(a) or True):
        out = await auth_api.forgot_password(auth_api.ForgotBody(email="ME@example.com "), _request(), db)
    assert out.sent is True
    (to, subject, text, html), = sent
    assert to == "me@example.com" and "Reset" in subject
    link = next(w for w in text.split() if w.startswith("http"))
    assert link.startswith("https://") and "/app#reset=" in link and link in html
    token = link.split("#reset=", 1)[1]
    assert (await verify_reset_token(token, db)).id == user.id


async def test_forgot_answers_the_same_for_an_unknown_address(db, user):
    """The response must not say who has signed up."""
    with patch.object(auth_api.notify, "send_email") as send:
        out = await auth_api.forgot_password(auth_api.ForgotBody(email="nobody@example.com"), _request(), db)
    assert out.sent is True
    send.assert_not_called()


async def test_reset_sets_the_password_and_signs_in(db, user):
    token = create_reset_token(user)
    out = await auth_api.reset_password(auth_api.ResetBody(token=token, password=NEW), db)
    await db.refresh(user)
    assert verify_password(NEW, user.password_hash) and not verify_password(OLD, user.password_hash)
    assert security._decode_uid(out.token) == user.id and out.email == user.email


async def test_reset_refuses_a_used_link(db, user):
    token = create_reset_token(user)
    await auth_api.reset_password(auth_api.ResetBody(token=token, password=NEW), db)
    with pytest.raises(HTTPException) as e:
        await auth_api.reset_password(auth_api.ResetBody(token=token, password="another-one-1"), db)
    assert e.value.status_code == 400


def test_reset_keeps_the_registration_password_rules():
    with pytest.raises(ValueError):
        auth_api.ResetBody(token="t", password="short")


# --------------------------------------------------------------------------
# Throttling
# --------------------------------------------------------------------------
async def test_login_is_cut_off_after_repeated_failures(db, user):
    body = auth_api.LoginBody(email=user.email, password="wrong-password")
    for _ in range(auth_api.LOGIN_ATTEMPTS):
        with pytest.raises(HTTPException) as e:
            await auth_api.login(body, _request(), db)
        assert e.value.status_code == 401
    with pytest.raises(HTTPException) as e:
        await auth_api.login(auth_api.LoginBody(email=user.email, password=OLD), _request(), db)
    assert e.value.status_code == 429, "the right password is refused too until the window passes"


async def test_the_login_throttle_is_per_caller_and_address(db, user):
    """A neighbour behind the same NAT, or the same person on another
    account, is not locked out by someone else's failures."""
    body = auth_api.LoginBody(email=user.email, password="wrong-password")
    for _ in range(auth_api.LOGIN_ATTEMPTS):
        with pytest.raises(HTTPException):
            await auth_api.login(body, _request("203.0.113.7"), db)
    out = await auth_api.login(auth_api.LoginBody(email=user.email, password=OLD), _request("198.51.100.2"), db)
    assert out.email == user.email


async def test_reset_requests_are_capped_per_address(db, user):
    with patch.object(auth_api.notify, "send_email", return_value=True):
        for _ in range(auth_api.RESET_REQUESTS):
            await auth_api.forgot_password(auth_api.ForgotBody(email=user.email), _request(), db)
        with pytest.raises(HTTPException) as e:
            await auth_api.forgot_password(auth_api.ForgotBody(email=user.email), _request(), db)
    assert e.value.status_code == 429


def test_forgot_over_http_needs_no_account():
    """The FastAPI wiring of the optional ``Request`` parameter, end to end."""
    main = pytest.importorskip("app.main", reason="needs the analysis stack (mediapipe/opencv)")
    from fastapi.testclient import TestClient

    # The context manager runs startup, which is what creates the tables.
    with patch.object(auth_api.notify, "send_email", return_value=True), TestClient(main.app) as c:
        r = c.post("/auth/forgot", json={"email": "nobody-at-all@example.invalid"})
    assert r.status_code == 200 and r.json() == {"sent": True}
