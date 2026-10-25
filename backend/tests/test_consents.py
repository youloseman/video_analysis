"""Sign-up consents (2026-10-24): one required, one optional and separate.

1. No account without agreeing to the Terms and Privacy Policy -- and the
   moment it was given is on the row.
2. News and offers need their own yes, default no, recorded with a time.
3. The Account switch gives and withdraws it; the one-click unsubscribe in an
   issue withdraws it too.
4. The form in the app sends both and cannot submit without the first.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api import auth as auth_api

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


def _request(ip="203.0.113.77"):
    return SimpleNamespace(headers={}, client=SimpleNamespace(host=ip))


@pytest.fixture(autouse=True)
def _no_throttle(monkeypatch):
    monkeypatch.setattr(auth_api, "_throttle", lambda *a, **k: None)


async def test_no_account_without_agreeing(db):
    with pytest.raises(HTTPException) as e:
        await auth_api.register(
            auth_api.RegisterBody(email="noconsent@example.com", password="long-enough-pw"),
            _request(), db,
        )
    assert e.value.status_code == 422


async def test_consent_is_recorded_and_marketing_defaults_to_no(db):
    from sqlalchemy import select

    from app.models.user import User

    await auth_api.register(
        auth_api.RegisterBody(email="plain@example.com", password="long-enough-pw", accept_privacy=True),
        _request(), db,
    )
    await auth_api.register(
        auth_api.RegisterBody(email="news@example.com", password="long-enough-pw",
                              accept_privacy=True, marketing=True),
        _request(), db,
    )
    plain = (await db.execute(select(User).where(User.email == "plain@example.com"))).scalar_one()
    news = (await db.execute(select(User).where(User.email == "news@example.com"))).scalar_one()
    assert plain.privacy_accepted_at is not None and news.privacy_accepted_at is not None
    assert plain.marketing_opt_in is False and plain.marketing_changed_at is None
    assert news.marketing_opt_in is True and news.marketing_changed_at is not None


async def test_the_account_switch_gives_and_withdraws(db):
    from sqlalchemy import select

    from app.models.user import User

    await auth_api.register(
        auth_api.RegisterBody(email="switch@example.com", password="long-enough-pw", accept_privacy=True),
        _request(), db,
    )
    u = (await db.execute(select(User).where(User.email == "switch@example.com"))).scalar_one()
    u.news_opt_out = True       # left the list from an issue earlier
    out = await auth_api.set_marketing(auth_api.MarketingBody(opt_in=True), u, db)
    assert out.marketing_opt_in is True and u.news_opt_out is False
    out = await auth_api.set_marketing(auth_api.MarketingBody(opt_in=False), u, db)
    assert out.marketing_opt_in is False and u.news_opt_out is True and u.marketing_changed_at


def test_the_issue_unsubscribe_withdraws_consent_too():
    main = (STATIC.parent / "main.py").read_text(encoding="utf-8")
    block = main[main.index('if list == "news":'):]
    block = block[:block.index("return HTMLResponse(page)")]
    assert "user.news_opt_out = True" in block and "user.marketing_opt_in = False" in block


def test_the_sign_up_form_asks_for_both_separately():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    form = html[html.index('id="authConsent"'):]
    form = form[:form.index('id="authErr"')]
    assert 'id="authPrivacy"' in form and 'id="authMarketing"' in form
    # neither is pre-ticked
    assert "checked" not in form
    js = html[html.index("async function submitAuth("):]
    js = js[:js.index("function logout(")]
    assert "!$('#authPrivacy').checked" in js
    assert "accept_privacy:true,marketing" in js


def test_the_account_page_has_the_switch_and_the_policy_says_it():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="acctMarketing"' in html and "'/auth/me/marketing'" in html
    policy = (STATIC / "privacy.html").read_text(encoding="utf-8")
    assert 'id="emails"' in policy and "only if you said yes" in policy
