"""Coach orders from the home page without signing up (services/guest.py).

Pinned: the account is made from the email with no password, reused for the
same guest, and never attached to a real account; a guest cannot log in until
the emailed link sets a password, and that link dies once used; a run or bike
clip is filed as a teaser entry once analysed, and a failed analysis keeps the
order and tells the coach; Stripe brings a guest back to the home page; the
sale is marked GUEST and the receipt carries the password link; the home
page's coach buttons open the form instead of the app.
"""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException

from app.api import auth, billing
from app.api.billing import CheckoutIn, _apply_event, start_checkout
from app.core.config import settings
from app.core.security import is_guest, verify_password, verify_reset_token
from app.services import coach_brief, guest, notify, pricing


class _FakeRequest:
    base_url = "https://getflapp.com/"


# --------------------------------------------------------------------------
# the account
# --------------------------------------------------------------------------
async def test_a_guest_account_is_made_from_the_email_with_no_password(db):
    user = await guest.account_for(db, "new@example.com", marketing=False)
    assert is_guest(user)
    assert user.privacy_accepted_at is not None and not user.marketing_opt_in
    # No password matches it, the empty one included.
    assert not verify_password("", user.password_hash)
    assert not verify_password(user.password_hash, user.password_hash)


async def test_a_second_order_reuses_the_guest_account(db):
    first = await guest.account_for(db, "again@example.com", marketing=False)
    second = await guest.account_for(db, "again@example.com", marketing=True)
    assert second.id == first.id
    # The same password link keeps working: the hash is not replaced.
    assert second.password_hash == first.password_hash
    assert second.marketing_opt_in


async def test_a_real_account_is_never_given_a_clip_without_a_password(db, make_user):
    await make_user(email="taken@example.com")
    with pytest.raises(HTTPException) as exc:
        await guest.account_for(db, "taken@example.com", marketing=False)
    assert exc.value.status_code == 409
    assert "Log in" in exc.value.detail


async def test_signing_up_on_a_guest_email_points_at_forgot_password(db):
    await guest.account_for(db, "later@example.com", marketing=False)
    with pytest.raises(HTTPException) as exc:
        await auth.register(auth.RegisterBody(
            email="later@example.com", password="long-enough-1", accept_privacy=True,
        ), None, db)
    assert exc.value.status_code == 409
    assert "Forgot password" in exc.value.detail


async def test_the_emailed_link_sets_a_password_once(db):
    user = await guest.account_for(db, "link@example.com", marketing=False)
    link = guest.set_password_link(user)
    assert "/app#reset=" in link
    token = link.split("#reset=", 1)[1]
    assert (await verify_reset_token(token, db)).id == user.id
    await auth.reset_password(auth.ResetBody(token=token, password="my-new-pass-1"), db)
    await db.refresh(user)
    assert not is_guest(user)
    assert await verify_reset_token(token, db) is None


def test_the_order_is_refused_before_the_upload_when_it_cannot_be_sold():
    for sport, plan, ok in (("bike", "notes", False), ("ski", "notes", True), ("bike", "unlock", True)):
        with pytest.raises(HTTPException):
            guest.check_order(sport, plan, ok)
    guest.check_order("swim", "expert", True)
    with pytest.raises(HTTPException):
        guest.clean_email("not-an-email")
    assert guest.clean_email("  Me@Example.COM ") == "me@example.com"


# --------------------------------------------------------------------------
# a run or bike clip, filed by the server
# --------------------------------------------------------------------------
def _pending():
    return guest.pending_entry(client_id="h1", at_ms=1, job_id="j1", sport="bike",
                               position="triathlon", question="Knee hurts")


def test_an_analysed_guest_clip_is_filed_as_a_teaser():
    result = {
        "status": "completed", "sport_type": "bike", "cycling_position": "triathlon",
        "camera_side": "right", "technique_score": 71, "letter_grade": "C",
        "keyframe_base64": "FRAME", "angle_statistics": {"knee": {"mean": 140}},
        "detected_issues": [{"type": "saddle_low", "severity": "minor"}],
    }
    entry, frame = guest.finish_entry(_pending(), {"status": "completed", "result": result})
    assert frame == "FRAME" and "keyframe" not in entry
    assert entry["score"] == 71 and entry["cameraSide"] == "right"
    assert entry["locked"] is True and "pending" not in entry
    # No measurements in the entry: a later unlock rebuilds it from the stored result.
    assert "angleStats" not in entry and "metrics" not in entry


def test_a_failed_analysis_keeps_the_order_and_tells_the_coach():
    entry, frame = guest.finish_entry(_pending(), {"status": "failed", "error": "No rider found"})
    assert frame is None and entry["analysisFailed"] == "No rider found"
    html = coach_brief.render_brief_html(order_id=7, paid_ms=1793196120000, result=None,
                                         entry=entry, media={}, clip=None)
    assert "Автоанализ не смог прочитать" in html and "переснять" in html
    # The question survives a run with no result to carry it.
    assert "Knee hurts" in html


# --------------------------------------------------------------------------
# paying
# --------------------------------------------------------------------------
async def test_stripe_brings_a_guest_back_to_the_home_page(db, monkeypatch):
    monkeypatch.setattr(billing, "settings", dataclasses.replace(settings, stripe_secret_key="sk_test_x"))
    user = await guest.account_for(db, "pay@example.com", marketing=False)
    calls = {}

    class _S:
        url = "https://checkout.stripe.test/s"

    class _Cust:
        id = "cus_g"

    monkeypatch.setattr(billing.stripe.Customer, "create", lambda **kw: _Cust())
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", lambda **kw: calls.update(kw) or _S())
    out = await start_checkout(CheckoutIn(plan="notes", analysis_client_id="h1"), _FakeRequest(), user, db,
                               return_to=guest.RETURN_TO)
    assert out["url"].startswith("https://checkout.stripe.test/")
    assert calls["success_url"] == "https://getflapp.com/?order=paid&plan=notes"
    assert calls["cancel_url"] == "https://getflapp.com/?order=cancelled"
    assert calls["metadata"]["analysis_client_id"] == "h1"


async def test_a_guest_sale_is_marked_and_the_receipt_carries_the_link(db):
    user = await guest.account_for(db, "sold@example.com", marketing=False)
    sale = await _apply_event(db, {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_guest_1", "client_reference_id": str(user.id), "customer": "cus_1",
        "mode": "payment", "amount_total": 1200, "currency": "usd",
        "metadata": {"user_id": str(user.id), "plan": "notes", "tier": "", "analysis_client_id": "h1"},
    }}})
    assert sale["guest"] is True
    subject, _, _ = notify.sale_email(sale, "https://getflapp.com/app#reviews")
    assert "GUEST" in subject
    s, text, html = guest.welcome_email("notes", "swim", "https://getflapp.com/app#reset=abc")
    assert "#reset=abc" in text and "#reset=abc" in html
    assert "72 hours" in text and f"{guest.LINK_TTL.days} days" in text
    assert "automatic analysis" not in text           # swim: nothing is measured
    assert "automatic analysis" in guest.welcome_email("expert", "bike", "x")[1]


async def test_an_ordinary_sale_is_not_marked_guest(db, make_user):
    user = await make_user()
    sale = await _apply_event(db, {"type": "checkout.session.completed", "data": {"object": {
        "id": "cs_plain_1", "client_reference_id": str(user.id), "customer": "cus_2",
        "mode": "payment", "amount_total": 1200, "currency": "usd",
        "metadata": {"user_id": str(user.id), "plan": "notes", "tier": "", "analysis_client_id": "h1"},
    }}})
    assert sale["guest"] is False
    assert "GUEST" not in notify.sale_email(sale, "x")[0]


# --------------------------------------------------------------------------
# the home page
# --------------------------------------------------------------------------
def test_the_home_page_coach_buttons_open_the_form_not_the_app():
    landing = pricing.render_landing_pricing()
    assert 'data-guest-order="notes"' in landing and 'data-guest-order="expert"' in landing
    assert 'data-guest-sport="swim"' in landing
    assert '<dialog class="go" id="guestOrder"' in landing
    assert 'href="/app#pricing">Get my plan' not in landing and 'href="/app#swim"' not in landing
    # The form's prices are the price list's.
    assert "Coach Notes · US$12" in landing
    # The app keeps its own buttons.
    assert "data-guest-order" not in pricing.render_pricing("app")


def test_the_form_shows_an_unsigned_example_for_each_sport_and_product():
    examples = pricing._guest_examples()
    assert examples.count('class="go-excard"') == 6
    for plan in ("notes", "expert"):
        for sport in ("bike", "run", "swim"):
            assert f'data-plan="{plan}" data-sport="{sport}"' in examples
    # our wording, so never under the coach's name
    assert "Kovalenko" not in examples and "Illustrative wording" in examples
    assert 'id="goEx"' in pricing.render_landing_pricing()
