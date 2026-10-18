"""The re-check: the Expert Review again, cheaper, on a NEW clip.

Who may buy it is the checkout's rule, not the price's -- the price is just a
second Stripe price -- so this pins the rules: only after a DELIVERED Coach
Notes or review, inside the window, once per delivery, never on the clip the
coach already saw, never for an account whose plan still holds an unused
review. And that it joins the review queue and the weekly cap instead of
sitting outside them.
"""
from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException

from app.api import billing, me
from app.api.billing import CheckoutIn, create_checkout
from app.core.config import settings
from app.models.order import ORDER_DELIVERED, ORDER_PAID, ORDER_REFUNDED, Order

DAY = billing.DAY_MS


def _use(monkeypatch, **kw):
    base = dict(stripe_secret_key="sk_test_x", stripe_price_expert_recheck="price_rc",
                recheck_window_days=45)
    monkeypatch.setattr(billing, "settings", dataclasses.replace(settings, **{**base, **kw}))


async def _delivered(db, user, plan="notes", *, ago_days=3, clip="old", sid="cs_src"):
    now = billing._now_ms()
    o = Order(user_id=user.id, stripe_session_id=sid, plan=plan, status=ORDER_DELIVERED,
              created_at_ms=now - (ago_days + 1) * DAY, updated_at_ms=now,
              delivered_at_ms=now - ago_days * DAY, analysis_client_id=clip)
    db.add(o)
    await db.commit()
    return o


async def _clip(db, user, cid, days_ago):
    await me._upsert(db, user, {"id": cid, "at": billing._now_ms() - int(days_ago * DAY),
                                "jobId": "j-" + cid, "sport": "bike"})


# --------------------------------------------------------------------------
# who is offered it
# --------------------------------------------------------------------------

async def test_a_delivered_notes_order_opens_a_recheck_until_the_window_closes(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    src = await _delivered(db, user, ago_days=3)
    offer = await billing.recheck_offer(db, user)
    assert offer["order_id"] == src.id
    assert offer["until_ms"] == src.delivered_at_ms + 45 * DAY


async def test_nothing_is_offered_without_a_delivery(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    assert await billing.recheck_offer(db, user) is None
    now = billing._now_ms()
    db.add(Order(user_id=user.id, stripe_session_id="cs_p", plan="notes", status=ORDER_PAID,
                 created_at_ms=now, updated_at_ms=now, analysis_client_id="old"))
    await db.commit()
    assert await billing.recheck_offer(db, user) is None      # paid, not yet written


async def test_the_window_closes(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _delivered(db, user, ago_days=46)
    assert await billing.recheck_offer(db, user) is None


async def test_the_window_is_a_setting_not_a_constant(db, make_user, monkeypatch):
    _use(monkeypatch, recheck_window_days=10)
    user = await make_user()
    await _delivered(db, user, ago_days=12)
    assert await billing.recheck_offer(db, user) is None


async def test_a_delivered_review_and_a_credit_review_both_count(db, make_user, monkeypatch):
    _use(monkeypatch)
    for plan in ("expert", "expert_credit"):
        user = await make_user(email=f"{plan}@x.test")
        await _delivered(db, user, plan=plan, sid=f"cs_{plan}")
        assert (await billing.recheck_offer(db, user)) is not None


async def test_a_recheck_does_not_open_another_recheck(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _delivered(db, user, plan="expert_recheck", ago_days=1, sid="cs_rc")
    assert await billing.recheck_offer(db, user) is None


async def test_one_recheck_per_delivery(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _delivered(db, user, ago_days=5)
    now = billing._now_ms()
    db.add(Order(user_id=user.id, stripe_session_id="cs_rc", plan="expert_recheck",
                 status=ORDER_PAID, created_at_ms=now - DAY, updated_at_ms=now,
                 analysis_client_id="new"))
    await db.commit()
    assert await billing.recheck_offer(db, user) is None


async def test_a_refunded_recheck_gives_the_offer_back(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _delivered(db, user, ago_days=5)
    now = billing._now_ms()
    db.add(Order(user_id=user.id, stripe_session_id="cs_rc", plan="expert_recheck",
                 status=ORDER_REFUNDED, created_at_ms=now - DAY, updated_at_ms=now))
    await db.commit()
    assert await billing.recheck_offer(db, user) is not None


async def test_a_later_delivery_opens_a_fresh_recheck(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _delivered(db, user, ago_days=20, sid="cs_a")
    now = billing._now_ms()
    db.add(Order(user_id=user.id, stripe_session_id="cs_rc", plan="expert_recheck",
                 status=ORDER_PAID, created_at_ms=now - 15 * DAY, updated_at_ms=now))
    await db.commit()
    assert await billing.recheck_offer(db, user) is None
    await _delivered(db, user, plan="expert", ago_days=2, sid="cs_b")
    assert (await billing.recheck_offer(db, user)) is not None


async def test_an_unused_included_review_comes_first(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    user.expert_credits = 1
    await db.commit()
    await _delivered(db, user)
    assert await billing.recheck_offer(db, user) is None


async def test_the_endpoint_says_no_when_the_price_is_not_configured(db, make_user, monkeypatch):
    _use(monkeypatch, stripe_price_expert_recheck=None)
    user = await make_user()
    await _delivered(db, user)
    assert await billing.my_recheck(user, db) == {"available": False}


async def test_the_endpoint_carries_the_deadline(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    src = await _delivered(db, user, ago_days=3, clip="old")
    out = await billing.my_recheck(user, db)
    assert out["available"] is True and out["source_analysis"] == "old"
    assert out["until_ms"] == src.delivered_at_ms + 45 * DAY and out["since_ms"] == src.created_at_ms


# --------------------------------------------------------------------------
# checkout
# --------------------------------------------------------------------------

async def test_checkout_refuses_without_a_delivery(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert_recheck", analysis_client_id="new"), None, user, db)
    assert exc.value.status_code == 403


async def test_checkout_refuses_the_clip_the_coach_already_saw(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _clip(db, user, "old", 5)
    await _delivered(db, user, clip="old")
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert_recheck", analysis_client_id="old"), None, user, db)
    assert exc.value.status_code == 409 and "after your review" in exc.value.detail


async def test_checkout_refuses_a_clip_filmed_before_the_advice(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _clip(db, user, "older", 20)
    await _delivered(db, user, clip="old", ago_days=3)
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert_recheck", analysis_client_id="older"), None, user, db)
    assert exc.value.status_code == 409


async def test_checkout_needs_a_clip_and_it_must_be_the_users(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    await _delivered(db, user)
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert_recheck"), None, user, db)
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert_recheck", analysis_client_id="nope"), None, user, db)
    assert exc.value.status_code == 404


async def test_checkout_refuses_an_account_that_still_holds_a_review(db, make_user, monkeypatch):
    _use(monkeypatch)
    user = await make_user()
    user.expert_credits = 1
    await db.commit()
    await _clip(db, user, "new", 0)
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert_recheck", analysis_client_id="new"), None, user, db)
    assert exc.value.status_code == 409 and "already includes" in exc.value.detail


async def test_a_valid_recheck_goes_to_stripe_as_a_one_time_payment(db, make_user, monkeypatch):
    _use(monkeypatch)
    seen = {}

    class _S:
        url = "https://stripe.test/pay"

    def fake_create(**kw):
        seen.update(kw)
        return _S()

    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)
    monkeypatch.setattr(billing, "_base_url", lambda r: "https://getflapp.com")
    user = await make_user()
    user.stripe_customer_id = "cus_1"
    await db.commit()
    await _delivered(db, user, ago_days=3, clip="old")
    await _clip(db, user, "new", 0)
    out = await create_checkout(CheckoutIn(plan="expert_recheck", analysis_client_id="new"), None, user, db)
    assert out == {"url": "https://stripe.test/pay"}
    assert seen["mode"] == "payment"
    assert seen["line_items"] == [{"price": "price_rc", "quantity": 1}]
    assert seen["metadata"]["plan"] == "expert_recheck" and seen["metadata"]["tier"] == ""
    assert seen["metadata"]["analysis_client_id"] == "new"


async def test_the_recheck_shares_the_reviews_weekly_cap(db, make_user, monkeypatch):
    _use(monkeypatch, expert_review_slots_per_week=1)
    user = await make_user()
    now = billing._now_ms()
    db.add(Order(user_id=user.id, stripe_session_id="cs_rc", plan="expert_recheck",
                 status=ORDER_PAID, created_at_ms=now, updated_at_ms=now))
    await db.commit()
    assert await billing.review_slots_left(db) == 0


def test_a_recheck_order_is_written_like_a_review_and_shows_in_the_queue():
    from app.services import expert_review as er
    assert "expert_recheck" in billing.REVIEW_ORDER_PLANS
    assert not er.is_notes_plan("expert_recheck")
    assert [s["key"] for s in er.sections_for("bike", "expert_recheck")] == \
           [s["key"] for s in er.sections_for("bike", "expert")]


def test_the_price_is_in_the_catalogue_and_the_map():
    from app.services import pricing
    assert "expert_recheck" in settings.plan_price_map
    card = pricing.CARD_BY_ID["recheck"]
    assert card.kind == "single" and card.prices[0].plan == "expert_recheck"
    assert card.prices[0].amount < pricing.CARD_BY_ID["expert"].prices[0].amount


# --------------------------------------------------------------------------
# the delivery email says it exists
# --------------------------------------------------------------------------

def test_the_delivery_emails_mention_the_recheck_only_when_given_one():
    from app.services import expert_review as er
    line = "Made the changes? Film again and book a re-check."
    for mail in (er.notes_ready_email, er.ready_email):
        _s, text, html = mail({"first": "x", "verdict": "y"}, "https://x/app", recheck=line)
        assert line in text and line in html
        _s, text, html = mail({"first": "x", "verdict": "y"}, "https://x/app")
        assert "re-check" not in text and "re-check" not in html


def test_the_line_states_the_catalogue_price_and_the_window(monkeypatch):
    from types import SimpleNamespace
    _use(monkeypatch)
    line = billing._recheck_line(SimpleNamespace(plan="notes"), SimpleNamespace(expert_credits=0))
    assert line and "$27" in line and "45 days" in line


def test_no_line_for_a_recheck_a_held_review_or_an_unconfigured_price(monkeypatch):
    from types import SimpleNamespace
    _use(monkeypatch)
    assert billing._recheck_line(SimpleNamespace(plan="expert_recheck"), SimpleNamespace(expert_credits=0)) is None
    assert billing._recheck_line(SimpleNamespace(plan="notes"), SimpleNamespace(expert_credits=1)) is None
    _use(monkeypatch, stripe_price_expert_recheck=None)
    assert billing._recheck_line(SimpleNamespace(plan="notes"), SimpleNamespace(expert_credits=0)) is None


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------

def _spa():
    from conftest import read_spa
    return read_spa()


def test_the_page_offers_the_recheck_where_the_athlete_just_read_the_advice():
    spa = _spa()
    # after delivered Notes, and under a delivered review
    assert "renderRecheck(hostId+'Recheck')" in spa
    assert "renderRecheck('rrRecheck')" in spa
    # the button books through the one checkout
    assert "startCheckout('expert_recheck')" in spa
    assert "analysisId=await pickAnalysis('expert_recheck')" in spa


def test_the_page_never_writes_the_price_down_itself():
    """The amount is the catalogue's (/billing/plans); test_pricing_catalog
    scans the SPA for a transcribed price, this pins that the block reads it."""
    spa = _spa()
    block = spa[spa.index("function recheckBlockHtml"):spa.index("async function renderRecheck")]
    assert "p.display" in block and "$" not in block.replace("${", "")


def test_the_picker_only_offers_clips_filmed_after_the_advice():
    spa = _spa()
    i = spa.index("function pickAnalysis")
    body = spa[i:spa.index("modal.classList.remove('hidden')", i)]
    assert "since_ms" in body and "source_analysis" in body


def test_the_return_from_stripe_knows_the_recheck():
    spa = _spa()
    assert "if(plan==='expert_recheck')" in spa
    assert "expert_recheck:'Expert Review · re-check'" in spa
