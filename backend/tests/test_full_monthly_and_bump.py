"""Full by the month, and the Expert Review order bump on Enthusiast.

Two decisions of 2026-10-26, both about where the included review lives:

* Full is sold by the month too, but the review comes with the YEARLY plan
  only. A US$39 review inside a US$13 month would be a discount on the review,
  bought and cancelled -- so neither the purchase nor a monthly renewal grants
  one.
* An Enthusiast checkout may add a review at US$35 as one more line on the
  first invoice. The webhook turns it into a credit, once.
"""

from __future__ import annotations

import dataclasses

from app.api import billing
from app.api.billing import CheckoutIn, _apply_event, create_checkout
from app.core.config import settings
from app.models.user import FULL_EXPERT_CREDITS, TIER_ENTHUSIAST, TIER_FULL, TIER_STARTER
from app.services import pricing


def sub_checkout(user_id, plan, tier, *, session_id="cs_1", bump=False):
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "client_reference_id": str(user_id),
            "customer": "cus_1",
            "mode": "subscription",
            "amount_total": 1300,
            "currency": "usd",
            "metadata": {
                "user_id": str(user_id), "plan": plan, "tier": tier,
                "review_bump": "1" if bump else "",
            },
        }},
    }


def renewal(price, invoice_id="in_1"):
    return {
        "type": "invoice.payment_succeeded",
        "data": {"object": {
            "id": invoice_id, "customer": "cus_1", "billing_reason": "subscription_cycle",
            "lines": {"data": [{"price": {"id": price}}]},
        }},
    }


# --------------------------------------------------------------------------
# Full monthly
# --------------------------------------------------------------------------
def test_full_monthly_is_a_plan_on_the_full_tier():
    assert billing.tier_for_plan("full_monthly") == TIER_FULL
    assert billing.tier_for_price(settings.stripe_price_full_m) == TIER_FULL
    assert settings.plan_price_map["full_monthly"] == settings.stripe_price_full_m


def test_the_full_card_sells_both_periods_at_the_same_saving_as_enthusiast():
    prices = {p.interval: p for p in pricing.CARD_BY_ID["full"].prices}
    assert prices["month"].plan == "full_monthly" and prices["month"].amount == 1300
    assert pricing.annual_saving_pct("full") == pricing.annual_saving_pct("enthusiast")


async def test_full_monthly_gives_full_but_no_review(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, sub_checkout(user.id, "full_monthly", "full"))
    await db.refresh(user)
    assert user.tier == TIER_FULL
    assert (user.expert_credits or 0) == 0


async def test_full_yearly_still_gives_the_review(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, sub_checkout(user.id, "full_yearly", "full"))
    await db.refresh(user)
    assert user.expert_credits == FULL_EXPERT_CREDITS


async def test_a_monthly_renewal_grants_nothing(db, make_user):
    user = await make_user(tier=TIER_FULL, stripe_customer_id="cus_1")
    await _apply_event(db, renewal(settings.stripe_price_full_m))
    await db.refresh(user)
    assert (user.expert_credits or 0) == 0
    await _apply_event(db, renewal(settings.stripe_price_full_y, invoice_id="in_2"))
    await db.refresh(user)
    assert user.expert_credits == FULL_EXPERT_CREDITS


# --------------------------------------------------------------------------
# The order bump
# --------------------------------------------------------------------------
class _Req:
    base_url = "https://getflapp.com/"


def _checkout_spy(monkeypatch):
    monkeypatch.setattr(
        billing, "settings", dataclasses.replace(settings, stripe_secret_key="sk_test_x"),
    )
    calls = {}

    class _S:
        url = "https://checkout.stripe.test/s"

    class _Cust:
        id = "cus_new"

    def fake_create(**kw):
        calls.update(kw)
        return _S()

    monkeypatch.setattr(billing.stripe.Customer, "create", lambda **kw: _Cust())
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)
    return calls


async def test_the_bump_adds_the_review_line_to_an_enthusiast_checkout(db, make_user, monkeypatch):
    calls = _checkout_spy(monkeypatch)
    user = await make_user(tier=TIER_STARTER)
    await create_checkout(CheckoutIn(plan="enthusiast_yearly", add_review=True), _Req(), user, db)
    prices = [li["price"] for li in calls["line_items"]]
    assert prices == [settings.stripe_price_enthusiast_y, settings.stripe_price_expert_bump]
    assert calls["metadata"]["review_bump"] == "1"


async def test_the_bump_is_ignored_on_full_and_when_not_asked(db, make_user, monkeypatch):
    calls = _checkout_spy(monkeypatch)
    user = await make_user(tier=TIER_STARTER)
    await create_checkout(CheckoutIn(plan="full_yearly", add_review=True), _Req(), user, db)
    assert len(calls["line_items"]) == 1 and calls["metadata"]["review_bump"] == ""
    await create_checkout(CheckoutIn(plan="enthusiast_monthly"), _Req(), user, db)
    assert len(calls["line_items"]) == 1


async def test_a_bumped_enthusiast_purchase_grants_one_review_once(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    for _ in range(3):
        await _apply_event(db, sub_checkout(user.id, "enthusiast_yearly", "enthusiast", bump=True))
    await db.refresh(user)
    assert user.tier == TIER_ENTHUSIAST
    assert user.expert_credits == 1


async def test_an_enthusiast_purchase_without_the_bump_grants_nothing(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, sub_checkout(user.id, "enthusiast_monthly", "enthusiast"))
    await db.refresh(user)
    assert (user.expert_credits or 0) == 0


def test_the_catalogue_offers_the_bump_only_with_a_price():
    assert pricing.catalog({}, "price_x")["review_bump"]["available"] is True
    off = pricing.catalog({}, None)["review_bump"]
    assert off["available"] is False and off["display"] == "$35" and off["was"] == "$39"
