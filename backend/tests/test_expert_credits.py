"""Expert Review credits: the "1 Expert Review included" on the Full tier.

That bullet sat on the pricing card for weeks with no code behind it -- no way
to grant the entitlement, and no way for the customer to claim it. Somebody
buying Full would have had to email and ask, and nothing in the system would
have agreed with them.

A credit is a deliverable we owe rather than a capability the tier confers, so
the properties worth pinning down are the ones money has: granted exactly once
per payment, spent exactly once, and not quietly destroyed by a downgrade.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import billing
from app.api.billing import CheckoutIn, _apply_event, create_checkout
from app.core.config import settings
from app.models.order import ORDER_PAID, Order
from app.models.user import (
    FULL_EXPERT_CREDITS,
    TIER_ADMIN,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
)

PRICE_FULL_Y = settings.stripe_price_full_y


def full_checkout(user_id, *, session_id="cs_full_1", customer="cus_1"):
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "client_reference_id": str(user_id),
            "customer": customer,
            "mode": "subscription",
            "amount_total": 9900,
            "currency": "usd",
            "metadata": {
                "user_id": str(user_id), "plan": "full_yearly", "tier": "full",
            },
        }},
    }


def _no_stripe(monkeypatch):
    """Prove the redemption path never reaches Stripe: with no secret key,
    anything that called _require_stripe() would 503."""
    monkeypatch.setattr(
        billing, "settings", dataclasses.replace(settings, stripe_secret_key=None),
    )


# --------------------------------------------------------------------------
# Granting
# --------------------------------------------------------------------------
async def test_buying_full_grants_the_included_review(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, full_checkout(user.id))
    await db.refresh(user)
    assert user.tier == TIER_FULL
    assert user.expert_credits == FULL_EXPERT_CREDITS


async def test_a_redelivered_webhook_does_not_grant_twice(db, make_user):
    """Stripe retries until it gets a 2xx, so "grant on purchase" without an
    idempotency key means "grant once per delivery attempt"."""
    user = await make_user(tier=TIER_STARTER)
    for _ in range(3):
        await _apply_event(db, full_checkout(user.id))
    await db.refresh(user)
    assert user.expert_credits == FULL_EXPERT_CREDITS


async def test_a_second_genuine_purchase_does_grant_again(db, make_user):
    """A different Checkout Session is a different payment, not a retry."""
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, full_checkout(user.id, session_id="cs_a"))
    await _apply_event(db, full_checkout(user.id, session_id="cs_b"))
    await db.refresh(user)
    assert user.expert_credits == 2 * FULL_EXPERT_CREDITS


async def test_the_cheaper_tier_grants_nothing(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    event = full_checkout(user.id)
    event["data"]["object"]["metadata"].update(
        {"plan": "enthusiast_yearly", "tier": "enthusiast"},
    )
    await _apply_event(db, event)
    await db.refresh(user)
    assert user.tier == TIER_ENTHUSIAST
    assert user.expert_credits == 0


async def test_an_admin_who_pays_still_gets_what_they_paid_for(db, make_user):
    """Billing deliberately never moves an admin's tier, so a grant keyed off
    the resulting tier would silently swallow their purchase."""
    user = await make_user(tier=TIER_ADMIN)
    await _apply_event(db, full_checkout(user.id))
    await db.refresh(user)
    assert user.tier == TIER_ADMIN          # untouched, as designed
    assert user.expert_credits == FULL_EXPERT_CREDITS


async def test_cancelling_does_not_take_back_an_unspent_review(db, make_user):
    """They paid for the term that included it."""
    user = await make_user(tier=TIER_FULL, stripe_customer_id="cus_1")
    user.expert_credits = 1
    await db.commit()
    await _apply_event(db, {
        "type": "customer.subscription.deleted",
        "data": {"object": {"customer": "cus_1", "status": "canceled"}},
    })
    await db.refresh(user)
    assert user.tier == TIER_STARTER
    assert user.expert_credits == 1


# --------------------------------------------------------------------------
# Spending
# --------------------------------------------------------------------------
async def test_redeeming_creates_a_paid_order_without_charging(db, make_user, monkeypatch):
    _no_stripe(monkeypatch)
    user = await make_user(tier=TIER_FULL)
    user.expert_credits = 1
    await db.commit()

    out = await create_checkout(
        CheckoutIn(plan="expert", analysis_client_id="h123"), None, user, db,
    )
    assert out["mode"] == "credit"
    assert out["credits_left"] == 0

    order = (
        await db.execute(select(Order).where(Order.user_id == user.id))
    ).scalar_one()
    assert order.plan == "expert_credit"
    assert order.amount_total == 0
    assert order.status == ORDER_PAID
    assert order.analysis_client_id == "h123"


async def test_redeeming_without_a_clip_refuses_rather_than_burning_the_credit(
    db, make_user, monkeypatch,
):
    """The paid path can survive a missing clip -- the reviewer picks one in the
    queue. A credit is spent irreversibly, so it must not be spent on nothing."""
    _no_stripe(monkeypatch)
    user = await make_user(tier=TIER_FULL)
    user.expert_credits = 1
    await db.commit()

    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="expert"), None, user, db)
    assert exc.value.status_code == 400
    await db.refresh(user)
    assert user.expert_credits == 1


async def test_double_clicking_spends_one_credit_and_makes_one_order(
    db, make_user, monkeypatch,
):
    _no_stripe(monkeypatch)
    user = await make_user(tier=TIER_FULL)
    user.expert_credits = 2
    await db.commit()

    first = await create_checkout(
        CheckoutIn(plan="expert", analysis_client_id="h123"), None, user, db,
    )
    second = await create_checkout(
        CheckoutIn(plan="expert", analysis_client_id="h123"), None, user, db,
    )
    assert first["order_id"] == second["order_id"]

    orders = (
        await db.execute(select(Order).where(Order.user_id == user.id))
    ).scalars().all()
    assert len(orders) == 1
    await db.refresh(user)
    assert user.expert_credits == 1


async def test_a_second_clip_can_use_a_second_credit(db, make_user, monkeypatch):
    """The idempotency key is per (account, analysis), not per account -- two
    credits are two reviews, of two different clips."""
    _no_stripe(monkeypatch)
    user = await make_user(tier=TIER_FULL)
    user.expert_credits = 2
    await db.commit()

    await create_checkout(
        CheckoutIn(plan="expert", analysis_client_id="h1"), None, user, db,
    )
    out = await create_checkout(
        CheckoutIn(plan="expert", analysis_client_id="h2"), None, user, db,
    )
    assert out["credits_left"] == 0
    orders = (
        await db.execute(select(Order).where(Order.user_id == user.id))
    ).scalars().all()
    assert len(orders) == 2


async def test_without_a_credit_the_purchase_goes_to_stripe_as_before(
    db, make_user, monkeypatch,
):
    """No credit, no Stripe key -> the ordinary 503. Proves the redemption
    branch is the only thing that bypasses checkout."""
    _no_stripe(monkeypatch)
    user = await make_user(tier=TIER_ENTHUSIAST)
    assert user.expert_credits == 0
    with pytest.raises(HTTPException) as exc:
        await create_checkout(
            CheckoutIn(plan="expert", analysis_client_id="h1"), None, user, db,
        )
    assert exc.value.status_code == 503
