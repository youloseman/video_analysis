"""Stripe webhook handling.

This is the code path that decides whether a paying customer gets what they
paid for. It is also the one path that cannot be exercised by clicking around:
the events only ever arrive from Stripe, and a mistake shows up as a customer
who paid and stayed on the free plan.

``_apply_event`` is tested directly, against the raw dict shape production
actually processes (the HTTP layer verifies the signature and then hands the
parsed JSON straight here), so no live Stripe or signature is needed.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.api.billing import (
    _apply_event,
    has_live_subscription,
    tier_for_plan,
    tier_for_price,
)
from app.core.config import settings
from app.models.order import ORDER_PAID, Order
from app.models.user import (
    TIER_ADMIN,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
)

# Planted in the environment by conftest before ``settings`` is loaded, so these
# are the ids production's own env-driven price->tier map resolves.
PRICE_ENTHUSIAST_M = settings.stripe_price_enthusiast_m
PRICE_FULL_Y = settings.stripe_price_full_y


def checkout_completed(user_id, *, mode="subscription", tier="enthusiast",
                       customer="cus_1", session_id="cs_test_1", plan="enthusiast_monthly",
                       amount=2900, currency="usd"):
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "client_reference_id": str(user_id) if user_id is not None else None,
            "customer": customer,
            "mode": mode,
            "amount_total": amount,
            "currency": currency,
            "metadata": {"user_id": str(user_id), "plan": plan, "tier": tier},
        }},
    }


def subscription_event(etype, *, customer="cus_1", status="active", price=PRICE_ENTHUSIAST_M):
    return {
        "type": etype,
        "data": {"object": {
            "customer": customer,
            "status": status,
            "items": {"data": [{"price": {"id": price}}]},
        }},
    }


# --------------------------------------------------------------------------
# 1. checkout.session.completed (subscription)
# --------------------------------------------------------------------------
async def test_checkout_completed_activates_the_subscription(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, checkout_completed(user.id))
    await db.refresh(user)
    assert user.tier == TIER_ENTHUSIAST
    assert user.subscription_status == "active"
    assert user.stripe_customer_id == "cus_1"


async def test_checkout_falls_back_to_the_customer_id_when_the_reference_is_missing(db, make_user):
    """``client_reference_id`` is absent when the subscription was started from
    the Stripe dashboard rather than our checkout."""
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_known")
    event = checkout_completed(None, customer="cus_known")
    event["data"]["object"]["client_reference_id"] = None
    await _apply_event(db, event)
    await db.refresh(user)
    assert user.tier == TIER_ENTHUSIAST


async def test_unknown_customer_is_ignored_without_raising(db):
    """Stripe retries anything that is not a 2xx, so an event we cannot match
    must still be accepted rather than error-looping forever."""
    await _apply_event(db, checkout_completed(9999, customer="cus_nobody"))


# --------------------------------------------------------------------------
# 2. checkout.session.completed (one-time Expert Review)
# --------------------------------------------------------------------------
async def test_expert_review_purchase_is_recorded_as_an_order(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, checkout_completed(
        user.id, mode="payment", tier="", plan="expert", session_id="cs_test_expert",
    ))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.user_id == user.id
    assert order.plan == "expert"
    assert order.status == ORDER_PAID
    assert (order.amount_total, order.currency) == (2900, "usd")


async def test_expert_review_does_not_change_the_plan(db, make_user):
    """It is an add-on, not a subscription -- buying it must not grant a tier."""
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, checkout_completed(
        user.id, mode="payment", tier="", plan="expert", session_id="cs_test_expert",
    ))
    await db.refresh(user)
    assert user.tier == TIER_STARTER


async def test_a_retried_webhook_does_not_create_a_second_order(db, make_user):
    """Stripe re-delivers until it gets a 2xx; the same session must insert once."""
    user = await make_user()
    event = checkout_completed(
        user.id, mode="payment", tier="", plan="expert", session_id="cs_test_dup",
    )
    await _apply_event(db, event)
    await _apply_event(db, event)
    orders = (await db.execute(select(Order))).scalars().all()
    assert len(orders) == 1


async def test_two_separate_purchases_are_two_orders(db, make_user):
    user = await make_user()
    for sid in ("cs_test_a", "cs_test_b"):
        await _apply_event(db, checkout_completed(
            user.id, mode="payment", tier="", plan="expert", session_id=sid,
        ))
    orders = (await db.execute(select(Order))).scalars().all()
    assert len(orders) == 2


# --------------------------------------------------------------------------
# 3. customer.subscription.created / .updated
# --------------------------------------------------------------------------
@pytest.mark.parametrize("etype", [
    "customer.subscription.created", "customer.subscription.updated",
])
async def test_subscription_change_syncs_the_tier_from_the_price(db, make_user, etype):
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event(etype, price=PRICE_FULL_Y))
    await db.refresh(user)
    assert user.tier == TIER_FULL
    assert user.subscription_status == "active"


async def test_a_trialing_subscription_grants_the_tier(db, make_user):
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event(
        "customer.subscription.updated", status="trialing",
    ))
    await db.refresh(user)
    assert user.tier == TIER_ENTHUSIAST


async def test_past_due_records_the_status_without_granting_a_tier(db, make_user):
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event(
        "customer.subscription.updated", status="past_due",
    ))
    await db.refresh(user)
    assert user.subscription_status == "past_due"
    assert user.tier == TIER_STARTER


async def test_an_unknown_price_does_not_grant_a_tier(db, make_user):
    """A price created in the Stripe dashboard but never added to the env must
    not silently map to a plan."""
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event(
        "customer.subscription.updated", price="price_never_configured",
    ))
    await db.refresh(user)
    assert user.tier == TIER_STARTER


# --------------------------------------------------------------------------
# 4. customer.subscription.deleted
# --------------------------------------------------------------------------
async def test_cancellation_drops_the_user_to_the_free_plan(db, make_user):
    user = await make_user(tier=TIER_FULL, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event("customer.subscription.deleted"))
    await db.refresh(user)
    assert user.tier == TIER_STARTER
    assert user.subscription_status == "canceled"


# --------------------------------------------------------------------------
# Admin accounts are outside billing entirely
# --------------------------------------------------------------------------
async def test_billing_never_downgrades_an_admin(db, make_user):
    user = await make_user(tier=TIER_ADMIN, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event("customer.subscription.deleted"))
    await db.refresh(user)
    assert user.tier == TIER_ADMIN


async def test_billing_never_overwrites_an_admin_with_a_paid_tier(db, make_user):
    user = await make_user(tier=TIER_ADMIN, stripe_customer_id="cus_1")
    await _apply_event(db, subscription_event(
        "customer.subscription.updated", price=PRICE_FULL_Y,
    ))
    await db.refresh(user)
    assert user.tier == TIER_ADMIN


# --------------------------------------------------------------------------
# Events we do not handle
# --------------------------------------------------------------------------
async def test_an_unhandled_event_type_is_a_no_op(db, make_user):
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    await _apply_event(db, {
        "type": "invoice.payment_succeeded",
        "data": {"object": {"customer": "cus_1"}},
    })
    await db.refresh(user)
    assert user.tier == TIER_STARTER


# --------------------------------------------------------------------------
# Plan/price mapping + the double-subscription guard
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("plan", "tier"), [
    ("enthusiast_monthly", TIER_ENTHUSIAST),
    ("enthusiast_yearly", TIER_ENTHUSIAST),
    ("full_yearly", TIER_FULL),
    ("expert", None),          # one-time add-on: grants nothing
    ("nonsense", None),
])
def test_plan_to_tier_mapping(plan, tier):
    assert tier_for_plan(plan) == tier


def test_price_to_tier_mapping():
    assert tier_for_price(PRICE_ENTHUSIAST_M) == TIER_ENTHUSIAST
    assert tier_for_price(PRICE_FULL_Y) == TIER_FULL
    assert tier_for_price("price_unknown") is None
    assert tier_for_price(None) is None


@pytest.mark.parametrize(("status", "expected"), [
    ("active", True),
    ("trialing", True),
    ("past_due", True),      # still billing -- a 2nd checkout would double-charge
    ("canceled", False),
    ("unpaid", False),
    ("incomplete", False),
    (None, False),
])
async def test_live_subscription_detection(make_user, status, expected):
    user = await make_user(stripe_customer_id="cus_1", subscription_status=status)
    assert has_live_subscription(user) is expected


async def test_a_user_with_no_stripe_customer_has_no_live_subscription(make_user):
    user = await make_user(subscription_status="active")
    assert has_live_subscription(user) is False
