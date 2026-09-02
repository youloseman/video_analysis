"""Reading the price id out of a webhook, whichever way Stripe spells it.

The live destination (`flapp_webhook`) is pinned to API version
**2026-06-24.dahlia**. These handlers were written against an older one, and
across that gap Stripe moved the field they depend on: an invoice line and a
subscription item now carry `pricing.price_details.price` -- a bare id string
-- while the embedded `price` object is deprecated.

Nothing here fails loudly. `tier_for_price(None)` is None, so a renewal simply
returns before granting the Expert Review it was paid for, and a subscription
update records the status while leaving the tier alone. The customer sees a
missing entitlement; the log says nothing; Stripe's dashboard shows 200.

So the price reader accepts both shapes rather than betting on either, and
these tests hold both open. The old shape is not legacy trivia to be deleted
later: webhook destinations are pinned per endpoint, and a sandbox or a second
destination can still be on the older version.
"""

from __future__ import annotations

import pytest

from app.api.billing import _apply_event, _price_id_from_item
from app.core.config import settings
from app.models.user import (
    FULL_EXPERT_CREDITS,
    TIER_FULL,
    TIER_STARTER,
)

PRICE_FULL_Y = settings.stripe_price_full_y


def legacy_item(price: str) -> dict:
    """What the handlers were written against."""
    return {"price": {"id": price}}


def dahlia_item(price: str) -> dict:
    """What a newer API version sends: an id string, one level deeper."""
    return {"pricing": {"price_details": {"price": price}}}


SHAPES = {"legacy": legacy_item, "dahlia": dahlia_item}


# --------------------------------------------------------------------------
# the reader
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_both_shapes_read_the_same_price(shape):
    assert _price_id_from_item(SHAPES[shape]("price_x")) == "price_x"


def test_a_bare_string_price_is_accepted():
    """An expanded=false item can carry the id directly."""
    assert _price_id_from_item({"price": "price_x"}) == "price_x"


def test_the_new_shape_wins_when_both_are_present():
    """During a version transition Stripe sends the deprecated field too. The
    one the current version documents is the one to believe."""
    item = {"pricing": {"price_details": {"price": "price_new"}},
            "price": {"id": "price_old"}}
    assert _price_id_from_item(item) == "price_new"


@pytest.mark.parametrize("item", [
    None, {}, "price_x", 7, [],
    {"price": None}, {"price": {}}, {"price": {"id": None}},
    {"pricing": None}, {"pricing": {}}, {"pricing": {"price_details": {}}},
    {"pricing": {"price_details": {"price": None}}},
])
def test_nothing_readable_yields_none_rather_than_raising(item):
    """A raise here is a 500, and Stripe retries a 500 for days."""
    assert _price_id_from_item(item) is None


def test_an_empty_string_is_not_a_price():
    assert _price_id_from_item({"price": ""}) is None


# --------------------------------------------------------------------------
# ... and the two events that depend on it
# --------------------------------------------------------------------------

def renewal(shape: str, *, customer="cus_1", invoice_id="in_1"):
    return {
        "type": "invoice.payment_succeeded",
        "data": {"object": {
            "id": invoice_id, "customer": customer,
            "billing_reason": "subscription_cycle",
            "lines": {"data": [SHAPES[shape](PRICE_FULL_Y)]},
        }},
    }


def subscription(shape: str, *, customer="cus_1"):
    return {
        "type": "customer.subscription.updated",
        "data": {"object": {
            "customer": customer, "status": "active",
            "items": {"data": [SHAPES[shape](PRICE_FULL_Y)]},
        }},
    }


@pytest.mark.parametrize("shape", sorted(SHAPES))
async def test_a_renewal_grants_the_review_in_either_shape(db, make_user, shape):
    user = await make_user(tier=TIER_FULL, stripe_customer_id="cus_1")
    await _apply_event(db, renewal(shape))
    await db.refresh(user)
    assert user.expert_credits == FULL_EXPERT_CREDITS


@pytest.mark.parametrize("shape", sorted(SHAPES))
async def test_a_subscription_change_syncs_the_tier_in_either_shape(
    db, make_user, shape,
):
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    await _apply_event(db, subscription(shape))
    await db.refresh(user)
    assert user.tier == TIER_FULL


@pytest.mark.parametrize("shape", sorted(SHAPES))
async def test_an_unconfigured_price_still_grants_nothing(db, make_user, shape):
    """Tolerating both shapes must not turn into tolerating any price."""
    user = await make_user(tier=TIER_STARTER, stripe_customer_id="cus_1")
    event = subscription(shape)
    event["data"]["object"]["items"]["data"] = [SHAPES[shape]("price_unknown")]
    await _apply_event(db, event)
    await db.refresh(user)
    assert user.tier == TIER_STARTER
    assert user.subscription_status == "active"
