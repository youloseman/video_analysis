"""Selling one report at a time.

The unlock exists because the product is used episodically: somebody changes
their saddle height, films it, and wants that one answer. A subscription is
priced for a habit they do not have, so at $9/month they convert at roughly
zero and are worth nothing. $4 for the report in front of them is the version
of the offer that matches what they came for.

The fence between the two is *time*, not price: an unlock opens one report,
and a report on its own cannot be compared with anything. Trends, before/after
and the overlay video -- everything that needs a second analysis to mean
something -- stay with the subscription. So the tests worth writing are about
what an unlock does NOT open, and about the tier still winning over it.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import billing, me
from app.api.billing import CheckoutIn, _apply_event, create_checkout
from app.core.config import settings
from app.models.analysis import Analysis
from app.models.order import Order
from app.models.user import TIER_ENTHUSIAST, TIER_STARTER
from app.services.result_gating import (
    ACCESS_FULL,
    ACCESS_PREVIEW,
    ACCESS_TEASER,
    access_for_stored,
)

FULL_RESULT = {
    "status": "completed",
    "sport_type": "run",
    "technique_score": 74,
    "letter_grade": "C",
    "detected_issues": [{"type": "overstriding", "value": "0.21 x leg length"}],
    "angle_statistics": {"knee": {"mean": 142.0}, "hip": {"mean": 31.0}},
    "training_plan": {"top_3_priorities": [{"drill": "strides"}]},
    "ai_recommendations": {"report": "lift your cadence"},
    "sport_specific_metrics": {"cadence_spm": 168},
}


async def store(db, user, *, client_id="h1", preview=False, unlocked=None,
                result=FULL_RESULT):
    row = Analysis(
        user_id=user.id, client_id=client_id, job_id="j1", created_at_ms=1,
        data={"id": client_id}, result=result, preview=preview,
        unlocked_at_ms=unlocked,
    )
    db.add(row)
    await db.commit()
    return row


def unlock_event(user_id, *, client_id="h1", session_id="cs_unlock_1"):
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "client_reference_id": str(user_id),
            "customer": "cus_1",
            "mode": "payment",
            "amount_total": 400,
            "currency": "usd",
            "metadata": {
                "user_id": str(user_id), "plan": "unlock", "tier": "",
                "analysis_client_id": client_id or "",
            },
        }},
    }


# --------------------------------------------------------------------------
# Who may read what
# --------------------------------------------------------------------------
async def test_a_teaser_stays_shut(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user)
    assert access_for_stored(user, row) == ACCESS_TEASER


async def test_an_unlocked_report_opens(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user, unlocked=123)
    assert access_for_stored(user, row) == ACCESS_FULL


async def test_the_free_preview_is_still_the_preview(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user, preview=True)
    assert access_for_stored(user, row) == ACCESS_PREVIEW


async def test_subscribing_opens_everything_that_was_ever_run(db, make_user):
    """Most of the reason the full result is stored. Somebody who paid $4 twice
    and then subscribes should not find their old reports still shut -- and this
    is a better argument for a plan than any wording on the pricing page."""
    user = await make_user(tier=TIER_ENTHUSIAST)
    old_teaser = await store(db, user, client_id="h-old")
    assert access_for_stored(user, old_teaser) == ACCESS_FULL


async def test_an_unlock_does_not_leak_to_the_next_report(db, make_user):
    """One report, not an account-wide switch."""
    user = await make_user(tier=TIER_STARTER)
    bought = await store(db, user, client_id="h1", unlocked=123)
    other = await store(db, user, client_id="h2")
    assert access_for_stored(user, bought) == ACCESS_FULL
    assert access_for_stored(user, other) == ACCESS_TEASER


# --------------------------------------------------------------------------
# Buying one
# --------------------------------------------------------------------------
async def test_an_unlock_must_name_the_report_it_buys(db, make_user, monkeypatch):
    """Unlike an Expert Review there is no human downstream to work out which
    report was meant, so an unnamed unlock buys nothing at all."""
    monkeypatch.setattr(
        billing, "settings", dataclasses.replace(settings, stripe_secret_key="sk_test_x"),
    )
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="unlock"), None, user, db)
    assert exc.value.status_code == 400


async def test_paying_opens_exactly_that_report(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user)
    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    assert row.unlocked_at_ms
    assert access_for_stored(user, row) == ACCESS_FULL


async def test_paying_records_the_order_too(db, make_user):
    """The entitlement lives on the analysis; the order is the receipt. Both,
    because one answers "may I read this" and the other answers "did I pay"."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    await _apply_event(db, unlock_event(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.plan == "unlock"
    assert order.amount_total == 400


async def test_a_redelivered_webhook_changes_nothing(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user)
    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    first = row.unlocked_at_ms

    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    assert row.unlocked_at_ms == first
    assert len((await db.execute(select(Order))).scalars().all()) == 1


async def test_paying_for_a_deleted_report_does_not_explode(db, make_user):
    """Deleted between paying and the webhook landing. The order stands as the
    record that they paid; there is simply nothing left to open."""
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, unlock_event(user.id, client_id="h-gone"))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.plan == "unlock"


async def test_an_unlock_never_reaches_another_account(db, make_user):
    mine = await make_user(tier=TIER_STARTER)
    theirs = await make_user(tier=TIER_STARTER)
    ours = await store(db, mine, client_id="h1")
    not_ours = await store(db, theirs, client_id="h1")

    await _apply_event(db, unlock_event(mine.id, client_id="h1"))
    await db.refresh(ours)
    await db.refresh(not_ours)
    assert ours.unlocked_at_ms
    assert not_ours.unlocked_at_ms is None


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------
async def test_the_result_endpoint_serves_the_whole_report_once_unlocked(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, unlocked=123)
    out = await me.get_analysis_result("h1", user, db)
    assert out["access"] == ACCESS_FULL
    assert out["result"]["angle_statistics"] == FULL_RESULT["angle_statistics"]
    assert out["result"]["training_plan"] == FULL_RESULT["training_plan"]


async def test_the_result_endpoint_still_trims_a_teaser(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    out = await me.get_analysis_result("h1", user, db)
    assert out["access"] == ACCESS_TEASER
    assert "angle_statistics" not in out["result"]
    assert "training_plan" not in out["result"]


async def test_a_report_from_before_full_results_were_stored_says_so(db, make_user):
    """Rather than returning a shell that looks like a report we lost -- and
    rather than selling an unlock that would reveal nothing."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, result=None)
    with pytest.raises(HTTPException) as exc:
        await me.get_analysis_result("h1", user, db)
    assert exc.value.status_code == 409


async def test_the_history_list_says_what_each_entry_may_show(db, make_user):
    """The client stored whatever it was shown that day, so it needs telling
    when it is now entitled to more."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, client_id="h1", unlocked=123)
    await store(db, user, client_id="h2")
    await store(db, user, client_id="h3", result=None)

    by_id = {e["id"]: e for e in await me.list_analyses(user, db)}
    assert by_id["h1"]["access"] == ACCESS_FULL
    assert by_id["h2"]["access"] == ACCESS_TEASER
    assert by_id["h2"]["sellable"] is True
    assert by_id["h3"]["sellable"] is False


async def test_reading_another_accounts_report_is_a_404(db, make_user):
    mine = await make_user()
    theirs = await make_user()
    await store(db, theirs, client_id="h1")
    with pytest.raises(HTTPException) as exc:
        await me.get_analysis_result("h1", mine, db)
    assert exc.value.status_code == 404
