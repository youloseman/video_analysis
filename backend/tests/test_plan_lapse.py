"""A paid plan that ends: 30 more days for the clips, and two emails.

What matters: a failed card does not take a year of footage at the next
sweep; the owner is told when it happens and again before the clips go, and
only when there is something to lose; once the grace is over the free plan's
period applies as before.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.api import billing
from app.models.analysis import Analysis
from app.models.user import TIER_COACH, TIER_FULL, TIER_STARTER, PlanLapse
from app.services import notify, plan_lapse
from app.services.retention import (
    DAY_MS,
    LAPSE_GRACE_DAYS,
    YEAR_FROM_MS,
    job_ids_to_keep,
)
from app.services.tier_grants import expire_tier_grants, grant_tier

T = YEAR_FROM_MS + 200 * DAY_MS   # the moment the plan ends, well into the year era


async def clip(db, user, job_id, days_before_lapse):
    db.add(Analysis(user_id=user.id, client_id=job_id, job_id=job_id,
                    created_at_ms=int(T - days_before_lapse * DAY_MS), data={"id": job_id}))
    await db.commit()


@pytest.fixture
def outbox(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send_email",
                        lambda to, subject, text, html: sent.append((to, subject, text)) or True)
    return sent


# --------------------------------------------------------------------------
# Writing the lapse down
# --------------------------------------------------------------------------
async def test_a_cancelled_subscription_starts_the_grace_once(make_user, db):
    u = await make_user(tier=TIER_FULL, stripe_customer_id="cus_1")
    await billing._on_subscription_deleted(db, {"customer": "cus_1"})
    await db.commit()
    row = await db.get(PlanLapse, u.id)
    assert u.tier == TIER_STARTER and row.from_tier == TIER_FULL
    first = row.lapsed_at_ms
    # Stripe sends the end twice (an update to "canceled", then "deleted").
    await billing._on_subscription_deleted(db, {"customer": "cus_1"})
    await db.commit()
    assert (await db.get(PlanLapse, u.id)).lapsed_at_ms == first


async def test_a_coach_pilot_that_keeps_its_plan_writes_nothing(make_user, db):
    u = await make_user(tier=TIER_COACH, stripe_customer_id="cus_2",
                        tier_until=datetime.now(timezone.utc) + timedelta(days=20))
    await billing._on_subscription_deleted(db, {"customer": "cus_2"})
    await db.commit()
    assert u.tier == TIER_COACH and await db.get(PlanLapse, u.id) is None


async def test_a_grant_that_falls_to_free_starts_the_grace_one_over_a_plan_does_not(make_user, db):
    free = await make_user(tier=TIER_STARTER)
    paying = await make_user(tier=TIER_FULL, subscription_status="active")
    await grant_tier(db, free.email, TIER_FULL, 10)
    await grant_tier(db, paying.email, TIER_COACH, 10)
    assert await expire_tier_grants(db, datetime.now(timezone.utc) + timedelta(days=11)) == 2
    assert (await db.get(PlanLapse, free.id)).from_tier == TIER_FULL
    assert await db.get(PlanLapse, paying.id) is None


# --------------------------------------------------------------------------
# What the sweep keeps
# --------------------------------------------------------------------------
async def test_the_clips_keep_the_paid_period_for_thirty_days_then_go(make_user, db):
    u = await make_user(tier=TIER_STARTER)
    await clip(db, u, "old", 60)
    await plan_lapse.upsert_lapse(db, u.id, TIER_FULL, T)
    await db.commit()
    assert await job_ids_to_keep(db, T + 10 * DAY_MS) == {"old"}
    assert await job_ids_to_keep(db, T + (LAPSE_GRACE_DAYS + 1) * DAY_MS) == set()


async def test_paid_again_the_clips_are_on_the_plan_not_the_grace(make_user, db):
    u = await make_user(tier=TIER_STARTER)
    await clip(db, u, "old", 60)
    await plan_lapse.upsert_lapse(db, u.id, TIER_FULL, T)
    u.tier = TIER_FULL
    await db.commit()
    assert await job_ids_to_keep(db, T + 100 * DAY_MS) == {"old"}


# --------------------------------------------------------------------------
# The two emails
# --------------------------------------------------------------------------
async def test_a_notice_now_a_reminder_five_days_before_then_silence(make_user, db, outbox):
    u = await make_user(tier=TIER_STARTER, email="rider@gmail.com")
    await clip(db, u, "a", 60)
    await clip(db, u, "b", 3)
    await plan_lapse.upsert_lapse(db, u.id, TIER_FULL, T)
    await db.commit()
    assert await plan_lapse.send_lapse_emails(db, T + DAY_MS // 2) == 1
    assert await plan_lapse.send_lapse_emails(db, T + DAY_MS) == 0
    to, subject, text = outbox[0]
    assert to == "rider@gmail.com" and "has ended" in subject and "2 uploaded clips" in text
    end = T + LAPSE_GRACE_DAYS * DAY_MS
    assert await plan_lapse.send_lapse_emails(db, end - 4 * DAY_MS) == 1
    assert "will be deleted" in outbox[1][1]
    assert await plan_lapse.send_lapse_emails(db, end - DAY_MS) == 0
    assert len(outbox) == 2


async def test_nothing_to_lose_nothing_sent(make_user, db, outbox):
    u = await make_user(tier=TIER_STARTER)
    await plan_lapse.upsert_lapse(db, u.id, TIER_FULL, T)
    await db.commit()
    assert await plan_lapse.send_lapse_emails(db, T + DAY_MS) == 0
    assert outbox == [] and (await db.get(PlanLapse, u.id)).emails_sent == plan_lapse.SENT_ALL


async def test_back_on_a_plan_no_email(make_user, db, outbox):
    u = await make_user(tier=TIER_FULL)
    await clip(db, u, "a", 60)
    await plan_lapse.upsert_lapse(db, u.id, TIER_FULL, T)
    await db.commit()
    assert await plan_lapse.send_lapse_emails(db, T + DAY_MS) == 0 and outbox == []


async def test_a_failed_send_is_tried_again(make_user, db, monkeypatch):
    u = await make_user(tier=TIER_STARTER)
    await clip(db, u, "a", 60)
    await plan_lapse.upsert_lapse(db, u.id, TIER_FULL, T)
    await db.commit()
    monkeypatch.setattr(notify, "send_email", lambda *a: False)
    assert await plan_lapse.send_lapse_emails(db, T + DAY_MS) == 0
    assert (await db.get(PlanLapse, u.id)).emails_sent == plan_lapse.SENT_NONE


def test_the_privacy_policy_promises_the_thirty_days():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "privacy.html").read_text(encoding="utf-8")
    assert "kept on the paid period for 30 more days" in html
