"""Quota accounting.

The failure modes worth guarding are the boundary ones: an off-by-one that
gives away a free analysis every window, or a window edge that lets an event
from last month count against this month's allowance (or vice versa).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models.usage import UsageEvent
from app.models.user import (
    TIER_ADMIN,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
    tier_limit,
)
from app.services.usage_limits import (
    check_quota,
    next_reset,
    record_usage,
    usage_in_window,
    window_start_ms,
)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


async def add_events(db, user_id: int, when: list[datetime]) -> None:
    for t in when:
        db.add(UsageEvent(user_id=user_id, created_at_ms=ms(t), kind="video"))
    await db.commit()


# --------------------------------------------------------------------------
# Tier table
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        (TIER_STARTER, (10, "month")),
        (TIER_ENTHUSIAST, (30, "month")),
        (TIER_FULL, (120, "month")),
        (TIER_ADMIN, (15, "day")),
    ],
)
def test_tier_limits(tier, expected):
    assert tier_limit(tier) == expected


def test_unknown_tier_falls_back_to_the_free_plan():
    """A tier string written by a future Stripe price (or a typo) must not be
    read as "unlimited"."""
    assert tier_limit("platinum_unicorn") == tier_limit(TIER_STARTER)


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def test_month_window_starts_at_the_first_of_the_month():
    now = datetime(2026, 7, 30, 13, 45, tzinfo=timezone.utc)
    assert window_start_ms("month", now) == ms(
        datetime(2026, 7, 1, tzinfo=timezone.utc)
    )


def test_day_window_is_a_rolling_24h():
    now = datetime(2026, 7, 30, 13, 45, tzinfo=timezone.utc)
    assert window_start_ms("day", now) == ms(now - timedelta(hours=24))


def test_next_reset_rolls_into_the_next_month():
    now = datetime(2026, 7, 30, 13, 45, tzinfo=timezone.utc)
    assert next_reset("month", now) == datetime(2026, 8, 1, tzinfo=timezone.utc)


def test_next_reset_rolls_over_the_year_end():
    """December is the case a naive ``month + 1`` gets wrong."""
    now = datetime(2026, 12, 31, 23, 59, tzinfo=timezone.utc)
    assert next_reset("month", now) == datetime(2027, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------
async def test_only_events_inside_the_window_count(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    await add_events(db, user.id, [
        month_start - timedelta(seconds=1),   # last month -- must NOT count
        month_start + timedelta(seconds=1),   # this month -- counts
        now - timedelta(minutes=5),           # this month -- counts
    ])
    assert await usage_in_window(db, user.id, "month") == 2


async def test_usage_is_counted_per_user(db, make_user):
    a = await make_user()
    b = await make_user()
    now = datetime.now(timezone.utc)
    await add_events(db, a.id, [now, now])
    await add_events(db, b.id, [now])
    assert await usage_in_window(db, a.id, "month") == 2
    assert await usage_in_window(db, b.id, "month") == 1


async def test_quota_allows_up_to_the_limit_then_blocks(db, make_user):
    """The boundary: the Nth analysis on an N/month plan is allowed, the N+1th
    is not. Reads the cap rather than restating it -- this test is about the
    boundary, and should not fail again the next time the number moves."""
    user = await make_user(tier=TIER_STARTER)
    cap, _ = tier_limit(TIER_STARTER)
    now = datetime.now(timezone.utc)

    allowed, used, limit, window = await check_quota(db, user)
    assert (allowed, used, limit, window) == (True, 0, cap, "month")

    await add_events(db, user.id, [now] * (cap - 1))
    allowed, used, _, _ = await check_quota(db, user)
    assert (allowed, used) == (True, cap - 1)      # the last one is still allowed

    await add_events(db, user.id, [now])
    allowed, used, _, _ = await check_quota(db, user)
    assert (allowed, used) == (False, cap)   # one past the cap is blocked


async def test_admin_quota_is_a_rolling_day(db, make_user):
    """Admin is capped daily, so an event from 25h ago must not count."""
    user = await make_user(tier=TIER_ADMIN)
    # Read the cap rather than restating it: this test is about the *window*,
    # and it should not fail again the next time the number moves.
    cap, _ = tier_limit(TIER_ADMIN)
    now = datetime.now(timezone.utc)
    # One event outside the window, then one short of a full day's worth in it.
    await add_events(db, user.id, [now - timedelta(hours=25)] + [now] * (cap - 1))
    allowed, used, limit, window = await check_quota(db, user)
    assert (used, limit, window) == (cap - 1, cap, "day")
    assert allowed is True

    await add_events(db, user.id, [now])
    allowed, used, _, _ = await check_quota(db, user)
    assert (allowed, used) == (False, cap)


async def test_record_usage_moves_the_counter(db, make_user):
    user = await make_user(tier=TIER_ENTHUSIAST)
    await record_usage(db, user.id, "video")
    await record_usage(db, user.id, "photo")
    assert await usage_in_window(db, user.id, "month") == 2


async def test_photo_and_video_share_one_quota(db, make_user):
    """Both endpoints record into the same table on purpose -- a plan is N
    analyses, not N videos plus N photos."""
    user = await make_user(tier=TIER_STARTER)
    cap, _ = tier_limit(TIER_STARTER)
    kinds = ["video", "photo"] * cap
    for kind in kinds[:cap]:
        await record_usage(db, user.id, kind)
    allowed, used, _, _ = await check_quota(db, user)
    assert (allowed, used) == (False, cap)
