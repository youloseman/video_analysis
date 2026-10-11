"""A tier that was given back has to actually come back.

Every tier used to come from Stripe and leave through Stripe. A tier handed
over by a person — "full for ten days, thanks for the feedback" — had no
webhook coming, so nothing in the system would ever have revoked it. The
promise was really "full forever, unless somebody remembers".

Two things can go wrong here and both are expensive in opposite directions. A
grant that never expires gives the product away. A sweep that is too eager
revokes a subscription somebody is paying for, which is worse: they notice,
and they are right to be angry. Most of what is below is about the second.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.models.user import TIER_ADMIN, TIER_ENTHUSIAST, TIER_FULL, TIER_STARTER, User
from app.services.tier_grants import (
    active_grants,
    as_utc,
    clear_grant,
    expire_tier_grants,
    grant_tier,
)


def test_a_naive_datetime_from_sqlite_is_read_as_utc():
    """SQLite has no timezone type, Postgres does. Anything doing arithmetic
    on tier_until has to survive both, or it works in production and crashes
    on a laptop -- the divergence this project has been bitten by before."""
    naive = datetime(2026, 10, 21, 10, 0, 0)
    assert as_utc(naive) == datetime(2026, 10, 21, 10, 0, 0, tzinfo=timezone.utc)
    aware = datetime(2026, 10, 21, 10, 0, 0, tzinfo=timezone.utc)
    assert as_utc(aware) is aware
    assert as_utc(None) is None


def _now():
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# granting
# --------------------------------------------------------------------------

async def test_a_grant_sets_the_tier_and_an_end_date(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    out = await grant_tier(db, user.email, TIER_FULL, 10)
    assert out is not None
    assert out.tier == TIER_FULL
    assert out.tier_until is not None
    # Ten days, give or take the second it took to run.
    assert timedelta(days=9, hours=23) < out.tier_until - _now() < timedelta(days=10)


async def test_the_email_match_is_case_insensitive(db, make_user):
    """People type their address the way they feel like typing it, and the one
    in the outreach list came out of a database rather than off a business
    card."""
    user = await make_user(tier=TIER_STARTER, email="Mixed.Case@Example.COM")
    assert await grant_tier(db, "mixed.case@example.com", TIER_FULL, 10) is not None
    await db.refresh(user)
    assert user.tier == TIER_FULL


async def test_an_unknown_address_is_not_an_error(db):
    """A typo in a hand-typed address must not look like a success."""
    assert await grant_tier(db, "nobody@example.com", TIER_FULL, 10) is None


async def test_granting_again_replaces_rather_than_extends(db, make_user):
    """Ten days to somebody with three left should leave ten. That is what a
    person means when they say it."""
    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 30)
    first = as_utc(user.tier_until)
    await grant_tier(db, user.email, TIER_FULL, 10)
    await db.refresh(user)
    assert as_utc(user.tier_until) < first


async def test_an_admin_is_never_granted_over(db, make_user):
    """The account that runs the service must not acquire an expiry date and
    get demoted by a housekeeping job at 3am."""
    user = await make_user(tier=TIER_ADMIN)
    out = await grant_tier(db, user.email, TIER_FULL, 10)
    assert out is not None
    await db.refresh(user)
    assert user.tier == TIER_ADMIN
    assert user.tier_until is None


@pytest.mark.parametrize("bad", [0, -1, -365])
async def test_a_nonsense_duration_is_refused(db, make_user, bad):
    user = await make_user()
    with pytest.raises(ValueError):
        await grant_tier(db, user.email, TIER_FULL, bad)


async def test_an_unknown_tier_is_refused(db, make_user):
    user = await make_user()
    with pytest.raises(ValueError):
        await grant_tier(db, user.email, "platinum", 10)


async def test_the_admin_tier_cannot_be_handed_out(db, make_user):
    """Not a typo guard — admin is unbounded and granting it for ten days
    would create an account nothing could demote."""
    user = await make_user()
    with pytest.raises(ValueError):
        await grant_tier(db, user.email, TIER_ADMIN, 10)


# --------------------------------------------------------------------------
# expiry
# --------------------------------------------------------------------------

async def test_an_expired_grant_goes_back_to_starter(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 10)
    assert await expire_tier_grants(db, now=_now() + timedelta(days=11)) == 1
    await db.refresh(user)
    assert user.tier == TIER_STARTER
    assert user.tier_until is None


async def test_a_live_grant_is_left_alone(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 10)
    assert await expire_tier_grants(db, now=_now() + timedelta(days=9)) == 0
    await db.refresh(user)
    assert user.tier == TIER_FULL


async def test_a_bought_tier_is_never_swept(db, make_user):
    """The expensive mistake. A Stripe subscription has no end date on the
    row, and the sweep must key on that and nothing else — a WHERE clause that
    matched on tier instead would cancel paying customers."""
    payer = await make_user(tier=TIER_FULL)
    assert payer.tier_until is None
    assert await expire_tier_grants(db, now=_now() + timedelta(days=3650)) == 0
    await db.refresh(payer)
    assert payer.tier == TIER_FULL


async def test_an_admin_with_an_expiry_is_still_not_demoted(db, make_user):
    """Belt and braces: grant_tier refuses admins, but if a date ever reached
    an admin row by another route the sweep must still not act on it."""
    user = await make_user(tier=TIER_ADMIN)
    user.tier_until = _now() - timedelta(days=1)
    await db.commit()
    assert await expire_tier_grants(db, now=_now()) == 0
    await db.refresh(user)
    assert user.tier == TIER_ADMIN


async def test_sweeping_twice_changes_nothing_the_second_time(db, make_user):
    """It runs every ten minutes forever and once at every boot."""
    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 1)
    later = _now() + timedelta(days=2)
    assert await expire_tier_grants(db, now=later) == 1
    assert await expire_tier_grants(db, now=later) == 0


async def test_it_expires_several_at_once(db, make_user):
    for _ in range(3):
        u = await make_user(tier=TIER_STARTER)
        await grant_tier(db, u.email, TIER_FULL, 5)
    live = await make_user(tier=TIER_STARTER)
    await grant_tier(db, live.email, TIER_ENTHUSIAST, 60)

    assert await expire_tier_grants(db, now=_now() + timedelta(days=6)) == 3
    await db.refresh(live)
    assert live.tier == TIER_ENTHUSIAST


# --------------------------------------------------------------------------
# Stripe always wins
# --------------------------------------------------------------------------

def test_clear_grant_removes_the_date():
    user = User(email="x@example.com", password_hash="x", tier=TIER_FULL)
    user.tier_until = _now() + timedelta(days=5)
    clear_grant(user)
    assert user.tier_until is None


async def test_buying_a_plan_during_a_grant_makes_it_permanent(db, make_user):
    """Somebody on a ten-day thank-you subscribes on day three. If the date
    survived, the sweep would cancel the plan they are paying for a week
    later, and the only evidence would be an angry email."""
    from app.api.billing import _set_tier

    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 10)
    await db.refresh(user)
    assert user.tier_until is not None

    _set_tier(user, TIER_ENTHUSIAST)   # the webhook path
    await db.commit()

    assert user.tier_until is None
    assert await expire_tier_grants(db, now=_now() + timedelta(days=30)) == 0
    await db.refresh(user)
    assert user.tier == TIER_ENTHUSIAST


async def test_a_lapse_during_a_grant_also_clears_it(db, make_user):
    """The same single place handles the way down: a cancelled subscription
    drops the tier AND the date, so an expired grant cannot resurrect later."""
    from app.api.billing import _set_tier

    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 10)
    _set_tier(user, TIER_STARTER)
    await db.commit()
    assert user.tier_until is None


# --------------------------------------------------------------------------
# seeing what is outstanding
# --------------------------------------------------------------------------

async def test_active_grants_lists_only_granted_tiers(db, make_user):
    """The only place the outstanding promises are visible. A paying customer
    appearing in this list would be alarming and wrong."""
    granted = await make_user(tier=TIER_STARTER)
    await grant_tier(db, granted.email, TIER_FULL, 10)
    await make_user(tier=TIER_FULL)        # a payer
    await make_user(tier=TIER_STARTER)     # free

    rows = await active_grants(db)
    assert len(rows) == 1
    assert rows[0][1] == granted.email


async def test_active_grants_is_ordered_by_what_ends_soonest(db, make_user):
    a = await make_user(tier=TIER_STARTER)
    b = await make_user(tier=TIER_STARTER)
    await grant_tier(db, a.email, TIER_FULL, 30)
    await grant_tier(db, b.email, TIER_FULL, 2)
    rows = await active_grants(db)
    assert [r[1] for r in rows] == [b.email, a.email]


async def test_the_whole_cycle(db, make_user):
    """Grant, see it, let it run out, see it gone — the thing the letters
    promise, start to finish."""
    user = await make_user(tier=TIER_STARTER)
    await grant_tier(db, user.email, TIER_FULL, 10)
    assert len(await active_grants(db)) == 1

    await expire_tier_grants(db, now=_now() + timedelta(days=10, seconds=1))
    assert await active_grants(db) == []

    stored = (
        await db.execute(select(User).where(User.id == user.id))
    ).scalar_one()
    assert stored.tier == TIER_STARTER
