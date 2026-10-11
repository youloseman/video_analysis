"""Tiers given rather than bought, and the clock that takes them back.

Until now a tier was only ever set by Stripe, and only ever removed by Stripe:
a subscription starts, a webhook raises the tier; it lapses, a webhook drops
it. That works for everything the billing code does and for nothing else.

The thing it does not cover is a tier handed over by a person. "Full for ten
days, as thanks for the feedback" has no webhook coming — nothing in the
system would ever have revoked it — so the promise was really "full forever,
unless somebody remembers in ten days". With one account that is a calendar
reminder. With a batch of outreach replies it is a list nobody keeps, and the
free plan quietly stops being the plan most accounts are on.

So a granted tier carries an end date (``users.tier_until``) and a sweep puts
it back. Three rules hold it together:

* **Stripe always wins.** Every path that sets a tier from a payment clears
  ``tier_until`` — see ``clear_grant``. A grant can neither outlive a purchase
  nor cut one short.
* **Admins are never touched**, the same rule billing already follows.
* **Expiry is idempotent and restart-safe.** It is a date on a row, not a
  timer in memory: a process that dies mid-grant loses nothing, and a sweep
  that runs twice does the same thing once.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal
from app.models.user import TIER_ADMIN, TIER_STARTER, VALID_TIERS, User

logger = structlog.get_logger()

# How often the sweep runs. A grant is measured in days, so being up to ten
# minutes late costs nothing and this keeps the query rare.
SWEEP_INTERVAL_S = 600


def _now() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime | None) -> datetime | None:
    """A datetime read back from the database, as an aware UTC one.

    Postgres hands back what was written: timezone-aware. SQLite has no
    timezone type and hands back a naive datetime holding the same UTC
    instant. Subtracting one from ``now()`` therefore works in production and
    raises ``TypeError`` on a developer's machine — the two-backends trap this
    project has been bitten by before, and the reason anything that does
    arithmetic on ``tier_until`` goes through here first.

    Nothing in the expiry path needs it: that comparison happens in SQL, where
    the database knows its own types. It is the human-facing arithmetic —
    "how many days are left" — that has to work in both.
    """
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def clear_grant(user: User) -> None:
    """Forget any end date on this account's tier.

    Called from every billing path that sets a tier, so that a real
    subscription is never on a borrowed clock. Cheap, idempotent, and far
    safer than remembering to do it at six call sites.
    """
    user.tier_until = None


async def grant_tier(
    db: AsyncSession, email: str, tier: str, days: int,
) -> User | None:
    """Give ``email`` ``tier`` for ``days``. Returns the user, or None.

    Refuses to touch an admin, for the same reason billing does: the account
    that runs the service must not be demoted by a housekeeping job.

    An existing grant is replaced rather than extended — granting ten days to
    somebody who has three left should leave ten, which is what a person means
    when they say it.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"unknown tier {tier!r}; expected one of {VALID_TIERS}")
    if tier == TIER_ADMIN:
        raise ValueError("refusing to grant the admin tier")
    if days <= 0:
        raise ValueError("days must be positive")

    user = (
        await db.execute(select(User).where(User.email.ilike(email.strip())))
    ).scalar_one_or_none()
    if user is None:
        return None
    if user.tier == TIER_ADMIN:
        logger.info("TIER_GRANT_SKIPPED_ADMIN", email=email)
        return user

    user.tier = tier
    user.tier_until = _now() + timedelta(days=days)
    await db.commit()
    logger.info(
        "TIER_GRANTED", user_id=user.id, tier=tier, days=days,
        until=user.tier_until.isoformat(),
    )
    return user


async def expire_tier_grants(db: AsyncSession, now: datetime | None = None) -> int:
    """Put expired grants back on the free plan. Returns how many.

    One UPDATE rather than a read-then-write loop: the set of rows is small
    and the operation is naturally expressed as "everything past its date".
    Admins are excluded in the WHERE clause, so there is no path by which a
    sweep can demote one.
    """
    cutoff = now or _now()
    result = await db.execute(
        update(User)
        .where(
            User.tier_until.is_not(None),
            User.tier_until <= cutoff,
            User.tier != TIER_ADMIN,
        )
        .values(tier=TIER_STARTER, tier_until=None)
        .returning(User.id, User.email)
    )
    rows = result.fetchall()
    if rows:
        await db.commit()
        for uid, email in rows:
            logger.info("TIER_GRANT_EXPIRED", user_id=uid, email=email)
    return len(rows)


async def active_grants(db: AsyncSession) -> list[tuple[int, str, str, datetime]]:
    """Every live grant: ``(id, email, tier, until)``. For the admin script."""
    rows = (
        await db.execute(
            select(User.id, User.email, User.tier, User.tier_until)
            .where(User.tier_until.is_not(None))
            .order_by(User.tier_until)
        )
    ).all()
    return [(r[0], r[1], r[2], as_utc(r[3])) for r in rows]


async def grant_sweeper_loop() -> None:
    """Background task: expire grants forever. Never dies on an error.

    Separate from the upload sweeper in core/jobs.py on purpose. That one is
    about files and the in-memory job store and is documented as such; this is
    about accounts. Sharing a loop would save one task and cost the next
    reader the time it takes to work out why user rows are being written by
    something called a file reaper.
    """
    while True:
        try:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            async with SessionLocal() as db:
                n = await expire_tier_grants(db)
            if n:
                logger.info("TIER_GRANTS_SWEPT", expired=n)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- the sweeper must never die
            logger.warning("TIER_SWEEP_FAILED", err=str(e))
