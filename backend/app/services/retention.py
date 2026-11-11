"""How long an uploaded clip is kept, and what pins one in place.

Storage used to have exactly one rule: delete everything ``job_ttl_hours``
(6h) after the analysis was accepted. That rule was written when a clip existed
only to be analysed, and it is why the Expert Review template opens by
apologising that the footage is gone -- somebody buys a review on Tuesday for
Monday's ride, and by then there is nothing to watch.

The obvious repair -- "keep a clip once its review is bought" -- does not work,
and it is worth writing down why, because it is the first thing anyone
proposes. A review is bought days after the analysis, from the history screen.
At six hours' retention the clip is long gone before the purchase happens, so
there is nothing left to pin. Retention has to be decided when the file is
written, by somebody who does not yet know whether it will ever be reviewed.

So the policy has two parts:

* a **baseline** by tier -- a paying athlete's footage outlives a free one's,
  which is also the honest version of "your history is worth something";
* a **pin** on top, for a clip with a review that has been paid for and not yet
  delivered. That one cannot expire on a deadline of ours while somebody is
  waiting on work from us.

This module answers one question -- *which stored jobs must survive this
sweep?* -- and the sweeper in ``app.core.jobs`` deletes everything else. Keeping
the policy here means the sweeper stays a filesystem routine that can be tested
without a database, and this stays a database query with no opinion about
directories.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis
from app.models.order import ORDER_DELIVERED, ORDER_REFUNDED, Order
from app.models.user import (
    TIER_ADMIN,
    TIER_COACH,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
    PlanLapse,
    User,
)

DAY_MS = 24 * 60 * 60 * 1000

# Baseline retention, in days, by the tier of the account that uploaded.
# A year on every paid plan (Artur, 2026-11-10): last season's clip next to
# this season's is what a paid history is for, and a measured CdA arrives
# months after the position was filmed. privacy.html sections 2 and 10 say so.
RETENTION_DAYS: dict[str, int] = {
    TIER_STARTER: 7,
    TIER_ENTHUSIAST: 365,
    TIER_FULL: 365,
    TIER_COACH: 365,
    TIER_ADMIN: 30,   # staff, not a paid plan
}
# What was promised before the year: a clip uploaded under that policy keeps
# the period it was uploaded under. Footage is not kept longer than the page
# said when it was taken.
LEGACY_RETENTION_DAYS: dict[str, int] = {
    TIER_STARTER: 7,
    TIER_ENTHUSIAST: 30,
    TIER_FULL: 30,
    TIER_COACH: 60,
    TIER_ADMIN: 30,
}
# 2026-11-11 00:00 UTC. Clips analysed before it are on the legacy periods.
YEAR_FROM_MS = 1_794_355_200_000
# An unknown tier is treated as the free one. Erring towards the shorter period
# is the safe direction for a mistake here: keeping someone's footage longer
# than promised is the failure that matters.
DEFAULT_RETENTION_DAYS = RETENTION_DAYS[TIER_STARTER]
MAX_RETENTION_DAYS = max(RETENTION_DAYS.values())

PAID_TIERS = frozenset({TIER_ENTHUSIAST, TIER_FULL, TIER_COACH})
# A paid plan that ends keeps its clips on the paid period this much longer,
# and services/plan_lapse.py emails the owner twice (Artur, 2026-11-11). A
# failed card used to cost up to a year of footage at the next sweep.
LAPSE_GRACE_DAYS = 30

# How long a clip survives after its review ships. Not zero: the athlete has
# just been told to re-read the report, and the reviewer may need to correct
# something they wrote about footage that would otherwise already be deleted.
REVIEW_GRACE_DAYS = 7


def retention_days(tier: str | None, created_at_ms: int | None = None) -> int:
    """Days a clip is kept, by the uploader's tier now.

    The tier is read at sweep time, not at upload: a plan that ends puts the
    free period on the clips from then on (privacy.html section 10 says so).
    """
    table = RETENTION_DAYS
    if created_at_ms is not None and created_at_ms < YEAR_FROM_MS:
        table = LEGACY_RETENTION_DAYS
    return table.get(tier or "", DEFAULT_RETENTION_DAYS)


def tier_for_clips(
    tier: str | None, lapse_from: str | None, lapsed_at_ms: int | None, now_ms: int,
) -> str | None:
    """The tier whose period a clip is kept on: the plan that ended, while its
    grace runs; the account's own tier otherwise."""
    if (
        tier == TIER_STARTER and lapse_from in PAID_TIERS and lapsed_at_ms is not None
        and now_ms < lapsed_at_ms + LAPSE_GRACE_DAYS * DAY_MS
    ):
        return lapse_from
    return tier


def _now_ms(now_ms: int | None = None) -> int:
    if now_ms is not None:
        return now_ms
    return int(datetime.now(timezone.utc).timestamp() * 1000)


async def job_ids_to_keep(
    db: AsyncSession, now_ms: int | None = None,
) -> set[str]:
    """Every stored job id that must survive this sweep.

    Two queries rather than one: the baseline is bounded by time (and so can be
    windowed cheaply), while a pinned clip is deliberately unbounded -- an order
    nobody has fulfilled yet keeps its footage however old it gets, which is the
    entire point of a pin and exactly what a time window would defeat.
    """
    now = _now_ms(now_ms)
    keep: set[str] = set()

    # 1. Baseline: still inside its owner's retention period. The window is the
    #    longest period anyone gets, so rows older than that need no per-row
    #    check -- they are past every tier's limit by definition.
    window_start = now - MAX_RETENTION_DAYS * DAY_MS
    rows = await db.execute(
        select(
            Analysis.job_id, Analysis.created_at_ms, User.tier,
            PlanLapse.from_tier, PlanLapse.lapsed_at_ms,
        )
        .join(User, User.id == Analysis.user_id)
        .outerjoin(PlanLapse, PlanLapse.user_id == Analysis.user_id)
        .where(
            Analysis.job_id.is_not(None),
            Analysis.created_at_ms >= window_start,
        )
    )
    for job_id, created_at_ms, tier, lapse_from, lapsed_at_ms in rows:
        tier = tier_for_clips(tier, lapse_from, lapsed_at_ms, now)
        if now - (created_at_ms or 0) <= retention_days(tier, created_at_ms or 0) * DAY_MS:
            keep.add(str(job_id))

    # 2. Pinned: an Expert Review that is paid for and still owed, or one
    #    delivered recently enough that the reviewer might still amend it.
    #    Joined on (user_id, client_id) because ``client_id`` is minted on the
    #    device and is only unique per account -- joining on it alone would let
    #    one athlete's order pin another athlete's footage.
    grace_start = now - REVIEW_GRACE_DAYS * DAY_MS
    pinned = await db.execute(
        select(Analysis.job_id)
        .join(
            Order,
            and_(
                Order.user_id == Analysis.user_id,
                Order.analysis_client_id == Analysis.client_id,
            ),
        )
        .where(
            Analysis.job_id.is_not(None),
            or_(
                Order.status.not_in((ORDER_DELIVERED, ORDER_REFUNDED)),
                and_(
                    Order.status == ORDER_DELIVERED,
                    Order.delivered_at_ms.is_not(None),
                    Order.delivered_at_ms >= grace_start,
                ),
            ),
        )
    )
    keep.update(str(job_id) for (job_id,) in pinned)
    return keep


async def user_owns_job(db: AsyncSession, user_id: int, job_id: str) -> bool:
    """Whether one of this user's stored analyses points at ``job_id``.

    The in-memory job store empties on every deploy and forgets finished jobs
    after ``job_ttl_hours`` regardless; the files under that id outlive both.
    This is how a signed-in athlete's own footage is recognised once the store
    no longer remembers who uploaded it -- "the video stopped playing half an
    hour later" was a redeploy between the upload and the replay, not an
    expiry, and the file was on the volume the whole time.
    """
    if not job_id:
        return False
    found = await db.scalar(
        select(Analysis.id)
        .where(Analysis.user_id == user_id, Analysis.job_id == job_id)
        .limit(1)
    )
    return found is not None


async def job_ids_for_user(
    db: AsyncSession, user_id: int, client_ids: list[str] | None = None,
) -> list[str]:
    """Stored job ids behind a user's analyses -- for deleting the files too.

    Making clips outlive their analysis created a new way to break the privacy
    policy: deleting a history entry (or a whole account) would leave the
    footage sitting on the volume for the rest of its retention period, which
    is precisely what "delete this" is supposed to prevent.
    """
    query = select(Analysis.job_id).where(
        Analysis.user_id == user_id, Analysis.job_id.is_not(None),
    )
    if client_ids is not None:
        query = query.where(Analysis.client_id.in_(client_ids))
    rows = await db.execute(query)
    return [str(job_id) for (job_id,) in rows]
