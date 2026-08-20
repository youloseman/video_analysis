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
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
    User,
)

DAY_MS = 24 * 60 * 60 * 1000

# Baseline retention, in days, by the tier of the account that uploaded.
RETENTION_DAYS: dict[str, int] = {
    TIER_STARTER: 7,
    TIER_ENTHUSIAST: 30,
    TIER_FULL: 30,
    TIER_ADMIN: 30,
}
# An unknown tier is treated as the free one. Erring towards the shorter period
# is the safe direction for a mistake here: keeping someone's footage longer
# than promised is the failure that matters.
DEFAULT_RETENTION_DAYS = RETENTION_DAYS[TIER_STARTER]
MAX_RETENTION_DAYS = max(RETENTION_DAYS.values())

# How long a clip survives after its review ships. Not zero: the athlete has
# just been told to re-read the report, and the reviewer may need to correct
# something they wrote about footage that would otherwise already be deleted.
REVIEW_GRACE_DAYS = 7


def retention_days(tier: str | None) -> int:
    return RETENTION_DAYS.get(tier or "", DEFAULT_RETENTION_DAYS)


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
        select(Analysis.job_id, Analysis.created_at_ms, User.tier)
        .join(User, User.id == Analysis.user_id)
        .where(
            Analysis.job_id.is_not(None),
            Analysis.created_at_ms >= window_start,
        )
    )
    for job_id, created_at_ms, tier in rows:
        if now - (created_at_ms or 0) <= retention_days(tier) * DAY_MS:
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
