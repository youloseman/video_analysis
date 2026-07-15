"""Per-user analysis quota: count usage in a window, enforce the tier limit.

Signed-in users are limited by their tier (see ``models.user.TIER_LIMITS``):
starter/enthusiast/full get a monthly quota, admin a small daily one. The count
is DB-backed (``usage_events``) so it survives restarts and is shared across
workers -- unlike the in-memory IP limiter used for anonymous visitors.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.usage import UsageEvent
from app.models.user import User, tier_limit


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def window_start_ms(window: str, now: datetime | None = None) -> int:
    """Epoch-ms start of the current quota window.

    - ``"month"`` -> first instant of the current calendar month (UTC).
    - ``"day"``   -> 24h ago (rolling), matching the legacy IP limiter feel.
    """
    now = now or datetime.now(timezone.utc)
    if window == "day":
        start = now - timedelta(hours=24)
    else:  # "month" (default)
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return int(start.timestamp() * 1000)


def next_reset(window: str, now: datetime | None = None) -> datetime:
    """When the current window resets (for user-facing messaging)."""
    now = now or datetime.now(timezone.utc)
    if window == "day":
        return now + timedelta(hours=24)
    # First day of next month, midnight UTC.
    year, month = now.year, now.month
    return datetime(
        year + (month // 12), (month % 12) + 1, 1, tzinfo=timezone.utc,
    )


async def usage_in_window(db: AsyncSession, user_id: int, window: str) -> int:
    start = window_start_ms(window)
    total = await db.scalar(
        select(func.count(UsageEvent.id)).where(
            UsageEvent.user_id == user_id,
            UsageEvent.created_at_ms >= start,
        )
    )
    return int(total or 0)


async def check_quota(db: AsyncSession, user: User) -> tuple[bool, int, int, str]:
    """Return (allowed, used, limit, window) for this user without recording."""
    limit, window = tier_limit(user.tier)
    used = await usage_in_window(db, user.id, window)
    return used < limit, used, limit, window


async def record_usage(db: AsyncSession, user_id: int, kind: str) -> None:
    db.add(UsageEvent(user_id=user_id, created_at_ms=_now_ms(), kind=kind))
    await db.commit()
