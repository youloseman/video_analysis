"""How many re-measurements a paying athlete gets, and why it is not a quota.

A correction is a repair of an analysis that was already paid for. Charging it
against the monthly quota would read as a penalty for our mistake, so it does
not: what is bounded instead is how much of the server one account can spend
per day repairing things. A re-measurement costs about half an analysis on a
video (no detector, but the overlay still renders) and a fraction of a second
on a photo; the daily caps sit at roughly a third of each tier's monthly
analysis quota, which nobody adjusting their own fit will ever reach and a
script would reach in a minute.

The counter lives in memory. It resets on deploy, which is fine for what it is
-- an abuse guard, not accounting. Nothing here is billed.
"""

from __future__ import annotations

import time
from collections import defaultdict

from app.models.user import TIER_ADMIN, TIER_ENTHUSIAST, TIER_FULL

DAILY_RECOMPUTES: dict[str, int] = {
    TIER_ENTHUSIAST: 10,
    TIER_FULL: 40,
    TIER_ADMIN: 15,
}
# Rounds of adjustment one analysis may go through. Offsets accumulate, so
# three covers "moved the hip, then the foot, then nudged the foot again".
PER_ANALYSIS = 3

_WINDOW_S = 24 * 3600
_used: dict[int, list[float]] = defaultdict(list)


def daily_limit(tier: str | None) -> int:
    """0 for a tier that has no corrections at all (the free plan)."""
    return DAILY_RECOMPUTES.get(tier or "", 0)


def take_recompute(
    user_id: int, tier: str | None, now: float | None = None,
) -> tuple[bool, int, int]:
    """Spend one re-measurement if the account has one left today.

    Returns ``(allowed, used_today, limit)``; ``used_today`` counts this one
    when it was allowed.
    """
    limit = daily_limit(tier)
    now = time.time() if now is None else now
    stamps = [t for t in _used[user_id] if now - t < _WINDOW_S]
    if len(stamps) >= limit:
        _used[user_id] = stamps
        return False, len(stamps), limit
    stamps.append(now)
    _used[user_id] = stamps
    return True, len(stamps), limit


def reset() -> None:
    """Tests only."""
    _used.clear()


__all__ = ["DAILY_RECOMPUTES", "PER_ANALYSIS", "daily_limit", "take_recompute", "reset"]
