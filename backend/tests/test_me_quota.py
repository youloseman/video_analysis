"""/auth/me carries the analysis allowance, so Account and Analyze can show
the count from the first load instead of only after an analysis."""
from __future__ import annotations

from app.api import auth as auth_api
from app.models.user import TIER_FULL, TIER_STARTER, tier_limit
from app.services.usage_limits import record_usage


async def test_me_says_what_is_left_this_window(db, make_user):
    user = await make_user(tier=TIER_FULL)
    limit, window = tier_limit(TIER_FULL)
    out = await auth_api.me(user, db)
    assert out.quota.limit == limit and out.quota.remaining == limit and out.quota.window == window
    await record_usage(db, user.id, "video")
    await record_usage(db, user.id, "photo")
    out = await auth_api.me(user, db)
    assert out.quota.remaining == limit - 2


async def test_remaining_never_goes_below_zero(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    limit, _ = tier_limit(TIER_STARTER)
    for _ in range(limit + 2):
        await record_usage(db, user.id, "video")
    assert (await auth_api.me(user, db)).quota.remaining == 0
