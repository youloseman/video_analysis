#!/usr/bin/env python
"""Give an account a tier for a number of days, or look at the live grants.

The outreach letters promise "full for ten days as a thank you". This is how
that promise is made, and services/tier_grants.py is what takes it back when
the ten days are up.

Run it on the box that can reach the database — from a laptop the Railway
``DATABASE_URL`` is an internal hostname that does not resolve:

    railway ssh --service video-analysis "python scripts/grant_tier.py list"
    railway ssh --service video-analysis "python scripts/grant_tier.py grant someone@example.com --days 10"
    railway ssh --service video-analysis "python scripts/grant_tier.py revoke someone@example.com"

Locally (SQLite) it works the same with no arguments about hostnames.

``list`` is the one to run first and the one worth running regularly: it is
the only place the outstanding promises are visible.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.db import SessionLocal  # noqa: E402
from app.models.user import TIER_FULL, VALID_TIERS  # noqa: E402
from app.services.tier_grants import (  # noqa: E402
    active_grants,
    as_utc,
    expire_tier_grants,
    grant_tier,
)


async def cmd_grant(email: str, tier: str, days: int) -> int:
    async with SessionLocal() as db:
        user = await grant_tier(db, email, tier, days)
    if user is None:
        print(f"no account with email {email!r}")
        return 1
    if user.tier_until is None:
        print(f"{user.email} is {user.tier} and was left alone (admin)")
        return 0
    until = as_utc(user.tier_until)
    left = until - datetime.now(timezone.utc)
    print(f"{user.email} -> {user.tier} until "
          f"{until:%Y-%m-%d %H:%M} UTC ({left.days} days)")
    return 0


async def cmd_revoke(email: str) -> int:
    """Expire a grant now, by moving its date into the past and sweeping.

    Written this way rather than as a direct downgrade so that revoking goes
    through exactly the code path that expiry does -- one behaviour, tested
    once, no second way for an account to lose a tier.
    """
    from sqlalchemy import select

    from app.models.user import User

    async with SessionLocal() as db:
        user = (
            await db.execute(select(User).where(User.email.ilike(email.strip())))
        ).scalar_one_or_none()
        if user is None:
            print(f"no account with email {email!r}")
            return 1
        if user.tier_until is None:
            print(f"{user.email} is {user.tier} with no grant on it — nothing to revoke")
            return 1
        user.tier_until = datetime.now(timezone.utc)
        await db.commit()
        n = await expire_tier_grants(db)
    print(f"revoked {n} grant(s) for {email}")
    return 0


async def cmd_list() -> int:
    async with SessionLocal() as db:
        rows = await active_grants(db)
    if not rows:
        print("no live grants")
        return 0
    now = datetime.now(timezone.utc)
    print(f"{'id':>4}  {'email':34s} {'tier':12s} {'until (UTC)':17s} left")
    for uid, email, tier, until in rows:
        days = (until - now).days
        left = f"{days}d" if days >= 0 else "EXPIRED (next sweep)"
        print(f"{uid:>4}  {email[:34]:34s} {tier:12s} {until:%Y-%m-%d %H:%M}  {left}")
    return 0


async def cmd_sweep() -> int:
    async with SessionLocal() as db:
        n = await expire_tier_grants(db)
    print(f"expired {n} grant(s)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grant", help="give an account a tier for N days")
    g.add_argument("email")
    g.add_argument("--tier", default=TIER_FULL, choices=sorted(VALID_TIERS))
    g.add_argument("--days", type=int, default=10)

    r = sub.add_parser("revoke", help="end a grant now")
    r.add_argument("email")

    sub.add_parser("list", help="every live grant and when it ends")
    sub.add_parser("sweep", help="expire anything past its date now")

    a = p.parse_args()
    if a.cmd == "grant":
        return asyncio.run(cmd_grant(a.email, a.tier, a.days))
    if a.cmd == "revoke":
        return asyncio.run(cmd_revoke(a.email))
    if a.cmd == "sweep":
        return asyncio.run(cmd_sweep())
    return asyncio.run(cmd_list())


if __name__ == "__main__":
    raise SystemExit(main())
