#!/usr/bin/env python
"""Keep the hip-over-crank reading for Enthusiasts to the end of their period.

The reading moved from Enthusiast to Full on 2026-10-26. Somebody who paid for
Enthusiast while it was in the plan keeps it until the period they paid for
runs out: this writes that moment into ``users.crank_until`` once, read from
the subscription in Stripe (or from ``tier_until`` for a hand-granted plan).
``models.user.crank_access`` does the rest; nothing needs undoing later.

Run it on the box that can reach the database and Stripe. Without ``--apply``
it only prints what it would write:

    railway ssh --service video-analysis "python scripts/keep_crank.py"
    railway ssh --service video-analysis "python scripts/keep_crank.py --apply"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stripe  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.models.user import TIER_ENTHUSIAST, User  # noqa: E402
from app.services.tier_grants import as_utc  # noqa: E402

LIVE = ("active", "trialing", "past_due")


def period_end(sub: Any) -> int | None:
    """When a subscription's current period ends, as a Unix time.

    Newer Stripe API versions moved ``current_period_end`` from the
    subscription onto its items, so both places are read.
    """
    end = sub.get("current_period_end")
    if end:
        return int(end)
    items = (sub.get("items") or {}).get("data") or []
    ends = [int(i["current_period_end"]) for i in items if i.get("current_period_end")]
    return max(ends) if ends else None


def paid_until(user: User) -> tuple[datetime | None, str]:
    if user.tier_until is not None:
        return as_utc(user.tier_until), "grant"
    if not user.stripe_customer_id:
        return None, "no Stripe customer"
    subs = stripe.Subscription.list(customer=user.stripe_customer_id, status="all", limit=20)
    ends = [period_end(s) for s in subs.auto_paging_iter() if s.get("status") in LIVE]
    ends = [e for e in ends if e]
    if not ends:
        return None, "no live subscription"
    return datetime.fromtimestamp(max(ends), timezone.utc), "stripe"


async def run(apply: bool) -> int:
    stripe.api_key = settings.stripe_secret_key
    async with SessionLocal() as db:
        users = (
            await db.execute(select(User).where(User.tier == TIER_ENTHUSIAST).order_by(User.id))
        ).scalars().all()
        if not users:
            print("no Enthusiast accounts")
            return 0
        for user in users:
            until, source = paid_until(user)
            shown = f"{until:%Y-%m-%d %H:%M} UTC" if until else "-"
            print(f"{user.id:>6}  {user.email:<40} {shown:<22} {source}")
            if apply and until is not None:
                user.crank_until = until
        if apply:
            await db.commit()
            print("written")
        else:
            print("dry run: nothing written (add --apply)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write crank_until")
    return asyncio.run(run(parser.parse_args().apply))


if __name__ == "__main__":
    raise SystemExit(main())
