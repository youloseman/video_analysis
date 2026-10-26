#!/usr/bin/env python
"""Print the written feedback athletes left on their reports. Reads only.

For picking quotes for the pricing page: every row with a message, newest
first, with the rating, the sport and the plan it was written on. The email is
there so the author can be asked before their words go on the site -- in-app
feedback is private by design (models/feedback.py), so a quote needs a yes.

Run it on the box that can reach the database:

    railway ssh --service video-analysis "python scripts/feedback_quotes.py"
    railway ssh --service video-analysis "python scripts/feedback_quotes.py --positive"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.db import SessionLocal  # noqa: E402
from app.models.feedback import Feedback  # noqa: E402


async def run(positive_only: bool) -> int:
    async with SessionLocal() as db:
        q = select(Feedback).where(Feedback.message.is_not(None)).order_by(Feedback.created_at_ms.desc())
        if positive_only:
            q = q.where(Feedback.rating == 1)
        rows = (await db.execute(q)).scalars().all()
    rows = [r for r in rows if (r.message or "").strip()]
    if not rows:
        print("no written feedback")
        return 0
    for r in rows:
        ctx = r.context or {}
        when = datetime.fromtimestamp(r.created_at_ms / 1000, timezone.utc).strftime("%Y-%m-%d")
        verdict = {1: "+1", -1: "-1"}.get(r.rating, " 0")
        print(f"[{verdict}] {when}  {ctx.get('sport') or '?'}/{ctx.get('kind') or '?'}  "
              f"tier={ctx.get('tier') or '?'}  {r.email or 'anonymous'}")
        print("    " + " ".join((r.message or "").split()))
        print()
    print(f"{len(rows)} with text")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--positive", action="store_true", help="only the thumbs-up ones")
    return asyncio.run(run(parser.parse_args().positive))


if __name__ == "__main__":
    raise SystemExit(main())
