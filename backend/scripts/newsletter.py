#!/usr/bin/env python3
"""Send one product-news issue to every registered athlete who has not opted out.

    # who would get it (nothing is sent)
    python scripts/newsletter.py 2026-10-kona-crank --list
    # one copy to one address, to look at it in a real mail client
    python scripts/newsletter.py 2026-10-kona-crank --to you@example.com
    # everyone; safe to re-run -- the log skips anyone already sent
    python scripts/newsletter.py 2026-10-kona-crank --send

Runs where the database and the mail key are (the production container).
Needs NEWSLETTER_ADDRESS (the sender's postal address, required in every
commercial email under CASL) and PUBLIC_BASE_URL. The send log lives next
to the uploads, one file per issue, so a crash halfway resumes cleanly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.db import SessionLocal  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services import newsletter as NL  # noqa: E402
from app.services.notify import send_email  # noqa: E402

PAUSE_S = 0.6  # ~2 a second: well inside Resend's default rate limit


def _unsub_url(user_id: int) -> str:
    from app.main import unsubscribe_token

    base = (settings.public_base_url or "https://getflapp.com").rstrip("/")
    return f"{base}/unsubscribe?u={user_id}&t={unsubscribe_token(user_id, 'news')}&list=news"


async def _recipients() -> list[User]:
    async with SessionLocal() as db:
        users = (await db.execute(select(User).order_by(User.id))).scalars().all()
    return [u for u in users if NL.is_recipient(u)]


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("issue")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print who would get it")
    g.add_argument("--to", help="send one copy to this address only")
    g.add_argument("--send", action="store_true", help="send to every recipient")
    a = ap.parse_args()

    issue = NL.load_issue(a.issue)
    address = os.environ.get("NEWSLETTER_ADDRESS", "")
    people = await _recipients()

    if a.list:
        for u in people:
            print(u.id, u.email, u.tier)
        print(f"{len(people)} recipients")
        return 0

    if a.to:
        subject, text, html = NL.render(issue, _unsub_url(0), address)
        ok = send_email(a.to, "[TEST] " + subject, text, html)
        print("sent" if ok else "NOT sent", a.to)
        return 0 if ok else 1

    log_path = Path(settings.uploads_dir) / f"newsletter-{issue['slug']}.json"
    sent: dict[str, float] = json.loads(log_path.read_text()) if log_path.is_file() else {}
    todo = [u for u in people if str(u.id) not in sent]
    print(f"{len(people)} recipients, {len(sent)} already sent, {len(todo)} to go")
    NL.render(issue, "x", address)  # refuses up front if the issue cannot go out
    failed = 0
    for u in todo:
        subject, text, html = NL.render(issue, _unsub_url(u.id), address)
        if send_email(u.email, subject, text, html):
            sent[str(u.id)] = time.time()
            log_path.write_text(json.dumps(sent))
        else:
            failed += 1
        time.sleep(PAUSE_S)
    print(f"done: {len(sent)} sent in total, {failed} failed this run")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
