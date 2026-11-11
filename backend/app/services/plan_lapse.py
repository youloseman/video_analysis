"""A paid plan that ends: thirty more days for the clips, and two emails.

Retention is read from the account's tier at sweep time (services/retention.py),
which is right for a plan that is running and harsh for one that just stopped:
a card that failed overnight would take a year of footage with it at the next
sweep, with nothing said. So (Artur, 2026-11-11):

* the drop from a paid plan to the free one is written down (``PlanLapse``);
* for ``LAPSE_GRACE_DAYS`` the clips keep the paid plan's period;
* the owner is told twice -- when it happens, and ``REMINDER_DAYS`` before the
  clips go -- and only if there are clips the free plan would delete.

Nothing here touches files. The sweep keeps what retention says to keep; this
module only decides who is told what.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis
from app.models.user import TIER_STARTER, PlanLapse, User
from app.services import notify
from app.services.retention import (
    DAY_MS,
    LAPSE_GRACE_DAYS,
    MAX_RETENTION_DAYS,
    PAID_TIERS,
    RETENTION_DAYS,
    retention_days,
)

logger = structlog.get_logger()

REMINDER_DAYS = 5
PLAN_NAMES = {"enthusiast": "Enthusiast", "full": "Full", "coach": "Coach"}

SENT_NONE, SENT_NOTICE, SENT_ALL = 0, 1, 2


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def grace_ends_ms(lapse: PlanLapse) -> int:
    return lapse.lapsed_at_ms + LAPSE_GRACE_DAYS * DAY_MS


async def upsert_lapse(db: AsyncSession, user_id: int, from_tier: str, now_ms: int) -> None:
    """Start (or restart) the grace for one account. The caller commits."""
    row = await db.get(PlanLapse, user_id)
    if row is None:
        db.add(PlanLapse(user_id=user_id, from_tier=from_tier,
                         lapsed_at_ms=now_ms, emails_sent=SENT_NONE))
    else:
        row.from_tier, row.lapsed_at_ms, row.emails_sent = from_tier, now_ms, SENT_NONE
    logger.info("PLAN_LAPSED", user_id=user_id, from_tier=from_tier)


async def record_lapse(
    db: AsyncSession, user: User, from_tier: str | None, now_ms: int | None = None,
) -> None:
    """After a tier change: a paid plan that became the free one starts its grace.

    Anything else -- an upgrade, an admin, a coach grant that kept the plan, a
    second webhook for a lapse already recorded (``from_tier`` is then the free
    plan) -- writes nothing.
    """
    if from_tier not in PAID_TIERS or user.tier != TIER_STARTER:
        return
    await upsert_lapse(db, user.id, from_tier, now_ms if now_ms is not None else _now_ms())


async def clips_at_stake(db: AsyncSession, lapse: PlanLapse, now_ms: int) -> int:
    """Stored clips that are kept now and that the free plan deletes when the
    grace ends."""
    end = grace_ends_ms(lapse)
    free_ms = RETENTION_DAYS[TIER_STARTER] * DAY_MS
    rows = await db.execute(
        select(Analysis.created_at_ms).where(
            Analysis.user_id == lapse.user_id,
            Analysis.job_id.is_not(None),
            Analysis.created_at_ms >= now_ms - MAX_RETENTION_DAYS * DAY_MS,
        )
    )
    n = 0
    for (created,) in rows:
        created = created or 0
        kept_now = now_ms - created <= retention_days(lapse.from_tier, created) * DAY_MS
        if kept_now and end - created > free_ms:
            n += 1
    return n


def _date(ms: int) -> str:
    d = datetime.fromtimestamp(ms / 1000, timezone.utc)
    return f"{d.day} {d:%B %Y}"


def lapse_email(kind: str, from_tier: str, clips: int, ends_ms: int) -> tuple[str, str, str]:
    """(subject, text, html). ``kind`` is "notice" or "reminder"."""
    plan = PLAN_NAMES.get(from_tier, "paid")
    when = _date(ends_ms)
    noun = "clip" if clips == 1 else "clips"
    base = notify.base_url()
    pricing, history = f"{base}/app#pricing", f"{base}/app#history"
    if kind == "reminder":
        subject = f"Your Flapp clips will be deleted on {when}"
        lead = (f"In {REMINDER_DAYS} days, on {when}, we delete the {clips} uploaded "
                f"{noun} your {plan} plan was keeping.")
    else:
        subject = f"Your Flapp plan has ended. Your clips are kept until {when}"
        lead = (f"Your {plan} plan on Flapp has ended. The {clips} uploaded {noun} "
                f"behind your saved analyses are kept until {when}.")
    after = ("After that the free plan's 7 days apply and older clips are deleted for good. "
             "Your saved analyses stay: scores, frames and notes.")
    text = (
        f"{lead}\n\n{after}\n\n"
        f"To keep the clips, restart your plan before {when}:\n{pricing}\n\n"
        f"Or download the overlay videos you want to keep, from History:\n{history}\n"
    )
    esc = notify._esc_html
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:16px;'
        'line-height:1.6;color:#14294B;max-width:520px">'
        '<p style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#2F6DE0;'
        'margin:0 0 6px">Flapp</p>'
        f'<p style="margin:0 0 12px">{esc(lead)}</p>'
        f'<p style="margin:0 0 22px;color:#5A6478">{esc(after)}</p>'
        f'<p style="margin:0 0 18px"><a href="{esc(pricing)}" style="background:#2F6DE0;color:#fff;'
        'text-decoration:none;padding:12px 20px;border-radius:8px;display:inline-block;'
        f'font-weight:600">Keep my clips</a></p>'
        f'<p style="font-size:13px;color:#5A6478;margin:0">Or download the overlay videos you want '
        f'from <a href="{esc(history)}" style="color:#2F6DE0">History</a>.</p></div>'
    )
    return subject, text, html


async def send_lapse_emails(db: AsyncSession, now_ms: int | None = None) -> int:
    """Send what is due. Returns how many went. Safe to run every few minutes."""
    now = now_ms if now_ms is not None else _now_ms()
    rows = (
        await db.execute(
            select(PlanLapse, User)
            .join(User, User.id == PlanLapse.user_id)
            .where(PlanLapse.emails_sent < SENT_ALL)
        )
    ).all()
    sent = 0
    for lapse, user in rows:
        end = grace_ends_ms(lapse)
        # Paid again, or too late to be useful: nothing more to say.
        if user.tier != TIER_STARTER or now >= end:
            lapse.emails_sent = SENT_ALL
            continue
        clips = await clips_at_stake(db, lapse, now)
        if clips == 0:
            lapse.emails_sent = SENT_ALL
            continue
        remind_at = end - REMINDER_DAYS * DAY_MS
        if lapse.emails_sent == SENT_NONE:
            # A notice that is itself late covers the reminder too.
            kind, stage = "notice", SENT_ALL if now >= remind_at else SENT_NOTICE
        elif now >= remind_at:
            kind, stage = "reminder", SENT_ALL
        else:
            continue
        subject, text, html = lapse_email(kind, lapse.from_tier, clips, end)
        if notify.send_email(user.email, subject, text, html):
            lapse.emails_sent = stage
            sent += 1
            logger.info("PLAN_LAPSE_EMAIL", user_id=user.id, kind=kind, clips=clips)
    await db.commit()
    return sent
