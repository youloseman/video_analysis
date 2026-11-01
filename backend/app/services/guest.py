"""Coach orders sent from the home page, without signing up first.

Artur, 2026-11-01: Coach Notes and an Expert Review are one-off purchases, and
a password in front of a $12 order is a step people leave at. So the home page
takes the clip, the email and the payment in one go. The clip is stored as it
comes -- no quality check before paying: if the automatic analysis cannot read
it, the coach still looks; if the coach cannot either, the athlete is asked to
film again or refunded.

What it does NOT do is skip the account. One is made from the email (no
password -- core/security.is_guest), because that is where the coach's answer
is delivered, where the History, the player and the PDF already live, and what
turns a one-off buyer into somebody who can open the app afterwards. After
paying they are emailed a link that sets a password and signs them in.

An email that already has a real account is refused: attaching a clip to it
without a password would let anyone put footage into anyone's account.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any

import structlog
from fastapi import HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_reset_token, guest_password_hash, is_guest
from app.models.analysis import Analysis
from app.models.user import TIER_STARTER, User
from app.services import notify
from app.services.expert_review import PLAN_EXPERT, PLAN_NOTES
from app.services.keyframe_store import split_keyframe
from app.services.result_gating import gate_free_result

logger = structlog.get_logger()

SPORTS = ("bike", "run", "swim")
PLANS = (PLAN_NOTES, PLAN_EXPERT)
# The "set your password" link rides in the receipt and again in the delivery
# email, which can come five days later.
LINK_TTL = timedelta(days=14)
# Where Stripe sends a guest back: the page they ordered from.
RETURN_TO = "/"
# Uploads per address per hour, paid or not. A clip is up to 200 MB of disk and,
# for bike and run, a slot of analysis before anyone has paid.
ORDERS_PER_IP = 6
ORDERS_WINDOW_S = 60 * 60

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def clean_email(raw: str | None) -> str:
    v = (raw or "").strip().lower()
    if not _EMAIL_RE.match(v) or len(v) > 320:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Enter a valid email address.")
    return v


def check_order(sport: str, plan: str, accept_privacy: bool) -> None:
    """The cheap refusals, before a byte of the clip is stored."""
    if sport not in SPORTS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"sport must be one of {list(SPORTS)}")
    if plan not in PLANS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Choose Coach Notes or an Expert Review.")
    if not accept_privacy:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "Please agree to the Terms and the Privacy Policy to send your clip.",
        )


async def account_for(db: AsyncSession, email: str, marketing: bool) -> User:
    """The guest account behind this email: found, or made now."""
    user = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if user is not None and not is_guest(user):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This email already has a Flapp account. Log in and send the clip from the app "
            "— it takes the same minute.",
        )
    if user is None:
        user = User(
            email=email, password_hash=guest_password_hash(), tier=TIER_STARTER,
            privacy_accepted_at=now, marketing_opt_in=bool(marketing),
            marketing_changed_at=now if marketing else None,
        )
        db.add(user)
        logger.info("GUEST_ACCOUNT_CREATED")
    elif marketing and not user.marketing_opt_in:
        # A yes given on a second order counts; a missing tick never withdraws
        # one given before.
        user.marketing_opt_in, user.marketing_changed_at = True, now
    await db.commit()
    await db.refresh(user)
    return user


def set_password_link(user: User) -> str:
    """Sets a password and signs them in (the app's reset form). Dies once used."""
    from app.api.auth import reset_link

    return reset_link(create_reset_token(user, LINK_TTL))


# --------------------------------------------------------------------------
# A run or bike clip: analysed on the server, filed in the History here
# --------------------------------------------------------------------------
def pending_entry(*, client_id: str, at_ms: int, job_id: str, sport: str,
                  position: str | None, question: str | None) -> dict[str, Any]:
    """The History entry for a clip still being analysed.

    The app files its own entries from the result it was shown. Nobody is
    watching a guest's run, so the server files it, in the shape the app
    gives a teaser -- which is what a starter account is entitled to; paying
    $4 or subscribing later rebuilds it from the stored result, the same way
    it does for any teaser (index.html rehydrateEntry).
    """
    return {
        "id": client_id, "at": at_ms, "kind": "video", "jobId": job_id, "sport": sport,
        "position": position, "locked": True, "lockReason": "teaser", "guest": True,
        "question": question, "pending": True,
    }


def finish_entry(entry: dict[str, Any], job: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """``(entry, keyframe)`` once the analysis has run, or failed.

    A failed run is kept, not deleted: the order stands, and the coach reviews
    the footage by eye (Artur, 2026-11-01). ``analysisFailed`` is what tells
    the coach's brief so.
    """
    entry = {k: v for k, v in entry.items() if k != "pending"}
    result = job.get("result") or {}
    if job.get("status") != "completed" or not result:
        entry["analysisFailed"] = str(job.get("error") or "The analysis could not read this clip.")[:300]
        return entry, None
    teaser = gate_free_result(result)
    locked = teaser.get("locked") or {}
    ssm, qb = result.get("sport_specific_metrics") or {}, result.get("quality") or {}
    entry.update({
        "sport": result.get("sport_type") or entry.get("sport"),
        "position": result.get("cycling_position") or entry.get("position"),
        "cameraSide": result.get("camera_side"),
        "score": teaser.get("technique_score"),
        "grade": teaser.get("letter_grade"),
        "lockReason": locked.get("reason") or "locked",
        "gated": bool(result.get("quality_gate_triggered")),
        "confidence": (ssm.get("analysis_confidence") or {}).get("level") or qb.get("confidence"),
        "warnings": list(result.get("warnings") or qb.get("warnings") or ssm.get("quality_warnings") or [])[:4],
        "keyframe": teaser.get("keyframe_base64") or teaser.get("thumbnail_base64"),
    })
    if locked.get("lock_summary"):
        entry["lockSummary"] = locked["lock_summary"]
    return split_keyframe(entry)


async def file_analysis(user_id: int, client_id: str, job_id: str) -> None:
    """Put a finished guest run into the History row written at upload.

    Queued as a background task right after the analysis itself; FastAPI runs
    a request's background tasks in order, so the job has finished (or
    failed) by the time this starts.
    """
    from app.api.me import _attach_result
    from app.core.db import SessionLocal
    from app.core.jobs import JOBS

    job = JOBS.get(job_id) or {}
    try:
        async with SessionLocal() as db:
            row = (await db.execute(select(Analysis).where(
                Analysis.user_id == user_id, Analysis.client_id == client_id,
            ))).scalar_one_or_none()
            if row is None:
                return
            entry, frame = finish_entry(dict(row.data or {}), job)
            row.data = entry
            row.score = entry.get("score")
            if frame:
                row.keyframe = frame
            _attach_result(row, job_id)
            await db.commit()
        logger.info("GUEST_ANALYSIS_FILED", job_id=job_id, failed=bool(entry.get("analysisFailed")))
    except Exception as e:  # noqa: BLE001 -- the order and the clip stand either way
        logger.warning("GUEST_ANALYSIS_FILE_FAILED", job_id=job_id, err=str(e))


# --------------------------------------------------------------------------
# The receipt: what happens next, and the password link
# --------------------------------------------------------------------------
_WHAT = {
    PLAN_NOTES: ("Coach Notes", "within 72 hours"),
    PLAN_EXPERT: ("an Expert Review", "within five days"),
}


def welcome_email(plan: str, sport: str | None, link: str) -> tuple[str, str, str]:
    what, when = _WHAT.get(plan, _WHAT[PLAN_NOTES])
    days = LINK_TTL.days
    analysed = sport in ("bike", "run")
    subject = "Your clip is with the coach — set a password to follow it"
    paras = [
        f"Thanks — we have your clip and your payment for {what}. A pro triathlete "
        f"watches it and writes back {when}. We'll email you when it's ready.",
        "We made a Flapp account with this email so you can read the answer. "
        f"Set a password with the button below (the link works for {days} days, once).",
        ("In the app you'll find your clip and its automatic analysis under History."
         if analysed else "In the app you'll find your clip under History."),
        "If the clip turns out too hard to read, we'll ask you to film it again or refund you.",
    ]
    text = "\n\n".join(paras[:2]) + f"\n\n{link}\n\n" + "\n\n".join(paras[2:]) + "\n\n— Flapp"
    safe = escape(link, quote=True)
    body = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:15px;'
        'line-height:1.6;color:#14294B;max-width:520px">'
        + "".join(f"<p>{escape(p)}</p>" for p in paras[:2])
        + f'<p><a href="{safe}" style="display:inline-block;padding:12px 20px;background:#2F6DE0;'
        'color:#fff;font-weight:700;text-decoration:none;border-radius:8px">Set my password</a></p>'
        + "".join(f'<p style="color:#555">{escape(p)}</p>' for p in paras[2:])
        + f'<p style="color:#555">If the button does nothing, copy this address into your '
        f'browser:<br><a href="{safe}">{safe}</a></p><p>— Flapp</p></div>'
    )
    return subject, text, body


async def send_welcome(user_id: int, plan: str, sport: str | None) -> bool:
    """After the payment, never before: an unpaid upload gets no email."""
    from app.core.db import SessionLocal

    async with SessionLocal() as db:
        user = await db.get(User, user_id)
        if not is_guest(user):
            return False
        link, email = set_password_link(user), user.email
    subject, text, html = welcome_email(plan, sport, link)
    sent = await run_in_threadpool(notify.send_email, email, subject, text, html)
    logger.info("GUEST_WELCOME_SENT", user_id=user_id, sent=sent)
    return sent
