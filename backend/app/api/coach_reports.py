"""A report handed to an athlete who has no account (model B).

``POST /me/athletes/{id}/analyses/{cid}/send`` mints a read-only link to the
report and, when the coach has their address, emails it. ``/r/{token}`` is the
page: the same branded document as the coach's PDF, no sign-in, for 90 days or
until the coach switches it off. A linked athlete has the report in their own app
already, and the coach's note reaches them there, so they are not sent a link.
"""

from __future__ import annotations

import secrets
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.athletes import _owned, _require_coach
from app.core.db import get_session
from app.core.net import client_ip
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.coach_upload import ReportShare
from app.models.user import User

logger = structlog.get_logger()
router = APIRouter(tags=["coach-reports"])

SHARE_DAYS = 90
DAY_MS = 24 * 3600 * 1000
SENDS_PER_HOUR = 30
VIEWS_PER_IP_HOUR = 60
PDFS_PER_IP_HOUR = 20


def _now_ms() -> int:
    return int(time.time() * 1000)


def _url(token: str) -> str:
    from app.services.notify import base_url

    return f"{base_url()}/r/{token}"


async def _live_share(db: AsyncSession, coach_id: int, client_id: str) -> ReportShare | None:
    return (await db.execute(select(ReportShare).where(
        ReportShare.coach_user_id == coach_id, ReportShare.client_id == client_id,
        ReportShare.revoked_at_ms.is_(None), ReportShare.expires_at_ms > _now_ms(),
    ).order_by(ReportShare.id.desc()))).scalars().first()


async def _coach_row(db: AsyncSession, user: User, athlete_id: int, client_id: str) -> Analysis:
    await _owned(db, user, athlete_id)
    row = (await db.execute(select(Analysis).where(
        Analysis.user_id == user.id, Analysis.client_id == client_id,
        Analysis.athlete_id == athlete_id,
    ))).scalar_one_or_none()
    if row is None:
        raise HTTPException(404, "Analysis not found.")
    return row


class CardEdit(BaseModel):
    key: str = Field(max_length=200)
    hidden: bool = False
    title: str | None = Field(default=None, max_length=300)
    why: str | None = Field(default=None, max_length=2000)
    do: str | None = Field(default=None, max_length=3000)


class ReviewBody(BaseModel):
    # None: Flapp's text. "": none at all.
    coachText: str | None = Field(default=None, max_length=8000)
    cards: list[CardEdit] = Field(default_factory=list, max_length=12)


async def _review_state(db: AsyncSession, user: User, row: Analysis) -> dict[str, Any]:
    """What the review screen edits: Flapp's plan cards and coach text, with the
    coach's edits laid over them. The numbers are not here: they stay as measured."""
    from app.api.me import _ensure_coaching, _refresh_fit_plan, saved_report_meta
    from app.services.report_pdf import render_report_html

    await _refresh_fit_plan(db, row)
    await _ensure_coaching(db, row)
    result, meta = await saved_report_meta(db, reader=user, owner=user, row=row, athlete_id=row.athlete_id)
    flapp: dict[str, Any] = {}
    render_report_html(result, {**meta, "edits": {}}, out=flapp)
    edits = result.get("coach_edits") or {}
    base = {c["key"]: c for c in flapp.get("cards") or []}
    cards = []
    for c in edits.get("cards") or []:
        b = base.pop(c.get("key"), None)
        if b is None:
            continue
        cards.append({**b, "orig": {k: b[k] for k in ("title", "why", "do")}, "hidden": bool(c.get("hidden")),
                      **{k: c[k] for k in ("title", "why", "do") if isinstance(c.get(k), str)}})
    cards += [{**b, "orig": {k: b[k] for k in ("title", "why", "do")}, "hidden": False} for b in base.values()]
    text = edits.get("coachText")
    return {
        "coachText": text if isinstance(text, str) else flapp.get("coach") or "",
        "coachTextFlapp": flapp.get("coach") or "",
        "edited": bool(edits.get("cards") or isinstance(text, str)),
        "cards": cards,
        "reviewedAt": edits.get("reviewedAt"),
        "reviewedBy": edits.get("reviewedBy"),
    }


def _set_edits(row: Analysis, edits: dict[str, Any] | None) -> None:
    res = dict(row.result or {})
    if edits:
        res["coach_edits"] = edits
    else:
        res.pop("coach_edits", None)
    row.result = res   # reassigned, not mutated: JSON columns don't track edits


@router.get("/me/athletes/{athlete_id}/analyses/{client_id}/review")
async def get_review(
    athlete_id: int, client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    row = await _coach_row(db, user, athlete_id, client_id)
    if not row.result:
        raise HTTPException(409, "A photo analysis has no report to review.")
    return await _review_state(db, user, row)


@router.put("/me/athletes/{athlete_id}/analyses/{client_id}/review")
async def put_review(
    athlete_id: int, client_id: str, body: ReviewBody,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Save the coach's words over Flapp's. Saving after sending keeps the report
    reviewed: the link shows what the coach left there."""
    _require_coach(user)
    row = await _coach_row(db, user, athlete_id, client_id)
    if not row.result:
        raise HTTPException(409, "A photo analysis has no report to review.")
    edits = dict((row.result or {}).get("coach_edits") or {})
    edits["coachText"] = body.coachText
    edits["cards"] = [c.model_dump() for c in body.cards]
    _set_edits(row, edits)
    await db.commit()
    logger.info("REPORT_REVIEW_SAVED", coach_user_id=user.id, athlete_id=athlete_id,
                cards=len(body.cards), hidden=sum(c.hidden for c in body.cards),
                text=None if body.coachText is None else len(body.coachText))
    return await _review_state(db, user, row)


@router.delete("/me/athletes/{athlete_id}/analyses/{client_id}/review")
async def reset_review(
    athlete_id: int, client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Back to Flapp's words. An approval stays: it was given."""
    _require_coach(user)
    row = await _coach_row(db, user, athlete_id, client_id)
    edits = dict((row.result or {}).get("coach_edits") or {})
    kept = {k: edits[k] for k in ("reviewedAt", "reviewedBy") if k in edits}
    _set_edits(row, kept)
    await db.commit()
    return await _review_state(db, user, row)


@router.get("/me/athletes/{athlete_id}/analyses/{client_id}/share")
async def get_share(
    athlete_id: int, client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    await _coach_row(db, user, athlete_id, client_id)
    share = await _live_share(db, user.id, client_id)
    if share is None:
        return {"url": None}
    return {"url": _url(share.token), "expires_at_ms": share.expires_at_ms}


@router.post("/me/athletes/{athlete_id}/analyses/{client_id}/send")
async def send_report(
    athlete_id: int, client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Make the link if there is none, and mail it if the athlete has an address."""
    from app.api.auth import _throttle
    from app.api.coach_link import coach_display
    from app.api.me import _ensure_coaching

    _require_coach(user)
    _throttle(f"report-send:{user.id}", SENDS_PER_HOUR, 3600)
    athlete = await _owned(db, user, athlete_id)
    row = await _coach_row(db, user, athlete_id, client_id)
    if not row.result:
        raise HTTPException(409, "A photo analysis has no report to send. Send the frame and numbers from the report itself.")
    # Written now, while a coach is asking: an anonymous page view must not be
    # what triggers a billed model call.
    await _ensure_coaching(db, row)
    # Sending is approving: the report leaves the draft with the coach's name on it.
    who = await coach_display(db, user)
    edits = dict((row.result or {}).get("coach_edits") or {})
    if not edits.get("reviewedAt"):
        edits.update(reviewedAt=_now_ms(), reviewedBy=who["name"] or "")
        _set_edits(row, edits)
        await db.commit()
    share = await _live_share(db, user.id, client_id)
    if share is None:
        now = _now_ms()
        share = ReportShare(
            token=secrets.token_urlsafe(24), coach_user_id=user.id, athlete_id=athlete_id,
            client_id=client_id, created_at_ms=now, expires_at_ms=now + SHARE_DAYS * DAY_MS,
        )
        db.add(share)
        await db.commit()
    url = _url(share.token)
    mailed = bool(athlete.email) and _mail(athlete.email, athlete.name, who["name"], row.sport, url)
    logger.info("REPORT_SENT", coach_user_id=user.id, athlete_id=athlete_id, mailed=mailed)
    return {"url": url, "expires_at_ms": share.expires_at_ms, "mailed": mailed, "email": athlete.email}


@router.delete("/me/athletes/{athlete_id}/analyses/{client_id}/share", status_code=status.HTTP_204_NO_CONTENT)
async def stop_sharing(
    athlete_id: int, client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    _require_coach(user)
    await _coach_row(db, user, athlete_id, client_id)
    share = await _live_share(db, user.id, client_id)
    if share is not None:
        share.revoked_at_ms = _now_ms()
        await db.commit()


def _mail(to: str, athlete: str, coach: str, sport: str | None, url: str) -> bool:
    """From the coach, about the athlete's own clip. It names no price."""
    from app.api.coach_link import in_sentence
    from app.services import notify

    esc = notify._esc_html
    noun = "ride" if sport == "bike" else "run"
    mid = in_sentence(coach)
    subject = f"{coach} sent you your {noun} report"
    text = (
        f"Hi {athlete},\n\n{coach} sent you the report on your {noun}.\n\nOpen it: {url}\n\n"
        f"The link works for {SHARE_DAYS} days. Measured by Flapp, sent because {mid} uploaded your clip; "
        "write to support@getflapp.com if that is wrong.\n"
    )
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:16px;'
        'line-height:1.6;color:#14294B;max-width:520px">'
        f'<h1 style="font-size:22px;margin:0 0 8px">{esc(coach)} sent you your {esc(noun)} report</h1>'
        f'<p style="margin:0 0 22px;color:#5A6478">Hi {esc(athlete)}. It opens without an account.</p>'
        f'<p style="margin:0 0 24px"><a href="{esc(url)}" style="background:#2F6DE0;color:#fff;text-decoration:none;'
        'padding:12px 20px;border-radius:8px;display:inline-block;font-weight:600">Open the report</a></p>'
        f'<p style="font-size:12px;color:#8A94A3;margin:0">The link works for {SHARE_DAYS} days. Measured by Flapp, '
        f'sent because {esc(mid)} uploaded your clip. If that is wrong, write to support@getflapp.com.</p></div>'
    )
    return notify.send_email(to, subject, text, html)


_GONE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="robots" content="noindex">'
    '<meta name="viewport" content="width=device-width, initial-scale=1"><title>This link has ended</title></head>'
    '<body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:480px;margin:48px auto;padding:0 16px;color:#14294B">'
    "<h1 style='font-size:22px'>This link has ended</h1><p>Ask your coach to send the report again.</p></body></html>"
)


@router.get("/r/{token}", include_in_schema=False)
async def shared_report(
    token: str, request: Request, db: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    from app.api.auth import _throttle
    from app.api.me import _refresh_fit_plan, render_saved_report

    _throttle(f"report-view:{client_ip(request)}", VIEWS_PER_IP_HOUR, 3600)
    headers = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex", "Referrer-Policy": "no-referrer"}
    share = (await db.execute(select(ReportShare).where(ReportShare.token == token))).scalar_one_or_none()
    if share is None or share.revoked_at_ms or share.expires_at_ms <= _now_ms():
        return HTMLResponse(_GONE, status_code=404, headers=headers)
    coach = await db.get(User, share.coach_user_id)
    row = (await db.execute(select(Analysis).where(
        Analysis.user_id == share.coach_user_id, Analysis.client_id == share.client_id,
    ))).scalar_one_or_none()
    if coach is None or row is None or not row.result:
        return HTMLResponse(_GONE, status_code=404, headers=headers)
    await _refresh_fit_plan(db, row)
    html = await render_saved_report(db, reader=coach, owner=coach, row=row, athlete_id=share.athlete_id,
                                     pdf_url=f"/r/{token}/pdf")
    logger.info("REPORT_VIEWED", share_id=share.id)
    return HTMLResponse(html, headers=headers)


@router.get("/r/{token}/pdf", include_in_schema=False)
async def shared_report_pdf(
    token: str, request: Request, db: AsyncSession = Depends(get_session),
) -> Any:
    """The athlete's report as a file, from the link their coach sent."""
    from app.api.auth import _throttle
    from app.api.me import _refresh_fit_plan, saved_report_pdf

    _throttle(f"report-pdf:{client_ip(request)}", PDFS_PER_IP_HOUR, 3600)
    headers = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex", "Referrer-Policy": "no-referrer"}
    share = (await db.execute(select(ReportShare).where(ReportShare.token == token))).scalar_one_or_none()
    if share is None or share.revoked_at_ms or share.expires_at_ms <= _now_ms():
        return HTMLResponse(_GONE, status_code=404, headers=headers)
    coach = await db.get(User, share.coach_user_id)
    row = (await db.execute(select(Analysis).where(
        Analysis.user_id == share.coach_user_id, Analysis.client_id == share.client_id,
    ))).scalar_one_or_none()
    if coach is None or row is None or not row.result:
        return HTMLResponse(_GONE, status_code=404, headers=headers)
    await _refresh_fit_plan(db, row)
    resp = await saved_report_pdf(db, reader=coach, owner=coach, row=row, athlete_id=share.athlete_id)
    resp.headers["X-Robots-Tag"] = "noindex"
    logger.info("REPORT_PDF_VIEWED", share_id=share.id)
    return resp
