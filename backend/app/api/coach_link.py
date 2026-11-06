"""The coach link (model A) and what it opens: ``/me/coach-link``,
``/me/athletes/{id}/analyses``, ``/me/athletes/{id}/analyses/{cid}/note``.

The athlete's own account joined to a roster row. The athlete does it, from
their side, with a consent box; the coach only hands over a code. From then
on the coach reads the athlete's history -- what the athlete's plan shows the
athlete, no more -- and writes notes on it. ``models.coach_link`` says why the
consent is a row of its own; docs/COACH_AREA_PLAN_RU.md section 1 says why
there are two models on one roster.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.athletes import _now_ms, _owned, _require_coach
from app.core.db import get_session
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.athlete import Athlete
from app.models.coach import CoachProfile
from app.models.coach_link import SOURCE_CODE, CoachLink
from app.models.user import User, has_coach

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["coach-link"])

CONSENT_TEXT = (
    "I want this coach to see my analyses, my annotated frames and my uploaded "
    "clips, and to write notes on my reports. I can switch this off at any time."
)


class LinkIn(BaseModel):
    code: str = Field(min_length=4, max_length=64)
    consent: bool = False


async def coach_display(db: AsyncSession, coach: User) -> dict[str, Any]:
    """The coach as the athlete sees them: the profile's name, else the
    account's local part."""
    prof = (
        await db.execute(select(CoachProfile).where(CoachProfile.user_id == coach.id))
    ).scalar_one_or_none()
    d = (prof.data if prof else None) or {}
    return {
        "name": d.get("name") or coach.email.split("@")[0],
        "tagline": d.get("tagline") or "",
    }


async def _live_link_for_user(db: AsyncSession, user_id: int) -> CoachLink | None:
    return (
        await db.execute(
            select(CoachLink).where(
                CoachLink.user_id == user_id, CoachLink.revoked_at_ms.is_(None),
            ).order_by(CoachLink.consented_at_ms.desc())
        )
    ).scalars().first()


async def _link_out(db: AsyncSession, link: CoachLink | None) -> dict[str, Any]:
    if link is None:
        return {"coach": None, "consent_text": CONSENT_TEXT}
    athlete = (
        await db.execute(select(Athlete).where(Athlete.id == link.athlete_id))
    ).scalar_one_or_none()
    coach = (
        await db.execute(select(User).where(User.id == athlete.coach_user_id))
    ).scalar_one_or_none() if athlete else None
    if athlete is None or coach is None:
        return {"coach": None, "consent_text": CONSENT_TEXT}
    return {
        "coach": {
            **(await coach_display(db, coach)),
            "since_ms": link.consented_at_ms,
            "athlete_name": athlete.name,
        },
        "consent_text": CONSENT_TEXT,
    }


@router.get("/coach-link")
async def get_coach_link(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Who, if anyone, this account has let in: the athlete's side."""
    return await _link_out(db, await _live_link_for_user(db, user.id))


@router.post("/coach-link", status_code=status.HTTP_201_CREATED)
async def create_coach_link(
    body: LinkIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The athlete enters the coach's code and ticks the box.

    One coach at a time: a second code is refused until the first link is
    taken off, so "who can see my clips" is never a list the athlete has to
    remember. Linking to one's own roster is refused too -- a coach's own
    clips are already theirs.
    """
    if not body.consent:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tick the box to let your coach see your analyses.",
        )
    code = body.code.strip()
    athlete = (
        await db.execute(
            select(Athlete).where(Athlete.upload_token == code, Athlete.archived.is_(False))
        )
    ).scalar_one_or_none()
    if athlete is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="That code doesn't match any coach's athlete. Ask your coach for the code on your card.",
        )
    if athlete.coach_user_id == user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="That's your own roster.")
    coach = (await db.execute(select(User).where(User.id == athlete.coach_user_id))).scalar_one_or_none()
    if coach is None or not has_coach(coach.tier):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That coach's plan has ended.")
    live = await _live_link_for_user(db, user.id)
    if live is not None and live.athlete_id == athlete.id:
        return await _link_out(db, live)
    if live is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You're already linked to a coach. Unlink them first.",
        )
    if athlete.linked_user_id is not None and athlete.linked_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That code is already in use by another account. Ask your coach for a fresh one.",
        )
    link = CoachLink(
        athlete_id=athlete.id, user_id=user.id, consented_at_ms=_now_ms(), source=SOURCE_CODE,
    )
    db.add(link)
    athlete.linked_user_id = user.id
    await db.commit()
    logger.info("COACH_LINKED", athlete_id=athlete.id, user_id=user.id, coach_user_id=coach.id)
    return await _link_out(db, link)


@router.delete("/coach-link", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_coach_link(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    """The athlete takes the access back. Immediate; notes already written stay."""
    live = await _live_link_for_user(db, user.id)
    if live is None:
        return
    live.revoked_at_ms = _now_ms()
    await db.execute(
        update(Athlete)
        .where(Athlete.id == live.athlete_id, Athlete.linked_user_id == user.id)
        .values(linked_user_id=None)
    )
    await db.commit()
    logger.info("COACH_UNLINKED", athlete_id=live.athlete_id, user_id=user.id)


async def linked_owner(db: AsyncSession, coach: User, athlete_id: int) -> User | None:
    """The athlete's own account behind roster row ``athlete_id``, if the row
    is the coach's and the link is live. None for a model-B athlete (the
    coach's own rows) and for anything that is not theirs."""
    athlete = (
        await db.execute(
            select(Athlete).where(Athlete.id == athlete_id, Athlete.coach_user_id == coach.id)
        )
    ).scalar_one_or_none()
    if athlete is None or athlete.linked_user_id is None:
        return None
    live = (
        await db.execute(
            select(CoachLink.id).where(
                CoachLink.athlete_id == athlete.id,
                CoachLink.user_id == athlete.linked_user_id,
                CoachLink.revoked_at_ms.is_(None),
            )
        )
    ).scalar_one_or_none()
    if live is None:
        return None
    return (
        await db.execute(select(User).where(User.id == athlete.linked_user_id))
    ).scalar_one_or_none()


async def reader_owner(db: AsyncSession, user: User, athlete_id: int | None) -> User:
    """Whose rows a read of ``/me/analyses/...`` should hit.

    No athlete named, or a roster row that is the coach's own (model B):
    the caller's. A linked athlete (model A): the athlete's account, and the
    access level is theirs too. Anything else is a 404 -- never a hint that
    the id exists.
    """
    if athlete_id is None:
        return user
    athlete = (
        await db.execute(
            select(Athlete).where(Athlete.id == athlete_id, Athlete.coach_user_id == user.id)
        )
    ).scalar_one_or_none()
    if athlete is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    owner = await linked_owner(db, user, athlete_id)
    return owner or user


@router.get("/athletes/{athlete_id}/analyses")
async def athlete_analyses(
    athlete_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    """One athlete's history, for the coach.

    A linked athlete's own rows, gated by the athlete's plan, each tagged
    ``linkedAthleteId`` so the client fetches frames and results with
    ``?athlete_id=``; a model-B athlete's rows are the coach's own, filtered.
    """
    from app.api import me as me_api

    _require_coach(user)
    row = await _owned(db, user, athlete_id)
    owner = await linked_owner(db, user, athlete_id)
    if owner is None:
        return await me_api._list_rows(db, user, athlete_id=row.id)
    out = await me_api._list_rows(db, owner)
    for e in out:
        e["athleteId"] = row.id
        e["linkedAthleteId"] = row.id
    return out


class NoteIn(BaseModel):
    note: str = Field(default="", max_length=4000)


@router.put("/athletes/{athlete_id}/analyses/{client_id}/note")
async def write_note(
    athlete_id: int,
    client_id: str,
    body: NoteIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The coach's note on an athlete's report, under the coach's name.

    Written onto the entry the way delivered Coach Notes are (billing
    ``_write_notes_to_entry``): the same fields, so the athlete's read-only
    note block and the export pick it up unchanged. A bought note on the
    entry is never overwritten.
    """
    _require_coach(user)
    row_a = await _owned(db, user, athlete_id)
    owner = (await linked_owner(db, user, athlete_id)) or user
    q = select(Analysis).where(Analysis.user_id == owner.id, Analysis.client_id == client_id)
    if owner.id == user.id:
        q = q.where(Analysis.athlete_id == row_a.id)
    row = (await db.execute(q)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    data = dict(row.data or {})
    if data.get("coachOrderId"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This report carries bought Coach Notes; they stay as written.",
        )
    note = body.note.strip()
    who = await coach_display(db, user)
    had = bool(data.get("coachNote"))
    data["coachNote"] = note
    data["coachName"] = who["name"]
    data["coachBio"] = who["tagline"]
    row.data = data   # reassigned, not mutated: JSON columns don't track edits
    await db.commit()
    if owner.id != user.id and note and not had:
        _mail_note(owner, who["name"], row)
    logger.info("COACH_NOTE_WRITTEN", coach_user_id=user.id, athlete_id=athlete_id,
                analysis=client_id, linked=owner.id != user.id, chars=len(note))
    return {"coachNote": note, "coachName": who["name"], "coachBio": who["tagline"]}


def _mail_note(athlete: User, coach_name: str, row: Analysis) -> None:
    """Tell a linked athlete their coach wrote on a report. A service mail:
    it follows from something they asked for, and names no price."""
    from app.core.config import settings
    from app.services import notify

    base = settings.public_base_url or "https://getflapp.com"
    noun = "ride" if row.sport == "bike" else "run"
    subject = f"{coach_name} left a note on your {noun} report"
    url = f"{base}/app#history"
    text = (
        f"{coach_name} read your {noun} analysis on Flapp and left a note on the report.\n\n"
        f"Open it: {url}\n\n"
        "You linked this coach yourself; you can unlink them any time from Account > My coach.\n"
    )
    esc = notify._esc_html
    html = (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:16px;'
        'line-height:1.6;color:#14294B;max-width:520px">'
        '<p style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#2F6DE0;'
        'margin:0 0 6px">Flapp</p>'
        f'<h1 style="font-size:22px;margin:0 0 8px">{esc(coach_name)} left a note</h1>'
        f'<p style="margin:0 0 22px;color:#5A6478">On your {esc(noun)} report.</p>'
        f'<p style="margin:0 0 24px"><a href="{esc(url)}" style="background:#2F6DE0;color:#fff;'
        'text-decoration:none;padding:12px 20px;border-radius:8px;display:inline-block;'
        'font-weight:600">Open the report</a></p>'
        '<p style="font-size:12px;color:#8A94A3;margin:0">You linked this coach yourself; '
        'unlink any time from Account &rarr; My coach.</p></div>'
    )
    notify.send_email(athlete.email, subject, text, html)
