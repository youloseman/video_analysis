"""How clips reach a coach without a chat thread.

``/me/coach-uploads``: the coach's batch, one file per request so a phone on a
bad connection can retry the one that failed. ``/a/{token}``: the athlete's
intake link, no account needed. Both feed the lane (services/coach_lane.py), which
files the report under the athlete.

Photos do not come through here. privacy.html promises a photo is never written
to disk, and a queue needs a file to come back to, so the batch screen sends each
photo to ``/analyze-photo`` at once and saves the entry as the app always does.
"""

from __future__ import annotations

import re
import secrets
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.athletes import _require_coach, owned_athlete_id
from app.core.db import get_session
from app.core.net import client_ip
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.athlete import Athlete
from app.models.coach import CoachProfile
from app.models.coach_upload import (
    MAX_FILENAME_CHARS,
    OPEN_STATUSES,
    SOURCE_BATCH,
    SOURCE_INTAKE,
    STATUS_DONE,
    STATUS_NO_GO,
    STATUS_QUEUED,
    CoachUpload,
)
from app.models.user import User, has_coach
from app.services import coach_lane
from app.services.intake_page import render_intake_page

logger = structlog.get_logger()
router = APIRouter(tags=["coach-uploads"])

_BATCH_ID = re.compile(r"^[a-z0-9]{6,24}$")
INTAKE_PER_IP_HOUR = 12
INTAKE_PER_ATHLETE_DAY = 10
DAY_MS = 24 * 3600 * 1000
# What the coach sees when they open the batch screen with no batch in hand.
RECENT_DAYS = 14
RECENT_LIMIT = 60
# 1 January 2015. Older than the app's every clip, and it stops a zero or a
# seconds-for-milliseconds slip passing for a date.
FILMED_FLOOR_MS = 1_420_070_400_000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _clean_sport(sport: str | None, *, auto: bool = False) -> str:
    """``auto`` lets a coach's batch say "read it from the clip" (coach_lane.AUTO)."""
    if sport not in ("run", "bike") and not (auto and sport == coach_lane.AUTO):
        raise HTTPException(400, "sport must be 'run' or 'bike'")
    return sport  # type: ignore[return-value]


def _clean_position(sport: str, position: str | None) -> str | None:
    from app.main import DEFAULT_BIKE_POSITION, VALID_POSITIONS

    if sport not in ("bike", coach_lane.AUTO):
        return None
    if position == coach_lane.AUTO or (sport == coach_lane.AUTO and not position):
        return coach_lane.AUTO
    position = position or DEFAULT_BIKE_POSITION
    if position not in VALID_POSITIONS:
        raise HTTPException(400, f"invalid position; valid: {sorted(VALID_POSITIONS)}")
    return position


async def _accept(
    db: AsyncSession, coach: User, video: UploadFile, *, sport: str, position: str | None,
    camera_side: str | None, athlete_id: int | None, profile_id: int | None,
    batch_id: str | None, source: str, extra: dict[str, Any] | None = None,
    filmed_at_ms: Any = None,
) -> CoachUpload:
    """Everything both doors share: the checks, the clip, the rows, the queue."""
    from app.core.config import settings
    from app.main import ALLOWED_SUFFIXES, _store_clip
    from app.services.guest import pending_entry
    from app.services.usage_limits import check_quota

    if not settings.model_path.exists():
        raise HTTPException(503, "pose model not installed on the server")
    suffix = Path(video.filename or "").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"unsupported file type '{suffix}'; allowed: {sorted(ALLOWED_SUFFIXES)}")
    backlog = int(await db.scalar(
        select(func.count(CoachUpload.id)).where(CoachUpload.status.in_(OPEN_STATUSES))
    ) or 0)
    if backlog >= coach_lane.MAX_LANE_BACKLOG:
        raise HTTPException(503, "The queue is full right now. Please try again in a little while.",
                            headers={"Retry-After": "600"})
    # What is already in line counts: it will spend quota when its turn comes.
    _allowed, used, limit, _window = await check_quota(db, coach)
    mine_open = int(await db.scalar(select(func.count(CoachUpload.id)).where(
        CoachUpload.coach_user_id == coach.id, CoachUpload.status.in_(OPEN_STATUSES),
    )) or 0)
    if used + mine_open >= limit:
        raise HTTPException(429, f"This plan's {limit} analyses are used up for now.")
    job_id, _path, size = await _store_clip(video, suffix)
    now = _now_ms()
    client_id = f"h{now}{secrets.token_hex(2)}"
    entry = pending_entry(client_id=client_id, at_ms=now, job_id=job_id, sport=sport,
                          position=position, question=None)
    entry.pop("guest", None)
    # Held out of History until it is a report (me._list_rows).
    entry["coachHold"] = True
    entry["athleteId"] = athlete_id
    if profile_id:
        entry["profileId"] = profile_id
    entry.update(extra or {})
    # Scanned and filed by the lane; the column stays NULL until then, so the
    # squad's numbers do not count a clip that may yet turn out a NO-GO.
    if video.filename:
        entry["fileName"] = video.filename[:MAX_FILENAME_CHARS]
    # Kept apart from ``at``, which is when the row was made: retention counts
    # from there, and a clip filmed in May must not expire the day it arrives.
    filmed = _filmed_at(filmed_at_ms, now)
    if filmed:
        entry["filmedAt"] = filmed
    db.add(Analysis(user_id=coach.id, client_id=client_id, created_at_ms=now, job_id=job_id,
                    sport=sport, kind="video", score=None, data=entry))
    up = CoachUpload(
        coach_user_id=coach.id, athlete_id=athlete_id, profile_id=profile_id,
        batch_id=batch_id, source=source, client_id=client_id, job_id=job_id,
        filename=(video.filename or "")[:MAX_FILENAME_CHARS] or None,
        sport=sport, bike_position=position, camera_side=camera_side,
        status=STATUS_QUEUED, created_at_ms=now,
    )
    db.add(up)
    await db.commit()
    await db.refresh(up)
    coach_lane.enqueue(up.id)
    logger.info("COACH_UPLOAD", upload_id=up.id, source=source, bytes=size, coach_user_id=coach.id)
    return up


def _filmed_at(raw: Any, now: int) -> int | None:
    """The file's own date, when it can be one: phones and copies can report
    anything, and a date nobody could have filmed on is worse than none."""
    return raw if isinstance(raw, int) and FILMED_FLOOR_MS <= raw <= now else None


def _item(up: CoachUpload, names: dict[int, str]) -> dict[str, Any]:
    return {
        "id": up.id, "batch_id": up.batch_id, "source": up.source,
        "client_id": up.client_id, "filename": up.filename, "sport": up.sport,
        "position": up.bike_position,
        "athlete_id": up.athlete_id, "athlete_name": names.get(up.athlete_id or 0),
        "status": up.status, "reasons": up.reasons or [], "error": up.error,
        "created_at_ms": up.created_at_ms, "finished_at_ms": up.finished_at_ms,
    }


@router.post("/me/coach-uploads", status_code=status.HTTP_201_CREATED)
async def upload_to_batch(
    video: UploadFile = File(...),
    batch_id: str = Form(...),
    sport: str = Form(...),
    position: str | None = Form(None),
    camera_side: str | None = Form(None),
    athlete_id: int | None = Form(None),
    profile_id: int | None = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
    filmed_at_ms: int | None = Form(None),
) -> dict[str, Any]:
    """One file of a batch. It goes in line and the request returns."""
    from app.api.me import _owned_profile_id

    _require_coach(user)
    if not _BATCH_ID.match(batch_id or ""):
        raise HTTPException(400, "bad batch id")
    sport = _clean_sport(sport, auto=True)
    position = _clean_position(sport, position)
    if camera_side not in (None, "", "left", "right"):
        raise HTTPException(400, "camera_side must be 'left' or 'right'")
    if athlete_id is not None:
        athlete_id = await owned_athlete_id(db, user, athlete_id)
        if athlete_id is None:
            raise HTTPException(404, "Athlete not found.")
        # Somebody else's footage: asked once (privacy.html section 3).
        prof = await db.get(CoachProfile, user.id)
        if not ((prof.data if prof else None) or {}).get("others_ok_at_ms"):
            raise HTTPException(409, "Confirm you have permission to upload other people's footage first.")
    in_batch = (await db.execute(select(CoachUpload.batch_id, CoachUpload.status).where(
        CoachUpload.coach_user_id == user.id,
        (CoachUpload.batch_id == batch_id) | CoachUpload.status.in_(OPEN_STATUSES),
    ))).all()
    if sum(1 for b, _s in in_batch if b == batch_id) >= coach_lane.MAX_BATCH_FILES:
        raise HTTPException(400, f"A batch holds {coach_lane.MAX_BATCH_FILES} files.")
    if any(b != batch_id and b is not None for b, _s in in_batch):
        raise HTTPException(409, "Your last batch is still running. Let it finish first.")
    up = await _accept(
        db, user, video, sport=sport, position=position, camera_side=camera_side or None,
        athlete_id=athlete_id, profile_id=await _owned_profile_id(db, user, profile_id),
        batch_id=batch_id, source=SOURCE_BATCH, filmed_at_ms=filmed_at_ms,
    )
    return _item(up, {})


async def _names(db: AsyncSession, user: User) -> dict[int, str]:
    rows = (await db.execute(select(Athlete.id, Athlete.name).where(
        Athlete.coach_user_id == user.id,
    ))).all()
    return dict(rows)  # type: ignore[arg-type]


@router.get("/me/coach-uploads")
async def list_uploads(
    batch_id: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The lane as this coach sees it: one batch, or the last two weeks."""
    if not has_coach(user.tier):
        return {"items": [], "max_files": coach_lane.MAX_BATCH_FILES}
    q = select(CoachUpload).where(CoachUpload.coach_user_id == user.id)
    if batch_id:
        q = q.where(CoachUpload.batch_id == batch_id)
    else:
        q = q.where(CoachUpload.created_at_ms >= _now_ms() - RECENT_DAYS * DAY_MS)
    rows = (await db.execute(q.order_by(CoachUpload.id.desc()).limit(RECENT_LIMIT))).scalars().all()
    names = await _names(db, user)
    return {"items": [_item(r, names) for r in rows], "max_files": coach_lane.MAX_BATCH_FILES}


async def _own_upload(db: AsyncSession, user: User, upload_id: int) -> CoachUpload:
    _require_coach(user)
    up = await db.get(CoachUpload, upload_id)
    if up is None or up.coach_user_id != user.id:
        raise HTTPException(404, "Clip not found.")
    return up


@router.post("/me/coach-uploads/{upload_id}/run-anyway")
async def run_anyway(
    upload_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The coach has looked at a NO-GO and wants the numbers regardless. The
    check advises; the analysis's own quality warnings still apply."""
    up = await _own_upload(db, user, upload_id)
    if up.status != STATUS_NO_GO:
        raise HTTPException(409, "Only a clip that was stopped by the check can be run anyway.")
    up.status = STATUS_QUEUED
    up.run_anyway = True
    up.finished_at_ms = None
    await db.commit()
    coach_lane.enqueue(up.id)
    return _item(up, await _names(db, user))


@router.delete("/me/coach-uploads/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_upload(
    upload_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    """Take a clip off the list. One that became a report keeps the report; one
    that did not takes its footage with it."""
    from app.core.jobs import delete_job_files

    up = await _own_upload(db, user, upload_id)
    if up.status in OPEN_STATUSES and up.status != STATUS_QUEUED:
        raise HTTPException(409, "That clip is being analysed right now.")
    if up.status != STATUS_DONE:
        row = (await db.execute(select(Analysis).where(
            Analysis.user_id == user.id, Analysis.client_id == up.client_id,
        ))).scalar_one_or_none()
        if row is not None and (row.data or {}).get("coachHold"):
            await db.delete(row)
            delete_job_files(up.job_id)
    await db.delete(up)
    await db.commit()


# --------------------------------------------------------------------------
# The intake link: /a/{token}. The athlete needs no account; the clip lands
# under their row on the coach's roster, on the coach's quota.
# --------------------------------------------------------------------------
async def _intake_target(db: AsyncSession, token: str) -> tuple[Athlete, User] | None:
    """The live roster row behind a link: not archived, and its coach still on
    the plan. None for everything else, without saying which."""
    athlete = (await db.execute(select(Athlete).where(
        Athlete.upload_token == token, Athlete.archived.is_(False),
    ))).scalar_one_or_none()
    if athlete is None:
        return None
    coach = await db.get(User, athlete.coach_user_id)
    if coach is None or not has_coach(coach.tier):
        return None
    return athlete, coach


@router.get("/a/{token}", include_in_schema=False)
async def intake_page(token: str, db: AsyncSession = Depends(get_session)) -> HTMLResponse:
    from app.api.coach_link import coach_display

    target = await _intake_target(db, token)
    headers = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex", "Referrer-Policy": "no-referrer"}
    if target is None:
        return HTMLResponse(render_intake_page(None), status_code=404, headers=headers)
    athlete, coach = target
    who = await coach_display(db, coach)
    prof = await db.get(CoachProfile, coach.id)
    logo = ((prof.data if prof else None) or {}).get("logo")
    return HTMLResponse(render_intake_page({
        "token": token, "coach": who["name"], "tagline": who["tagline"], "logo": logo,
        "athlete": athlete.name, "sport": athlete.sport, "linked": athlete.linked_user_id is not None,
    }), headers=headers)


@router.post("/a/{token}/clip", status_code=status.HTTP_201_CREATED)
async def intake_clip(
    token: str,
    request: Request,
    video: UploadFile = File(...),
    sport: str = Form(...),
    position: str | None = Form(None),
    accept: bool = Form(False),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    from app.api.auth import _throttle

    _throttle(f"intake:{client_ip(request)}", INTAKE_PER_IP_HOUR, 3600)
    target = await _intake_target(db, token)
    if target is None:
        raise HTTPException(404, "This link isn't active any more. Ask your coach for a new one.")
    athlete, coach = target
    if athlete.linked_user_id is not None:
        raise HTTPException(409, "Your account is linked to your coach. Upload in the app and they will see it.")
    if not accept:
        raise HTTPException(400, "Tick the box so your coach can see this clip.")
    since = _now_ms() - DAY_MS
    today = int(await db.scalar(select(func.count(CoachUpload.id)).where(
        CoachUpload.athlete_id == athlete.id, CoachUpload.source == SOURCE_INTAKE,
        CoachUpload.created_at_ms >= since,
    )) or 0)
    if today >= INTAKE_PER_ATHLETE_DAY:
        raise HTTPException(429, "That's plenty for one day. Your coach has your clips.")
    sport = _clean_sport(sport)
    up = await _accept(
        db, coach, video, sport=sport, position=_clean_position(sport, position), camera_side=None,
        athlete_id=athlete.id, profile_id=None, batch_id=None, source=SOURCE_INTAKE,
        extra={"intakeConsentAt": _now_ms()},
    )
    return {"id": up.client_id, "status": up.status}


@router.get("/a/{token}/clips/{client_id}")
async def intake_clip_status(
    token: str, client_id: str, db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Whether the clip passed, so an athlete still by the camera can film again.
    The verdict and its reason, never the numbers: the report is the coach's."""
    target = await _intake_target(db, token)
    if target is None:
        raise HTTPException(404, "not found")
    up = (await db.execute(select(CoachUpload).where(
        CoachUpload.client_id == client_id, CoachUpload.athlete_id == target[0].id,
    ))).scalar_one_or_none()
    if up is None:
        raise HTTPException(404, "not found")
    return {"status": up.status, "reasons": [r.get("text") for r in (up.reasons or [])]}
