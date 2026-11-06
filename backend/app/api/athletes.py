"""A coach's roster: GET / POST / PATCH / DELETE ``/me/athletes``.

Coach and admin only; everybody else gets a 402 that names the plan. An
athlete is a label the coach owns -- see ``models.athlete`` -- so the rules
mirror the equipment profiles': the list rides with how many analyses each
athlete holds and when they last filmed, archiving keeps the history, and
deleting unfiles rather than removes it.

``owned_athlete_id`` is what ``api.me`` calls when a history entry names an
athlete: taken on trust, a client-supplied id would be a way to file a clip
under somebody else's athlete, and the squad table reads from exactly that
column.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.athlete import (
    ATHLETE_SPORTS,
    DEFAULT_SPORT,
    MAX_NAME_CHARS,
    MAX_NOTE_CHARS,
    Athlete,
    new_upload_token,
)
from app.models.profile import Profile
from app.models.user import User, athlete_limit, has_coach

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["athletes"])


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_coach(user: User) -> None:
    if not has_coach(user.tier):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="A roster of athletes -- their clips, their trends, their "
                   "reports under your name -- is part of the Coach plan.",
        )


def _clean_name(v: str) -> str:
    name = " ".join((v or "").split())[:MAX_NAME_CHARS]
    if not name:
        raise ValueError("Give the athlete a name.")
    return name


def _clean_sport(v: str | None) -> str:
    return v if v in ATHLETE_SPORTS else DEFAULT_SPORT


def _clean_note(v: str | None) -> str | None:
    note = " ".join((v or "").split())[:MAX_NOTE_CHARS]
    return note or None


class AthleteIn(BaseModel):
    name: str = Field(max_length=MAX_NAME_CHARS * 2)
    email: EmailStr | None = None
    sport: str = DEFAULT_SPORT
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS * 2)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _clean_name(v)

    @field_validator("sport")
    @classmethod
    def _sport(cls, v: str) -> str:
        return _clean_sport(v)

    @field_validator("note")
    @classmethod
    def _note(cls, v: str | None) -> str | None:
        return _clean_note(v)


class AthletePatch(BaseModel):
    """Only the fields sent change; ``email: null`` clears it."""

    name: str | None = Field(default=None, max_length=MAX_NAME_CHARS * 2)
    email: EmailStr | None = None
    sport: str | None = None
    note: str | None = Field(default=None, max_length=MAX_NOTE_CHARS * 2)
    archived: bool | None = None
    # True mints a new intake link; the old one stops working at once.
    rotate_token: bool = False


def _out(a: Athlete, stats: tuple[int, int | None, int | None] | None = None) -> dict[str, Any]:
    n, last_at, last_score = stats or (0, None, None)
    return {
        "id": a.id, "name": a.name, "email": a.email, "sport": a.sport,
        "note": a.note, "archived": a.archived, "created_at_ms": a.created_at_ms,
        "upload_token": a.upload_token,
        "analyses": n, "last_at_ms": last_at, "last_score": last_score,
    }


async def _owned(db: AsyncSession, user: User, athlete_id: int) -> Athlete:
    row = (
        await db.execute(
            select(Athlete).where(
                Athlete.id == athlete_id, Athlete.coach_user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        # 404 rather than 403: ids are sequential, and confirming one exists
        # would be a way to count somebody else's squad.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Athlete not found.")
    return row


async def owned_athlete_id(db: AsyncSession, user: User, raw: Any) -> int | None:
    """A client-supplied athlete id, if it is one of the caller's; else None."""
    try:
        aid = int(raw)
    except (TypeError, ValueError):
        return None
    owned = await db.scalar(
        select(Athlete.id).where(Athlete.id == aid, Athlete.coach_user_id == user.id)
    )
    return int(owned) if owned else None


async def _refuse_if_no_room(db: AsyncSession, user: User) -> None:
    live = int(
        await db.scalar(
            select(func.count(Athlete.id)).where(
                Athlete.coach_user_id == user.id, Athlete.archived.is_(False),
            )
        )
        or 0
    )
    limit = athlete_limit(user.tier)
    if live >= limit:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Your plan keeps {limit} athletes. Archive one who has moved "
                "on, or write to support@getflapp.com for a bigger squad."
            ),
        )


async def _stats(db: AsyncSession, user: User) -> dict[int, tuple[int, int | None, int | None]]:
    """Per athlete: how many analyses, when the latest was, and its score.

    Two queries rather than a correlated subquery per row: the squad is
    fifteen people, the history a thousand rows, and this runs on every open
    of the Athletes screen.
    """
    counts = (
        await db.execute(
            select(
                Analysis.athlete_id,
                func.count(Analysis.id),
                func.max(Analysis.created_at_ms),
            )
            .where(Analysis.user_id == user.id, Analysis.athlete_id.is_not(None))
            .group_by(Analysis.athlete_id)
        )
    ).all()
    out: dict[int, tuple[int, int | None, int | None]] = {}
    if not counts:
        return out
    latest_at = {aid: last for aid, _n, last in counts}
    scores = (
        await db.execute(
            select(Analysis.athlete_id, Analysis.score, Analysis.created_at_ms)
            .where(
                Analysis.user_id == user.id,
                Analysis.athlete_id.in_(list(latest_at)),
            )
        )
    ).all()
    last_score = {
        aid: score for aid, score, at in scores if latest_at.get(aid) == at
    }
    for aid, n, last in counts:
        out[int(aid)] = (int(n), int(last) if last is not None else None, last_score.get(aid))
    return out


@router.get("/athletes")
async def list_athletes(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The roster, with ``limit`` and ``used`` so the client can say
    "6 of 15" without knowing the tier table; ``allowed`` says whether this
    plan has one at all."""
    if not has_coach(user.tier):
        return {"allowed": False, "athletes": [], "limit": 0, "used": 0}
    rows = (
        await db.execute(
            select(Athlete)
            .where(Athlete.coach_user_id == user.id)
            .order_by(Athlete.archived, Athlete.name)
        )
    ).scalars().all()
    stats = await _stats(db, user)
    live = sum(1 for a in rows if not a.archived)
    return {
        "allowed": True,
        "athletes": [_out(a, stats.get(a.id)) for a in rows],
        "limit": athlete_limit(user.tier),
        "used": live,
    }


@router.post("/athletes", status_code=status.HTTP_201_CREATED)
async def create_athlete(
    body: AthleteIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    await _refuse_if_no_room(db, user)
    row = Athlete(
        coach_user_id=user.id, name=body.name, email=body.email, sport=body.sport,
        note=body.note, created_at_ms=_now_ms(), upload_token=new_upload_token(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info("ATHLETE_CREATED", user_id=user.id, athlete_id=row.id)
    return _out(row)


@router.patch("/athletes/{athlete_id}")
async def update_athlete(
    athlete_id: int,
    body: AthletePatch,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    row = await _owned(db, user, athlete_id)
    if body.name is not None:
        try:
            row.name = _clean_name(body.name)
        except ValueError:
            pass
    if "email" in body.model_fields_set:
        row.email = body.email
    if body.sport is not None:
        row.sport = _clean_sport(body.sport)
    if "note" in body.model_fields_set:
        row.note = _clean_note(body.note)
    if body.archived is not None:
        # Bringing somebody back can pass the limit (archived three, downgraded).
        if not body.archived and row.archived:
            await _refuse_if_no_room(db, user)
        row.archived = body.archived
    if body.rotate_token:
        row.upload_token = new_upload_token()
    await db.commit()
    return _out(row)


@router.delete("/athletes/{athlete_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_athlete(
    athlete_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    """Remove an athlete. Their analyses and setups stay, unfiled.

    Not a cascade, for the same reason profiles are not: the clips were run
    on this account and are its history. Archive is the softer option and the
    one the UI offers first.
    """
    _require_coach(user)
    row = await _owned(db, user, athlete_id)
    await db.execute(
        update(Analysis)
        .where(Analysis.user_id == user.id, Analysis.athlete_id == row.id)
        .values(athlete_id=None)
    )
    await db.execute(
        update(Profile)
        .where(Profile.user_id == user.id, Profile.athlete_id == row.id)
        .values(athlete_id=None)
    )
    await db.delete(row)
    await db.commit()
    logger.info("ATHLETE_DELETED", user_id=user.id, athlete_id=athlete_id)
