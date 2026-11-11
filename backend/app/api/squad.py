"""Races, clip reviews and groups: what a coach keeps about the squad.

All under ``/me``, Coach plan and admin only, every row scoped by the coach
who owns the roster (``api.athletes._owned`` answers 404 for anybody else's).

``roster_extras`` is what ``GET /me/athletes`` adds to each roster row, so the
Squad screen gets the next race, the groups and the count of unseen clips in
the request it already makes.

"Unseen" has a start date. The mark did not exist before ``UNSEEN_SINCE_MS``,
and counting every older clip as unseen would greet each coach with their
whole history as an inbox; clips from before it are simply not counted.
A written note marks its clip as seen: nobody writes on a report they have
not looked at.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.athletes import _owned, _require_coach
from app.core.db import get_session
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.athlete import Athlete
from app.models.squad import (
    DEFAULT_PRIORITY,
    MAX_GROUP_NAME_CHARS,
    MAX_GROUPS,
    MAX_RACE_NAME_CHARS,
    MAX_RACES_PER_ATHLETE,
    RACE_PRIORITIES,
    AthleteGroup,
    AthleteGroupMember,
    AthleteRace,
    ClipReview,
)
from app.models.user import User

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["squad"])

# The day the review mark shipped. Clips older than this are never "unseen".
UNSEEN_SINCE_MS = int(datetime(2026, 11, 11, tzinfo=timezone.utc).timestamp() * 1000)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


# --------------------------------------------------------------------------
# Races
# --------------------------------------------------------------------------
def _clean_race_name(v: str) -> str:
    name = " ".join((v or "").split())[:MAX_RACE_NAME_CHARS]
    if not name:
        raise ValueError("Give the race a name.")
    return name


def _clean_race_day(v: str) -> str:
    try:
        day = date.fromisoformat((v or "").strip())
    except ValueError as e:
        raise ValueError("The race date should look like 2026-12-13.") from e
    if not 2000 <= day.year <= 2100:
        raise ValueError("That race date is out of range.")
    return day.isoformat()


def _clean_priority(v: str | None) -> str:
    p = (v or "").strip().upper()
    return p if p in RACE_PRIORITIES else DEFAULT_PRIORITY


class RaceIn(BaseModel):
    name: str = Field(max_length=MAX_RACE_NAME_CHARS * 2)
    race_on: str = Field(max_length=32)
    priority: str = DEFAULT_PRIORITY

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _clean_race_name(v)

    @field_validator("race_on")
    @classmethod
    def _day(cls, v: str) -> str:
        return _clean_race_day(v)

    @field_validator("priority")
    @classmethod
    def _priority(cls, v: str) -> str:
        return _clean_priority(v)


class RacePatch(BaseModel):
    """Only the fields sent change."""

    name: str | None = Field(default=None, max_length=MAX_RACE_NAME_CHARS * 2)
    race_on: str | None = Field(default=None, max_length=32)
    priority: str | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, v: str | None) -> str | None:
        return None if v is None else _clean_race_name(v)

    @field_validator("race_on")
    @classmethod
    def _day(cls, v: str | None) -> str | None:
        return None if v is None else _clean_race_day(v)


def _race_out(r: AthleteRace, today: str) -> dict[str, Any]:
    days = (date.fromisoformat(r.race_on) - date.fromisoformat(today)).days
    return {
        "id": r.id, "athlete_id": r.athlete_id, "name": r.name, "race_on": r.race_on,
        "priority": r.priority, "days_to_go": days, "past": days < 0,
    }


def next_race(races: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The nearest race still ahead (today counts); ``races`` sorted by date."""
    return next((r for r in races if not r["past"]), None)


async def _owned_race(db: AsyncSession, user: User, athlete_id: int, race_id: int) -> AthleteRace:
    row = (
        await db.execute(
            select(AthleteRace).where(
                AthleteRace.id == race_id, AthleteRace.athlete_id == athlete_id,
                AthleteRace.coach_user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Race not found.")
    return row


@router.post("/athletes/{athlete_id}/races", status_code=status.HTTP_201_CREATED)
async def add_race(
    athlete_id: int,
    body: RaceIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    athlete = await _owned(db, user, athlete_id)
    held = int(
        await db.scalar(
            select(func.count(AthleteRace.id)).where(AthleteRace.athlete_id == athlete.id)
        )
        or 0
    )
    if held >= MAX_RACES_PER_ATHLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An athlete keeps {MAX_RACES_PER_ATHLETE} races. Remove one that has passed.",
        )
    row = AthleteRace(
        athlete_id=athlete.id, coach_user_id=user.id, name=body.name,
        race_on=body.race_on, priority=body.priority, created_at_ms=_now_ms(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info("RACE_ADDED", user_id=user.id, athlete_id=athlete.id, race_on=row.race_on)
    return _race_out(row, _today())


@router.patch("/athletes/{athlete_id}/races/{race_id}")
async def update_race(
    athlete_id: int,
    race_id: int,
    body: RacePatch,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    row = await _owned_race(db, user, athlete_id, race_id)
    if body.name is not None:
        row.name = body.name
    if body.race_on is not None:
        row.race_on = body.race_on
    if body.priority is not None:
        row.priority = _clean_priority(body.priority)
    await db.commit()
    return _race_out(row, _today())


@router.delete("/athletes/{athlete_id}/races/{race_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_race(
    athlete_id: int,
    race_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    _require_coach(user)
    row = await _owned_race(db, user, athlete_id, race_id)
    await db.delete(row)
    await db.commit()


# --------------------------------------------------------------------------
# Clip reviews
# --------------------------------------------------------------------------
async def _clip(db: AsyncSession, coach: User, athlete: Athlete, client_id: str) -> Analysis:
    """The analysis ``client_id`` names on this athlete: a linked athlete's own
    row while the link is live, else the coach's clip filed under them. The
    same lookup, in the same order, as the coach's note (api/coach_link.py)."""
    from app.api.coach_link import linked_owner

    owner = await linked_owner(db, coach, athlete.id)
    row = None
    if owner is not None:
        row = (
            await db.execute(
                select(Analysis).where(
                    Analysis.user_id == owner.id, Analysis.client_id == client_id,
                )
            )
        ).scalar_one_or_none()
    if row is None:
        row = (
            await db.execute(
                select(Analysis).where(
                    Analysis.user_id == coach.id, Analysis.client_id == client_id,
                    Analysis.athlete_id == athlete.id,
                )
            )
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.")
    return row


async def mark_reviewed(db: AsyncSession, coach: User, analysis_id: int) -> int:
    """Stamp ``analysis_id`` as seen by ``coach``; the first stamp is kept.
    Adds to the session, the caller commits."""
    held = (
        await db.execute(
            select(ClipReview).where(
                ClipReview.coach_user_id == coach.id, ClipReview.analysis_id == analysis_id,
            )
        )
    ).scalar_one_or_none()
    if held is not None:
        return int(held.reviewed_at_ms)
    at = _now_ms()
    db.add(ClipReview(coach_user_id=coach.id, analysis_id=analysis_id, reviewed_at_ms=at))
    return at


async def reviews_for(db: AsyncSession, coach: User, owner_id: int) -> dict[str, int]:
    """``client_id`` -> when ``coach`` marked it, over ``owner_id``'s analyses."""
    rows = (
        await db.execute(
            select(Analysis.client_id, ClipReview.reviewed_at_ms)
            .join(ClipReview, ClipReview.analysis_id == Analysis.id)
            .where(ClipReview.coach_user_id == coach.id, Analysis.user_id == owner_id)
        )
    ).all()
    return {cid: int(at) for cid, at in rows}


def _unseen(coach: User):
    """Clips young enough to count, with no mark from this coach."""
    return (
        Analysis.created_at_ms >= UNSEEN_SINCE_MS,
        ~exists().where(
            ClipReview.analysis_id == Analysis.id, ClipReview.coach_user_id == coach.id,
        ),
    )


async def unreviewed_counts(db: AsyncSession, coach: User, rows: list[Athlete]) -> dict[int, int]:
    """Per roster row: clips this coach has not marked. The coach's own clips
    filed under the athlete, plus a linked athlete's own (swim reviews are a
    different product and never land in the squad)."""
    out: dict[int, int] = {}
    if not rows:
        return out
    ids = [a.id for a in rows]
    mine = (
        await db.execute(
            select(Analysis.athlete_id, func.count(Analysis.id))
            .where(Analysis.user_id == coach.id, Analysis.athlete_id.in_(ids), *_unseen(coach))
            .group_by(Analysis.athlete_id)
        )
    ).all()
    for aid, n in mine:
        out[int(aid)] = int(n)
    linked = {a.linked_user_id: a.id for a in rows if a.linked_user_id is not None}
    if linked:
        theirs = (
            await db.execute(
                select(Analysis.user_id, func.count(Analysis.id))
                .where(
                    Analysis.user_id.in_(list(linked)),
                    func.coalesce(Analysis.sport, "") != "swim",
                    *_unseen(coach),
                )
                .group_by(Analysis.user_id)
            )
        ).all()
        for uid, n in theirs:
            aid = linked[uid]
            out[aid] = out.get(aid, 0) + int(n)
    return out


@router.put("/athletes/{athlete_id}/analyses/{client_id}/reviewed")
async def mark_clip_reviewed(
    athlete_id: int,
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    athlete = await _owned(db, user, athlete_id)
    row = await _clip(db, user, athlete, client_id)
    at = await mark_reviewed(db, user, row.id)
    await db.commit()
    return {"id": client_id, "reviewedAt": at}


@router.delete(
    "/athletes/{athlete_id}/analyses/{client_id}/reviewed",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def unmark_clip_reviewed(
    athlete_id: int,
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    _require_coach(user)
    athlete = await _owned(db, user, athlete_id)
    row = await _clip(db, user, athlete, client_id)
    await db.execute(
        delete(ClipReview).where(
            ClipReview.coach_user_id == user.id, ClipReview.analysis_id == row.id,
        )
    )
    await db.commit()


@router.post("/athletes/{athlete_id}/reviewed-all")
async def mark_all_reviewed(
    athlete_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Everything of this athlete's that is still unseen, in one go."""
    from app.api.coach_link import linked_owner

    _require_coach(user)
    athlete = await _owned(db, user, athlete_id)
    ids = list(
        (
            await db.execute(
                select(Analysis.id).where(
                    Analysis.user_id == user.id, Analysis.athlete_id == athlete.id,
                    *_unseen(user),
                )
            )
        ).scalars()
    )
    owner = await linked_owner(db, user, athlete.id)
    if owner is not None:
        ids += list(
            (
                await db.execute(
                    select(Analysis.id).where(
                        Analysis.user_id == owner.id,
                        func.coalesce(Analysis.sport, "") != "swim",
                        *_unseen(user),
                    )
                )
            ).scalars()
        )
    at = _now_ms()
    for aid in ids:
        db.add(ClipReview(coach_user_id=user.id, analysis_id=aid, reviewed_at_ms=at))
    await db.commit()
    return {"marked": len(ids)}


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------
def _clean_group_name(v: str) -> str:
    name = " ".join((v or "").split())[:MAX_GROUP_NAME_CHARS]
    if not name:
        raise ValueError("Give the group a name.")
    return name


class GroupIn(BaseModel):
    name: str = Field(max_length=MAX_GROUP_NAME_CHARS * 2)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        return _clean_group_name(v)


class GroupMembersIn(BaseModel):
    group_ids: list[int] = Field(default_factory=list, max_length=MAX_GROUPS)


async def _groups(db: AsyncSession, user: User) -> list[AthleteGroup]:
    return list(
        (
            await db.execute(
                select(AthleteGroup)
                .where(AthleteGroup.coach_user_id == user.id)
                .order_by(AthleteGroup.name)
            )
        ).scalars()
    )


async def _owned_group(db: AsyncSession, user: User, group_id: int) -> AthleteGroup:
    row = (
        await db.execute(
            select(AthleteGroup).where(
                AthleteGroup.id == group_id, AthleteGroup.coach_user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    return row


async def _refuse_a_twin(db: AsyncSession, user: User, name: str, but: int | None = None) -> None:
    """Two groups a coach cannot tell apart are one group typed twice."""
    for g in await _groups(db, user):
        if g.id != but and g.name.casefold() == name.casefold():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="You already have a group with that name.",
            )


async def _members(db: AsyncSession, group_ids: list[int]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {g: [] for g in group_ids}
    if not group_ids:
        return out
    rows = (
        await db.execute(
            select(AthleteGroupMember.group_id, AthleteGroupMember.athlete_id)
            .where(AthleteGroupMember.group_id.in_(group_ids))
            .order_by(AthleteGroupMember.athlete_id)
        )
    ).all()
    for gid, aid in rows:
        out[int(gid)].append(int(aid))
    return out


def _group_out(g: AthleteGroup, athlete_ids: list[int]) -> dict[str, Any]:
    return {"id": g.id, "name": g.name, "athlete_ids": athlete_ids, "athletes": len(athlete_ids)}


@router.get("/athlete-groups")
async def list_groups(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    groups = await _groups(db, user)
    members = await _members(db, [g.id for g in groups])
    return {"groups": [_group_out(g, members[g.id]) for g in groups], "limit": MAX_GROUPS}


@router.post("/athlete-groups", status_code=status.HTTP_201_CREATED)
async def create_group(
    body: GroupIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    if len(await _groups(db, user)) >= MAX_GROUPS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A squad keeps {MAX_GROUPS} groups. Remove one you no longer use.",
        )
    await _refuse_a_twin(db, user, body.name)
    row = AthleteGroup(coach_user_id=user.id, name=body.name, created_at_ms=_now_ms())
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info("GROUP_CREATED", user_id=user.id, group_id=row.id)
    return _group_out(row, [])


@router.patch("/athlete-groups/{group_id}")
async def rename_group(
    group_id: int,
    body: GroupIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    row = await _owned_group(db, user, group_id)
    await _refuse_a_twin(db, user, body.name, but=row.id)
    row.name = body.name
    await db.commit()
    return _group_out(row, (await _members(db, [row.id]))[row.id])


@router.delete("/athlete-groups/{group_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_group(
    group_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    """Remove a group. Its athletes stay on the roster, in their other groups."""
    _require_coach(user)
    row = await _owned_group(db, user, group_id)
    # Spelled out rather than left to ON DELETE CASCADE: SQLite only honours
    # that with a pragma the app does not set.
    await db.execute(delete(AthleteGroupMember).where(AthleteGroupMember.group_id == row.id))
    await db.delete(row)
    await db.commit()


@router.put("/athletes/{athlete_id}/groups")
async def set_athlete_groups(
    athlete_id: int,
    body: GroupMembersIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The groups this athlete is in, replaced as a whole. An id that is not
    one of the coach's groups is a 404 for the whole call: nothing half-saved."""
    _require_coach(user)
    athlete = await _owned(db, user, athlete_id)
    wanted = sorted(set(body.group_ids))
    mine = {g.id for g in await _groups(db, user)}
    if any(g not in mine for g in wanted):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Group not found.")
    await db.execute(delete(AthleteGroupMember).where(AthleteGroupMember.athlete_id == athlete.id))
    for gid in wanted:
        db.add(AthleteGroupMember(group_id=gid, athlete_id=athlete.id))
    await db.commit()
    return {"athlete_id": athlete.id, "group_ids": wanted}


# --------------------------------------------------------------------------
# What the roster listing carries, and what leaves with an athlete
# --------------------------------------------------------------------------
async def roster_extras(
    db: AsyncSession, coach: User, rows: list[Athlete], today: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Per roster row: ``races`` (by date), ``next_race``, ``group_ids`` and
    ``unreviewed``. Three queries for the whole squad, not three per athlete."""
    today = today or _today()
    ids = [a.id for a in rows]
    out: dict[int, dict[str, Any]] = {
        i: {"races": [], "next_race": None, "group_ids": [], "unreviewed": 0} for i in ids
    }
    if not ids:
        return out
    races = (
        await db.execute(
            select(AthleteRace)
            .where(AthleteRace.coach_user_id == coach.id, AthleteRace.athlete_id.in_(ids))
            .order_by(AthleteRace.race_on, AthleteRace.id)
        )
    ).scalars()
    for r in races:
        out[r.athlete_id]["races"].append(_race_out(r, today))
    members = (
        await db.execute(
            select(AthleteGroupMember.athlete_id, AthleteGroupMember.group_id)
            .where(AthleteGroupMember.athlete_id.in_(ids))
            .order_by(AthleteGroupMember.group_id)
        )
    ).all()
    for aid, gid in members:
        out[int(aid)]["group_ids"].append(int(gid))
    unseen = await unreviewed_counts(db, coach, rows)
    for i in ids:
        out[i]["next_race"] = next_race(out[i]["races"])
        out[i]["unreviewed"] = unseen.get(i, 0)
    return out


async def forget_athlete(db: AsyncSession, athlete_id: int) -> None:
    """Races and group seats go with the athlete. Review marks stay with the
    analyses, which stay too (unfiled)."""
    await db.execute(delete(AthleteRace).where(AthleteRace.athlete_id == athlete_id))
    await db.execute(delete(AthleteGroupMember).where(AthleteGroupMember.athlete_id == athlete_id))
