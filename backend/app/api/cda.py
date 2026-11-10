"""Measured CdA on a setup: ``/me/profiles/{id}/cda`` and the coach's view of
an athlete's setups, ``/me/athletes/{id}/setups``. models/cda.py says what a
row is and why Flapp never calculates one.

Who may write where:

* the owner, on their own bike setups;
* a coach, on a setup kept on the coach's account for a roster athlete
  (``profiles.athlete_id``), and -- while the athlete's coach link is live --
  on the linked athlete's own setups. A coach's row is stamped with who typed
  it; the athlete sees "added by your coach" and may delete it. The coach may
  delete only what the coach typed.

Every miss is a 404, never a 403: ids are sequential, and confirming one
exists would be a way to count somebody else's bikes.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user
from app.models.athlete import Athlete
from app.models.cda import (
    CDA_PROTOCOLS,
    CDA_RANGE,
    CDA_SOURCES,
    MAX_NOTE_CHARS,
    NOT_A_RIDE_SOURCES,
    CdaMeasurement,
)
from app.models.profile import Profile
from app.models.user import User, has_coach

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["cda"])

# Rows kept per setup. A setup measured more than this is a test campaign,
# and the history screen is not where that lives.
MAX_PER_PROFILE = 50


def _now_ms() -> int:
    return int(time.time() * 1000)


def _not_found(what: str = "Setup not found.") -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=what)


class CdaIn(BaseModel):
    value_m2: float
    measured_on: str
    source: str
    protocol: str = "repeated_loops"
    note: str | None = None

    @field_validator("value_m2")
    @classmethod
    def _value(cls, v: float) -> float:
        lo, hi = CDA_RANGE
        if not (lo <= v <= hi):
            raise ValueError(
                f"CdA is in m², between {lo:.3f} and {hi:.3f}. A TT position is "
                f"usually 0.18-0.30, a road position 0.28-0.40."
            )
        return round(float(v), 4)

    @field_validator("measured_on")
    @classmethod
    def _day(cls, v: str) -> str:
        try:
            d = date.fromisoformat(v.strip()[:10])
        except ValueError as exc:
            raise ValueError("Give the date it was measured (YYYY-MM-DD).") from exc
        # A day ahead is the far side of a time zone, not the future.
        if d > date.today() + timedelta(days=1) or d.year < 2000:
            raise ValueError("That date is not one a measurement could have been taken on.")
        return d.isoformat()

    @field_validator("source")
    @classmethod
    def _source(cls, v: str) -> str:
        if v not in CDA_SOURCES:
            raise ValueError("Pick where the number came from.")
        return v

    @field_validator("protocol")
    @classmethod
    def _protocol(cls, v: str) -> str:
        if v not in CDA_PROTOCOLS:
            raise ValueError("Say how it was measured.")
        return v

    @field_validator("note")
    @classmethod
    def _note(cls, v: str | None) -> str | None:
        v = (v or "").strip()[:MAX_NOTE_CHARS]
        return v or None


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
async def _names(db: AsyncSession, ids: set[int]) -> dict[int, str]:
    """The coach's display name for each writer who is not the owner."""
    from app.api.coach_link import coach_display

    out: dict[int, str] = {}
    for uid in ids:
        u = await db.get(User, uid)
        if u is not None:
            out[uid] = (await coach_display(db, u))["name"]
    return out


def _row_out(r: CdaMeasurement, owner_id: int, names: dict[int, str], reader_id: int) -> dict[str, Any]:
    by = r.entered_by_user_id
    return {
        "id": r.id, "profile_id": r.profile_id, "value_m2": r.value_m2,
        "measured_on": r.measured_on, "source": r.source, "protocol": r.protocol,
        "note": r.note, "created_at_ms": r.created_at_ms,
        # None = the setup's owner typed it. Otherwise the coach's name, for
        # "added by your coach, <name>".
        "entered_by_coach": names.get(by) if by is not None and by != owner_id else None,
        # Whether the reader typed it: a coach reading the athlete's setup
        # sees "added by you" on their own rows.
        "by_reader": by is not None and by == reader_id,
        # Whether THIS reader may delete it: the owner always, a coach only
        # their own row.
        "can_delete": reader_id == owner_id or (by is not None and by == reader_id),
    }


async def rows_by_profile(
    db: AsyncSession, profiles: list[Profile], reader_id: int,
) -> dict[int, list[dict[str, Any]]]:
    """Every measurement on these setups, newest first, keyed by profile id."""
    if not profiles:
        return {}
    owner_of = {p.id: p.user_id for p in profiles}
    rows = (
        await db.execute(
            select(CdaMeasurement)
            .where(CdaMeasurement.profile_id.in_(list(owner_of)))
            .order_by(CdaMeasurement.measured_on.desc(), CdaMeasurement.created_at_ms.desc())
        )
    ).scalars().all()
    writers = {
        r.entered_by_user_id for r in rows
        if r.entered_by_user_id is not None and r.entered_by_user_id != owner_of.get(r.profile_id)
    }
    names = await _names(db, writers)
    out: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r.profile_id, []).append(_row_out(r, owner_of[r.profile_id], names, reader_id))
    return out


def _setup_out(p: Profile, rows: list[dict[str, Any]], whose: str) -> dict[str, Any]:
    return {
        "id": p.id, "name": p.name, "kind": p.kind, "archived": p.archived,
        "position": p.bike_position, "whose": whose, "cda": rows,
    }


async def athlete_setups(db: AsyncSession, coach: User, athlete: Athlete) -> list[dict[str, Any]]:
    """An athlete's bike setups as the coach sees them: the ones the coach
    keeps for them, and the athlete's own while the link is live."""
    from app.api.coach_link import linked_owner

    mine = (
        await db.execute(
            select(Profile).where(
                Profile.user_id == coach.id, Profile.athlete_id == athlete.id,
                Profile.sport == "bike",
            ).order_by(Profile.archived, Profile.created_at_ms)
        )
    ).scalars().all()
    theirs: list[Profile] = []
    owner = await linked_owner(db, coach, athlete.id)
    if owner is not None:
        theirs = (
            await db.execute(
                select(Profile).where(
                    Profile.user_id == owner.id, Profile.sport == "bike",
                    Profile.athlete_id.is_(None),
                ).order_by(Profile.archived, Profile.created_at_ms)
            )
        ).scalars().all()
    by = await rows_by_profile(db, [*mine, *theirs], coach.id)
    return (
        [_setup_out(p, by.get(p.id, []), "athlete") for p in theirs]
        + [_setup_out(p, by.get(p.id, []), "coach") for p in mine]
    )


@router.get("/athletes/{athlete_id}/setups")
async def list_athlete_setups(
    athlete_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    from app.api.athletes import _owned, _require_coach

    _require_coach(user)
    athlete = await _owned(db, user, athlete_id)
    return {"setups": await athlete_setups(db, user, athlete)}


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------
async def _writable_profile(
    db: AsyncSession, user: User, profile_id: int, athlete_id: int | None,
) -> Profile:
    """The setup this caller may add to (or delete from), else 404."""
    from app.api.coach_link import linked_owner

    p = await db.get(Profile, profile_id)
    if p is None or p.sport != "bike":
        raise _not_found()
    if athlete_id is None:
        if p.user_id != user.id:
            raise _not_found()
        return p
    # A coach writing for a roster athlete.
    if not has_coach(user.tier):
        raise _not_found()
    athlete = (
        await db.execute(
            select(Athlete).where(Athlete.id == athlete_id, Athlete.coach_user_id == user.id)
        )
    ).scalar_one_or_none()
    if athlete is None:
        raise _not_found()
    if p.user_id == user.id and p.athlete_id == athlete.id:
        return p
    owner = await linked_owner(db, user, athlete.id)
    if owner is not None and p.user_id == owner.id and p.athlete_id is None:
        return p
    raise _not_found()


@router.post("/profiles/{profile_id}/cda", status_code=status.HTTP_201_CREATED)
async def add_cda(
    profile_id: int,
    body: CdaIn,
    athlete_id: int | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    p = await _writable_profile(db, user, profile_id, athlete_id)
    n = len((await db.execute(
        select(CdaMeasurement.id).where(CdaMeasurement.profile_id == p.id)
    )).scalars().all())
    if n >= MAX_PER_PROFILE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"This setup already holds {MAX_PER_PROFILE} measurements. Delete an old one first.",
        )
    # A tunnel or a CFD model is never a ride, whatever the form said.
    protocol = "not_a_ride" if body.source in NOT_A_RIDE_SOURCES else body.protocol
    row = CdaMeasurement(
        profile_id=p.id, user_id=p.user_id, value_m2=body.value_m2,
        measured_on=body.measured_on, source=body.source, protocol=protocol,
        note=body.note, entered_by_user_id=user.id, created_at_ms=_now_ms(),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    logger.info("CDA_ADDED", profile_id=p.id, owner_id=p.user_id, by=user.id, source=row.source)
    names = await _names(db, {user.id}) if user.id != p.user_id else {}
    return _row_out(row, p.user_id, names, user.id)


@router.delete("/profiles/{profile_id}/cda/{measurement_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cda(
    profile_id: int,
    measurement_id: int,
    athlete_id: int | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    p = await _writable_profile(db, user, profile_id, athlete_id)
    row = await db.get(CdaMeasurement, measurement_id)
    if row is None or row.profile_id != p.id:
        raise _not_found("Measurement not found.")
    # The owner removes anything on their setup; a coach only what they typed.
    if user.id != p.user_id and row.entered_by_user_id != user.id:
        raise _not_found("Measurement not found.")
    await db.delete(row)
    await db.commit()
    logger.info("CDA_DELETED", profile_id=p.id, by=user.id)


# --------------------------------------------------------------------------
# The roster's column
# --------------------------------------------------------------------------
async def latest_by_athlete(db: AsyncSession, rows: list[Athlete]) -> dict[int, dict[str, Any]]:
    """The newest measurement across each roster athlete's bike setups: the
    ones the coach keeps for them and, when linked, their own."""
    if not rows:
        return {}
    ids = [a.id for a in rows]
    linked = {a.linked_user_id: a.id for a in rows if a.linked_user_id is not None}
    q = select(CdaMeasurement, Profile).join(Profile, Profile.id == CdaMeasurement.profile_id)
    found = list((await db.execute(
        q.where(Profile.athlete_id.in_(ids), Profile.user_id == rows[0].coach_user_id)
    )).all())
    if linked:
        found += list((await db.execute(
            q.where(Profile.user_id.in_(list(linked)), Profile.athlete_id.is_(None))
        )).all())
    best: dict[int, tuple[tuple[str, int], dict[str, Any]]] = {}
    for m, p in found:
        aid = p.athlete_id if p.athlete_id is not None else linked.get(p.user_id)
        if aid is None:
            continue
        key = (m.measured_on, m.created_at_ms)
        if aid not in best or key > best[aid][0]:
            best[aid] = (key, {
                "value_m2": m.value_m2, "measured_on": m.measured_on, "source": m.source,
                "protocol": m.protocol, "profile_name": p.name,
            })
    return {aid: v for aid, (_k, v) in best.items()}
