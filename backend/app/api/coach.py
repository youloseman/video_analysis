"""The coach profile of a Full account: GET / PUT ``/me/coach``.

Full and admin only. The report a coach hands an athlete carries the
coach's name and mark and the coach's own notes; a plan that does not
include handing reports to athletes has no use for any of it, so the other
tiers get a 402 that says which plan does. See ``models.coach``.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user
from app.models.coach import (
    LOGO_PREFIXES,
    MAX_CUE_CHARS,
    MAX_CUES,
    MAX_LOGO_BYTES,
    MAX_NAME_CHARS,
    MAX_TAGLINE_CHARS,
    CoachProfile,
)
from app.models.user import TIER_ADMIN, TIER_FULL, User

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["coach"])

COACH_TIERS = (TIER_FULL, TIER_ADMIN)


def coach_allowed(user: User | None) -> bool:
    return user is not None and user.tier in COACH_TIERS


def _require_coach(user: User) -> None:
    if not coach_allowed(user):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="The coach profile -- your name and mark on the report, and your "
                   "cue library -- is part of the Full plan.",
        )


class CoachIn(BaseModel):
    name: str = Field(default="", max_length=MAX_NAME_CHARS)
    tagline: str = Field(default="", max_length=MAX_TAGLINE_CHARS)
    logo: str | None = Field(default=None, max_length=MAX_LOGO_BYTES * 2)
    cues: list[str] = Field(default_factory=list, max_length=MAX_CUES)


def _clean(body: CoachIn) -> dict[str, Any]:
    name = " ".join(body.name.split())[:MAX_NAME_CHARS]
    tagline = " ".join(body.tagline.split())[:MAX_TAGLINE_CHARS]
    cues: list[str] = []
    seen: set[str] = set()
    for raw in body.cues:
        cue = " ".join(str(raw).split())[:MAX_CUE_CHARS]
        key = cue.lower()
        if cue and key not in seen:
            seen.add(key)
            cues.append(cue)
    logo = (body.logo or "").strip() or None
    if logo:
        if not logo.startswith(LOGO_PREFIXES):
            raise HTTPException(422, "The logo must be a PNG, JPEG or WebP image.")
        payload = logo.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(payload, validate=True)
        except Exception:  # noqa: BLE001
            raise HTTPException(422, "The logo image could not be read.")
        if len(raw_bytes) > MAX_LOGO_BYTES:
            raise HTTPException(
                413, f"The logo is too large -- keep it under {MAX_LOGO_BYTES // 1024} KB "
                     "(it is shown about 48 px tall).",
            )
    return {"name": name, "tagline": tagline, "logo": logo, "cues": cues[:MAX_CUES]}


def _public(row: CoachProfile | None) -> dict[str, Any]:
    d = (row.data if row else None) or {}
    return {
        "name": d.get("name") or "",
        "tagline": d.get("tagline") or "",
        "logo": d.get("logo") or None,
        "cues": list(d.get("cues") or []),
        "updated_at_ms": row.updated_at_ms if row else None,
    }


async def _row_for(db: AsyncSession, user: User) -> CoachProfile | None:
    return (
        await db.execute(select(CoachProfile).where(CoachProfile.user_id == user.id))
    ).scalar_one_or_none()


@router.get("/coach")
async def get_coach(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The coach profile, or an empty one; ``allowed`` says whether this
    plan may set it, so the client can show the card or the upgrade line
    without a second call."""
    row = await _row_for(db, user) if coach_allowed(user) else None
    return {"allowed": coach_allowed(user), **_public(row)}


@router.put("/coach")
async def put_coach(
    body: CoachIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_coach(user)
    data = _clean(body)
    row = await _row_for(db, user)
    now = int(time.time() * 1000)
    if row is None:
        row = CoachProfile(user_id=user.id, updated_at_ms=now, data=data)
        db.add(row)
    else:
        # Reassigned, not mutated: JSON columns do not track in-place edits.
        row.data = data
        row.updated_at_ms = now
    await db.commit()
    logger.info("COACH_PROFILE_SAVED", user_id=user.id, cues=len(data["cues"]), logo=bool(data["logo"]))
    return {"allowed": True, **_public(row)}
