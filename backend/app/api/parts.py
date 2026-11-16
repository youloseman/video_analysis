"""``/me/parts`` -- which parts fit this frame (services/parts.py).

Read-only over files shipped with the code. Signed-in only, like the rest of a
setup: the panel lives on a bike profile.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user
from app.models.part_report import MAX_NOTE_CHARS, PartReport
from app.models.profile import Profile
from app.models.user import TIER_ADMIN, User
from app.services import bike_catalog, parts

logger = structlog.get_logger()

router = APIRouter(prefix="/me/parts", tags=["parts"])

# Beyond these the question is a different bike, not a part.
MAX_REACH_MM = 120.0
MAX_STACK_MM = 120.0
MAX_ANGLE_DEG = 25.0


@router.get("/frames")
async def list_frames(
    family: str | None = None,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    return {"frames": parts.frames_for(family)}


@router.get("/frames/{frame_id}")
async def suggest_parts(
    frame_id: str,
    size: str | None = None,
    reach_mm: float = Query(0.0, ge=-MAX_REACH_MM, le=MAX_REACH_MM),
    stack_mm: float = Query(0.0, ge=-MAX_STACK_MM, le=MAX_STACK_MM),
    angle_deg: float = Query(0.0, ge=-MAX_ANGLE_DEG, le=MAX_ANGLE_DEG),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    out = parts.suggest(frame_id, size=size, reach_mm=reach_mm, stack_mm=stack_mm, angle_deg=angle_deg)
    if out is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="We hold no parts data for that frame yet.")
    try:
        from app.services import analytics

        analytics.capture_bg(analytics.person_id(user), "parts_suggested", {
            "frame": frame_id, "reach_mm": reach_mm, "stack_mm": stack_mm, "angle_deg": angle_deg,
            "helps": len(out["helps"]), "incompatible": len(out["incompatible"]),
        })
    except Exception:  # noqa: BLE001 -- analytics never fails a request
        pass
    return out


# --------------------------------------------------------------------------
# What riders tell us back: a queue, never a write into the data
# (models/part_report.py says why).
# --------------------------------------------------------------------------
class ReportIn(BaseModel):
    part_id: str | None = None
    fits: bool
    note: str | None = None

    @field_validator("note")
    @classmethod
    def _note(cls, v: str | None) -> str | None:
        v = " ".join((v or "").split())[:MAX_NOTE_CHARS]
        return v or None


# A rider reporting the same part twice in a day is one report.
REPORTS_PER_USER_PER_DAY = 20


@router.post("/frames/{frame_id}/reports", status_code=status.HTTP_201_CREATED)
async def report_part(
    frame_id: str,
    body: ReportIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    frame = parts.load_frames().get(frame_id)
    if frame is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="We hold no parts data for that frame yet.")
    if body.part_id is not None and body.part_id not in {p["id"] for p in frame["parts"]}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="That part is not on record for this frame.")
    if body.part_id is None and not body.note:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Name the part in the note.")
    since = int(time.time() * 1000) - 86_400_000
    today = await db.scalar(
        select(func.count(PartReport.id)).where(PartReport.user_id == user.id, PartReport.created_at_ms >= since)
    )
    if int(today or 0) >= REPORTS_PER_USER_PER_DAY:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="That is a lot of reports for one day. Thank you; try again tomorrow.")
    row = PartReport(
        user_id=user.id, frame_id=frame_id, part_id=body.part_id, fits=body.fits,
        note=body.note, created_at_ms=int(time.time() * 1000),
    )
    db.add(row)
    await db.commit()
    logger.info("PART_REPORTED", frame=frame_id, part=body.part_id, fits=body.fits, user_id=user.id)
    try:
        from app.services import analytics

        analytics.capture_bg(analytics.person_id(user), "part_reported", {
            "frame": frame_id, "part": body.part_id, "fits": body.fits, "with_note": bool(body.note),
        })
    except Exception:  # noqa: BLE001
        pass
    return {"id": row.id, "queued": True}


def _require_admin(user: User) -> None:
    if user.tier != TIER_ADMIN:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")


@router.get("/reports")
async def list_reports(
    open_only: bool = True,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The review queue (admin). Each row next to what the file says now."""
    _require_admin(user)
    q = select(PartReport).order_by(PartReport.created_at_ms.desc()).limit(200)
    if open_only:
        q = q.where(PartReport.reviewed_at_ms.is_(None))
    rows = (await db.execute(q)).scalars().all()
    frames = parts.load_frames()

    def on_file(r: PartReport) -> str | None:
        f = frames.get(r.frame_id) or {}
        p = next((p for p in f.get("parts") or [] if p["id"] == r.part_id), None)
        return p["status"] if p else None

    return {"reports": [{
        "id": r.id, "frame_id": r.frame_id, "part_id": r.part_id, "fits": r.fits,
        "note": r.note, "at": r.created_at_ms, "reviewed_at": r.reviewed_at_ms,
        "status_on_file": on_file(r),
    } for r in rows]}


@router.post("/reports/{report_id}/reviewed")
async def mark_reviewed(
    report_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _require_admin(user)
    r = await db.get(PartReport, report_id)
    if r is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    r.reviewed_at_ms = int(time.time() * 1000)
    await db.commit()
    return {"id": r.id, "reviewed_at": r.reviewed_at_ms}


@router.get("/demand")
async def frame_demand(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Which bikes riders have filed, and which of them we hold data for --
    the order to fill the data in (admin). Counts only, no names."""
    _require_admin(user)
    rows = (await db.execute(
        select(Profile.bike_family, Profile.bike_generation, func.count(Profile.id))
        .where(Profile.sport == "bike", Profile.archived.is_(False))
        .group_by(Profile.bike_family, Profile.bike_generation)
    )).all()
    held = {f["family"] for f in parts.load_frames().values()}
    by_family: dict[str, dict[str, Any]] = {}
    for fam, gen, n in rows:
        key = fam or "(not set)"
        d = by_family.setdefault(key, {
            "family": key, "label": bike_catalog.family_label(fam) or key,
            "profiles": 0, "with_generation": 0, "data_held": fam in held,
        })
        d["profiles"] += int(n)
        d["with_generation"] += int(n) if gen else 0
    return {"families": sorted(by_family.values(), key=lambda d: -d["profiles"])}
