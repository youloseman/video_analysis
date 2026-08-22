"""Off-bike mobility screens: measure once, informs every bike analysis after.

Two photos taken lying on the floor. What they buy is the difference between
"you could go lower" and "you could go lower if your hips went there" -- see
``services/video_analysis/biomechanics/mobility`` for what the module is and is
not allowed to claim.

The measurement lives on the ACCOUNT, not on an analysis. Your hamstrings are
not a property of the clip you uploaded, and nobody is going to lie on the
floor before every ride.

Deliberately outside the analysis quota. Measuring your own range is setup, not
a report -- charging a free rider one of their ten monthly analyses to do it
would price the thing that makes their fit advice correct. The cost is bounded
instead by a small per-account cooldown: the screens take seconds to shoot but
weeks to change, so nobody legitimately needs to run them in a loop.
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_session
from app.core.security import get_current_user
from app.models.user import User
from app.services.video_analysis.biomechanics import mobility as M

logger = structlog.get_logger()
router = APIRouter(prefix="/mobility", tags=["mobility"])

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_PHOTO_BYTES = 15 * 1024 * 1024

# A brake, not a quota. The natural flow is two photos back to back, plus a
# retake or two when the first attempt gets refused for a bent knee -- so a
# per-request cooldown would fight the actual usage. A small burst over a
# rolling minute leaves that alone and still stops a MediaPipe loop.
SCREEN_BURST = 8
SCREEN_WINDOW_S = 60.0
_recent_screens: dict[int, deque[float]] = {}

VALID_GOALS = ("comfort", "speed")


class GoalIn(BaseModel):
    goal: str | None = Field(
        None, description="comfort | speed | null to clear",
    )


def stored_profile(user: User) -> dict[str, Any] | None:
    """The mobility profile saved on this account, or None if never measured."""
    return M.profile_from_stored(
        hamstring_deg=user.mobility_hamstring_deg,
        hip_flexion_deg=user.mobility_hip_flexion_deg,
        goal=user.mobility_goal,
        measured_at=user.mobility_measured_at,
    )


def _catalogue() -> list[dict[str, Any]]:
    """How to shoot each screen, served rather than duplicated in the client.

    The instructions and the validation live in the same module: a client copy
    of "keep the knee straight" would drift from the check that rejects a bent
    one, and the athlete would be told off for following our own guidance.
    """
    out = []
    for key, spec in M.MOBILITY_SCREENS.items():
        out.append({
            "screen": key,
            "label": spec["label"],
            "measures": spec["measures"],
            "unit": spec["unit"],
            "why": spec["why"],
            "setup": list(spec["setup"]),
            "source": spec["source"],
            "bands": {
                "good": spec["tiers"][0][0],
                "moderate": spec["tiers"][1][0],
            },
        })
    return out


@router.get("")
async def get_mobility(user: User = Depends(get_current_user)) -> dict[str, Any]:
    """This account's mobility profile, plus how to shoot the screens."""
    return {
        "profile": stored_profile(user),
        "screens": _catalogue(),
        "goals": list(VALID_GOALS),
    }


@router.post("/screen")
async def measure_screen(
    screen: str = Form(..., description="hamstring | hip_flexion"),
    photo: UploadFile = File(..., description="Side-on photo of the screen."),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Measure one screen from one photo and save it to the account.

    A screen that cannot be measured is a 422 with the reason, and NOTHING is
    written. "We could not see your knee" must never be filed as "your hip does
    not go there" -- the first is our problem and the second is a finding, and
    storing one as the other would put a fabricated limitation on the account
    for every future analysis.
    """
    if screen not in M.MOBILITY_SCREENS:
        raise HTTPException(
            400, f"unknown screen; valid: {sorted(M.MOBILITY_SCREENS)}",
        )
    if not settings.model_path.exists():
        raise HTTPException(503, "pose model not installed on the server")

    suffix = Path(photo.filename or "").suffix.lower() or ".jpg"
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            400,
            f"unsupported image type '{suffix}'; "
            f"allowed: {sorted(ALLOWED_IMAGE_SUFFIXES)}",
        )
    data = await photo.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(
            413, f"file too large (> {MAX_PHOTO_BYTES // (1024 * 1024)} MB)",
        )

    # Checked here rather than at the top: a rejected upload must not count
    # against the burst. Somebody who picks the wrong file and immediately
    # picks the right one has not asked us for any CPU, and holding their own
    # typo against them is a strange way to run a form.
    now = time.monotonic()
    hits = _recent_screens.setdefault(user.id, deque())
    while hits and now - hits[0] > SCREEN_WINDOW_S:
        hits.popleft()
    if len(hits) >= SCREEN_BURST:
        raise HTTPException(
            429,
            "That is a lot of screens in a minute. Give it a moment -- your "
            "range will not have changed.",
            headers={"Retry-After": str(int(SCREEN_WINDOW_S - (now - hits[0])) + 1)},
        )
    hits.append(now)

    from app.services.video_analysis.photo_analyzer import detect_pose_in_image

    def _measure() -> dict[str, Any]:
        pose = detect_pose_in_image(data)
        result = M.measure_screen(screen, pose["world"])
        result["capture_warnings"] = pose["warnings"]
        return result

    try:
        result = await run_in_threadpool(_measure)
    except ValueError as e:
        # Undecodable image / no pose found -- the caller can fix both.
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("MOBILITY_SCREEN_FAILED", err=str(e), screen=screen)
        raise HTTPException(500, "could not measure that photo")

    if "error" in result:
        # A refusal, not a low reading. Nothing is saved.
        raise HTTPException(422, result["error"])

    if screen == "hamstring":
        user.mobility_hamstring_deg = result["value"]
    else:
        user.mobility_hip_flexion_deg = result["value"]
    user.mobility_measured_at = datetime.now(timezone.utc)
    await db.commit()

    logger.info(
        "MOBILITY_MEASURED",
        user_id=user.id, screen=screen, value=result["value"], tier=result["tier"],
    )
    return {"measurement": result, "profile": stored_profile(user)}


@router.patch("")
async def set_goal(
    body: GoalIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Set (or clear) the comfort-vs-speed preference.

    A preference, not a measurement: it says which end of a fit window the
    rider is aiming at. It never moves the window.
    """
    goal = (body.goal or "").strip().lower() or None
    if goal is not None and goal not in VALID_GOALS:
        raise HTTPException(400, f"goal must be one of {list(VALID_GOALS)} or null")
    user.mobility_goal = goal
    await db.commit()
    return {"profile": stored_profile(user)}


@router.delete("", status_code=200)
async def clear_mobility(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Forget the measurements.

    Worth having as a first-class action rather than a support request: a
    reading taken on a bad day follows the account into every future fit, and
    the rider is the only one who can tell us it was wrong.
    """
    user.mobility_hamstring_deg = None
    user.mobility_hip_flexion_deg = None
    user.mobility_measured_at = None
    await db.commit()
    logger.info("MOBILITY_CLEARED", user_id=user.id)
    return {"profile": None}
