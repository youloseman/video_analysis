"""POST /prescan: what a clip shows, from a few frames the browser cut out.

Open to visitors -- on a phone the footage comes first, before any account --
so it is throttled per IP the way sign-up is, and capped in size: at most
eight frames of at most 400 KB, which is far more than a 640 px JPEG needs.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.auth import _throttle
from app.core.db import get_session
from app.core.net import client_ip
from app.core.security import optional_user
from app.models.user import User, has_coach

router = APIRouter(tags=["prescan"])

MAX_FRAMES = 8
MAX_FRAME_BYTES = 400_000
PER_IP = 40            # scans per IP ...
WINDOW_S = 60 * 60     # ... per hour


@router.post("/prescan")
async def prescan(
    request: Request,
    frames: list[UploadFile] = File(...),
    mode: str = Form("video"),
    # The clip's own size: the frames sent are shrunk to 640 px, and the
    # rider's size is judged in the frame the full analysis will see.
    video_w: int | None = Form(None),
    video_h: int | None = Form(None),
    # The tile chosen on Analyze: which bars apply when the frames cannot say.
    sport: str | None = Form(None),
    user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    _throttle(f"prescan:{client_ip(request)}", PER_IP, WINDOW_S)
    if not frames or len(frames) > MAX_FRAMES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Send 1 to {MAX_FRAMES} frames.")
    images = []
    for f in frames:
        raw = await f.read(MAX_FRAME_BYTES + 1)
        if len(raw) > MAX_FRAME_BYTES:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Frames must be small JPEGs.")
        images.append(raw)
    from app.services.prescan import scan_images

    ok = all(isinstance(v, int) and 0 < v <= 10000 for v in (video_w, video_h))
    dims = (video_w, video_h) if ok else (None, None)
    # A coach's own grid, when a coach is the one asking.
    bars = None
    if isinstance(user, User) and has_coach(user.tier):
        from app.api.coach import capture_for

        bars = await capture_for(db, user.id)
    return await run_in_threadpool(scan_images, images, photo=(mode == "photo"),
                                   video_w=dims[0], video_h=dims[1], thresholds=bars,
                                   sport_hint=sport if sport in ("run", "bike") else None)
