"""POST /prescan: what a clip shows, from a few frames the browser cut out.

Open to visitors -- on a phone the footage comes first, before any account --
so it is throttled per IP the way sign-up is, and capped in size: at most
eight frames of at most 400 KB, which is far more than a 640 px JPEG needs.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.api.auth import _throttle
from app.core.net import client_ip

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

    return await run_in_threadpool(scan_images, images, photo=(mode == "photo"))
