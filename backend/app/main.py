"""FastAPI service for standalone video technique analysis (Milestone 3).

Endpoints
    GET  /                       -> redirect to interactive docs (/docs)
    GET  /health                 -> liveness + whether the pose model is present
    POST /analyze                -> upload a side-view clip, returns a job id
    GET  /jobs/{job_id}          -> job status + full result JSON when done
    GET  /jobs/{job_id}/overlay  -> the annotated overlay .mp4 (if generated)

Async job model: MediaPipe analysis is CPU-bound (~30-60 s per clip), so the
POST returns immediately with a ``job_id`` to poll. Work runs in a background
thread (Starlette runs sync BackgroundTasks in a threadpool, so the event loop
stays free for polling).

Job state is IN-MEMORY: fine for a single worker / MVP, but it does not survive
a restart and is not shared across workers. Move it to Redis or a DB when we add
persistence + multi-worker scaling (M4).
"""

from __future__ import annotations

import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import academy as academy_routes
from app.api import auth as auth_routes
from app.api import me as me_routes
from app.core.config import settings
from app.core.db import get_session, init_db
from app.core.security import optional_user
from app.models.user import User
from app.services.result_gating import gate_free_result, is_free
from app.services.usage_limits import (
    check_quota,
    next_reset,
    record_usage,
)
from app.services.video_analysis.runner import (
    DEFAULT_BIKE_POSITION,
    VALID_POSITIONS,
    _json_safe,
    run_analysis,
)

logger = structlog.get_logger()

# Max upload size (bytes). Phone clips are a few MB; cap to avoid abuse.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".m4v", ".mkv", ".webm"}
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
MAX_PHOTO_BYTES = 30 * 1024 * 1024  # 30 MB

# In-memory job store. job_id -> job dict.
JOBS: dict[str, dict[str, Any]] = {}

# Per-IP rate limiter (rolling 24h). In-memory: single-instance only, resets on
# restart -- consistent with the in-memory job store. Move to Redis when we
# scale past one replica (M4b).
_RATE_WINDOW_S = 24 * 3600
_rate_hits: dict[str, deque] = {}


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Railway sits behind a proxy, so prefer XFF."""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_state(ip: str) -> tuple[int, int]:
    """Return (used, retry_after_seconds) for this IP after pruning old hits."""
    if settings.rate_limit_per_day <= 0:
        return 0, 0
    now = time.time()
    dq = _rate_hits.setdefault(ip, deque())
    while dq and now - dq[0] > _RATE_WINDOW_S:
        dq.popleft()
    retry = int(_RATE_WINDOW_S - (now - dq[0])) + 1 if dq else 0
    return len(dq), retry


def _rate_record(ip: str) -> None:
    _rate_hits.setdefault(ip, deque()).append(time.time())


async def _enforce_quota(
    request: Request, user: User | None, db: AsyncSession, noun: str,
) -> None:
    """Raise 429 if the caller is over quota. Signed-in users are limited by
    their tier (DB-backed monthly/daily); anonymous visitors keep the legacy
    per-IP daily limit. Does NOT record usage -- call after the upload validates
    so a rejected file never burns quota."""
    if user is not None:
        allowed, used, limit, window = await check_quota(db, user)
        if not allowed:
            reset = next_reset(window)
            when = "tomorrow" if window == "day" else "next month"
            logger.info(
                "QUOTA_EXCEEDED", user_id=user.id, tier=user.tier,
                used=used, limit=limit, window=window,
            )
            unit = "day" if window == "day" else "month"
            raise HTTPException(
                status_code=429,
                detail=(
                    f"You've used all {limit} {noun}s on your plan this {unit}. "
                    f"Your limit resets {when}."
                ),
                headers={"Retry-After": str(
                    max(1, int((reset - datetime.now(timezone.utc)).total_seconds()))
                )},
            )
        return
    # Anonymous: legacy per-IP daily limiter.
    ip = _client_ip(request)
    limit = settings.rate_limit_per_day
    used, retry_after = _rate_state(ip)
    if limit > 0 and used >= limit:
        hours = max(1, round(retry_after / 3600))
        logger.info("RATE_LIMITED", ip=ip, used=used, limit=limit)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Daily limit reached — you can analyze {limit} {noun}s per day. "
                f"Sign in for a higher limit, or try again in about {hours}h."
            ),
            headers={"Retry-After": str(retry_after)},
        )


async def _record_and_headers(
    response: Response, request: Request, user: User | None,
    db: AsyncSession, kind: str,
) -> None:
    """Record one usage event and set X-RateLimit-* headers for the caller."""
    if user is not None:
        await record_usage(db, user.id, kind)
        _, used, limit, _window = await check_quota(db, user)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - used))
        return
    ip = _client_ip(request)
    limit = settings.rate_limit_per_day
    if limit > 0:
        used, _ = _rate_state(ip)
        _rate_record(ip)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - used - 1))


def _small_keyframe(data_uri: str | None, max_w: int = 720, quality: int = 82) -> str | None:
    """Downscale an annotated image (data URI) to a small JPEG data URI for the
    history thumbnail. Returns None on failure."""
    if not data_uri or "," not in data_uri:
        return None
    try:
        import base64

        import cv2
        import numpy as np

        raw = base64.b64decode(data_uri.split(",", 1)[1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w > max_w:
            img = cv2.resize(img, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    except Exception:  # noqa: BLE001
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


# Hide the interactive API docs in production (they expose the full endpoint
# surface to end users). They stay on locally for development. Railway sets
# RAILWAY_ENVIRONMENT on every deploy; VA_ENABLE_DOCS=1 can force them back on.
_docs_on = os.environ.get("VA_ENABLE_DOCS") == "1" or not os.environ.get("RAILWAY_ENVIRONMENT")
app = FastAPI(
    title="Flapp",
    version="0.6.0",
    description="Flapp — side-view running & cycling form analysis with AI coaching.",
    lifespan=lifespan,
    docs_url="/docs" if _docs_on else None,
    redoc_url="/redoc" if _docs_on else None,
    openapi_url="/openapi.json" if _docs_on else None,
)

# Permissive CORS so a browser frontend (M6) can call this directly. Lock the
# origin list down for production via the VA_CORS_ORIGINS env var (comma-sep).
_origins = os.environ.get("VA_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server-rendered Academy (SEO): /academy hub + article pages, sitemap, robots.
app.include_router(academy_routes.router)
# Accounts: /auth/register, /auth/login, /auth/me.
app.include_router(auth_routes.router)
# Per-user cloud history/progress: /me/analyses.
app.include_router(me_routes.router)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class JobCreated(BaseModel):
    job_id: str
    status: str
    poll_url: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    sport: str | None = None
    cycling_position: str | None = None
    error: str | None = None
    overlay_available: bool = False
    overlay_url: str | None = None
    overlay_failed: bool = False
    result: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Background worker
# --------------------------------------------------------------------------
def _process_job(
    job_id: str, input_path: str, sport: str,
    cycling_position: str | None, overlay_path: str | None,
    free: bool = False,
) -> None:
    """Run the analysis for a job (executed in a threadpool by BackgroundTasks).

    ``free`` (starter/anonymous): render the teaser keyframe (skeleton, no angle
    numbers, watermark) and trim the paid fields out of the served result.
    """
    job = JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "processing"
    logger.info("JOB_START", job_id=job_id, sport=sport, position=cycling_position)
    try:
        result = run_analysis(
            input_path, sport, cycling_position,
            # No overlay video for free users -- they get the teaser keyframe only.
            overlay_path=None if free else overlay_path,
            hide_angle_values=free,
        )
        safe = _json_safe(result)
        # Don't leak the server filesystem path; expose the API URL instead.
        if safe.get("overlay_video_path"):
            safe["overlay_video_path"] = f"/jobs/{job_id}/overlay"
        # Trim to the teaser payload for free callers (paid fields removed here,
        # not hidden client-side).
        if free:
            safe = gate_free_result(safe)
        job["result"] = safe
        if result.get("status") == "completed":
            job["status"] = "completed"
        else:
            job["status"] = "failed"
            job["error"] = result.get("error_message")
        logger.info(
            "JOB_DONE", job_id=job_id, status=job["status"],
            score=safe.get("technique_score"), grade=safe.get("letter_grade"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("JOB_FAILED", job_id=job_id, err=str(e))
        job["status"] = "failed"
        job["error"] = str(e)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def root(request: Request) -> HTMLResponse:
    """Serve the single-page frontend.

    OG/Twitter ``og:url`` and ``og:image`` are stored as root-relative paths in
    the static file and rewritten to absolute URLs here, using the request's
    own origin — so link previews resolve on any host (localhost, Railway,
    custom domain) without hardcoding a base URL.
    """
    html_doc = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    origin = str(request.base_url).rstrip("/")
    html_doc = html_doc.replace('content="/og-image.png"', f'content="{origin}/og-image.png"')
    html_doc = html_doc.replace('property="og:url" content="/"', f'property="og:url" content="{origin}/"')
    return HTMLResponse(html_doc)


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico() -> FileResponse:
    # No .ico asset; hand back the SVG so browsers requesting /favicon.ico still
    # get the brand mark instead of a 404.
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/og-image.png", include_in_schema=False)
def og_image() -> FileResponse:
    return FileResponse(STATIC_DIR / "og-image.png", media_type="image/png")


@app.get("/privacy", include_in_schema=False)
def privacy() -> FileResponse:
    """Serve the privacy policy."""
    return FileResponse(STATIC_DIR / "privacy.html")


@app.get("/terms", include_in_schema=False)
def terms() -> FileResponse:
    """Serve the terms of service."""
    return FileResponse(STATIC_DIR / "terms.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_present": settings.model_path.exists(),
        "active_jobs": sum(
            1 for j in JOBS.values() if j["status"] in ("queued", "processing")
        ),
    }


@app.post("/analyze", response_model=JobCreated, status_code=202)
async def analyze_endpoint(
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    video: UploadFile = File(..., description="Side-view video clip (mp4/mov/...)."),
    sport: str = Form(..., description="run | bike"),
    position: str | None = Form(
        None, description="Cycling position (bike only): "
        "road_hoods | road_drops | tt_aero | triathlon | casual.",
    ),
    overlay: bool = Form(
        True, description="Also render the annotated overlay video.",
    ),
    user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_session),
) -> JobCreated:
    """Accept a clip + params, kick off analysis, return a job id to poll."""
    ip = _client_ip(request)
    await _enforce_quota(request, user, db, "clip")

    if sport not in ("run", "bike"):
        raise HTTPException(400, "sport must be 'run' or 'bike'")

    cycling_position: str | None = None
    if sport == "bike":
        cycling_position = position or DEFAULT_BIKE_POSITION
        if cycling_position not in VALID_POSITIONS:
            raise HTTPException(
                400, f"invalid position; valid: {sorted(VALID_POSITIONS)}",
            )

    if not settings.model_path.exists():
        raise HTTPException(
            503, "pose model not installed on the server "
            "(backend/models/pose_landmarker_heavy.task)",
        )

    suffix = Path(video.filename or "").suffix.lower() or ".mp4"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400, f"unsupported file type '{suffix}'; "
            f"allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    data = await video.read()
    if len(data) == 0:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"file too large (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")

    job_id = uuid.uuid4().hex[:12]
    job_dir = settings.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(data)

    overlay_path = str(job_dir / "overlay.mp4") if overlay else None
    JOBS[job_id] = {
        "status": "queued",
        "sport": sport,
        "cycling_position": cycling_position,
        "result": None,
        "error": None,
        "overlay_path": overlay_path,
    }

    background_tasks.add_task(
        _process_job, job_id, str(input_path), sport, cycling_position,
        overlay_path, is_free(user),
    )
    await _record_and_headers(response, request, user, db, "video")
    logger.info("JOB_QUEUED", job_id=job_id, sport=sport, bytes=len(data), ip=ip)
    return JobCreated(job_id=job_id, status="queued", poll_url=f"/jobs/{job_id}")


@app.post("/analyze-photo", status_code=200)
async def analyze_photo_endpoint(
    request: Request,
    response: Response,
    photo: UploadFile = File(..., description="Side-view still photo (jpg/png/heic)."),
    sport: str = Form(..., description="run | bike"),
    position: str | None = Form(None, description="Cycling position (bike only)."),
    coaching: bool = Form(True, description="Include AI coaching."),
    user: User | None = Depends(optional_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """Analyze a single still photo. Synchronous (~5s) -- returns the full
    result inline, including an annotated image (data URI) + optional coaching.
    Shares the analysis quota with video analyses.
    """
    ip = _client_ip(request)
    await _enforce_quota(request, user, db, "photo")

    if sport not in ("run", "bike"):
        raise HTTPException(400, "sport must be 'run' or 'bike'")
    cycling_position: str | None = None
    if sport == "bike":
        # Respect an explicit choice; otherwise leave it None so the photo
        # analyzer auto-detects the position from the measured trunk angle
        # (a flat aero back -> tt_aero, upright -> casual, etc.). Substituting
        # DEFAULT_BIKE_POSITION here would silently disable that auto-detect and
        # score every un-picked photo against road_hoods.
        cycling_position = position or None
        if cycling_position is not None and cycling_position not in VALID_POSITIONS:
            raise HTTPException(400, f"invalid position; valid: {sorted(VALID_POSITIONS)}")
    if not settings.model_path.exists():
        raise HTTPException(503, "pose model not installed on the server")

    suffix = Path(photo.filename or "").suffix.lower() or ".jpg"
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            400, f"unsupported image type '{suffix}'; allowed: {sorted(ALLOWED_IMAGE_SUFFIXES)}",
        )
    data = await photo.read()
    if len(data) == 0:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, f"file too large (> {MAX_PHOTO_BYTES // (1024 * 1024)} MB)")

    free = is_free(user)
    from app.services.video_analysis.photo_analyzer import analyze_photo
    try:
        result = await run_in_threadpool(
            analyze_photo, data, sport, cycling_position, free,
        )
    except ValueError as e:
        # No pose detected / undecodable image -> user-actionable 422.
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("PHOTO_FAILED", err=str(e), ip=ip)
        raise HTTPException(500, "photo analysis failed")

    # Compact annotated frame for the client-side history record.
    result["keyframe_base64"] = _small_keyframe(result.get("thumbnail_base64"))

    # Free callers don't get AI coaching (it's a paid unlock).
    if coaching and not free:
        from app.services.video_analysis.llm_recommendations import (
            generate_photo_recommendations,
        )
        result["ai_recommendations"] = await run_in_threadpool(
            generate_photo_recommendations, sport, result,
        )

    if free:
        result = gate_free_result(result)

    await _record_and_headers(response, request, user, db, "photo")
    logger.info(
        "PHOTO_DONE", sport=sport, ip=ip,
        score=(result.get("score") or {}).get("overall_score"),
    )
    return _json_safe(result)


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    overlay_ready = bool(job.get("overlay_path")) and Path(job["overlay_path"]).exists()
    # Overlay was requested for this job but the file never materialized after a
    # completed run -- rendering failed (e.g. ffmpeg). Let the client say so
    # instead of silently hiding the video with no explanation.
    overlay_failed = (
        bool(job.get("overlay_path"))
        and job.get("status") == "completed"
        and not overlay_ready
    )
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        sport=job.get("sport"),
        cycling_position=job.get("cycling_position"),
        error=job.get("error"),
        overlay_available=overlay_ready,
        overlay_url=f"/jobs/{job_id}/overlay" if overlay_ready else None,
        overlay_failed=overlay_failed,
        result=job.get("result"),
    )


@app.get("/jobs/{job_id}/overlay")
def job_overlay(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    overlay_path = job.get("overlay_path")
    if not overlay_path or not Path(overlay_path).exists():
        raise HTTPException(404, "overlay not available for this job")
    return FileResponse(
        overlay_path, media_type="video/mp4", filename=f"{job_id}_overlay.mp4",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
