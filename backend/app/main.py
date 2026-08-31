"""FastAPI service for standalone video technique analysis (Milestone 3).

Endpoints
    GET  /                       -> redirect to interactive docs (/docs)
    GET  /health                 -> liveness + whether the pose model is present
    POST /analyze                -> upload a side-view clip, returns a job id
    GET  /jobs/{job_id}          -> job status + full result JSON when done
    GET  /jobs/{job_id}/overlay  -> the annotated overlay .mp4 (if generated)
    GET  /jobs/{job_id}/export   -> the analysis as an AI-readable .md / .json

Async job model: MediaPipe analysis is CPU-bound (~30-60 s per clip), so the
POST returns immediately with a ``job_id`` to poll. Work runs in a background
thread (Starlette runs sync BackgroundTasks in a threadpool, so the event loop
stays free for polling).

Job state is IN-MEMORY: fine for a single worker / MVP, but it does not survive
a restart and is not shared across workers. Move it to Redis or a DB when we add
persistence + multi-worker scaling (M4).

Because nothing else would ever delete them, a background sweeper drops jobs and
their upload directories once they are older than ``settings.job_ttl_hours``,
and concurrent analyses are capped by a semaphore (``max_concurrent_analyses``)
with a bounded queue in front of it (``max_queued_analyses``) -- MediaPipe
"heavy" is CPU/RAM-bound and the threadpool would otherwise start one per
thread and OOM the container.
"""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import hmac
import json
import math
import os
import secrets
import shutil
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
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
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import academy as academy_routes
from app.api import auth as auth_routes
from app.api import billing as billing_routes
from app.api.billing import log_billing_configuration
from app.api import changelog as changelog_routes
from app.api import examples as examples_routes
from app.api import feedback as feedback_routes
from app.api import me as me_routes
from app.api import mobility as mobility_routes
from app.api.mobility import stored_profile as _stored_mobility
from app.core.compression import (
    SelectiveGZipMiddleware,
    accepts_brotli,
    brotli_encode,
)
from app.core.config import settings
from app.core.db import SessionLocal, get_session, init_db
from app.core.jobs import (
    ANALYSIS_SLOTS,
    JOBS,
    SLOT_WAIT_TIMEOUT_S,
    authorized_job,
    pending_jobs,
    queue_ahead,
    log_storage_configuration,
    retained_job_ids,
    sweep_upload_dirs,
    sweeper_loop,
)
from app.core.net import client_ip
from app.core.security import get_current_user, optional_user
from app.models.user import User
from app.services import analytics, pricing
from app.services.analytics import log_analytics_configuration
from app.services.export import ai_export
from app.services.notify import log_email_configuration
from app.services.result_gating import (
    ACCESS_FULL,
    ACCESS_PREVIEW,
    ACCESS_TEASER,
    access_for,
    gate_for_access,
    is_free,
)
from app.services.usage_limits import (
    check_quota,
    next_reset,
    record_usage,
)
from app.services.video_analysis.bilateral_session import build_pair_result
from app.services.video_analysis.run_session import build_run_session
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

# Can we produce an overlay a BROWSER can play? ``video_visualizer`` re-encodes
# to H.264 with ffmpeg and, when ffmpeg is missing, falls back to OpenCV's mp4v
# muxer -- a file that plays in VLC but that no browser decodes. That fallback
# fails in the one way nothing catches: the file exists, so the job looks
# successful, and the athlete gets a silent black player with no explanation.
#
# So: the encoder's absence is treated as "no overlay for this job", which routes
# into the ``overlay_failed`` path the client already has a sentence for. The
# file is still written, so it stays available on disk for local debugging.
# ffmpeg is installed in the image (see Dockerfile); this guards the day it is not.
OVERLAY_ENCODER_PRESENT = shutil.which("ffmpeg") is not None

# Job store, concurrency cap and expiry all live in app.core.jobs (see there for
# why). Imported under the original names so the routes below read unchanged.

# Per-IP rate limiter (rolling 24h). In-memory: single-instance only, resets on
# restart -- consistent with the in-memory job store. Move to Redis when we
# scale past one replica (M4b).
_RATE_WINDOW_S = 24 * 3600
_rate_hits: dict[str, deque] = {}


def _client_ip(request: Request) -> str:
    """Best-effort client IP (shared with the feedback endpoint)."""
    return client_ip(request)


# Kept but unused by the analysis endpoints: analysis now requires an account,
# so quotas are per-user. A per-IP cap can't be layered on top without punishing
# whole households, gyms and offices behind one NAT address.
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
    request: Request, user: User, db: AsyncSession, noun: str,
) -> None:
    """Raise 429 if the caller is over their tier's quota (DB-backed
    monthly/daily). Does NOT record usage -- call after the upload validates so
    a rejected file never burns quota.

    Analysis requires an account, so there is no anonymous branch: the free
    plan *is* the signed-in Starter tier.
    """
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


def preview_available(user: User) -> bool:
    """Whether this account still has its one free preview to spend.

    The claim records the *job*, not a timestamp, so a run that failed can hand
    the preview back: it is the first analysis a new account ever runs -- the
    one most likely to be filmed badly, and the worst possible one to burn.

    A claimed job that has aged out of the store is treated as spent. The store
    keeps a job for ``job_ttl_hours``, which is far longer than anyone waits
    before retrying a failure, so the alternative reading -- "gone, therefore
    forgotten, therefore have another" -- would hand out a fresh preview every
    few hours forever.
    """
    if not is_free(user):
        return False
    claimed = user.free_preview_job_id
    if not claimed:
        return True
    job = JOBS.get(claimed)
    if not job:
        return False
    if job.get("status") == "failed":
        return True
    # A run that COMPLETED and then declined to publish a score did not deliver
    # what the preview exists to show either. The preview is one report worth
    # judging us by; a page reading "not scored" is an honest answer and not
    # that report. Same reasoning as the failed case above, and it became
    # reachable the day the quality gate started withholding the number
    # (2026-08-31) -- before that a gated clip still carried a score, so this
    # branch had nothing to catch.
    return bool((job.get("result") or {}).get("score_withheld"))


async def _record_and_headers(
    response: Response, request: Request, user: User,
    db: AsyncSession, kind: str,
) -> None:
    """Record one usage event and set X-RateLimit-* headers for the caller."""
    await record_usage(db, user.id, kind)
    _, used, limit, _window = await check_quota(db, user)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, limit - used))


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
    # Same class of problem as the two below: a missing volume breaks nothing
    # visible until somebody's footage is needed weeks later and is not there.
    log_storage_configuration()
    # Say in the deploy log whether checkout can actually charge anyone; every
    # way it can be broken is otherwise silent until a customer hits it.
    log_billing_configuration()
    # Same reason: a delivered review that silently announces itself to
    # nobody looks exactly like one that did.
    log_email_configuration()
    # And again: an empty dashboard reads as "nobody used it", whether that is
    # true or the key was never set.
    log_analytics_configuration()
    if not OVERLAY_ENCODER_PRESENT:
        logger.error(
            "OVERLAY_ENCODER_MISSING",
            detail="ffmpeg not on PATH -- overlay videos are disabled "
                   "(they would be encoded as mp4v, which no browser plays)",
        )
    # Clear anything the previous process left behind before serving. On a
    # volume that is now most of the disk: the job store did not survive the
    # restart but the files did, so without this pass every upload from the
    # previous process would be unreachable AND undeletable -- no live job
    # claims it, and the ids are gone from memory. What the database still
    # vouches for is kept.
    keep = await retained_job_ids()
    if keep is not None:
        swept = await run_in_threadpool(sweep_upload_dirs, keep)
        logger.info("SWEEP_STARTUP", dirs_deleted=swept, retained=len(keep))
    sweeper = asyncio.create_task(sweeper_loop())
    try:
        yield
    finally:
        sweeper.cancel()


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

# Compression. The whole SPA -- CSS, markup and JS -- is inlined in one ~330 KB
# document and was going over the wire uncompressed; gzip takes it to roughly
# 60 KB. Repeat visits were already cheap (the root route ETags and 304s), so
# this is specifically about first paint for a first-time visitor. It also
# covers the server-rendered Academy pages and the JSON API. Text only -- see
# app/core/compression.py for why the stock middleware would break the overlay
# video.
app.add_middleware(SelectiveGZipMiddleware, minimum_size=1000)

# CORS. The SPA is served by THIS app, so production needs no cross-origin
# access at all -- every request the product makes is same-origin, and CORS
# headers would only let other sites drive the API with someone else's browser.
# So: explicit VA_CORS_ORIGINS wins (comma-separated, or "*" to opt back into
# the old behaviour); otherwise wildcard locally (a frontend on :3000 during
# development) and NO cross-origin access in production, where the middleware
# is simply not installed.
_cors_env = os.environ.get("VA_CORS_ORIGINS", "").strip()
if _cors_env == "*":
    _cors_origins = ["*"]
elif _cors_env:
    _cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
else:
    _cors_origins = [] if os.environ.get("RAILWAY_ENVIRONMENT") else ["*"]

if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS_ENABLED", origins=_cors_origins)
else:
    logger.info("CORS_DISABLED", reason="same-origin only")

# Server-rendered Academy (SEO): /academy hub + article pages, sitemap, robots.
app.include_router(academy_routes.router)
# Accounts: /auth/register, /auth/login, /auth/me.
app.include_router(auth_routes.router)
# Per-user cloud history/progress: /me/analyses.
app.include_router(me_routes.router)
# Billing: /billing/checkout, /billing/webhook, /billing/portal, /billing/orders.
app.include_router(billing_routes.router)
# Expert Review fulfilment queue: /admin/orders (admin tier only).
app.include_router(billing_routes.admin_router)
# Result feedback: POST /feedback (anyone) + /admin/feedback* (admin tier only).
app.include_router(feedback_routes.router)
# Release notes: /changelog.json (in-app "What's new") + /changelog (public page).
app.include_router(changelog_routes.router)
# Public sample reports: /examples -- what a real analysis looks like to
# somebody who has not signed up, which is the one thing a score-only free
# tier cannot show them.
app.include_router(examples_routes.router)
# Off-bike mobility screens: /mobility (profile), /mobility/screen (measure).
app.include_router(mobility_routes.router)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class JobCreated(BaseModel):
    job_id: str
    status: str
    poll_url: str
    # Capability token for this job. The id alone is not a secret worth relying
    # on: it travels in a URL fragment, gets pasted into chats, and a job holds
    # someone's footage. Keep it out of the address bar (the client stores it in
    # sessionStorage) and pass it as ``?t=`` when polling or fetching the overlay.
    job_token: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | processing | completed | failed
    sport: str | None = None
    cycling_position: str | None = None
    # How many clips are ahead of this one while it waits for an analysis slot
    # (0 = next up). Only meaningful while status == "queued".
    queue_ahead: int = 0
    error: str | None = None
    overlay_available: bool = False
    overlay_url: str | None = None
    overlay_failed: bool = False
    result: dict[str, Any] | None = None
    # A two-sided session analyses two clips in one job, so "processing" lasts
    # about twice as long. Without a word for which clip is running, the wait
    # looks identical to a hang -- and a silent spinner is exactly what makes
    # people reload and lose the run.
    stage: str | None = None
    # Joint corrections (bike, paid): the adjustments currently applied to
    # this job's result, how many rounds are left, and -- when a re-measurement
    # failed -- why, with the previous result left standing.
    corrections: list[dict[str, Any]] | None = None
    rounds_left: int | None = None
    recompute_error: str | None = None


# --------------------------------------------------------------------------
# Background worker
# --------------------------------------------------------------------------
# The athlete's own question, on its way to a language model. Capped and
# stripped of newlines: the cap keeps one upload from spending the coach's
# whole token budget on a wall of text, and collapsing newlines stops the
# field being used to fake extra sections in the prompt. It is never used as
# an instruction -- the system prompt decides what the coach does with it.
FOCUS_MAX_CHARS = 200


def _clean_focus(raw: str | None) -> str | None:
    if not raw:
        return None
    text = " ".join(str(raw).split())
    return text[:FOCUS_MAX_CHARS] or None


def _process_job(
    job_id: str, input_path: str, sport: str,
    cycling_position: str | None, overlay_path: str | None,
    free: bool = False, preview: bool = False,
    athlete_height_cm: int | None = None,
    focus: str | None = None,
    mobility_profile: dict[str, Any] | None = None,
    camera_side_override: str | None = None,
) -> None:
    """Run the analysis for a job (executed in a threadpool by BackgroundTasks).

    ``free`` (starter): render the teaser keyframe (skeleton, no angle numbers,
    watermark), skip the paid LLM call, and trim the paid fields out of the
    served result.

    ``preview`` is the one exception per account: the coaching call runs and the
    keyframe keeps its numbers, because a score with nothing to read is a
    demonstration that the product exists rather than that it is worth paying
    for. The measurement table and the training plan are still withheld -- see
    services/result_gating.py for which half goes where and why.
    """
    job = JOBS.get(job_id)
    if job is None:
        return
    # Wait for a slot BEFORE flipping to "processing", so a job sitting in line
    # keeps reporting "queued" (with its position) rather than pretending to
    # work. This blocks a threadpool thread, which is cheap -- what it prevents
    # is N concurrent MediaPipe runs, which is not.
    if not ANALYSIS_SLOTS.acquire(timeout=SLOT_WAIT_TIMEOUT_S):
        logger.warning("JOB_SLOT_TIMEOUT", job_id=job_id)
        job["status"] = "failed"
        job["error"] = (
            "The server was busy for too long and gave up on this clip. "
            "Please try again in a few minutes."
        )
        return
    job["status"] = "processing"
    logger.info("JOB_START", job_id=job_id, sport=sport, position=cycling_position)
    try:
        result = run_analysis(
            input_path, sport, cycling_position,
            # No overlay video on a free plan, preview included: the annotated
            # video is a subscription feature, not part of the sample.
            overlay_path=None if free else overlay_path,
            # The preview shows its numbers. Burning them out of the frame while
            # the coaching talks about them would be a strange thing to show
            # somebody we are trying to convince.
            hide_angle_values=free and not preview,
            # AI coaching is a paid unlock, and the gate strips it from the
            # response anyway -- generating it for a teaser caller just burns a
            # billed Gemini call on output nobody ever sees. The preview is the
            # one free result where it IS shown, so there it runs.
            recommendations=(not free) or preview,
            # Same reasoning for the kinogram, which the gate withholds from
            # every free response INCLUDING the preview: rendering it would buy
            # a second video decode and nothing else.
            kinogram=not free,
            # The one real-world length a side view gets. Every tier, free
            # included -- it makes their centimetres correct rather than
            # unlocking anything, and a paywall in front of accuracy is a
            # different product from a paywall in front of detail.
            athlete_height_cm=athlete_height_cm,
            # What this athlete asked us to look at. Reaches the coach, never
            # the measurement: a question cannot change what the pose model saw.
            focus=focus,
            camera_side_override=camera_side_override,
            # Their off-bike range, if they have ever measured it. Also reaches
            # only the advice, never the measurement -- what it decides is
            # whether "get lower" is a sentence we are entitled to write.
            mobility_profile=mobility_profile,
            # The stabilized frames, kept beside the clip so the analysis can
            # be run again with the athlete's joint corrections applied. Same
            # directory, same retention: they expire with the footage.
            frames_store=str(Path(input_path).parent / "landmarks.npz"),
        )
        safe = _json_safe(result)
        # Don't leak the server filesystem path; expose the API URL instead.
        if safe.get("overlay_video_path"):
            safe["overlay_video_path"] = f"/jobs/{job_id}/overlay"
        # The WHOLE result is kept. It used to be trimmed here, which quietly
        # decided that a free analysis would never be worth anything later:
        # the paid half was thrown away before anything could store it, so
        # unlocking a report afterwards had nothing to reveal and upgrading
        # could not open the history somebody had already built. Trimming now
        # happens on the way out, per reader -- see ``job_status``.
        job["result"] = safe
        job["preview"] = preview
        if result.get("status") == "completed":
            job["status"] = "completed"
        else:
            job["status"] = "failed"
            job["error"] = result.get("error_message")
        logger.info(
            "JOB_DONE", job_id=job_id, status=job["status"],
            score=safe.get("technique_score"), grade=safe.get("letter_grade"),
        )
        if job["status"] == "completed":
            _schedule_ready_mail(job_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("JOB_FAILED", job_id=job_id, err=str(e))
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        ANALYSIS_SLOTS.release()


# --------------------------------------------------------------------------
# Re-measuring a clip with the athlete's joint corrections (bike, paid).
# See docs/LANDMARK_CORRECTION_PLAN_RU.md. The frames the analysis measured
# from are on disk beside the clip (landmark_store); a correction is a
# constant offset per joint applied to all of them, and the measuring half of
# the pipeline runs again without the detector.
# --------------------------------------------------------------------------

_RECOMPUTE_STAGE = "Measuring again with your adjustments…"
_RESET_STAGE = "Restoring the automatic measurement…"
_BASELINE_KEYS = (
    "knee_at_bdc", "knee_at_tdc", "trunk_angle_avg", "hip_angle_avg",
    "elbow_angle_avg", "shoulder_angle_avg", "forearm_tilt_avg",
    "head_alignment_avg", "pelvic_ratio", "saddle_height_assessment",
)


def _baseline_of(result: dict[str, Any]) -> dict[str, Any]:
    """The automatic reading, kept beside every corrected one so the
    adjustment stays visible as a delta rather than replacing history."""
    metrics = result.get("sport_specific_metrics") or {}
    return {
        "technique_score": result.get("technique_score"),
        "letter_grade": result.get("letter_grade"),
        "score_breakdown": result.get("score_breakdown"),
        "metrics": {k: metrics.get(k) for k in _BASELINE_KEYS},
    }


def _rounds_left(job: dict[str, Any]) -> int:
    from app.services.correction_limits import PER_ANALYSIS

    return max(0, PER_ANALYSIS - int(job.get("recompute_rounds") or 0))


def _correction_access(job: dict[str, Any], user: User) -> None:
    """The refusals every corrections call shares. 402 first: the plan is the
    reason most callers will be turned away, and it is the one with a fix."""
    if is_free(user):
        raise HTTPException(
            status_code=402,
            detail=(
                "Adjusting the joint points is part of a paid plan. The "
                "automatic measurement is unchanged."
            ),
        )
    if job.get("sport") != "bike":
        raise HTTPException(400, "Joint adjustment is available for cycling clips only for now.")
    if job.get("pair"):
        raise HTTPException(
            400, "A two-sided session cannot be adjusted yet -- adjust each clip "
            "on its own analysis.",
        )


def _frames_store_or_410(job: dict[str, Any]) -> Path:
    path = job.get("frames_store")
    if not path or not Path(path).is_file():
        raise HTTPException(
            410,
            "The measured frames of this clip are no longer stored, so its "
            "joints cannot be adjusted. Analyze the clip again to adjust it.",
        )
    return Path(path)


def _process_recompute(
    job_id: str, corrections: list[dict[str, Any]],
    mobility_profile: dict[str, Any] | None,
) -> None:
    """Run the measuring half again on the stored frames (threadpool).

    An empty ``corrections`` list is the reset: the automatic measurement,
    re-rendered. Whatever happens, the job ends ``completed`` -- on a failure
    the previous result is left standing and ``recompute_error`` says why,
    because a clip that WAS measured must not turn into one that was not.
    """
    job = JOBS.get(job_id)
    if job is None:
        return
    previous = job.get("result")
    if not ANALYSIS_SLOTS.acquire(timeout=SLOT_WAIT_TIMEOUT_S):
        job["status"], job["stage"] = "completed", None
        job["recompute_error"] = (
            "The server was busy for too long; your previous measurement is "
            "unchanged. Please try again in a few minutes."
        )
        return
    job["status"], job["stage"] = "processing", (
        _RECOMPUTE_STAGE if corrections else _RESET_STAGE
    )
    try:
        from app.services.video_analysis.biomechanics.corrections import (
            apply_corrections,
        )
        from app.services.video_analysis.landmark_store import load_frames
        from app.services.video_analysis.runner import analyze_from_frames

        frames, meta = load_frames(job["frames_store"])
        if corrections:
            apply_corrections(frames, corrections, job["sport"])
        params = job.get("params") or {}
        result = analyze_from_frames(
            frames, job["input_path"], job["sport"], job.get("cycling_position"),
            video_info=meta["video_info"],
            sampling_meta=meta.get("sampling_meta"),
            stabilizer_ctx=meta.get("stabilizer_ctx"),
            overlay_path=job.get("overlay_path"),
            recommendations=True, hide_angle_values=False, kinogram=True,
            athlete_height_cm=params.get("athlete_height_cm"),
            focus=params.get("focus"),
            mobility_profile=mobility_profile,
            camera_side_override=(
                params.get("camera_side_override") or meta.get("camera_side_override")
            ),
            corrections=corrections or None,
        )
        if result.get("status") != "completed":
            raise RuntimeError(result.get("error_message") or "re-measurement failed")
        safe = _json_safe(result)
        if safe.get("overlay_video_path"):
            safe["overlay_video_path"] = f"/jobs/{job_id}/overlay"
        safe["baseline"] = job.get("baseline") if corrections else None
        safe["plausibility_warnings"] = job.get("plausibility_warnings") or []
        job["result"] = safe
        job["corrections"] = list(corrections)
        job["recompute_error"] = None
        if not corrections:
            job["recompute_rounds"] = 0
            job["baseline"] = None
        # For the record beside the frames: an Expert Review opened later
        # should see what the athlete moved, and by how much.
        try:
            (Path(job["job_dir"]) / "corrections.json").write_text(
                json.dumps({
                    "corrections": job["corrections"],
                    "baseline": job.get("baseline"),
                    "plausibility_warnings": safe["plausibility_warnings"],
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("CORRECTIONS_RECORD_FAILED", job_id=job_id, err=str(e))
        logger.info(
            "RECOMPUTE_DONE", job_id=job_id, corrections=len(corrections),
            score=safe.get("technique_score"),
            baseline=((job.get("baseline") or {}).get("technique_score")),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("RECOMPUTE_FAILED", job_id=job_id, err=str(e))
        job["result"] = previous
        if corrections:
            # The round was not delivered, so it is not spent.
            job["recompute_rounds"] = max(0, int(job.get("recompute_rounds") or 0) - 1)
        job["recompute_error"] = (
            "We couldn't re-measure this clip with those adjustments. Your "
            "previous measurement is unchanged."
        )
    finally:
        job["status"], job["stage"] = "completed", None
        ANALYSIS_SLOTS.release()


def _process_pair_job(
    job_id: str, left_path: str, right_path: str, cycling_position: str | None,
    overlay_left: str | None, overlay_right: str | None,
    athlete_height_cm: int | None = None,
    focus: str | None = None,
    mobility_profile: dict[str, Any] | None = None,
    sport: str = "bike",
) -> None:
    """Analyse both side clips of one ride and merge them into one verdict.

    Runs inside a SINGLE analysis slot. Two clips take about twice as long, but
    claiming two slots for one job would let a pair monopolise a small server
    and could deadlock it outright when capacity is two.

    Which clip is which is not detected here -- the rider put each file in a
    labelled slot, so each analysis is told its side outright. That is the same
    override the single-clip form offers, and on a pair it is free: nobody
    uploads two clips without knowing which is which.

    Per-clip coaching prose is deliberately NOT generated (``recommendations``
    off). One report per leg is the contradiction this feature exists to end;
    the merged result gets one report written about the merged numbers, which
    is also one Gemini call instead of two.
    """
    job = JOBS.get(job_id)
    if job is None:
        return
    if not ANALYSIS_SLOTS.acquire(timeout=SLOT_WAIT_TIMEOUT_S):
        logger.warning("JOB_SLOT_TIMEOUT", job_id=job_id)
        job["status"] = "failed"
        job["error"] = (
            "The server was busy for too long and gave up on this session. "
            "Please try again in a few minutes."
        )
        return
    job["status"] = "processing"
    logger.info("PAIR_JOB_START", job_id=job_id, position=cycling_position)
    try:
        results = {}
        for n, (side, path, overlay) in enumerate(
            (("left", left_path, overlay_left), ("right", right_path, overlay_right)),
            start=1,
        ):
            job["stage"] = f"Analyzing the {side}-side clip ({n} of 2)…"
            results[side] = run_analysis(
                path, sport, cycling_position if sport == "bike" else None,
                overlay_path=overlay,
                recommendations=False,
                kinogram=True,
                athlete_height_cm=athlete_height_cm,
                focus=focus,
                mobility_profile=mobility_profile if sport == "bike" else None,
                # Bike only: a run side view sees both legs, so a user-set
                # unilateral lock there would claim a certainty the sport
                # does not have. The slots still say which clip is which.
                camera_side_override=side if sport == "bike" else None,
            )
            if results[side].get("status") != "completed":
                job["status"] = "failed"
                job["error"] = (
                    f"The {side}-side clip could not be analysed: "
                    f"{results[side].get('error_message') or 'unknown error'}"
                )
                return

        job["stage"] = "Merging both sides…"
        if sport == "run":
            # A run pair shares no rigid object to pool geometry against --
            # what it shares is the athlete. See run_session.py: the metrics
            # both clips measure independently become the session's own error
            # bar, and the merge is refused outright when either clip failed
            # the leg-identity check.
            session = build_run_session(results["left"], results["right"])
            # The score, the angle table and the findings cannot be merged --
            # a run pair shares no ruler to pool them against -- so they are
            # carried from ONE clip, and it has to be the same clip the
            # unmergeable summary values came from. It was hardcoded to the
            # left while run_session picked its own base by whichever summary
            # dict had more keys, so a session could show the left clip's score
            # above the right clip's knee angles, both labelled "both sides".
            base = results[session["base_side"]]
            merged = {k: v for k, v in base.items()
                      if k not in ("keyframe_base64", "overlay_video_path",
                                   "kinogram_base64", "ai_recommendations")}
            merged["run_session"] = session["session"]
            merged["keyframe_base64"] = base.get("keyframe_base64")
            if session["merged_summary"] is not None:
                merged["sport_specific_metrics"] = session["merged_summary"]
                merged["camera_side"] = "both"
        else:
            merged = build_pair_result(
                results["left"], results["right"], cycling_position,
                recommendations=True,
            )
        safe = _json_safe(merged)
        # BOTH overlays are kept and both are offered. The rider filmed two
        # clips; showing one and silently dropping the other was the first
        # version's mistake -- it made a two-sided session look like it had
        # only half happened.
        job["overlay_paths"] = {
            side: path for side, path in
            (("left", overlay_left), ("right", overlay_right))
            if path and Path(path).exists()
        }
        base_side = (safe.get("bilateral") or {}).get("base_side") or "left"
        chosen = job["overlay_paths"].get(base_side or "") or next(
            iter(job["overlay_paths"].values()), None,
        )
        if chosen:
            job["overlay_path"] = chosen
            safe["overlay_video_path"] = f"/jobs/{job_id}/overlay"
        if safe.get("bilateral") is not None:
            safe["bilateral"]["overlays"] = {
                side: f"/jobs/{job_id}/overlay?side={side}"
                for side in job["overlay_paths"]
            }
        job["result"] = safe
        job["preview"] = False
        job["status"] = "completed"
        job["stage"] = None
        logger.info(
            "PAIR_JOB_DONE", job_id=job_id,
            combined=(safe.get("bilateral") or {}).get("combined"),
            score=safe.get("technique_score"),
        )
        _schedule_ready_mail(job_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("PAIR_JOB_FAILED", job_id=job_id, err=str(e))
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        job["stage"] = None
        ANALYSIS_SLOTS.release()


def unsubscribe_token(user_id: int) -> str:
    """Signed, login-free proof that this address asked to stop.

    HMAC over the id with the app secret: it cannot be guessed for another
    account, it carries no password, and it survives a restart (unlike
    anything held in memory). An unsubscribe link that demands a login is a
    link most people answer with the spam button instead.
    """
    return hmac.new(
        settings.jwt_secret.encode(), f"unsub:{user_id}".encode(), hashlib.sha256,
    ).hexdigest()[:32]


# How long after a result lands we wait to see whether anybody came for it.
# The client polls every 2.5 s, so a tab that is still open marks the job seen
# almost immediately; this only has to outlast a slow network and a phone that
# woke up late.
READY_MAIL_GRACE_S = 45


def _schedule_ready_mail(job_id: str) -> None:
    """Email the athlete IF they are not there when their result lands.

    A timer, not a queue, because the job store is already in-memory and
    single-instance (see core/jobs.py): a durable notification would need
    durable jobs first, and promising delivery this cannot keep would be worse
    than the current honest gap -- a restart mid-grace loses the mail, and the
    result is still waiting in their history.
    """
    def check() -> None:
        job = JOBS.get(job_id)
        if not job or job.get("seen") or job.get("mailed"):
            return
        job["mailed"] = True
        try:
            asyncio.run(_send_ready_mail(job_id, job))
        except Exception as e:  # noqa: BLE001
            logger.warning("READY_MAIL_FAILED", job_id=job_id, err=str(e))

    timer = threading.Timer(READY_MAIL_GRACE_S, check)
    timer.daemon = True          # never hold a shutdown open for a courtesy mail
    timer.start()


async def _send_ready_mail(job_id: str, job: dict[str, Any]) -> None:
    """Look the owner up, respect their preference, send once."""
    user_id = job.get("owner_user_id")
    if not user_id:
        return
    from app.services.notify import analysis_ready_email, email_enabled, send_email
    if not email_enabled():
        return
    async with SessionLocal() as db:
        user = await db.get(User, user_id)
        if not user or not user.email or not user.notify_on_ready:
            return
        email = user.email
        token = unsubscribe_token(user.id)
        uid = user.id
    result = job.get("result") or {}
    base = (settings.public_base_url or "").rstrip("/")
    subject, text, html = analysis_ready_email(
        sport=result.get("sport_type") or "run",
        score=result.get("technique_score"),
        grade=result.get("letter_grade"),
        report_url=f"{base}/app#job={job_id}",
        unsubscribe_url=f"{base}/unsubscribe?u={uid}&t={token}",
    )
    await run_in_threadpool(send_email, email, subject, text, html)
    logger.info("READY_MAIL_SENT", job_id=job_id)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"

# Content types for the formats this app ships, declared rather than looked up.
#
# ``mimetypes`` seeds itself from the HOST -- the Windows registry here,
# /etc/mime.types in the container -- so the type of a file we serve depended on
# what the machine happened to know. It did not know webp: every image on the
# landing page, the capture guide and the sample reports went out as
# ``application/octet-stream``, which loses CDN image handling and can make a
# browser offer a download instead of a picture.
#
# Registered at import so it applies to StaticFiles and FileResponse alike.
for _mime, _ext in (
    ("image/webp", ".webp"),
    ("font/woff2", ".woff2"),
    ("image/avif", ".avif"),
):
    mimetypes.add_type(_mime, _ext)


class _RevalidatingStatic(StaticFiles):
    """Static media a deploy can actually replace.

    The shells below are served ``no-cache``, so a deploy changes the HTML for
    everyone at once -- but the media that HTML points at was mounted bare,
    with no cache headers at all. Browsers then fall back to HEURISTIC
    freshness (roughly a tenth of the file's age), which is how a hero video
    swapped in August went on playing the July clip for anyone who had already
    visited: fresh markup, stale film. The filenames here are stable by design,
    so correctness has to come from revalidation rather than from the URL.

    ETag and Last-Modified still do the real work -- the common answer is a
    304, so this costs a round-trip, not the four megabytes.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        response.headers.setdefault("Cache-Control", "no-cache, must-revalidate")
        return response


# Landing-page media (hero video, product screenshots). StaticFiles handles
# Range requests and conditional GETs, which the hero <video> relies on.
app.mount("/media", _RevalidatingStatic(directory=STATIC_DIR / "media"), name="media")


@lru_cache(maxsize=8)
def _static_version(filename: str) -> str:
    """A short hash of a static file's bytes, for cache-busting its URL.

    Cached for the process: these files do not change under a running server,
    and hashing 109 KB on every page load to learn that would be a strange way
    to save bandwidth. A deploy restarts the process, which is exactly when the
    answer can differ.
    """
    try:
        data = (STATIC_DIR / filename).read_bytes()
    except OSError:
        return "0"
    return hashlib.sha256(data).hexdigest()[:12]


def _serve_shell(request: Request, filename: str, canonical_path: str) -> Response:
    """Serve an inlined static HTML shell (landing page or SPA).

    OG/Twitter ``og:url`` and ``og:image`` are stored as root-relative paths in
    the static file and rewritten to absolute URLs here, using the request's
    own origin — so link previews resolve on any host (localhost, Railway,
    custom domain) without hardcoding a base URL.

    Behind Railway's proxy TLS terminates at the edge, so ``request.base_url``
    reports http:// even on an https:// request -- trust X-Forwarded-Proto (as
    the Academy pages already do) or the tags advertise insecure URLs.
    """
    html_doc = (STATIC_DIR / filename).read_text(encoding="utf-8")
    # Analytics goes in before the build stamp on purpose: switching it on
    # changes what the document contains, so it should change the build id and
    # bust the ETag with it -- exactly like a price change below.
    html_doc = analytics.inject(html_doc)
    # The landing page prints its prices as plain HTML -- it is a marketing
    # page, so the numbers have to be in the document for crawlers and in the
    # first paint, not fetched afterwards. Rendering them here from the same
    # catalogue the API serves is what stops the landing page from becoming a
    # fourth independent copy of the price list, which is how it last came to
    # disagree with the Terms of Service about the currency.
    #
    # Before the build stamp on purpose: a price change is a change to what the
    # page says, so it should change the build id and bust the ETag with it.
    if pricing.LANDING_TOKEN in html_doc:
        html_doc = html_doc.replace(
            pricing.LANDING_TOKEN, pricing.render_landing_pricing(),
        )
    if pricing.EXPERT_PRICE_TOKEN in html_doc:
        html_doc = html_doc.replace(
            pricing.EXPERT_PRICE_TOKEN, pricing.headline_price("expert"),
        )
    if pricing.FREE_LIMIT_TOKEN in html_doc:
        html_doc = html_doc.replace(
            pricing.FREE_LIMIT_TOKEN, str(pricing.starter_monthly_limit()),
        )
    # Point the stylesheet link at a content-hashed URL. Done BEFORE the build
    # stamp, so a CSS change also changes the document's own hash and neither
    # can go stale while the other updates.
    html_doc = html_doc.replace(
        '<link rel="stylesheet" href="/app.css">',
        f'<link rel="stylesheet" href="/app.css?v={_static_version("app.css")}">',
    )
    # Stamp a build id so every piece of result feedback records which build
    # produced it. Digested from the file as-is, before the per-host rewrites
    # below, so the same shell reports the same build on every domain.
    if "__BUILD__" in html_doc:
        html_doc = html_doc.replace(
            "__BUILD__", hashlib.sha256(html_doc.encode("utf-8")).hexdigest()[:12],
        )
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get(
        "host", request.url.netloc
    )
    origin = f"{proto}://{host}".rstrip("/")
    html_doc = html_doc.replace('content="/og-image.png"', f'content="{origin}/og-image.png"')
    html_doc = html_doc.replace(
        f'property="og:url" content="{canonical_path}"',
        f'property="og:url" content="{origin}{canonical_path}"',
    )
    html_doc = html_doc.replace(
        f'rel="canonical" href="{canonical_path}"',
        f'rel="canonical" href="{origin}{canonical_path}"',
    )
    # The whole page (JS + CSS) is inlined in this one document, so a cached
    # copy keeps running the previous build after a deploy -- which is how a
    # redesigned share card can keep rendering the old layout. This response
    # carried no cache headers at all, leaving it to browser heuristics; pin it
    # to always revalidate, with an ETag so an unchanged shell still costs a 304.
    body = html_doc.encode("utf-8")
    etag = '"' + hashlib.sha256(body).hexdigest()[:32] + '"'
    # Vary on our own account: this response can now come back in two encodings,
    # and a shared cache that missed that would hand brotli bytes to a client
    # that only reads gzip. The gzip middleware adds this header for the
    # responses IT compresses; a pre-encoded one never reaches that code.
    headers = {
        "Cache-Control": "no-cache, must-revalidate",
        "ETag": etag,
        "Vary": "Accept-Encoding",
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    # Brotli, when the client asked for it and the document has been compressed
    # before (see core/compression: keyed on this same ETag, so the cache is
    # exact by construction). Falls through to the gzip middleware whenever
    # brotli is missing, refused, or would not have helped.
    if accepts_brotli(request.headers.get("accept-encoding", "")):
        packed = brotli_encode(body, etag)
        if packed is not None:
            return Response(
                content=packed, media_type="text/html; charset=utf-8",
                headers={**headers, "Content-Encoding": "br"},
            )
    return HTMLResponse(html_doc, headers=headers)


@app.get("/", include_in_schema=False)
def root(request: Request) -> Response:
    """Marketing landing page. The app itself lives at /app; a script on the
    landing page forwards legacy hash deep-links (/#job=… etc.) to /app."""
    return _serve_shell(request, "landing.html", "/")


@app.get("/app", include_in_schema=False)
def app_shell(request: Request) -> Response:
    """The single-page application (analyze, results, history, pricing)."""
    return _serve_shell(request, "index.html", "/app")


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


# ---- PWA (installable app + Capacitor-ready asset set) -----------------------
# Icons are plain files; StaticFiles gives us content-type + conditional GETs.
app.mount("/icons", StaticFiles(directory=STATIC_DIR / "icons"), name="icons")


@app.get("/unsubscribe", include_in_schema=False)
async def unsubscribe(u: int, t: str, db: AsyncSession = Depends(get_session)) -> Response:
    """One click, no login, from the mail itself.

    Always answers the same way, whether or not the signature checked out: a
    page that says "no such account" turns an unsubscribe link into a way to
    test whether an address is registered here.
    """
    page = (
        '<!doctype html><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Unsubscribed · Flapp</title>'
        '<link rel="stylesheet" href="/tokens.css">'
        '<style>body{font-family:Manrope,Segoe UI,sans-serif;color:var(--c-ink);'
        'background:var(--c-bg);display:flex;min-height:100vh;align-items:center;'
        'justify-content:center;margin:0;padding:24px;text-align:center}'
        'div{max-width:34rem}h1{font-family:Archivo,sans-serif;color:var(--c-navy);'
        'font-size:24px;margin:0 0 12px}p{color:var(--c-ink-soft);line-height:1.6}'
        'a{color:var(--c-blue)}</style>'
        '<div><h1>Done — no more of those</h1>'
        '<p>We won\'t email you when an analysis finishes. Your reports are all '
        'still in your history.</p>'
        '<p><a href="/app">Back to Flapp</a></p></div>'
    )
    if hmac.compare_digest(t or "", unsubscribe_token(u)):
        user = await db.get(User, u)
        if user:
            user.notify_on_ready = False
            await db.commit()
            logger.info("UNSUBSCRIBED", user_id=u, kind="analysis_ready")
    return HTMLResponse(page)


@app.get("/tokens.css", include_in_schema=False)
def tokens_css() -> FileResponse:
    """The shared design tokens, linked by the SPA, the landing page and the
    Academy. Revalidated rather than cached hard: a colour that changes has to
    reach every surface on the next visit, and the file is ~2 KB."""
    return FileResponse(
        STATIC_DIR / "tokens.css",
        media_type="text/css",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


@app.get("/app.css", include_in_schema=False)
def app_css(request: Request) -> Response:
    """The SPA's stylesheet, lifted out of the document it used to be inlined in.

    Cached HARD, unlike tokens.css above, and the difference is the ``?v=``
    the shell appends: that hash is of this file's own bytes, so a changed
    stylesheet is a changed URL and there is nothing to revalidate. A CSS
    change reaches everyone on their next load; an unchanged one is never
    fetched again, which is the whole reason for taking it out of a document
    that is stamped per build.

    Served without the hash (someone typing the path) it still works -- it
    just falls back to revalidating, because then we cannot know the copy they
    hold is current.
    """
    path = STATIC_DIR / "app.css"
    versioned = request.query_params.get("v") == _static_version("app.css")
    return FileResponse(
        path,
        media_type="text/css",
        headers={
            "Cache-Control": (
                "public, max-age=31536000, immutable" if versioned
                else "no-cache, must-revalidate"
            ),
        },
    )


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest() -> FileResponse:
    """Web app manifest — makes /app installable to the home screen."""
    return FileResponse(
        STATIC_DIR / "manifest.webmanifest",
        media_type="application/manifest+json",
    )


@app.get("/sw.js", include_in_schema=False)
def service_worker() -> FileResponse:
    """Service worker, served from the root so its scope covers the whole site
    (/, /app, …). `no-cache` so browsers always revalidate and pick up a new SW
    on the next visit rather than pinning an old one."""
    return FileResponse(
        STATIC_DIR / "sw.js",
        media_type="text/javascript",
        headers={
            "Cache-Control": "no-cache, must-revalidate",
            # Explicitly allow root scope even though the file lives at /sw.js.
            "Service-Worker-Allowed": "/",
        },
    )


@app.get("/offline.html", include_in_schema=False)
def offline() -> FileResponse:
    """Offline fallback shown by the service worker when the network is down."""
    return FileResponse(STATIC_DIR / "offline.html", media_type="text/html")


@app.get("/apple-touch-icon.png", include_in_schema=False)
@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon() -> FileResponse:
    """iOS home-screen icon. iOS probes these root paths directly, so answer
    them (not just the <link> in the shell)."""
    return FileResponse(STATIC_DIR / "icons" / "apple-touch-icon.png", media_type="image/png")


@app.get("/privacy", include_in_schema=False)
def privacy() -> Response:
    """Serve the privacy policy.

    Read and rewritten rather than handed over as a file because the analytics
    snippet has to be on this page in particular: the opt-out button in the
    cookies section is wired to it, and an opt-out that only works on pages you
    have already left is not one.
    """
    doc = (STATIC_DIR / "privacy.html").read_text(encoding="utf-8")
    # Section 7 ships in two versions -- "we run analytics" and "we don't" --
    # and this is what keeps the true one. See services/analytics.py.
    doc = analytics.apply_disclosure(doc)
    return HTMLResponse(analytics.inject(doc))


@app.get("/terms", include_in_schema=False)
def terms() -> Response:
    """Serve the terms of service, with the plan table rendered from the
    catalogue rather than transcribed into it.

    This document is where a stale price stops being a typo: it is the one the
    customer is agreeing to. It is also where the last drift happened -- it
    promised Canadian dollars for months while the pricing page charged in USD.
    """
    doc = (STATIC_DIR / "terms.html").read_text(encoding="utf-8")
    if pricing.TERMS_TOKEN in doc:
        doc = doc.replace(pricing.TERMS_TOKEN, pricing.render_terms_table())
    return HTMLResponse(analytics.inject(doc))


@app.get("/health")
def health() -> dict[str, Any]:
    pending = pending_jobs()
    return {
        "status": "ok",
        "model_present": settings.model_path.exists(),
        # Deliberately alongside model_present: both are "a dependency the image
        # is supposed to bake in", and both degrade a feature rather than the
        # process when they go missing -- so neither shows up without asking.
        "overlay_encoder_present": OVERLAY_ENCODER_PRESENT,
        "active_jobs": len(pending),
        "running": sum(1 for j in pending if j["status"] == "processing"),
        "queued": sum(1 for j in pending if j["status"] == "queued"),
        "capacity": settings.max_concurrent_analyses,
        "stored_jobs": len(JOBS),
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
    camera_side: str | None = Form(
        None, description="Bike only: which side of the rider faces the "
        "camera ('left' | 'right'). Omit for automatic detection.",
    ),
    overlay: bool = Form(
        True, description="Also render the annotated overlay video.",
    ),
    focus: str | None = Form(
        None, description="Optional: what the athlete wants looked at closely.",
    ),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> JobCreated:
    """Accept a clip + params, kick off analysis, return a job id to poll."""
    ip = _client_ip(request)
    # Backpressure before anything expensive: past this depth the wait is longer
    # than anyone will sit through, so say so now instead of accepting the upload
    # and letting it rot in a queue. Checked before the quota so a rejected clip
    # never costs the caller an analysis.
    backlog = len(pending_jobs())
    if backlog >= settings.max_queued_analyses:
        logger.info("BACKPRESSURE", backlog=backlog, ip=ip)
        raise HTTPException(
            status_code=503,
            detail=(
                "We're at capacity right now — too many clips are being analyzed. "
                "Please try again in a few minutes."
            ),
            headers={"Retry-After": "120"},
        )
    await _enforce_quota(request, user, db, "clip")

    if sport not in ("run", "bike"):
        raise HTTPException(400, "sport must be 'run' or 'bike'")

    cycling_position: str | None = None
    side_override: str | None = None
    if sport == "bike":
        cycling_position = position or DEFAULT_BIKE_POSITION
        if cycling_position not in VALID_POSITIONS:
            raise HTTPException(
                400, f"invalid position; valid: {sorted(VALID_POSITIONS)}",
            )
        if camera_side:
            if camera_side not in ("left", "right"):
                raise HTTPException(
                    400, "camera_side must be 'left' or 'right'",
                )
            side_override = camera_side

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

    # The overlay video is a paid unlock. Leave ``overlay_path`` unset for a free
    # caller even when the form asked for one: the worker would skip rendering
    # anyway, and a recorded-but-missing path is what made /jobs report
    # ``overlay_failed`` -- i.e. an error message where the paywall belongs.
    free = is_free(user)
    # The one report a free account gets in a form worth judging us by. Claimed
    # here rather than on completion so two uploads in a row cannot both spend
    # it; handed back by ``preview_available`` if this run fails.
    preview = preview_available(user)
    overlay_path = str(job_dir / "overlay.mp4") if (overlay and not free) else None
    job_token = secrets.token_urlsafe(24)
    JOBS[job_id] = {
        "status": "queued",
        "sport": sport,
        "cycling_position": cycling_position,
        "result": None,
        "error": None,
        "overlay_path": overlay_path,
        "created_at": time.time(),
        "job_dir": str(job_dir),
        # Who may read this job back (see _authorized_job).
        "token": job_token,
        "owner_user_id": user.id,
        # What a re-measurement with joint corrections needs to reproduce the
        # run: the inputs that are not on disk, and the corrections so far.
        "params": {
            "athlete_height_cm": user.height_cm,
            "focus": _clean_focus(focus),
            "camera_side_override": side_override,
        },
        "frames_store": str(job_dir / "landmarks.npz"),
        "input_path": str(input_path),
        "corrections": [],
        "recompute_rounds": 0,
        "baseline": None,
    }

    if preview:
        user.free_preview_job_id = job_id
        await db.commit()
        logger.info("PREVIEW_CLAIMED", user_id=user.id, job_id=job_id)

    background_tasks.add_task(
        _process_job, job_id, str(input_path), sport, cycling_position,
        overlay_path, free, preview, user.height_cm, _clean_focus(focus),
        _stored_mobility(user) if sport == "bike" else None,
        side_override,
    )
    await _record_and_headers(response, request, user, db, "video")
    logger.info("JOB_QUEUED", job_id=job_id, sport=sport, bytes=len(data), ip=ip)
    return JobCreated(
        job_id=job_id, status="queued",
        poll_url=f"/jobs/{job_id}", job_token=job_token,
    )


@app.post("/analyze-pair", status_code=200)
async def analyze_pair_endpoint(
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    video_left: UploadFile = File(..., description="Clip filmed from the rider's LEFT side."),
    video_right: UploadFile = File(..., description="Clip filmed from the rider's RIGHT side."),
    sport: str = Form("bike", description="run | bike"),
    position: str | None = Form(
        None, description="Cycling position (bike only): road_hoods | "
        "road_drops | tt_aero | triathlon | casual.",
    ),
    overlay: bool = Form(True, description="Also render the annotated overlay video."),
    focus: str | None = Form(
        None, description="Optional: what the athlete wants looked at closely.",
    ),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> JobCreated:
    """Two side clips of one ride, analysed together into a single verdict.

    A side view can only measure the leg nearest the camera, so a complete
    bike fit takes two clips -- and measured separately they contradict each
    other by several degrees at the bottom of the stroke, where the metric is
    most sensitive and the pose model least sure of the hip. Merging them
    against one shared body ends that (see ``biomechanics.bilateral``).

    Bike only: a running side view already sees both legs, so there is nothing
    here for it to fix.

    Spends two clips of quota, because it really is two analyses, and is
    checked up front -- discovering the second one is unaffordable after the
    first has run would burn compute and still owe the rider an answer.
    """
    ip = _client_ip(request)
    backlog = len(pending_jobs())
    if backlog >= settings.max_queued_analyses:
        logger.info("BACKPRESSURE", backlog=backlog, ip=ip)
        raise HTTPException(
            status_code=503,
            detail=(
                "We're at capacity right now — too many clips are being analyzed. "
                "Please try again in a few minutes."
            ),
            headers={"Retry-After": "120"},
        )

    if is_free(user):
        # The single-clip analysis is the free tier's product; this is the one
        # that costs twice as much to run and answers a question one clip
        # cannot. Refused here rather than downgraded silently.
        raise HTTPException(
            status_code=402,
            detail=(
                "Two-sided analysis is part of a paid plan. On the free plan "
                "you can still analyze each side separately."
            ),
        )

    await _enforce_quota(request, user, db, "clip")
    _allowed, used, limit, window = await check_quota(db, user)
    if limit - used < 2:
        unit = "day" if window == "day" else "month"
        raise HTTPException(
            status_code=429,
            detail=(
                f"A two-sided session uses two clips, and you have "
                f"{max(0, limit - used)} left this {unit}. Analyze one side "
                f"now, or wait for your limit to reset."
            ),
        )

    if sport not in ("run", "bike"):
        raise HTTPException(400, "sport must be 'run' or 'bike'")
    cycling_position = None
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

    job_id = uuid.uuid4().hex[:12]
    job_dir = settings.uploads_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    total_bytes = 0
    for side, upload in (("left", video_left), ("right", video_right)):
        suffix = Path(upload.filename or "").suffix.lower() or ".mp4"
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                400, f"unsupported file type '{suffix}' for the {side} clip; "
                f"allowed: {sorted(ALLOWED_SUFFIXES)}",
            )
        data = await upload.read()
        if len(data) == 0:
            raise HTTPException(400, f"the {side} clip is empty")
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"the {side} clip is too large "
                f"(> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
            )
        path = job_dir / f"{side}{suffix}"
        path.write_bytes(data)
        paths[side] = str(path)
        total_bytes += len(data)

    overlay_left = str(job_dir / "overlay_left.mp4") if overlay else None
    overlay_right = str(job_dir / "overlay_right.mp4") if overlay else None
    job_token = secrets.token_urlsafe(24)
    JOBS[job_id] = {
        "status": "queued",
        "sport": sport,
        "cycling_position": cycling_position,
        "result": None,
        "error": None,
        # Set to the clip the merge is built on once both have run.
        "overlay_path": None,
        "created_at": time.time(),
        "job_dir": str(job_dir),
        "token": job_token,
        "owner_user_id": user.id,
        "stage": "Queued — two clips to analyze…",
        "pair": True,
    }

    background_tasks.add_task(
        _process_pair_job, job_id, paths["left"], paths["right"],
        cycling_position, overlay_left, overlay_right,
        user.height_cm, _clean_focus(focus),
        _stored_mobility(user) if sport == "bike" else None,
        sport,
    )
    # Two analyses, two clips of quota.
    await _record_and_headers(response, request, user, db, "video")
    await _record_and_headers(response, request, user, db, "video")
    logger.info(
        "PAIR_JOB_QUEUED", job_id=job_id, bytes=total_bytes, ip=ip,
        position=cycling_position,
    )
    return JobCreated(
        job_id=job_id, status="queued",
        poll_url=f"/jobs/{job_id}", job_token=job_token,
    )


@app.post("/analyze-photo", status_code=200)
async def analyze_photo_endpoint(
    request: Request,
    response: Response,
    photo: UploadFile = File(..., description="Side-view still photo (jpg/png/heic)."),
    sport: str = Form(..., description="run | bike"),
    position: str | None = Form(None, description="Cycling position (bike only)."),
    coaching: bool = Form(True, description="Include AI coaching."),
    focus: str | None = Form(
        None, description="Optional: what the athlete wants looked at closely.",
    ),
    user: User = Depends(get_current_user),
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
    preview = preview_available(user)
    from app.services.video_analysis.photo_analyzer import analyze_photo
    try:
        result = await run_in_threadpool(
            analyze_photo, data, sport, cycling_position, free and not preview,
        )
    except ValueError as e:
        # No pose detected / undecodable image -> user-actionable 422.
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("PHOTO_FAILED", err=str(e), ip=ip)
        raise HTTPException(500, "photo analysis failed")

    # Compact annotated frame for the client-side history record.
    result["keyframe_base64"] = _small_keyframe(result.get("thumbnail_base64"))

    # Free callers don't get AI coaching (it's a paid unlock) -- except on the
    # one preview, which exists precisely to show what the paid report reads
    # like.
    if coaching and (not free or preview):
        from app.services.video_analysis.llm_recommendations import (
            generate_photo_recommendations,
        )
        # The question rides in the result, which is also how it reaches the
        # report: the athlete sees what they asked, above the answer.
        result["focus"] = _clean_focus(focus)
        result["ai_recommendations"] = await run_in_threadpool(
            generate_photo_recommendations, sport, result,
        )

    if free:
        result = gate_for_access(
            result, ACCESS_PREVIEW if preview else ACCESS_TEASER,
        )
        if preview:
            # A photo has no job to name, so the claim records the analysis it
            # was spent on. Nothing hands this one back: the photo path either
            # raised already or produced a result.
            user.free_preview_job_id = f"photo:{int(time.time() * 1000)}"
            await db.commit()
            logger.info("PREVIEW_CLAIMED", user_id=user.id, kind="photo")

    await _record_and_headers(response, request, user, db, "photo")
    logger.info(
        "PHOTO_DONE", sport=sport, ip=ip,
        score=(result.get("score") or {}).get("overall_score"),
    )
    return _json_safe(result)


@app.post("/analyze-photo/recompute", status_code=200)
async def analyze_photo_recompute_endpoint(
    request: Request,
    response: Response,
    photo: UploadFile = File(..., description="The SAME photo the analysis was run on."),
    sport: str = Form(..., description="bike (the only sport with corrections so far)"),
    position: str | None = Form(None, description="Cycling position the result was filed under."),
    pose: str = Form(..., description="The `pose` object the analysis returned, verbatim (JSON)."),
    corrections: str = Form(
        "[]", description="JSON list of {landmark, dx, dy, frame_idx?} in normalized image units.",
    ),
    coaching: bool = Form(True, description="Regenerate the AI coaching for the corrected pose."),
    focus: str | None = Form(None, description="Optional: what the athlete wants looked at closely."),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Measure a photo again with the athlete's joint corrections applied.

    Synchronous like ``/analyze-photo`` and cheaper: no pose model runs, the
    signed ``pose`` from the first result is the baseline. Paid plans only, and
    it spends no analysis quota -- a correction repairs an analysis that was
    already paid for. It does count against a daily cap (see
    ``services/correction_limits``).
    """
    ip = _client_ip(request)
    if is_free(user):
        raise HTTPException(
            status_code=402,
            detail=(
                "Adjusting the joint points is part of a paid plan. The "
                "automatic measurement above is unchanged."
            ),
        )
    if sport != "bike":
        raise HTTPException(400, "joint adjustment is available for cycling photos only for now")
    cycling_position: str | None = position or None
    if cycling_position is not None and cycling_position not in VALID_POSITIONS:
        raise HTTPException(400, f"invalid position; valid: {sorted(VALID_POSITIONS)}")

    try:
        pose_obj = json.loads(pose)
        corr_obj = json.loads(corrections)
    except ValueError:
        raise HTTPException(400, "pose and corrections must be JSON")
    if not isinstance(corr_obj, list):
        raise HTTPException(400, "corrections must be a JSON list")

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

    from app.services.correction_limits import take_recompute

    allowed, used, limit = take_recompute(user.id, user.tier)
    response.headers["X-Recompute-Limit"] = str(limit)
    response.headers["X-Recompute-Remaining"] = str(max(0, limit - used))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have used today's {limit} joint adjustments. The cap "
                "resets over the next 24 hours; the automatic measurement is "
                "unchanged."
            ),
            headers={"Retry-After": "3600"},
        )

    from app.services.photo_corrections import recompute_photo

    try:
        result = await run_in_threadpool(
            recompute_photo, data, sport, cycling_position, pose_obj, corr_obj,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # noqa: BLE001
        logger.warning("PHOTO_RECOMPUTE_FAILED", err=str(e), ip=ip)
        raise HTTPException(500, "photo re-measurement failed")

    result["keyframe_base64"] = _small_keyframe(result.get("thumbnail_base64"))
    if coaching:
        from app.services.video_analysis.llm_recommendations import (
            generate_photo_recommendations,
        )
        result["focus"] = _clean_focus(focus)
        result["ai_recommendations"] = await run_in_threadpool(
            generate_photo_recommendations, sport, result,
        )

    logger.info(
        "PHOTO_RECOMPUTE", sport=sport, ip=ip, user_id=user.id,
        corrections=len(result.get("corrections") or []),
        score=(result.get("score") or {}).get("overall_score"),
        baseline=((result.get("baseline") or {}).get("score") or {}).get("overall_score"),
    )
    return _json_safe(result)


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(
    job_id: str,
    t: str | None = None,
    user: User | None = Depends(optional_user),
) -> JobStatus:
    job = authorized_job(job_id, t, user.id if user else None)
    # Somebody is here for it. Recorded before anything else so the ready-mail
    # check can tell "left the page" from "watching the spinner" -- see
    # _maybe_notify_ready.
    if job.get("status") == "completed":
        job["seen"] = True
    # Gate on read. The stored result is complete; what this caller may see
    # depends on their plan now -- so upgrading mid-analysis, or coming back to
    # a job after subscribing, shows the full thing without re-running it.
    access = access_for(user, bool(job.get("preview")))
    result = job.get("result")
    if result is not None:
        result = gate_for_access(result, access)
    overlay_ready = (
        OVERLAY_ENCODER_PRESENT
        and bool(job.get("overlay_path"))
        and Path(job["overlay_path"]).exists()
    )
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
        queue_ahead=queue_ahead(job) if job["status"] == "queued" else 0,
        error=job.get("error"),
        overlay_available=overlay_ready,
        overlay_url=f"/jobs/{job_id}/overlay" if overlay_ready else None,
        overlay_failed=overlay_failed,
        result=result,
        stage=job.get("stage"),
        corrections=(job.get("corrections") or None) if access == ACCESS_FULL else None,
        rounds_left=(
            _rounds_left(job)
            if (access == ACCESS_FULL and job.get("sport") == "bike" and job.get("frames_store"))
            else None
        ),
        recompute_error=job.get("recompute_error") if access == ACCESS_FULL else None,
    )


class CorrectionsIn(BaseModel):
    corrections: list[dict[str, Any]]


@app.get("/jobs/{job_id}/landmarks")
def job_landmarks(
    job_id: str,
    t: str | None = None,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """The measured joint points of every analyzed frame, for the editor.

    Only the near-side points the report reads (see
    ``corrections.DRAGGABLE_LANDMARKS``), as detected -- the corrections in
    force are returned beside them, and the client applies the offsets to
    draw, exactly as the server applies them to measure.
    """
    job = authorized_job(job_id, t, user.id)
    _correction_access(job, user)
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(409, "This analysis has not finished yet.")
    path = _frames_store_or_410(job)

    from app.services.video_analysis.biomechanics.corrections import (
        DRAGGABLE_LANDMARKS,
    )
    from app.services.video_analysis.landmark_store import load_frames

    frames, meta = load_frames(path)
    side = (job.get("result") or {}).get("camera_side") or "left"
    draggable = list(DRAGGABLE_LANDMARKS.get(side, ()))

    def _pt(lm: Any) -> list[Any]:
        x, y, v = float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0))
        return [
            None if math.isnan(x) else round(x, 4),
            None if math.isnan(y) else round(y, 4),
            round(0.0 if math.isnan(v) else v, 2),
        ]

    return {
        "job_id": job_id,
        "camera_side": side,
        "draggable": draggable,
        "fps": (meta.get("video_info") or {}).get("fps"),
        "frame_width": frames[0]["frame_width"] if frames else None,
        "frame_height": frames[0]["frame_height"] if frames else None,
        "frames": [
            {"i": f["frame_idx"], "p": [_pt(f["normalized_landmarks"][k]) for k in draggable]}
            for f in frames
        ],
        "corrections": job.get("corrections") or [],
        "rounds_left": _rounds_left(job),
    }


@app.post("/jobs/{job_id}/corrections", status_code=202)
async def job_corrections(
    job_id: str,
    body: CorrectionsIn,
    background_tasks: BackgroundTasks,
    response: Response,
    t: str | None = None,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Apply joint corrections to a finished analysis and measure it again.

    Cumulative: what is sent is added to what is already in force. Refused
    before anything is spent when the plan, the sport, the round count or the
    daily cap say no; the geometry check runs on the stored frames before
    the cap is spent, so an impossible move costs nothing. Then a background
    re-measurement, polled through ``/jobs/{id}`` like the original.
    """
    job = authorized_job(job_id, t, user.id)
    _correction_access(job, user)
    if job.get("status") in ("queued", "processing"):
        raise HTTPException(409, "This clip is still being measured -- wait for it to finish.")
    if not job.get("result"):
        raise HTTPException(409, "This analysis has no result to adjust.")
    path = _frames_store_or_410(job)
    if _rounds_left(job) <= 0:
        from app.services.correction_limits import PER_ANALYSIS

        raise HTTPException(
            status_code=429,
            detail=(
                f"This analysis has had its {PER_ANALYSIS} rounds of adjustment. "
                "Reset to the automatic measurement to start over, or analyze "
                "the clip again."
            ),
        )

    from app.services.video_analysis.biomechanics.corrections import (
        check_plausibility,
        normalize_corrections,
    )
    from app.services.video_analysis.landmark_store import load_frames

    side = (job.get("result") or {}).get("camera_side") or "left"
    try:
        merged = normalize_corrections(
            list(job.get("corrections") or []) + list(body.corrections), side,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    if not merged:
        raise HTTPException(422, "No adjustment to apply -- move a joint point first.")

    frames, _meta = await run_in_threadpool(load_frames, path)
    aspect = 1.0
    if frames and frames[0].get("frame_height"):
        aspect = frames[0]["frame_width"] / frames[0]["frame_height"]
    try:
        warnings = check_plausibility(frames, merged, side, aspect=aspect)
    except ValueError as e:
        raise HTTPException(422, str(e))

    from app.services.correction_limits import take_recompute

    allowed, used, limit = take_recompute(user.id, user.tier)
    response.headers["X-Recompute-Limit"] = str(limit)
    response.headers["X-Recompute-Remaining"] = str(max(0, limit - used))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"You have used today's {limit} joint adjustments. The cap "
                "resets over the next 24 hours; the current measurement is "
                "unchanged."
            ),
            headers={"Retry-After": "3600"},
        )

    if job.get("baseline") is None:
        job["baseline"] = _baseline_of(job["result"])
    job["plausibility_warnings"] = warnings
    job["recompute_rounds"] = int(job.get("recompute_rounds") or 0) + 1
    job["status"], job["stage"], job["recompute_error"] = "processing", _RECOMPUTE_STAGE, None
    background_tasks.add_task(_process_recompute, job_id, merged, _stored_mobility(user))
    logger.info(
        "RECOMPUTE_QUEUED", job_id=job_id, user_id=user.id,
        corrections=len(merged), round=job["recompute_rounds"],
    )
    return {
        "job_id": job_id, "status": "processing",
        "corrections": merged, "plausibility_warnings": warnings,
        "rounds_left": _rounds_left(job),
    }


@app.delete("/jobs/{job_id}/corrections", status_code=202)
async def job_corrections_reset(
    job_id: str,
    background_tasks: BackgroundTasks,
    response: Response,
    t: str | None = None,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Drop every correction and restore the automatic measurement.

    Costs a render (the overlay is re-drawn without the offsets), so it
    counts against the daily cap -- but not against the per-analysis rounds,
    which start over.
    """
    job = authorized_job(job_id, t, user.id)
    _correction_access(job, user)
    if job.get("status") in ("queued", "processing"):
        raise HTTPException(409, "This clip is still being measured -- wait for it to finish.")
    if not job.get("corrections"):
        return {
            "job_id": job_id, "status": job.get("status"), "corrections": [],
            "rounds_left": _rounds_left(job),
        }
    _frames_store_or_410(job)

    from app.services.correction_limits import take_recompute

    allowed, used, limit = take_recompute(user.id, user.tier)
    response.headers["X-Recompute-Limit"] = str(limit)
    response.headers["X-Recompute-Remaining"] = str(max(0, limit - used))
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"You have used today's {limit} joint adjustments; the reset can wait until tomorrow.",
            headers={"Retry-After": "3600"},
        )
    job["plausibility_warnings"] = []
    job["status"], job["stage"], job["recompute_error"] = "processing", _RESET_STAGE, None
    background_tasks.add_task(_process_recompute, job_id, [], _stored_mobility(user))
    logger.info("RECOMPUTE_RESET_QUEUED", job_id=job_id, user_id=user.id)
    return {"job_id": job_id, "status": "processing", "corrections": [], "rounds_left": _rounds_left(job)}


@app.get("/jobs/{job_id}/export")
def job_export(
    job_id: str,
    format: str = "md",
    t: str | None = None,
    user: User | None = Depends(optional_user),
) -> Response:
    """The analysis as a document another AI can read (Markdown or JSON).

    The athlete's own numbers, written so ChatGPT/Claude/Gemini interpret them
    correctly rather than from priors -- see ``services/export/ai_export.py``
    for what that costs and why the raw result JSON is not it.

    Paid feature, and the check is not cosmetic: a free result is trimmed to a
    score by ``gate_free_result`` before it is ever stored, so a free export
    would be a document containing one number and a page of caveats. Better to
    say "this is on the paid plans" than to ship that.
    """
    job = authorized_job(job_id, t, user.id if user else None)
    if is_free(user):
        raise HTTPException(
            status_code=402,
            detail="The AI export is available on paid plans.",
        )
    result = job.get("result")
    if job.get("status") != "completed" or not result:
        raise HTTPException(409, "this analysis has no result to export (yet)")

    if format == "json":
        payload = ai_export.build_json(result, job_id=job_id)
        filename = ai_export.export_filename(result, job_id=job_id, ext="json")
        return JSONResponse(
            payload,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    if format != "md":
        raise HTTPException(400, "format must be 'md' or 'json'")

    text = ai_export.build_markdown(result, job_id=job_id)
    filename = ai_export.export_filename(result, job_id=job_id, ext="md")
    logger.info("EXPORT", job_id=job_id, fmt=format, chars=len(text))
    return PlainTextResponse(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/jobs/{job_id}/overlay")
def job_overlay(
    job_id: str,
    t: str | None = None,
    side: str | None = None,
    user: User | None = Depends(optional_user),
) -> FileResponse:
    # Same gate as the poll. The token rides in the query string here because
    # this URL is consumed by <video src> and a download link, neither of which
    # can set a header.
    job = authorized_job(job_id, t, user.id if user else None)
    # A two-sided session rendered one overlay per clip, and both are the
    # athlete's own footage -- serving only the one the merge happened to be
    # built on hides half of what they filmed.
    overlay_path = job.get("overlay_path")
    if side in ("left", "right"):
        overlay_path = (job.get("overlay_paths") or {}).get(side)
    if not overlay_path or not Path(overlay_path).exists():
        raise HTTPException(404, "overlay not available for this job")
    suffix = f"_{side}" if side in ("left", "right") else ""
    return FileResponse(
        overlay_path, media_type="video/mp4",
        filename=f"{job_id}{suffix}_overlay.mp4",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
    )
