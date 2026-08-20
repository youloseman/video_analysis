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
import os
import secrets
import shutil
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
from app.api import feedback as feedback_routes
from app.api import me as me_routes
from app.core.compression import SelectiveGZipMiddleware
from app.core.config import settings
from app.core.db import get_session, init_db
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
from app.services import pricing
from app.services.export import ai_export
from app.services.notify import log_email_configuration
from app.services.result_gating import (
    ACCESS_PREVIEW,
    ACCESS_TEASER,
    gate_for_access,
    is_free,
)
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
    return bool(job and job.get("status") == "failed")


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


# --------------------------------------------------------------------------
# Background worker
# --------------------------------------------------------------------------
def _process_job(
    job_id: str, input_path: str, sport: str,
    cycling_position: str | None, overlay_path: str | None,
    free: bool = False, preview: bool = False,
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
        )
        safe = _json_safe(result)
        # Don't leak the server filesystem path; expose the API URL instead.
        if safe.get("overlay_video_path"):
            safe["overlay_video_path"] = f"/jobs/{job_id}/overlay"
        # Trim to what this caller may see (paid fields removed here, never
        # hidden client-side).
        if free:
            safe = gate_for_access(
                safe, ACCESS_PREVIEW if preview else ACCESS_TEASER,
            )
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
    finally:
        ANALYSIS_SLOTS.release()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"

# Landing-page media (hero video, product screenshots). StaticFiles handles
# Range requests and conditional GETs, which the hero <video> relies on.
app.mount("/media", StaticFiles(directory=STATIC_DIR / "media"), name="media")


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
    etag = '"' + hashlib.sha256(html_doc.encode("utf-8")).hexdigest()[:32] + '"'
    headers = {"Cache-Control": "no-cache, must-revalidate", "ETag": etag}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
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
def privacy() -> FileResponse:
    """Serve the privacy policy."""
    return FileResponse(STATIC_DIR / "privacy.html")


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
    return HTMLResponse(doc)


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
    overlay: bool = Form(
        True, description="Also render the annotated overlay video.",
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
    }

    if preview:
        user.free_preview_job_id = job_id
        await db.commit()
        logger.info("PREVIEW_CLAIMED", user_id=user.id, job_id=job_id)

    background_tasks.add_task(
        _process_job, job_id, str(input_path), sport, cycling_position,
        overlay_path, free, preview,
    )
    await _record_and_headers(response, request, user, db, "video")
    logger.info("JOB_QUEUED", job_id=job_id, sport=sport, bytes=len(data), ip=ip)
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


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(
    job_id: str,
    t: str | None = None,
    user: User | None = Depends(optional_user),
) -> JobStatus:
    job = authorized_job(job_id, t, user.id if user else None)
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
        result=job.get("result"),
    )


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
    user: User | None = Depends(optional_user),
) -> FileResponse:
    # Same gate as the poll. The token rides in the query string here because
    # this URL is consumed by <video src> and a download link, neither of which
    # can set a header.
    job = authorized_job(job_id, t, user.id if user else None)
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
