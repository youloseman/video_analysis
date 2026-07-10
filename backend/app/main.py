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
import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from app.core.config import settings
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

# In-memory job store. job_id -> job dict.
JOBS: dict[str, dict[str, Any]] = {}

app = FastAPI(
    title="Video Technique Analysis",
    version="0.3.0",
    description="Side-view running + cycling technique analysis (standalone).",
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
    result: dict[str, Any] | None = None


# --------------------------------------------------------------------------
# Background worker
# --------------------------------------------------------------------------
def _process_job(
    job_id: str, input_path: str, sport: str,
    cycling_position: str | None, overlay_path: str | None,
) -> None:
    """Run the analysis for a job (executed in a threadpool by BackgroundTasks)."""
    job = JOBS.get(job_id)
    if job is None:
        return
    job["status"] = "processing"
    logger.info("JOB_START", job_id=job_id, sport=sport, position=cycling_position)
    try:
        result = run_analysis(
            input_path, sport, cycling_position, overlay_path=overlay_path,
        )
        safe = _json_safe(result)
        # Don't leak the server filesystem path; expose the API URL instead.
        if safe.get("overlay_video_path"):
            safe["overlay_video_path"] = f"/jobs/{job_id}/overlay"
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
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


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
    video: UploadFile = File(..., description="Side-view video clip (mp4/mov/...)."),
    sport: str = Form(..., description="run | bike"),
    position: str | None = Form(
        None, description="Cycling position (bike only): "
        "road_hoods | road_drops | tt_aero | triathlon | casual.",
    ),
    overlay: bool = Form(
        True, description="Also render the annotated overlay video.",
    ),
) -> JobCreated:
    """Accept a clip + params, kick off analysis, return a job id to poll."""
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
        _process_job, job_id, str(input_path), sport, cycling_position, overlay_path,
    )
    logger.info("JOB_QUEUED", job_id=job_id, sport=sport, bytes=len(data))
    return JobCreated(job_id=job_id, status="queued", poll_url=f"/jobs/{job_id}")


@app.get("/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str) -> JobStatus:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    overlay_ready = bool(job.get("overlay_path")) and Path(job["overlay_path"]).exists()
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        sport=job.get("sport"),
        cycling_position=job.get("cycling_position"),
        error=job.get("error"),
        overlay_available=overlay_ready,
        overlay_url=f"/jobs/{job_id}/overlay" if overlay_ready else None,
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
