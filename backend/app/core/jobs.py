"""In-memory analysis job store: capacity, queueing, and expiry.

Split out of ``app.main`` so it can be tested without importing the analysis
core (mediapipe/opencv) -- this module deletes directories, which is not code to
ship untested.

Two problems live here:

*Capacity.* MediaPipe "heavy" is CPU- and RAM-bound, and Starlette runs sync
BackgroundTasks in a threadpool that will happily start one analysis per thread
(~40) and OOM the container before finishing any of them. ``ANALYSIS_SLOTS``
bounds how many run at once; ``max_queued_analyses`` bounds how many may wait.

*Expiry.* The store is a plain dict and uploads land on the container's
ephemeral disk, so nothing would ever delete either one -- a long-lived instance
grows until the heap or the disk fills. ``sweep_expired_jobs`` and
``sweep_orphan_upload_dirs`` are the reaper, and they are also what makes the
retention promise in the privacy policy true by something other than a redeploy.
"""

from __future__ import annotations

import asyncio
import hmac
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import structlog
from fastapi import HTTPException

from app.core.config import settings

logger = structlog.get_logger()

# job_id -> job dict. Not shared across workers and not persisted; single
# replica only (see DEPLOY.md).
JOBS: dict[str, dict[str, Any]] = {}

# Jobs waiting on this stay "queued" and report their place in line.
ANALYSIS_SLOTS = threading.BoundedSemaphore(settings.max_concurrent_analyses)
# Give up rather than wedge a threadpool thread forever if a slot never frees.
SLOT_WAIT_TIMEOUT_S = 30 * 60

PENDING_STATES = ("queued", "processing")


def pending_jobs() -> list[dict[str, Any]]:
    """Accepted jobs that have not finished (queued or running)."""
    return [j for j in JOBS.values() if j.get("status") in PENDING_STATES]


def queue_ahead(job: dict[str, Any]) -> int:
    """How many unfinished jobs were accepted before this one (0 = next up).

    Reported to the client so "queued" can say how long the line actually is,
    instead of spinning with no explanation while the semaphore is full.
    """
    mine = job.get("created_at") or 0.0
    return sum(
        1 for j in pending_jobs()
        if j is not job and (j.get("created_at") or 0.0) < mine
    )


def authorized_job(
    job_id: str, token: str | None, user_id: int | None,
) -> dict[str, Any]:
    """Fetch a job the caller is entitled to read, or raise 404.

    A job holds someone's footage, so possession of the id is not enough: ids
    travel in a URL fragment and get pasted around. Two ways in:

    * the account that created it -- survives a cleared browser, and a signed-in
      athlete should not lose their own result, or
    * the capability token handed out at upload, which is the only route for
      anonymous callers since they have no account to be recognised by.

    Failures are 404 rather than 403: a 403 would confirm the id exists.
    """
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    owner_id = job.get("owner_user_id")
    if owner_id is not None and user_id is not None and user_id == owner_id:
        return job
    expected = job.get("token")
    if expected and token and hmac.compare_digest(str(token), str(expected)):
        return job
    logger.info("JOB_FORBIDDEN", job_id=job_id, user_id=user_id)
    raise HTTPException(404, "unknown job_id")


def discard_job(job_id: str, job: dict[str, Any] | None = None) -> None:
    """Drop a job from the store and delete its upload directory."""
    job = job if job is not None else JOBS.get(job_id)
    job_dir = (job or {}).get("job_dir")
    JOBS.pop(job_id, None)
    if not job_dir:
        return
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except OSError as e:  # noqa: BLE001 -- cleanup must never break the server
        logger.warning("SWEEP_RMTREE_FAILED", job_id=job_id, err=str(e))


def sweep_expired_jobs(
    now: float | None = None, ttl_hours: float | None = None,
) -> int:
    """Delete jobs (and their files) past the TTL. Returns how many went.

    A job is kept for ``job_ttl_hours`` after it was accepted so the client can
    still poll its result and download the overlay. Unfinished jobs are never
    swept -- an analysis slower than the TTL would otherwise lose its own output
    directory mid-render.

    ``now`` and ``ttl_hours`` default to the wall clock and the configured TTL;
    both are parameters so the behaviour can be exercised without waiting hours
    or mutating the (frozen) settings object.
    """
    ttl_s = (settings.job_ttl_hours if ttl_hours is None else ttl_hours) * 3600
    if ttl_s <= 0:
        return 0
    now = now if now is not None else time.time()
    stale = [
        jid for jid, j in list(JOBS.items())
        if j.get("status") not in PENDING_STATES
        and now - (j.get("created_at") or 0.0) > ttl_s
    ]
    for jid in stale:
        discard_job(jid)
    return len(stale)


def sweep_orphan_upload_dirs(
    now: float | None = None,
    uploads_dir: Path | None = None,
    ttl_hours: float | None = None,
) -> int:
    """Delete upload directories with no live job behind them.

    The job store does not survive a restart but the files may (a mounted
    volume, or a container that restarts without being rebuilt), which would
    strand every upload from the previous process on disk forever.
    """
    ttl_s = (settings.job_ttl_hours if ttl_hours is None else ttl_hours) * 3600
    root = settings.uploads_dir if uploads_dir is None else uploads_dir
    if ttl_s <= 0 or not root.exists():
        return 0
    now = now if now is not None else time.time()
    removed = 0
    for entry in _iter_upload_dirs(root):
        if entry.name in JOBS:
            continue
        try:
            if now - entry.stat().st_mtime <= ttl_s:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        except OSError as e:  # noqa: BLE001
            logger.warning("SWEEP_ORPHAN_FAILED", path=str(entry), err=str(e))
    return removed


def _iter_upload_dirs(root: Path) -> list[Path]:
    try:
        return [p for p in root.iterdir() if p.is_dir()]
    except OSError:
        return []


async def sweeper_loop() -> None:
    """Background reaper; runs for the lifetime of the app."""
    from fastapi.concurrency import run_in_threadpool

    interval = max(30, settings.job_sweep_interval_s)
    while True:
        try:
            await asyncio.sleep(interval)
            jobs = await run_in_threadpool(sweep_expired_jobs)
            orphans = await run_in_threadpool(sweep_orphan_upload_dirs)
            if jobs or orphans:
                logger.info("SWEEP", jobs=jobs, orphan_dirs=orphans, live=len(JOBS))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- the reaper must never die
            logger.warning("SWEEP_FAILED", err=str(e))
