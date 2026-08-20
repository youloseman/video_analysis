"""In-memory analysis job store: capacity, queueing, and expiry.

Split out of ``app.main`` so it can be tested without importing the analysis
core (mediapipe/opencv) -- this module deletes directories, which is not code to
ship untested.

Two problems live here:

*Capacity.* MediaPipe "heavy" is CPU- and RAM-bound, and Starlette runs sync
BackgroundTasks in a threadpool that will happily start one analysis per thread
(~40) and OOM the container before finishing any of them. ``ANALYSIS_SLOTS``
bounds how many run at once; ``max_queued_analyses`` bounds how many may wait.

*Expiry.* The store is a plain dict, so nothing would ever drop an entry from
it -- a long-lived instance grows until the heap fills. ``sweep_expired_jobs``
is that reaper.

*Storage.* Files used to expire with their job, on the same six-hour clock,
because a clip existed only long enough to be analysed. It does not: an
athlete's history refers back to it, and an Expert Review bought a week later
has to be watched by a human. Uploads now live on a mounted volume with a
lifetime set by ``services.retention`` -- so ``sweep_upload_dirs`` takes the set
of ids that must survive and deletes the rest, and the two sweeps are separate
on purpose. What makes the privacy policy's retention promise true is the pair
of them running, not a redeploy wiping the disk.
"""

from __future__ import annotations

import asyncio
import hmac
import shutil
import threading
import time
from collections.abc import Collection
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


def job_dir_for(job_id: str) -> Path:
    """Where one upload's files live. Derived from the id, not stored."""
    return settings.uploads_dir / job_id


def job_file(job_id: str, stem: str) -> Path | None:
    """A file inside an upload directory, whatever extension it was saved with.

    The input keeps the extension it arrived with (``input.mov``, ``input.mp4``
    ...), so a caller that wants "the clip" cannot name the file it is after.
    """
    directory = job_dir_for(job_id)
    try:
        for candidate in sorted(directory.glob(f"{stem}.*")):
            if candidate.is_file():
                return candidate
    except OSError:
        return None
    return None


def forget_job(job_id: str) -> None:
    """Drop a job from the in-memory store, leaving its files alone.

    The store is a cache of analyses recent enough to still be polled; the
    files have their own, much longer, lifetime now (see services/retention.py).
    Conflating the two is what made every clip expire in six hours.
    """
    JOBS.pop(job_id, None)


def delete_job_files(job_id: str, job_dir: str | Path | None = None) -> bool:
    """Delete one upload directory. True if there was one to remove."""
    directory = Path(job_dir) if job_dir else job_dir_for(job_id)
    if not directory.exists():
        return False
    try:
        shutil.rmtree(directory, ignore_errors=True)
        return True
    except OSError as e:  # noqa: BLE001 -- cleanup must never break the server
        logger.warning("SWEEP_RMTREE_FAILED", job_id=job_id, err=str(e))
        return False


def discard_job(job_id: str, job: dict[str, Any] | None = None) -> None:
    """Forget a job AND delete its files -- for an explicit, immediate removal.

    Used where a person asked for the data to go (deleting a history entry or a
    whole account), which is not the same event as a clip reaching the end of
    its retention period.
    """
    job = job if job is not None else JOBS.get(job_id)
    job_dir = (job or {}).get("job_dir")
    forget_job(job_id)
    delete_job_files(job_id, job_dir)


def sweep_expired_jobs(
    now: float | None = None, ttl_hours: float | None = None,
) -> int:
    """Forget jobs past the TTL. Returns how many went.

    **Files are not touched here.** A job is dropped from the store once nobody
    could reasonably still be polling it, but the clip it produced may have
    weeks left -- an athlete's history entry, or an Expert Review somebody has
    paid for and not yet received. ``sweep_upload_dirs`` owns deletion, and it
    asks the database rather than the clock.

    Unfinished jobs are never swept: an analysis slower than the TTL would
    otherwise lose the store entry it is about to write its result into.
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
        forget_job(jid)
    return len(stale)


def sweep_upload_dirs(
    keep: Collection[str] | None = None,
    now: float | None = None,
    uploads_dir: Path | None = None,
    ttl_hours: float | None = None,
) -> int:
    """Delete stored uploads that nothing is entitled to keep.

    ``keep`` is the set of job ids retention says must survive (see
    ``services.retention.job_ids_to_keep``). A directory survives if it is in
    that set, if a live job is still using it, or if it is too young to judge --
    the grace window covers the gap between an analysis finishing and its owner
    saving it to history, during which nothing in the database refers to it yet.

    Passing no ``keep`` set deletes everything outside the grace window, which
    is the old behaviour and the right one for a caller that has no database:
    it must therefore never be the default in production. The startup sweep and
    the reaper both pass one.
    """
    ttl_s = (settings.job_ttl_hours if ttl_hours is None else ttl_hours) * 3600
    root = settings.uploads_dir if uploads_dir is None else uploads_dir
    keep = keep or ()
    if ttl_s <= 0 or not root.exists():
        return 0
    now = now if now is not None else time.time()
    removed = 0
    for entry in _iter_upload_dirs(root):
        if entry.name in JOBS or entry.name in keep:
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


async def retained_job_ids() -> set[str] | None:
    """Ask the database which stored uploads are still spoken for.

    ``None`` means the lookup failed and nothing should be deleted this round.
    That is the deliberate direction to fail in: sweeping on an unanswered
    question would delete footage somebody has paid to have reviewed, while not
    sweeping costs disk until the next pass ten minutes later.

    Imported lazily so this module keeps importing in tests that run with no
    database and no ML stack.
    """
    try:
        from app.core.db import SessionLocal
        from app.services.retention import job_ids_to_keep

        async with SessionLocal() as session:
            return await job_ids_to_keep(session)
    except Exception as e:  # noqa: BLE001 -- the reaper must never die
        logger.warning("RETENTION_QUERY_FAILED", err=str(e))
        return None


async def sweeper_loop() -> None:
    """Background reaper; runs for the lifetime of the app."""
    from fastapi.concurrency import run_in_threadpool

    interval = max(30, settings.job_sweep_interval_s)
    while True:
        try:
            await asyncio.sleep(interval)
            jobs = await run_in_threadpool(sweep_expired_jobs)
            # Files outlive their job now, so what may be deleted is a question
            # for the database. No answer -> no deletion this round.
            keep = await retained_job_ids()
            dirs = 0 if keep is None else await run_in_threadpool(
                sweep_upload_dirs, keep,
            )
            if jobs or dirs:
                logger.info(
                    "SWEEP", jobs_forgotten=jobs, dirs_deleted=dirs,
                    retained=(len(keep) if keep is not None else None),
                    live=len(JOBS),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- the reaper must never die
            logger.warning("SWEEP_FAILED", err=str(e))
