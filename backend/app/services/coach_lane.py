"""The coach's lane: batch files and intake clips, one at a time, in the background.

A coach drops ten clips and goes for a coffee. Queued the ordinary way those ten
would hold both analysis slots for a quarter of an hour and answer everybody else
with "we're at capacity". So they run here instead: one task, one clip at a time,
so the lane can never hold more than one slot and an athlete on the Analyze page
always has the other. Lane clips also wait outside the job store until their turn,
which keeps them out of ``max_queued_analyses``.

Each clip is checked first, on the coach's own GO / NO-GO grid. A NO-GO is not
analysed and costs no quota: it comes back as "re-film", with the reason, and the
coach can still say run it anyway. A clip we fail to check is run, not refused:
the check exists to save time, not to block anybody.

State lives in ``coach_uploads``. Railway restarts the service on every deploy, so
``lane_loop`` picks up whatever was left open when it starts.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from typing import Any

import structlog
from sqlalchemy import select
from starlette.concurrency import run_in_threadpool

from app.models.analysis import Analysis
from app.models.athlete import Athlete
from app.models.coach_upload import (
    OPEN_STATUSES,
    SOURCE_INTAKE,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_NO_GO,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SCANNING,
    CoachUpload,
)
from app.models.user import User, has_coach

logger = structlog.get_logger()

# Ten is what fits one sitting: a squad's clips from one session. It is also
# roughly what the lane clears in twenty minutes, which is the coffee.
MAX_BATCH_FILES = 10
# Past this many clips waiting across all coaches, an upload is told to wait.
MAX_LANE_BACKLOG = 60
SCAN_FRAMES = 6
SCAN_LONG_EDGE = 640

_queue: asyncio.Queue[int] = asyncio.Queue()


def _now_ms() -> int:
    return int(time.time() * 1000)


def enqueue(upload_id: int) -> None:
    _queue.put_nowait(upload_id)


def scan_clip(path: str, sport: str | None, thresholds: dict[str, Any] | None) -> dict[str, Any] | None:
    """GO / NO-GO on a stored clip, from a few frames cut out of it here.

    The browser does this for a clip on the Analyze page. Nobody is at a browser
    now, and the server can decode what Chrome cannot (an iPhone's HEVC). None
    when it could not look: the clip then runs as if nothing was said.
    """
    import cv2

    from app.services.prescan import scan_images

    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if total <= 0 or w <= 0 or h <= 0:
            return None
        images: list[bytes] = []
        for i in range(SCAN_FRAMES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / SCAN_FRAMES))
            ok, frame = cap.read()
            if not ok:
                continue
            k = SCAN_LONG_EDGE / max(frame.shape[:2])
            if k < 1:
                frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            done, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if done:
                images.append(jpg.tobytes())
    finally:
        cap.release()
    if not images:
        return None
    return (scan_images(images, video_w=w, video_h=h, thresholds=thresholds) or {}).get("capture")


def _shrink(path: str) -> None:
    from app.services.clip_store import compress_clip

    try:
        compress_clip(path)
    except Exception as e:  # noqa: BLE001 -- the clip is kept as it is
        logger.warning("LANE_SHRINK_FAILED", err=str(e))


def _run_job(job_id: str, input_path: str, up: dict[str, Any], coach_id: int) -> dict[str, Any]:
    """Register the job and run it. Blocking; called in a worker thread."""
    from pathlib import Path

    from app.core.jobs import JOBS
    from app.main import _process_job

    job_dir = str(Path(input_path).parent)
    overlay_path = str(Path(job_dir) / "overlay.mp4")
    JOBS[job_id] = {
        "status": "queued", "sport": up["sport"], "cycling_position": up["position"],
        "result": None, "error": None, "overlay_path": overlay_path,
        "created_at": time.time(), "job_dir": job_dir,
        "token": secrets.token_urlsafe(24), "owner_user_id": coach_id,
        # Nobody is waiting on a page for this one, and the coach hears once per
        # batch rather than once per clip (see _tell_the_coach).
        "mailed": True,
        "params": {"athlete_height_cm": None, "focus": None,
                   "camera_side_override": up["camera_side"], "crank_length_mm": None},
        "frames_store": str(Path(job_dir) / "landmarks.npz"),
        "input_path": input_path, "corrections": [], "recompute_rounds": 0, "baseline": None,
    }
    _process_job(
        job_id, input_path, up["sport"], up["position"], overlay_path,
        False, False, None, None, None, up["camera_side"], None,
    )
    return JOBS.get(job_id) or {}


def _file_entry(entry: dict[str, Any], job: dict[str, Any]) -> tuple[dict[str, Any], str | None, bool]:
    """``(entry, keyframe, ok)`` for a finished run.

    The guest flow's filing, with two differences: a coach is on a paid plan, so
    the frame kept is the real one and not the teaser's; and a run that failed
    stays held out of History, because only the batch screen has anything to say
    about it.
    """
    from app.services.guest import finish_entry

    filed, frame = finish_entry(entry, job)
    ok = not filed.get("analysisFailed")
    if ok:
        filed.pop("coachHold", None)
        # The app rebuilds this entry from the stored result the first time it
        # sees it (index.html rehydrateServerFiled): trends and angles included.
        filed["serverFiled"] = True
        frame = ((job.get("result") or {}).get("keyframe_base64")) or frame
    return filed, frame, ok


async def run_upload(upload_id: int) -> None:
    """Take one clip through the lane. Never holds a database session while
    the analysis runs: that is a minute or two per clip."""
    from app.api.coach import capture_for
    from app.api.me import _attach_result
    from app.core import db as core_db
    from app.core.jobs import job_file
    from app.services.usage_limits import check_quota, record_usage

    async with core_db.SessionLocal() as db:
        up = await db.get(CoachUpload, upload_id)
        if up is None or up.status != STATUS_QUEUED:
            return
        coach = await db.get(User, up.coach_user_id)
        if coach is None or not has_coach(coach.tier):
            await _close(db, up, coach, STATUS_FAILED, "The Coach plan on this account has ended.")
            return
        clip = job_file(up.job_id, "input")
        if clip is None:
            await _close(db, up, coach, STATUS_FAILED, "The clip is no longer stored.")
            return
        if not up.run_anyway:
            up.status = STATUS_SCANNING
            await db.commit()
            bars = await capture_for(db, coach.id)
            try:
                capture = await run_in_threadpool(scan_clip, str(clip), up.sport, bars)
            except Exception as e:  # noqa: BLE001 -- the check must never block a clip
                logger.warning("LANE_SCAN_FAILED", upload_id=upload_id, err=str(e))
                capture = None
            if capture:
                up.reasons = capture.get("reasons") or None
                if capture.get("verdict") == "no_go":
                    # It may wait weeks for "run anyway" or a delete, and an
                    # iPhone original is 60 MB against about 2 MB at 1080p.
                    await run_in_threadpool(_shrink, str(clip))
                    await _close(db, up, coach, STATUS_NO_GO)
                    return
        allowed, _used, limit, _window = await check_quota(db, coach)
        if not allowed:
            await _close(db, up, coach, STATUS_FAILED,
                         f"You've used all {limit} analyses on your plan for now.")
            return
        await record_usage(db, coach.id, "clip")
        up.status = STATUS_RUNNING
        await db.commit()
        params = {"sport": up.sport, "position": up.bike_position, "camera_side": up.camera_side}
        job_id, coach_id, client_id = up.job_id, coach.id, up.client_id

    logger.info("LANE_START", upload_id=upload_id, job_id=job_id)
    try:
        job = await run_in_threadpool(_run_job, job_id, str(clip), params, coach_id)
    except Exception as e:  # noqa: BLE001 -- a bad clip must not stop the lane
        logger.warning("LANE_RUN_FAILED", upload_id=upload_id, err=str(e))
        job = {"status": "failed", "error": "The analysis could not read this clip."}

    async with core_db.SessionLocal() as db:
        up = await db.get(CoachUpload, upload_id)
        coach = await db.get(User, coach_id)
        if up is None:
            return
        row = (await db.execute(select(Analysis).where(
            Analysis.user_id == coach_id, Analysis.client_id == client_id,
        ))).scalar_one_or_none()
        if row is None:     # the coach deleted it while it ran
            await _close(db, up, coach, STATUS_FAILED, "This clip was removed.")
            return
        entry, frame, ok = _file_entry(dict(row.data or {}), job)
        row.data = entry
        row.score = entry.get("score")
        if frame:
            row.keyframe = frame
        if not ok:
            await _close(db, up, coach, STATUS_FAILED, entry.get("analysisFailed"))
            return
        row.athlete_id = up.athlete_id
        row.profile_id = up.profile_id
        _attach_result(row, job_id)
        await _close(db, up, coach, STATUS_DONE)


async def _close(db: Any, up: CoachUpload, coach: User | None, status: str,
                 error: str | None = None) -> None:
    up.status = status
    up.finished_at_ms = _now_ms()
    if error:
        up.error = str(error)[:300]
    await db.commit()
    logger.info("LANE_DONE", upload_id=up.id, status=status)
    if coach is not None:
        await _tell_the_coach(db, up, coach)


async def _tell_the_coach(db: Any, up: CoachUpload, coach: User) -> None:
    """One mail per intake clip; one per batch, when the last file is through."""
    from app.services import notify

    try:
        if up.source == SOURCE_INTAKE:
            name = await db.scalar(select(Athlete.name).where(Athlete.id == up.athlete_id))
            who = name or "An athlete"
            msg = {
                STATUS_DONE: (f"{who}'s clip is analysed", "The report is in your squad."),
                STATUS_NO_GO: (f"{who}'s clip needs filming again",
                               "It didn't pass your framing check, so it wasn't analysed. You can still run it anyway."),
            }.get(up.status, (f"{who}'s clip couldn't be read", up.error or "The analysis could not read it."))
        else:
            rows = (await db.execute(select(CoachUpload.status).where(
                CoachUpload.coach_user_id == coach.id, CoachUpload.batch_id == up.batch_id,
            ))).scalars().all()
            if any(s in OPEN_STATUSES for s in rows):
                return
            done = sum(1 for s in rows if s == STATUS_DONE)
            refilm = sum(1 for s in rows if s == STATUS_NO_GO)
            failed = sum(1 for s in rows if s == STATUS_FAILED)
            parts = [f"{done} of {len(rows)} analysed"]
            if refilm:
                parts.append(f"{refilm} to film again")
            if failed:
                parts.append(f"{failed} couldn't be read")
            msg = (f"Your batch is ready: {done} of {len(rows)}", ", ".join(parts) + ".")
        subject, line = msg
        url = f"{notify.base_url()}/app#batch"
        esc = notify._esc_html
        notify.send_email(
            coach.email, subject,
            f"{subject}\n\n{line}\n\nOpen it: {url}\n",
            '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:16px;'
            'line-height:1.6;color:#14294B;max-width:520px">'
            '<p style="font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:#2F6DE0;'
            'margin:0 0 6px">Flapp</p>'
            f'<h1 style="font-size:22px;margin:0 0 8px">{esc(subject)}</h1>'
            f'<p style="margin:0 0 22px;color:#5A6478">{esc(line)}</p>'
            f'<a href="{esc(url)}" style="background:#2F6DE0;color:#fff;text-decoration:none;'
            'padding:12px 20px;border-radius:8px;display:inline-block;font-weight:600">Open it</a></div>',
        )
    except Exception as e:  # noqa: BLE001 -- a mail problem must not stop the lane
        logger.warning("LANE_MAIL_FAILED", upload_id=up.id, err=str(e))


async def requeue_open() -> int:
    """What the last process left open goes back in line. A clip it was halfway
    through starts again: the analysis writes nothing until it finishes."""
    from app.core import db as core_db

    async with core_db.SessionLocal() as db:
        rows = (await db.execute(
            select(CoachUpload).where(CoachUpload.status.in_(OPEN_STATUSES)).order_by(CoachUpload.id)
        )).scalars().all()
        for up in rows:
            up.status = STATUS_QUEUED
        await db.commit()
        ids = [up.id for up in rows]
    for i in ids:
        enqueue(i)
    if ids:
        logger.info("LANE_REQUEUED", n=len(ids))
    return len(ids)


async def lane_loop() -> None:
    try:
        await requeue_open()
    except Exception as e:  # noqa: BLE001 -- start anyway; new clips still run
        logger.warning("LANE_REQUEUE_FAILED", err=str(e))
    while True:
        upload_id = await _queue.get()
        try:
            await run_upload(upload_id)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- one clip must not stop the lane
            logger.warning("LANE_ITEM_FAILED", upload_id=upload_id, err=str(e))
