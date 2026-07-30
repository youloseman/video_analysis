"""Job store: expiry (which deletes directories) and queue accounting.

The sweeper is the only thing that ever frees disk or heap on a long-lived
instance, and it is also the only code in the service that calls ``rmtree``.
Both halves of that need to be pinned down: that it deletes what it should, and
that it does not delete a job that is still being analyzed.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core import jobs as jobstore
from app.core.config import settings

HOUR = 3600.0


@pytest.fixture(autouse=True)
def clean_store(tmp_path):
    """Empty store + a throwaway uploads dir for every test.

    ``settings`` is a frozen dataclass, so the directory is passed to the sweeps
    explicitly rather than patched onto it.
    """
    jobstore.JOBS.clear()
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    yield uploads
    jobstore.JOBS.clear()


def make_job(job_id: str, *, status="completed", age_h=0.0, uploads: Path | None = None):
    """Register a job with a real directory on disk, aged as requested."""
    job_dir = None
    if uploads is not None:
        job_dir = uploads / job_id
        job_dir.mkdir()
        (job_dir / "input.mp4").write_bytes(b"x" * 32)
    jobstore.JOBS[job_id] = {
        "status": status,
        "created_at": time.time() - age_h * HOUR,
        "job_dir": str(job_dir) if job_dir else None,
    }
    return job_dir


# --------------------------------------------------------------------------
# Expiry
# --------------------------------------------------------------------------
def test_a_finished_job_past_the_ttl_is_deleted_with_its_files(clean_store):
    job_dir = make_job("old", age_h=settings.job_ttl_hours + 1, uploads=clean_store)
    assert job_dir.exists()

    assert jobstore.sweep_expired_jobs() == 1
    assert "old" not in jobstore.JOBS
    assert not job_dir.exists()


def test_a_job_inside_the_ttl_is_left_alone(clean_store):
    job_dir = make_job("fresh", age_h=settings.job_ttl_hours - 1, uploads=clean_store)

    assert jobstore.sweep_expired_jobs() == 0
    assert "fresh" in jobstore.JOBS
    assert job_dir.exists()


@pytest.mark.parametrize("status", ["queued", "processing"])
def test_an_unfinished_job_is_never_swept(clean_store, status):
    """An analysis slower than the TTL must not have its own output directory
    deleted out from under it mid-render."""
    job_dir = make_job("slow", status=status, age_h=settings.job_ttl_hours * 5,
                       uploads=clean_store)

    assert jobstore.sweep_expired_jobs() == 0
    assert "slow" in jobstore.JOBS
    assert job_dir.exists()


def test_a_failed_job_expires_like_a_completed_one(clean_store):
    make_job("bad", status="failed", age_h=settings.job_ttl_hours + 1, uploads=clean_store)
    assert jobstore.sweep_expired_jobs() == 1


def test_ttl_of_zero_disables_expiry(clean_store):
    """The escape hatch for debugging a live instance -- it must actually
    disable the reaper, not just make everything expire instantly."""
    make_job("ancient", age_h=10_000, uploads=clean_store)
    assert jobstore.sweep_expired_jobs(ttl_hours=0) == 0
    assert "ancient" in jobstore.JOBS


def test_sweeping_a_job_whose_files_are_already_gone_does_not_raise(clean_store):
    job_dir = make_job("vanished", age_h=settings.job_ttl_hours + 1, uploads=clean_store)
    (job_dir / "input.mp4").unlink()
    job_dir.rmdir()
    assert jobstore.sweep_expired_jobs() == 1
    assert "vanished" not in jobstore.JOBS


def test_discard_job_tolerates_a_job_with_no_directory():
    jobstore.JOBS["nodir"] = {"status": "failed", "created_at": 0.0, "job_dir": None}
    jobstore.discard_job("nodir")
    assert "nodir" not in jobstore.JOBS


# --------------------------------------------------------------------------
# Orphan directories (files from a previous process; the store does not survive
# a restart but a mounted volume does)
# --------------------------------------------------------------------------
def test_an_orphan_directory_older_than_the_ttl_is_deleted(clean_store):
    orphan = clean_store / "leftover"
    orphan.mkdir()
    (orphan / "input.mp4").write_bytes(b"x")
    old = time.time() - (settings.job_ttl_hours + 1) * HOUR
    os.utime(orphan, (old, old))

    assert jobstore.sweep_orphan_upload_dirs(uploads_dir=clean_store) == 1
    assert not orphan.exists()


def test_a_recent_orphan_is_kept(clean_store):
    orphan = clean_store / "recent"
    orphan.mkdir()
    assert jobstore.sweep_orphan_upload_dirs(uploads_dir=clean_store) == 0
    assert orphan.exists()


def test_a_directory_with_a_live_job_is_never_treated_as_an_orphan(clean_store):
    """Same id in the store = someone is still polling it, however old it looks."""
    job_dir = make_job("live", status="processing", uploads=clean_store)
    old = time.time() - (settings.job_ttl_hours + 5) * HOUR
    os.utime(job_dir, (old, old))

    assert jobstore.sweep_orphan_upload_dirs(uploads_dir=clean_store) == 0
    assert job_dir.exists()


def test_orphan_sweep_survives_a_missing_uploads_dir(clean_store):
    assert jobstore.sweep_orphan_upload_dirs(uploads_dir=clean_store / "nope") == 0


# --------------------------------------------------------------------------
# Queue accounting
# --------------------------------------------------------------------------
def test_pending_counts_only_unfinished_jobs():
    for jid, status in [
        ("a", "queued"), ("b", "processing"), ("c", "completed"), ("d", "failed"),
    ]:
        make_job(jid, status=status)
    assert len(jobstore.pending_jobs()) == 2


def test_queue_position_counts_only_the_jobs_ahead():
    make_job("first", status="processing", age_h=0.3)
    make_job("second", status="queued", age_h=0.2)
    make_job("third", status="queued", age_h=0.1)

    assert jobstore.queue_ahead(jobstore.JOBS["first"]) == 0
    assert jobstore.queue_ahead(jobstore.JOBS["second"]) == 1
    assert jobstore.queue_ahead(jobstore.JOBS["third"]) == 2


def test_finished_jobs_do_not_inflate_the_queue():
    """A day of completed jobs sitting in the store must not make a new upload
    look like it is 200th in line."""
    for i in range(5):
        make_job(f"done{i}", status="completed", age_h=1)
    make_job("mine", status="queued", age_h=0.1)
    assert jobstore.queue_ahead(jobstore.JOBS["mine"]) == 0


def test_concurrency_cap_matches_the_configured_capacity():
    """Guards against the semaphore being built from a stale default."""
    acquired = 0
    try:
        while jobstore.ANALYSIS_SLOTS.acquire(blocking=False):
            acquired += 1
    finally:
        for _ in range(acquired):
            jobstore.ANALYSIS_SLOTS.release()
    assert acquired == settings.max_concurrent_analyses


# --------------------------------------------------------------------------
# Read authorization
#
# A job id is not a secret: it rides in the URL fragment and gets pasted into
# chats. The job behind it is someone's footage, so the id alone must not open it.
# --------------------------------------------------------------------------
def owned_job(job_id="j1", *, token="tok-secret", owner=None):
    jobstore.JOBS[job_id] = {
        "status": "completed", "created_at": time.time(), "job_dir": None,
        "token": token, "owner_user_id": owner,
    }
    return jobstore.JOBS[job_id]


def test_the_right_token_opens_the_job():
    owned_job()
    assert jobstore.authorized_job("j1", "tok-secret", None) is jobstore.JOBS["j1"]


def test_the_owning_account_opens_the_job_without_a_token():
    """A signed-in athlete who cleared sessionStorage must not lose their result."""
    owned_job(owner=7)
    assert jobstore.authorized_job("j1", None, 7) is jobstore.JOBS["j1"]


@pytest.mark.parametrize("token", [None, "", "wrong-token"])
def test_a_bad_token_is_refused(token):
    owned_job()
    with pytest.raises(HTTPException) as e:
        jobstore.authorized_job("j1", token, None)
    assert e.value.status_code == 404


def test_a_different_account_cannot_read_someone_elses_job():
    owned_job(owner=7)
    with pytest.raises(HTTPException) as e:
        jobstore.authorized_job("j1", None, 8)
    assert e.value.status_code == 404


def test_being_signed_in_does_not_open_an_anonymous_job():
    """No owner recorded -> only the token works, whoever is asking."""
    owned_job(owner=None)
    with pytest.raises(HTTPException):
        jobstore.authorized_job("j1", None, 7)


def test_an_unknown_id_and_an_unauthorized_id_are_indistinguishable():
    """Both answer 404 on purpose: a 403 would confirm the id exists."""
    owned_job()
    missing = pytest.raises(HTTPException)
    with missing as a:
        jobstore.authorized_job("does-not-exist", "tok-secret", None)
    with pytest.raises(HTTPException) as b:
        jobstore.authorized_job("j1", "wrong", None)
    assert a.value.status_code == b.value.status_code == 404
    assert a.value.detail == b.value.detail


def test_a_job_stored_without_a_token_is_not_open_to_everyone():
    """Defensive: an empty token must never compare equal to an empty guess."""
    jobstore.JOBS["legacy"] = {
        "status": "completed", "created_at": time.time(), "job_dir": None,
        "token": None, "owner_user_id": None,
    }
    with pytest.raises(HTTPException):
        jobstore.authorized_job("legacy", None, None)
    with pytest.raises(HTTPException):
        jobstore.authorized_job("legacy", "", None)
