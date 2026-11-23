"""The report is served before its overlay video has finished rendering.

The render was 12 s of a 1080p analysis and ~40 s of a 4K one, and the athlete
waited for it before seeing a single number. Now the job completes when the
report is measured, the poll says ``overlay_rendering`` until the file is
whole, the file is not served half-written, and the clip is compressed only
after the render that reads it is done.
"""

from __future__ import annotations

import threading
import time

import pytest
from fastapi import HTTPException

main = pytest.importorskip("app.main", reason="needs the analysis stack (mediapipe/opencv)")
from app.core.jobs import JOBS  # noqa: E402


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(main, "job_progress_hook", lambda *a, **k: None)
    monkeypatch.setattr(main.ANALYSIS_SLOTS, "acquire", lambda timeout=None: True)
    monkeypatch.setattr(main.ANALYSIS_SLOTS, "release", lambda: None)
    monkeypatch.setattr(main, "_schedule_ready_mail", lambda job_id: None)
    compressed: list[str] = []
    import app.services.clip_store as clip_store
    monkeypatch.setattr(clip_store, "compress_clip", lambda p: compressed.append(p))
    return compressed


def _job(tmp_path, jid):
    JOBS[jid] = {
        "status": "queued", "sport": "bike", "job_dir": str(tmp_path),
        "overlay_path": str(tmp_path / "overlay.mp4"), "token": "tok",
        "owner_user_id": None, "created_at": time.time(),
    }
    return JOBS[jid]


def test_the_job_completes_while_the_video_renders(tmp_path, monkeypatch, quiet):
    release = threading.Event()
    seen: dict = {}

    def fake_run(*a, overlay_done=None, **k):
        seen["callback"] = overlay_done

        def render():
            release.wait(5)
            (tmp_path / "overlay.mp4").write_bytes(b"\x00\x00\x00\x1cftypisom")
            overlay_done(str(tmp_path / "overlay.mp4"))

        threading.Thread(target=render, daemon=True).start()
        return {"status": "completed", "technique_score": 80,
                "overlay_video_path": str(tmp_path / "overlay.mp4"), "overlay_pending": True}

    monkeypatch.setattr(main, "run_analysis", fake_run)
    monkeypatch.setattr(main, "OVERLAY_ENCODER_PRESENT", True)
    job = _job(tmp_path, "defer01")
    worker = threading.Thread(
        target=main._process_job,
        args=("defer01", str(tmp_path / "in.mp4"), "bike", "tt_aero", str(tmp_path / "overlay.mp4")),
    )
    worker.start()
    try:
        deadline = time.time() + 5
        while job["status"] != "completed" and time.time() < deadline:
            time.sleep(0.01)
        assert seen["callback"] is not None
        assert job["status"] == "completed"
        assert "overlay_pending" not in job["result"]

        s = main.job_status("defer01", t="tok", user=None)
        assert s.overlay_rendering is True
        assert s.overlay_available is False and s.overlay_failed is False
        # The clip the render is reading is not re-encoded under it.
        assert quiet == []
    finally:
        release.set()
        worker.join(5)
    s = main.job_status("defer01", t="tok", user=None)
    assert s.overlay_rendering is False and s.overlay_available is True
    assert quiet == [str(tmp_path / "in.mp4")]
    JOBS.pop("defer01", None)


async def test_a_half_written_overlay_is_not_served(tmp_path):
    job = _job(tmp_path, "defer02")
    job["status"] = "completed"
    job["overlay_rendering"] = True
    (tmp_path / "overlay.mp4").write_bytes(b"partial")
    try:
        with pytest.raises(HTTPException) as err:
            await main.job_overlay("defer02", t="tok", side=None, mt=None, user=None, db=None)
        assert err.value.status_code == 409
    finally:
        JOBS.pop("defer02", None)


def test_a_failed_render_ends_as_overlay_failed(tmp_path, monkeypatch, quiet):
    def fake_run(*a, overlay_done=None, **k):
        overlay_done(None)
        return {"status": "completed", "technique_score": 80}

    monkeypatch.setattr(main, "run_analysis", fake_run)
    monkeypatch.setattr(main, "OVERLAY_ENCODER_PRESENT", True)
    _job(tmp_path, "defer03")
    main._process_job("defer03", str(tmp_path / "in.mp4"), "bike", "tt_aero", str(tmp_path / "overlay.mp4"))
    s = main.job_status("defer03", t="tok", user=None)
    assert s.status == "completed"
    assert s.overlay_rendering is False and s.overlay_failed is True
    JOBS.pop("defer03", None)


def test_a_crash_never_leaves_the_job_rendering(tmp_path, monkeypatch, quiet):
    def boom(*a, **k):
        raise RuntimeError("pipeline fell over")

    monkeypatch.setattr(main, "run_analysis", boom)
    job = _job(tmp_path, "defer04")
    main._process_job("defer04", str(tmp_path / "in.mp4"), "bike", "tt_aero", str(tmp_path / "overlay.mp4"))
    assert job["status"] == "failed" and job["overlay_rendering"] is False
    JOBS.pop("defer04", None)


def test_the_spa_waits_for_a_rendering_overlay():
    from conftest import read_spa

    spa = read_spa()
    assert "job.overlay_rendering" in spa and "function waitForOverlay(" in spa
    body = spa[spa.index("function waitForOverlay("):][:1800]
    assert "mountOverlay(" in body and "overlay_rendering" in body
