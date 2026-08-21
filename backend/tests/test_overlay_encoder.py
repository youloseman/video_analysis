"""An overlay no browser can play must not be offered as an overlay.

``video_visualizer`` re-encodes to H.264 with ffmpeg and, when ffmpeg is absent,
falls back to OpenCV's mp4v muxer. That fallback is the one failure nothing
catches on its own: the file is written, so the job completes, the API hands out
a URL, and the athlete gets a black player with no error and no explanation --
on a feature they paid for.

So the encoder's absence is treated as "no overlay for this job", which lands in
the ``overlay_failed`` branch the client already writes a sentence for.

Guarded by an import skip: importing ``app.main`` pulls in the mediapipe/opencv
stack the rest of the suite deliberately avoids (see conftest).
"""

from __future__ import annotations

import time
import uuid

import pytest

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

from app.core.jobs import JOBS  # noqa: E402


@pytest.fixture
def completed_job(tmp_path):
    """A finished job whose overlay file is really on disk."""
    job_id = uuid.uuid4().hex[:12]
    overlay = tmp_path / "overlay.mp4"
    overlay.write_bytes(b"\x00\x00\x00\x1cftypisom")   # enough to exist
    JOBS[job_id] = {
        "status": "completed",
        "sport": "run",
        "cycling_position": None,
        "result": {"technique_score": 71},
        "error": None,
        "overlay_path": str(overlay),
        "created_at": time.time(),
        "job_dir": str(tmp_path),
        "token": "tok",
        "owner_user_id": None,
    }
    yield job_id
    JOBS.pop(job_id, None)


def test_a_rendered_overlay_is_served_when_ffmpeg_is_there(completed_job, monkeypatch):
    monkeypatch.setattr(main, "OVERLAY_ENCODER_PRESENT", True)
    s = main.job_status(completed_job, t="tok", user=None)
    assert s.overlay_available is True
    assert s.overlay_url == f"/jobs/{completed_job}/overlay"
    assert s.overlay_failed is False


def test_without_ffmpeg_the_overlay_is_reported_failed_not_offered(
    completed_job, monkeypatch,
):
    """The file exists and is mp4v -- unplayable. Say so instead of linking it."""
    monkeypatch.setattr(main, "OVERLAY_ENCODER_PRESENT", False)
    s = main.job_status(completed_job, t="tok", user=None)
    assert s.overlay_available is False
    assert s.overlay_url is None
    # This is what makes the client explain itself rather than show a dead player.
    assert s.overlay_failed is True


def test_the_rest_of_the_result_is_untouched_by_a_missing_encoder(
    completed_job, monkeypatch,
):
    """Losing the encoder costs the video, not the analysis."""
    monkeypatch.setattr(main, "OVERLAY_ENCODER_PRESENT", False)
    s = main.job_status(completed_job, t="tok", user=None)
    assert s.status == "completed"
    # Not an exact dict any more. The job stores the complete result and it is
    # trimmed per reader on the way out, so a caller holding only the job token
    # -- no session, therefore no plan -- gets the teaser wrapper around it.
    # What this test is about is that the analysis survived the missing
    # encoder; which parts of it a given reader sees is tested next door, in
    # test_result_gating.py.
    assert s.result["technique_score"] == 71
    assert s.error is None


def test_a_job_that_never_asked_for_an_overlay_is_not_marked_failed(
    completed_job, monkeypatch,
):
    """``overlay_failed`` means "we owed you a video and could not make it" --
    not "there is no video", which is the normal free-tier case."""
    monkeypatch.setattr(main, "OVERLAY_ENCODER_PRESENT", False)
    JOBS[completed_job]["overlay_path"] = None
    s = main.job_status(completed_job, t="tok", user=None)
    assert s.overlay_available is False
    assert s.overlay_failed is False


def test_health_reports_whether_overlays_can_be_encoded():
    """Operators should not have to run an analysis to find this out."""
    assert "overlay_encoder_present" in main.health()
