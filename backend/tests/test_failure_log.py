"""A failed analysis tells the athlete something human and keeps the evidence.

Two things went wrong at once before 24.10, and both are pinned here:

* the exception's own text was the athlete's error message -- three people
  were told "'>' not supported between instances of 'NoneType' and 'int'";
* the only record of where it happened was a log line holding ``str(e)``,
  in a container log that the next deploy threw away.

The reports must also stay anonymous: a server-side event cannot see the
per-browser analytics opt-out that the privacy policy promises.
"""
from __future__ import annotations

import io

import pytest

from app.services import failure_log

CRASH_TEXT = "'>' not supported between instances of 'NoneType' and 'int'"


@pytest.fixture
def sent(monkeypatch):
    """Every report, captured inline instead of on a daemon thread."""
    out: list[tuple[str, str, dict]] = []

    class InlineThread:
        def __init__(self, target, args=(), daemon=None):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(failure_log.threading, "Thread", InlineThread)
    monkeypatch.setattr(
        failure_log.analytics, "capture",
        lambda distinct, event, props=None: out.append((distinct, event, props)) or True,
    )
    return out


def _raise_like_the_pipeline():
    value = None
    return value > 0          # the exact shape of the 23.10 crash


class TestTheAthletesSentence:
    def test_a_crash_never_reaches_the_athlete_as_its_own_text(self):
        try:
            _raise_like_the_pipeline()
        except TypeError as e:
            msg = failure_log.user_message(e)
        assert msg == failure_log.GENERIC
        assert "NoneType" not in msg and "not supported" not in msg

    def test_an_unreadable_video_says_so_without_the_server_path(self):
        e = RuntimeError("Could not open video: /data/uploads/ab12cd/input.mov")
        msg = failure_log.user_message(e)
        assert msg == failure_log.UNREADABLE
        assert "/data" not in msg and "input.mov" not in msg


class TestTheEvidence:
    def test_frames_name_the_line_that_raised_relative_to_the_app(self):
        try:
            _raise_like_the_pipeline()
        except TypeError as e:
            frames = failure_log._frames(e)
        assert frames and "_raise_like_the_pipeline" in frames[-1]
        assert all(not f.startswith(("/", "C:", "E:")) for f in frames)

    @pytest.mark.parametrize("raw", [
        "Could not open video: /data/uploads/ab12cd/input.mov",
        r"cannot read E:\PROJECTS\Video_analysis\backend\uploads\x\input.mp4",
    ])
    def test_filesystem_paths_are_stripped_from_the_message(self, raw):
        cleaned = failure_log._clean(raw)
        assert "<path>" in cleaned
        assert "uploads" not in cleaned and "input." not in cleaned

    def test_a_photo_is_described_by_its_shape_never_its_pixels(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (640, 480), "white").save(buf, format="JPEG")
        meta = failure_log.image_meta(buf.getvalue())
        assert (meta["img_w"], meta["img_h"], meta["img_format"]) == (640, 480, "JPEG")
        assert meta["img_mp"] == 0.3
        assert not any(isinstance(v, (bytes, bytearray)) for v in meta.values())

    def test_an_unreadable_photo_is_itself_a_finding_not_a_crash(self):
        meta = failure_log.image_meta(b"definitely not an image")
        assert meta.get("img_unreadable") is True
        assert meta["img_mb"] == 0.0


class TestAnonymity:
    def test_a_crash_report_carries_no_person(self, sent):
        try:
            _raise_like_the_pipeline()
        except TypeError as e:
            failure_log.report_crash("video", e, job_id="j1", sport="bike")
        (distinct, event, props), = sent
        assert event == "analysis_error"
        assert distinct == failure_log.DIAGNOSTICS_ID
        assert props["$process_person_profile"] is False
        assert props["err_type"] == "TypeError" and props["where"] == "video"
        assert "_raise_like_the_pipeline" in props["frames"][-1]

    def test_a_rejection_report_carries_no_person_and_drops_empty_fields(self, sent):
        failure_log.report_rejected("photo", "No body pose detected.", sport="bike",
                                    position=None, img_w=400)
        (distinct, event, props), = sent
        assert event == "analysis_rejected" and distinct == failure_log.DIAGNOSTICS_ID
        assert props["$process_person_profile"] is False
        assert "position" not in props and props["img_w"] == 400


class TestTheJobHandlers:
    @pytest.fixture
    def main(self, monkeypatch, tmp_path):
        main = pytest.importorskip("app.main")
        monkeypatch.setattr(main, "job_progress_hook", lambda *a, **k: None)
        monkeypatch.setattr(main.ANALYSIS_SLOTS, "acquire", lambda timeout=None: True)
        monkeypatch.setattr(main.ANALYSIS_SLOTS, "release", lambda: None)
        yield main
        for jid in ("crash01", "crash02", "unread01"):
            main.JOBS.pop(jid, None)

    def test_a_crashing_video_job_tells_the_athlete_something_human(self, main, monkeypatch, sent, tmp_path):
        def fake_run(*a, **k):
            return _raise_like_the_pipeline()
        monkeypatch.setattr(main, "run_analysis", fake_run)
        main.JOBS["crash01"] = {"status": "queued", "sport": "bike", "job_dir": str(tmp_path)}
        main._process_job("crash01", str(tmp_path / "input.mp4"), "bike", "road_hoods", None)
        job = main.JOBS["crash01"]
        assert job["status"] == "failed"
        assert job["error"] == failure_log.GENERIC
        assert CRASH_TEXT not in job["error"]
        crash = [p for _, ev, p in sent if ev == "analysis_error"]
        assert crash and crash[0]["where"] == "video" and crash[0]["job_id"] == "crash01"

    def test_a_crashing_pair_job_does_too(self, main, monkeypatch, sent, tmp_path):
        monkeypatch.setattr(main, "run_analysis", lambda *a, **k: _raise_like_the_pipeline())
        main.JOBS["crash02"] = {"status": "queued", "sport": "bike", "job_dir": str(tmp_path)}
        main._process_pair_job("crash02", "clipL.mp4", "clipR.mp4", "road_hoods", None, None)
        job = main.JOBS["crash02"]
        assert job["status"] == "failed" and job["error"] == failure_log.GENERIC
        assert any(ev == "analysis_error" and p["where"] == "pair" for _, ev, p in sent)

    def test_an_unopenable_clip_is_named_as_such(self, main, monkeypatch, sent, tmp_path):
        def fake_run(*a, **k):
            raise RuntimeError("Could not open video: /data/uploads/zz/input.mov")
        monkeypatch.setattr(main, "run_analysis", fake_run)
        main.JOBS["unread01"] = {"status": "queued", "sport": "bike", "job_dir": str(tmp_path)}
        main._process_job("unread01", str(tmp_path / "input.mov"), "bike", "road_hoods", None)
        assert main.JOBS["unread01"]["error"] == failure_log.UNREADABLE
