"""Clips in the wrong slots, and the hip on the overlay.

The two-sided upload trusts its slots: each clip is analysed as the side its
slot names. Measured on IMG_4527/IMG_4525 with the slots swapped, both clips
were analysed on the far leg -- no error, no gate, a full report about the
wrong leg -- while the depth vote had said the other side on 5 of 5 frames
for both. So the vote now checks the slot.

The rules pinned here:

* the single-clip override still wins, but a confident disagreement is
  reported on the result and said out loud in the warnings;
* a pair whose first clip reads as the other side is swapped and re-run, at
  the cost of one analysis, and the rider is told;
* a pair whose two clips read as the same side is refused with the side
  named, because no swap makes a pair of them;
* the bike hip callout grades against the position's own band, one-sided.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.services.video_analysis import runner
from app.services.video_analysis.pipeline import VideoAnalysisPipeline
from app.services.video_analysis.video_visualizer import _status_for_cfg


def _meta(votes: list[str], fallback: bool = False) -> dict[str, Any]:
    return {"votes": votes, "fallback": fallback, "quality_frames_used": len(votes)}


class TestTheOverrideReportsAConflict:
    def test_a_unanimous_vote_against_the_slot_is_confident(self):
        side, meta = runner._apply_camera_side_override(
            "right", _meta(["right"] * 5), "left", "bike",
        )
        assert side == "left", "the rider's word still stands on a single clip"
        c = meta["conflict"]
        assert c["detected"] == "right" and c["user"] == "left"
        assert c["votes_against"] == 5 and c["votes"] == 5
        assert c["confident"] is True

    def test_four_of_five_is_confident_three_of_five_is_not(self):
        _, m4 = runner._apply_camera_side_override(
            "right", _meta(["right"] * 4 + ["left"]), "left", "bike")
        _, m3 = runner._apply_camera_side_override(
            "right", _meta(["right"] * 3 + ["left"] * 2), "left", "bike")
        assert m4["conflict"]["confident"] is True
        assert m3["conflict"]["confident"] is False

    def test_a_fallback_vote_never_contradicts(self):
        """No quality frames -> the detector guessed "left" -- that guess must
        not be allowed to overrule a rider who said "right"."""
        _, meta = runner._apply_camera_side_override(
            "left", _meta([], fallback=True), "right", "bike")
        assert meta["conflict"]["confident"] is False

    def test_agreement_carries_no_conflict(self):
        _, meta = runner._apply_camera_side_override(
            "left", _meta(["left"] * 5), "left", "bike")
        assert meta["conflict"] is None

    def test_run_clips_are_untouched(self):
        side, meta = runner._apply_camera_side_override(
            "right", _meta(["right"] * 5), "left", "run")
        assert side == "right"
        assert "conflict" not in meta


class TestThePairJob:
    """Drive _process_pair_job with a fake analyser that answers by clip."""

    @pytest.fixture
    def job(self, monkeypatch, tmp_path):
        main = pytest.importorskip("app.main")
        calls: list[tuple[str, str]] = []

        def fake_run(path, sport, position, camera_side_override=None, **kw):
            calls.append((path, camera_side_override))
            truth = "left" if "L" in path else "right"
            conflict = None
            if camera_side_override and camera_side_override != truth:
                conflict = {"detected": truth, "user": camera_side_override,
                            "votes_against": 5, "votes": 5, "confident": True}
            return {
                "status": "completed", "camera_side": camera_side_override,
                "camera_side_conflict": conflict, "technique_score": 88,
                "sport_specific_metrics": {"knee_at_bdc": 140.0},
                "bilateral_geometry": None,
            }

        monkeypatch.setattr(main, "run_analysis", fake_run)
        monkeypatch.setattr(main, "job_progress_hook", lambda *a, **k: None)
        monkeypatch.setattr(main.ANALYSIS_SLOTS, "acquire", lambda timeout=None: True)
        monkeypatch.setattr(main.ANALYSIS_SLOTS, "release", lambda: None)
        monkeypatch.setattr(main, "_maybe_notify_ready", lambda *a, **k: None, raising=False)
        job_id = "pairslots01"
        main.JOBS[job_id] = {"status": "queued", "sport": "bike", "job_dir": str(tmp_path)}
        yield main, job_id, calls
        main.JOBS.pop(job_id, None)

    def test_swapped_clips_are_put_right_and_the_rider_is_told(self, job):
        main, job_id, calls = job
        main._process_pair_job(job_id, "clipR.mp4", "clipL.mp4", "road_hoods",
                               None, None, sport="bike")
        j = main.JOBS[job_id]
        assert j["status"] == "completed", j.get("error")
        # First analysis: right-side clip in the left slot -> conflict -> swap.
        # Then the left clip as left, and the right clip as right.
        assert calls == [("clipR.mp4", "left"), ("clipL.mp4", "left"), ("clipR.mp4", "right")]
        warnings = j["result"]["sport_specific_metrics"]["quality_warnings"]
        assert warnings and "each other's slots" in warnings[0]
        assert j["result"]["bilateral"]["slots_swapped"] is True

    def test_two_clips_of_one_side_are_refused_by_name(self, job):
        main, job_id, calls = job
        main._process_pair_job(job_id, "clipL.mp4", "clipL2.mp4", "road_hoods",
                               None, None, sport="bike")
        j = main.JOBS[job_id]
        assert j["status"] == "failed"
        assert "left side" in j["error"] and "film the right side" in j["error"]
        # The second clip was analysed once, as the right side, and refused.
        assert calls[-1] == ("clipL2.mp4", "right")

    def test_a_correctly_slotted_pair_runs_straight_through(self, job):
        main, job_id, calls = job
        main._process_pair_job(job_id, "clipL.mp4", "clipR.mp4", "road_hoods",
                               None, None, sport="bike")
        j = main.JOBS[job_id]
        assert j["status"] == "completed", j.get("error")
        assert calls == [("clipL.mp4", "left"), ("clipR.mp4", "right")]
        assert "slots_swapped" not in (j["result"].get("bilateral") or {})


class TestTheHipCallout:
    def test_the_overlay_hip_band_is_the_positions_own(self):
        from app.services.video_analysis.biomechanics.cycling_positions import (
            get_cycling_reference,
        )
        p = VideoAnalysisPipeline.__new__(VideoAnalysisPipeline)
        for pos in ("road_hoods", "tt_aero", "casual"):
            cfgs = p._get_angle_display_config(
                "bike", {"camera_side": "left"}, cycling_position=pos,
            )
            hip = next(c for c in cfgs if c["key"] == "left_hip")
            assert tuple(hip["optimal"]) == tuple(get_cycling_reference(pos)["hip_at_tdc"])
            assert hip.get("open_ok") is True

    def test_an_open_hip_is_green_a_closed_one_is_not(self):
        cfg = {"optimal": (55, 65), "open_ok": True}
        assert _status_for_cfg(cfg, 89.0) == "good", "open past the band is comfort, not a fault"
        assert _status_for_cfg(cfg, 60.0) == "good"
        assert _status_for_cfg(cfg, 40.0) == "bad"
        # Without the flag the same band is symmetric, as for every other joint.
        assert _status_for_cfg({"optimal": (55, 65)}, 89.0) == "bad"

    def test_the_hip_is_a_headline_callout_on_the_bike_only(self):
        from app.services.video_analysis.video_visualizer import VideoVisualizer

        assert "hip" in VideoVisualizer._HEADLINE_KEYS["bike"]
        assert "hip" not in VideoVisualizer._HEADLINE_KEYS["run"]


class TestWhatTheUploaderSaysBeforeFilming:
    """The drive side is explained BEFORE the take, not by the merge afterwards."""

    @pytest.fixture(scope="class")
    def html(self):
        from pathlib import Path
        return (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")

    def test_the_bike_pair_form_names_the_drive_side_and_what_each_clip_is_for(self, html):
        i = html.index("const PAIR_BIKE_NOTE=")
        note = html[i:html.index("\nfunction pairFileOf", i)]
        assert "drive side" in note and "chainring" in note
        assert "non-drive" in note
        assert "saddle verdict" in note

    def test_the_note_is_shown_for_bike_pairs_only(self, html):
        i = html.index("function renderPairCards(")
        fn = html[i:html.index("\nfunction ", i + 10)]
        assert "state.sport==='bike'" in fn and "PAIR_BIKE_NOTE" in fn

    def test_no_left_vs_right_answer_is_promised(self, html):
        """The session reports no asymmetry number on purpose."""
        assert "left-vs-right answer" not in html
