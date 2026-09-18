"""The hip is graded where it closes, not on its stroke average.

Every hip band in cycling_positions describes the CLOSED hip -- torso to
thigh at the top of the stroke, Retul's "hip closed angle", the number the 45
deg iliac-artery floor is about. What the report graded against those bands
was ``hip_angle_avg``, the mean over the whole revolution, which on a road
rider sits 20-30 deg above any closed-hip band: the check could only ever
read "open, fine", the fore/aft step of the plan never fired, and the
medical floor was unreachable. Measured on the 17 Sep pairs: min 61-65 /
mean 87-89 on hoods, min 49-54 / mean 74-82 in drops.

Now the analyzer samples the hip on the knee's own TDC frames and emits
``hip_at_tdc``; the served band, the scorer's severity map, the action plan,
the coach prompt, the overlay callout and the report tile all read that.
The mean stays, unbanded.
"""

from __future__ import annotations

import math

from app.services.video_analysis.biomechanics import technique_scorer
from app.services.video_analysis.biomechanics.cycling_analyzer import CyclingAnalyzer
from app.services.video_analysis.biomechanics.cycling_positions import (
    CYCLING_POSITIONS,
    get_cycling_reference,
)
from app.services.video_analysis.biomechanics.sport_configs import reference_bands


def _series(n, period, lo, hi, phase=0.0):
    mid, amp = (hi + lo) / 2, (hi - lo) / 2
    return [mid + amp * math.sin(2 * math.pi * (i / period) + phase) for i in range(n)]


def _analyzer():
    fps, n, period = 30.0, 240, 22.0    # ~82 rpm, 8 seconds
    an = CyclingAnalyzer(fps=fps, frame_aspect=9 / 16)
    an._near_side = an.camera_side = "right"
    knee = _series(n, period, 65.0, 145.0)
    # The hip closes when the knee flexes: same phase, torso-thigh 60 at TDC
    # and 110 at BDC, as on the real hoods clips.
    hip = _series(n, period, 60.0, 110.0)
    an.right_knee_angles = list(knee)
    an.left_knee_angles = [float("nan")] * n
    an.angle_history["right_knee"] = list(knee)
    an.angle_history["right_hip"] = list(hip)
    an.angle_timestamps = [i / fps for i in range(n)]
    return an


class TestTheAnalyzer:
    def test_the_hip_is_sampled_at_the_knees_tdc_frames(self):
        out = _analyzer()._get_bdc_tdc_angles()
        assert abs(out["right_hip_at_tdc"] - 60.0) < 2.0, out
        assert abs(out["right_hip_at_bdc"] - 110.0) < 2.0, out
        # and the mean of that same series is nowhere near the closed value
        assert abs(sum(_series(240, 22.0, 60.0, 110.0)) / 240 - 85.0) < 1.0

    def test_no_hip_series_means_no_hip_value_not_a_guess(self):
        an = _analyzer()
        del an.angle_history["right_hip"]
        out = an._get_bdc_tdc_angles()
        assert "right_hip_at_tdc" not in out
        assert "right_knee_at_tdc" in out


class TestTheBands:
    def test_every_position_has_a_closed_hip_band_and_it_differs_by_position(self):
        tdc = {p: CYCLING_POSITIONS[p]["hip_at_tdc"] for p in CYCLING_POSITIONS}
        assert tdc["road_drops"][0] < tdc["road_hoods"][0], "drops close the hip more than hoods"
        assert tdc["tt_aero"][0] < tdc["road_drops"][0]
        assert tdc["casual"][0] > tdc["road_hoods"][0]

    def test_the_served_band_is_for_the_closed_hip_and_the_mean_has_none(self):
        for pos in ("road_hoods", "road_drops"):
            bands = reference_bands("bike", pos)
            assert bands["hip_at_tdc"]["lo"], bands["hip_at_tdc"]["hi"] == get_cycling_reference(pos)["hip_at_tdc"]
            assert bands["hip_at_tdc"].get("source"), "the closed-hip evidence travels with the band"
            assert "hip_angle_avg" not in bands

    def test_hoods_and_drops_agree_on_the_saddle_and_differ_on_the_torso(self):
        """The legs do not change between hoods and drops -- same saddle,
        same crank -- so the knee windows must be identical. Everything the
        hands decide must not be."""
        h, d = get_cycling_reference("road_hoods"), get_cycling_reference("road_drops")
        assert h["knee_at_bdc"] == d["knee_at_bdc"] and h["knee_at_tdc"] == d["knee_at_tdc"]
        assert d["trunk_angle"][0] < h["trunk_angle"][0] and d["trunk_angle"][1] < h["trunk_angle"][1]
        assert d["elbow_angle"][0] < h["elbow_angle"][0]
        assert d["hip_at_tdc"][0] < h["hip_at_tdc"][0]


class TestTheGraders:
    def test_the_severity_map_reads_the_closed_hip_and_ignores_the_mean(self):
        sev = technique_scorer.build_severity_map(
            "road_hoods", {"near_side": "left", "hip_at_tdc": 62.0, "hip_angle_avg": 88.0},
        )
        assert sev["hip_at_tdc"] == "OPTIMAL"
        assert "hip_angle_max" not in sev
        sev = technique_scorer.build_severity_map("road_hoods", {"near_side": "left", "hip_angle_avg": 88.0})
        assert not any(k.startswith("hip") for k in sev), "a mean alone earns no hip verdict"

    def test_the_closed_hip_is_asymmetric_and_keeps_the_medical_floor(self):
        c = technique_scorer.classify_metric_severity
        assert c("road_hoods", "hip_at_tdc", 95.0) == "ACCEPTABLE", "open past the band is comfort"
        assert c("road_hoods", "hip_at_tdc", 62.0) == "OPTIMAL"
        assert c("tt_aero", "hip_at_tdc", 40.0) == "MEDICAL_RISK"
        assert c("road_drops", "hip_at_tdc", 35.0) in ("WARNING", "CRITICAL")

    def test_the_action_plan_fore_aft_step_fires_on_a_closed_hip(self):
        from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan

        plan = build_action_plan(
            position="road_drops", angle_statistics={},
            sport_specific_metrics={"knee_at_bdc": 141.0, "hip_at_tdc": 40.0, "hip_angle_avg": 75.0},
            technique_score=80, letter_grade="B", detected_issues=[],
        )
        step = next(d for d in plan.diagnostics if d.component == "saddle_fore_aft")
        assert step.metric_name == "hip_at_tdc"
        assert step.current_value == 40.0

    def test_the_overlay_callout_grades_the_hip_against_the_closed_band(self):
        from app.services.video_analysis.pipeline import VideoAnalysisPipeline

        p = VideoAnalysisPipeline.__new__(VideoAnalysisPipeline)
        cfgs = p._get_angle_display_config("bike", {"camera_side": "left"}, cycling_position="road_drops")
        hip = next(c for c in cfgs if c["key"] == "left_hip")
        assert tuple(hip["optimal"]) == tuple(get_cycling_reference("road_drops")["hip_at_tdc"])


class TestTheReport:
    def test_the_tile_reads_the_closed_hip_and_the_table_row_has_no_band(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index("function bikeTiles(")
        fn = html[i:html.index("\nfunction ", i + 10)]
        assert "s.hip_at_tdc" in fn and "['hip_at_tdc']" in fn
        assert "['hip_angle_avg']" not in fn, "the mean is no longer graded"
        j = html.index("function angleOptimal(")
        ao = html[j:html.index("\nfunction ", j + 10)]
        assert "if(base==='hip') return null;" in ao
        assert "hip_angle:[" not in html, "the position table's hip key must say what it is"


class TestTheShoulderBand:
    """Checked against sources on 2026-09-18 (see META): fitters put the
    torso-to-upper-arm angle on the hoods at about 90 (85-90, ~90, 90-100).
    The band this replaced was 90-120 -- floor on the textbook value, thirty
    degrees of room above it -- and every road clip in the repo failed it."""

    def test_the_road_bands_sit_around_ninety_not_above_it(self):
        h = CYCLING_POSITIONS["road_hoods"]["shoulder_angle"]
        d = CYCLING_POSITIONS["road_drops"]["shoulder_angle"]
        assert h[0] < 90 < h[1], h
        assert d[0] < h[0] and d[1] < h[1], "drops lower the torso, so the arm closes"
        for pos, meta in (("road_hoods", "BikeDynamics"), ("road_drops", "Derived")):
            from app.services.video_analysis.biomechanics.cycling_positions import get_position_meta
            assert meta in get_position_meta(pos, "shoulder_angle")["source"]

    def test_the_spa_table_mirrors_them(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        h = CYCLING_POSITIONS["road_hoods"]["shoulder_angle"]
        d = CYCLING_POSITIONS["road_drops"]["shoulder_angle"]
        assert f"road_hoods:{{trunk_angle:[40,55],elbow_angle:[145,165],shoulder_angle:[{h[0]},{h[1]}]" in html
        assert f"road_drops:{{trunk_angle:[30,45],elbow_angle:[130,160],shoulder_angle:[{d[0]},{d[1]}]" in html


class TestShortBikeClips:
    def test_a_short_bike_clip_is_nudged_not_refused(self):
        from app.services.video_analysis.capture_report import build_capture_report

        rep = build_capture_report(sport_type="bike", duration_s=5.5, frame_width=720, frame_height=1280,
                                   framing={"subject_height_px": 700, "subject_height_frac": 0.55},
                                   tracking_stability={})
        row = next(c for c in rep["checks"] if c["id"] == "duration")
        assert row["status"] == "warn" and row["impact"] == "low"
        assert "Analysed in full" in row["action"]
        assert "8-15 s" in row["target"]
        # the same length on a run clip is simply good
        rep = build_capture_report(sport_type="run", duration_s=5.5, frame_width=720, frame_height=1280,
                                   framing={"subject_height_px": 700, "subject_height_frac": 0.55},
                                   tracking_stability={"leg_swap_pct": 1.0})
        assert next(c for c in rep["checks"] if c["id"] == "duration")["status"] == "good"

    def test_the_pair_note_says_eight_seconds_and_that_shorter_still_runs(self):
        from pathlib import Path
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index("const PAIR_BIKE_NOTE=")
        note = html[i:html.index("\nfunction pairFileOf", i)]
        assert "8–15 seconds" in note and "shorter clips are analysed too" in note

    def test_one_bottom_sample_per_revolution_is_enough(self):
        from app.services.video_analysis.biomechanics import bilateral as B
        assert B._MIN_BOTTOM_SAMPLES_PER_REV == 1
        assert B._BOTTOM_MIN_Y <= 0.85 and B._BOTTOM_MAX_X >= 0.35
