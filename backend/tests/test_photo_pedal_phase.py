"""A single bike photo does not say where the crank was -- and the score says so.

Found on 15 race photos from the Ironman World Championship (2026-10-10):

1. The pedal position was guessed FROM the knee (>= 125 "bottom", <= 85 "top",
   anything between "mid-stroke"), and a "mid-stroke" knee was scored against
   60-155. A saddle far too low, photographed exactly at the bottom as the
   capture guide asks -- knee 118 -- was therefore filed as mid-stroke and
   scored 100. From a still, only a saddle that was too HIGH could ever be
   faulted.
2. The pelvic ratio (hip over trunk) was scored at any crank position against a
   band set on the stroke-AVERAGE hip. The one Kona frame actually at the
   bottom of the stroke read 5.7, scored 20 on that metric and 67 overall --
   the worst score in the set, on the most usable photo.
3. A frame where the model found the camera motorbike's passenger, not the
   rider -- no leg measured, trunk 83 deg -- printed 38/100.

Now: the guess is "bdc"/"tdc"/"uncertain"; knee and hip are scored only at a
named position; the server scores every position so the rider's correction
changes the number too; the pelvic ratio is shown, never scored, from a still;
and a bike photo with no measured leg gets no number.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="the photo path decodes and draws with OpenCV")

from app.services.video_analysis import photo_analyzer as PA  # noqa: E402

W, H = 720, 1280
ASP = W / H


def _bike_pose(knee_deg: float, knee_visible: bool = True):
    """A right-side TT rider whose near knee reads ``knee_deg`` on the image
    plane, built in aspect-corrected units and handed over normalized."""
    hip = np.array([0.38 * ASP, 0.375])
    a = math.radians(35)                                 # thigh forward-down
    thigh = np.array([math.sin(a), math.cos(a)])
    knee = hip + 0.21 * thigh
    back = -thigh                                        # knee -> hip
    t = math.radians(knee_deg)
    rot = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]])
    shin = rot @ back
    if shin[1] < 0:                                      # the foot goes down
        rot = rot.T
        shin = rot @ back
    ankle = knee + 0.21 * shin
    corrected = {
        24: hip, 26: knee, 28: ankle,
        30: ankle + np.array([-0.03, 0.01]), 32: ankle + np.array([0.06, 0.02]),
        12: np.array([0.72 * ASP, 0.30]), 14: np.array([0.82 * ASP, 0.36]),
        16: np.array([0.92 * ASP, 0.36]), 8: np.array([0.86 * ASP, 0.27]),
        0: np.array([0.93 * ASP, 0.28]),
    }
    norm, world = [], []
    for j in range(33):
        left = j in (7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31)
        p = corrected.get(j + 1 if left else j, np.array([0.88 * ASP, 0.27]))
        x, y = float(p[0] / ASP) - (0.01 if left else 0.0), float(p[1])
        vis = 0.6 if left else 1.0
        if j == 26 and not knee_visible:
            vis = 0.0
        z = 0.1 if left else -0.1
        norm.append(SimpleNamespace(x=x, y=y, z=z, visibility=vis))
        world.append(SimpleNamespace(x=(x - 0.5) * 1.2, y=(y - 0.5) * 2, z=z, visibility=vis))
    return world, norm


def _measure(knee_deg: float, *, knee_visible: bool = True, position: str = "tt_aero") -> dict:
    world, norm = _bike_pose(knee_deg, knee_visible)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    return PA.analyze_from_pose(image, world, norm, "right", [], "bike", position)


class TestTheGuessIsOnlyAGuess:
    def test_the_middle_is_uncertain_not_mid_stroke(self):
        assert PA._estimate_pedal_phase(118.0) == "uncertain"
        assert PA._estimate_pedal_phase(140.0) == "bdc"
        assert PA._estimate_pedal_phase(70.0) == "tdc"
        assert PA._estimate_pedal_phase(float("nan")) == "uncertain"

    def test_an_uncertain_position_scores_no_knee_and_no_hip(self):
        ranges = PA._get_cycling_optimal_ranges("tt_aero", "uncertain")
        assert "knee" not in ranges and "hip" not in ranges
        assert {"trunk", "shoulder", "elbow", "forearm_tilt", "head_alignment"} <= set(ranges)

    def test_the_pelvic_ratio_and_ankle_are_never_scored_from_a_still(self):
        for phase in PA.BIKE_PHOTO_PHASES:
            ranges = PA._get_cycling_optimal_ranges("tt_aero", phase)
            assert "pelvic_ratio" not in ranges and "ankle" not in ranges


class TestASaddleTooLowIsNoLongerInvisible:
    def test_the_frame_reads_as_built(self):
        res = _measure(118.0)
        assert abs(res["angles"]["knee"] - 118.0) < 1.0, res["angles"]["knee"]

    def test_an_uncertain_frame_carries_no_free_knee_points(self):
        res = _measure(118.0)
        assert res["pedal_phase"] == "uncertain"
        assert "knee" not in res["score"]["per_angle"]
        row = res["angles_with_context"]["knee"]
        assert row["status"] == "phase_dependent" and row["bands"]["bdc"]
        assert any("couldn't tell where the pedal was" in w for w in res["warnings"])

    def test_marked_as_the_bottom_of_the_stroke_the_low_knee_is_faulted(self):
        res = _measure(118.0)
        at_bdc = res["score_by_phase"]["bdc"]
        assert at_bdc["per_angle"]["knee"]["score"] < 100
        lo, _hi = res["angles_with_context"]["knee"]["bands"]["bdc"]
        assert res["angles"]["knee"] < lo, "below the bottom-of-stroke band: a saddle too low"

    def test_a_good_knee_at_the_bottom_is_not_faulted(self):
        res = _measure(142.0)
        assert res["pedal_phase"] == "bdc"
        assert res["score"]["per_angle"]["knee"]["score"] == 100


class TestEveryPositionIsScored:
    def test_the_score_is_the_one_for_the_guessed_position(self):
        res = _measure(142.0)
        assert set(res["score_by_phase"]) == set(PA.BIKE_PHOTO_PHASES)
        assert res["score"] == res["score_by_phase"][res["pedal_phase"]]

    def test_the_pelvic_ratio_row_is_shown_but_not_judged(self):
        res = _measure(142.0)
        row = res["angles_with_context"].get("pelvic_ratio")
        assert row is not None and row["status"] == "phase_dependent"
        assert "pelvic_ratio" not in res["score"]["per_angle"]


class TestTheTrunkIsReadOnTheNearSide:
    def test_a_right_side_photo_reads_the_right_shoulder_and_hip(self):
        # The far (left) side sits at visibility 0.6 in this pose, under the
        # threshold; reading it is what made the trunk NaN.
        res = _measure(142.0)
        assert not math.isnan(res["angles"]["trunk"])
        assert "pelvic_ratio" in res["angles"]


class TestNoLegNoNumber:
    def test_an_unmeasured_knee_withholds_the_score(self):
        res = _measure(140.0, knee_visible=False)
        assert math.isnan(res["angles"]["knee"])
        assert res["score"]["overall_score"] is None
        assert res["score"]["withheld"] == "leg_not_measured"
        assert "score_by_phase" not in res
        assert any(w.startswith("No score:") for w in res["warnings"])

    def test_the_picture_says_not_scored_instead_of_crashing(self):
        res = _measure(140.0, knee_visible=False)
        assert res["thumbnail_base64"].startswith("data:image/jpeg;base64,")


class TestTheReport:
    def _html(self) -> str:
        return (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")

    def test_the_picker_swaps_the_score_as_well_as_the_rows(self):
        html = self._html()
        i = html.index("function setPhotoPedalPhase(")
        fn = html[i:html.index("\n}", i)]
        assert "score_by_phase" in fn and "setScoreRing" in fn and "setGrade" in fn

    def test_the_picker_covers_the_knee_not_only_the_hip(self):
        html = self._html()
        i = html.index("function renderPhotoAngles(")
        fn = html[i:html.index("\nfunction ", i + 10)]
        assert "if(a.bands) a=rowForPhase(a,phase)" in fn
        assert "hipForPhase" not in html

    def test_coaching_is_not_written_over_a_withheld_photo(self):
        main = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
        assert main.count('.get("withheld")') >= 2


class TestTheStatusColours:
    def test_every_status_the_server_can_return_has_a_colour(self):
        import re
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index("const P_STATUS_CLS=")
        table = html[i:html.index("};", i)]
        assert re.search(r"\bacceptable:'medium'", table)
        assert re.search(r"\bneeds_work:'high'", table)


class TestTheSaddleIsJudgedLikeTheVideoJudgesIt:
    """Approved by Artur 2026-10-11: a knee marked at the bottom of the stroke
    and well under the band read "needs work" in the table while the overall
    stayed 94 "Excellent" -- 2 pts/deg against a band widened -12/+8."""

    def test_the_table_shows_the_reference_band_not_a_widened_one(self):
        from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference
        res = _measure(142.0)
        assert res["angles_with_context"]["knee"]["bands"]["bdc"] == list(get_cycling_reference("tt_aero")["knee_at_bdc"])

    def test_the_slope_is_the_videos(self):
        from app.services.video_analysis.biomechanics.technique_scorer import (
            SCORE_ANGLE_TOLERANCE,
            score_in_range,
        )
        band = (138.0, 145.0)
        # past the crank allowance the photo loses exactly what the video does
        crank_lo = PA._CRANK_SLACK_DEG["bdc"][0]
        for knee in (128.0, 124.0):
            photo = PA._score_single_angle(
                knee, *band, falloff=PA._knee_falloff(band), slack=PA._KNEE_SLACK["bdc"],
            )
            video = score_in_range(knee + crank_lo, *band, tolerance=SCORE_ANGLE_TOLERANCE)
            assert abs(photo - max(20.0, video)) <= 1.0, (knee, photo, video)

    def test_the_slack_is_what_a_frame_near_dead_centre_can_cost(self):
        assert PA._CRANK_SLACK_DEG["bdc"] == (3.0, 2.0)
        assert PA._CRANK_SLACK_DEG["tdc"] == (2.0, 3.0)
        assert PA._KNEE_SLACK["bdc"] == (5.0, 4.0)

    def test_a_red_knee_row_always_costs_points_and_an_amber_one_never_does(self):
        band, slack = (138.0, 145.0), PA._KNEE_SLACK["bdc"]
        for knee in range(120, 160):
            status = PA._classify_angle_status(float(knee), *band, slack=slack)
            score = PA._score_single_angle(float(knee), *band, falloff=PA._knee_falloff(band), slack=slack)
            if status == "needs_work":
                assert score < 100, knee
            else:
                assert score == 100, knee

    def test_a_low_saddle_marked_at_the_bottom_is_not_excellent(self):
        res = _measure(120.0)
        assert res["score_by_phase"]["bdc"]["grade"] not in ("Excellent", "Good")
        assert res["score_by_phase"]["bdc"]["per_angle"]["knee"]["score"] <= 30

    def test_the_client_status_follows_the_same_slack(self):
        html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
        i = html.index("function classifyAngle(")
        fn = html[i:html.index("\n}", i)]
        assert "slack" in fn
        assert "classifyAngle(a.value,band[0],band[1],a.slack&&a.slack[phase])" in html
