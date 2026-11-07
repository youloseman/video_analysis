"""A knee read high from the non-drive side is no reason to move the saddle.

Four same-session pairs, 2026-11-06, knee at the bottom of the stroke: MediaPipe
read the non-drive clips 150.8-153.1 where RTMPose read 144-148, and the two agreed
within a degree on the chainring side. Four riders told to lower a saddle that the
better model puts about in range. These pin that every surface moves together,
because a green tile over "lower the saddle" is worse than either alone.
"""
from __future__ import annotations

import pytest

from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan
from app.services.video_analysis.biomechanics.cycling_positions import (
    NON_DRIVE_KNEE_MARGIN_DEG,
    knee_bdc_band,
)
from app.services.video_analysis.biomechanics.sport_configs import reference_bands
from app.services.video_analysis.biomechanics.technique_scorer import (
    build_severity_map,
    score_cycling,
)

POS = "triathlon"  # band 138-145
MEASURED_NON_DRIVE = (152.79, 153.13, 151.56, 150.8)


def _summary(knee, side):
    near = "left" if side != "right" else "right"
    return {"knee_at_bdc": knee, f"{near}_knee_at_bdc": knee, "near_side": near, "camera_side": side,
            "knee_at_tdc": 66.0, f"{near}_knee_at_tdc": 66.0, "trunk_angle_avg": 22.0, "elbow_angle_avg": 90.0,
            "shoulder_angle_avg": 90.0}


def _saddle_actions(knee, side):
    plan = build_action_plan(POS, {}, _summary(knee, side), 90, "A", [])
    return [d.action for d in plan.diagnostics if d.component == "saddle_height"]


def test_only_the_non_drive_side_gets_the_room():
    assert knee_bdc_band(POS, "left") == (138, 145 + NON_DRIVE_KNEE_MARGIN_DEG)
    assert knee_bdc_band(POS, "right") == (138, 145)
    assert knee_bdc_band(POS, "both") == (138, 145), "a merged session is not one side's reading"
    assert knee_bdc_band(POS, None) == (138, 145)


@pytest.mark.parametrize("knee", MEASURED_NON_DRIVE)
def test_the_clips_that_were_told_to_lower_the_saddle_no_longer_are(knee):
    assert _saddle_actions(knee, "left") == []
    assert "lower_saddle" in _saddle_actions(knee, "right"), "the same reading from the other side still is"


def test_a_really_high_saddle_is_still_caught():
    assert "lower_saddle" in _saddle_actions(158.0, "left")


def test_a_low_saddle_gets_no_margin():
    assert "raise_saddle" in _saddle_actions(130.0, "left")


def test_the_score_follows_the_verdict():
    high = score_cycling(_summary(152.0, "left"), {}, cycling_position=POS)
    other = score_cycling(_summary(152.0, "right"), {}, cycling_position=POS)
    assert high["component_scores"]["knee_bdc"] == 100.0
    assert other["component_scores"]["knee_bdc"] < 100.0


def test_the_band_the_client_reads_says_so():
    left = reference_bands("bike", POS, camera_side="left")["knee_at_bdc"]
    right = reference_bands("bike", POS, camera_side="right")["knee_at_bdc"]
    assert (left["lo"], left["hi"]) == (138, 154) and "9 degrees wider" in left["source"]
    assert (right["lo"], right["hi"]) == (138, 145) and "9 degrees" not in right.get("source", "")
    assert reference_bands("bike", POS)["knee_at_bdc"]["hi"] == 145


def test_the_document_says_why_the_range_is_wider():
    from app.services.report_pdf import render_report_html

    def doc(side):
        return render_report_html({
            "sport_type": "bike", "technique_score": 90, "camera_side": side,
            "reference_bands": reference_bands("bike", POS, camera_side=side),
            "sport_specific_metrics": {"knee_at_bdc": 150.0},
        }, {})
    assert "9 degrees wider" in doc("left")
    assert "9 degrees wider" not in doc("right")


def test_severity_does_not_shout_inside_the_margin():
    assert build_severity_map(POS, _summary(152.0, "left"))["knee_at_bdc"] == "ACCEPTABLE"
    assert build_severity_map(POS, _summary(143.0, "left"))["knee_at_bdc"] == "OPTIMAL"
    assert build_severity_map(POS, _summary(152.0, "right"))["knee_at_bdc"] not in ("OPTIMAL", "ACCEPTABLE")
