"""The clip grades the head the way the photo does (2026-10-12).

It used to score "head position" -- the ear under a line through the hip and
shoulder joint centres -- at a tenth of the clip's score; every aero rider
passed, so it was free points, and a photo of the same rider reported a
helmet "+27 %, amber" while the clip said 100/100. Now the clip measures the
helmet above the top of the back off a few silhouette frames
(video_helmet.py) and grades it with the photo's rule and constants.
"""
from __future__ import annotations

from app.services.video_analysis.biomechanics.head_silhouette import KONA_PRO_FREE_UP_TO
from app.services.video_analysis.biomechanics.technique_scorer import (
    CYCLING_WEIGHTS,
    build_severity_map,
    helmet_component_score,
    score_cycling,
)
from app.services.video_analysis.photo_analyzer import (
    _HELMET_BAND,
    _HELMET_SCORE_FLOOR,
    _score_single_angle,
)

SUMMARY = {
    "frames_analyzed": 150, "near_side": "left", "left_knee_at_bdc": 142.0,
    "left_knee_at_tdc": 72.0, "trunk_angle_avg": 22.0, "elbow_angle_avg": 85.0,
    "shoulder_angle_avg": 88.0, "pelvic_ratio": 2.5, "forearm_tilt_avg": 20.0,
    "saddle_height_assessment": "optimal",
}


def test_the_old_head_score_is_out_of_the_rubric():
    assert "head_alignment" not in CYCLING_WEIGHTS
    out = score_cycling({**SUMMARY, "head_alignment_avg": 100.0}, {}, cycling_position="triathlon")
    assert "head_alignment" not in out["component_scores"]


def test_the_clip_and_the_photo_grade_a_helmet_alike():
    for v in (-4.0, 12.0, KONA_PRO_FREE_UP_TO, 26.0, 33.0, 60.0):
        photo = max(_score_single_angle(v, *_HELMET_BAND), _HELMET_SCORE_FLOOR)
        assert helmet_component_score(v) == photo, v


def test_a_high_helmet_on_aero_bars_costs_a_little_and_is_amber():
    ok = score_cycling({**SUMMARY, "helmet_above_back": 18.0}, {}, cycling_position="triathlon")
    high = score_cycling({**SUMMARY, "helmet_above_back": 33.0}, {}, cycling_position="triathlon")
    assert ok["component_scores"]["helmet_above_back"] == 100.0
    assert high["component_scores"]["helmet_above_back"] == 80.0
    assert high["overall_score"] < ok["overall_score"]
    sev = build_severity_map("triathlon", {**SUMMARY, "helmet_above_back": 33.0})
    assert sev["helmet_above_back"] == "ACCEPTABLE"


def test_off_the_bars_or_flagged_it_is_shown_and_not_graded():
    for summary, position in (
        ({**SUMMARY, "helmet_above_back": 40.0}, "road_hoods"),
        ({**SUMMARY, "helmet_above_back": 40.0, "helmet_check": True}, "triathlon"),
    ):
        out = score_cycling(summary, {}, cycling_position=position)
        assert "helmet_above_back" not in out["component_scores"]
        assert "helmet_above_back" not in out["coverage"]["missing"], "left the rubric, not missing"
        assert "helmet_above_back" not in build_severity_map(position, summary)


def test_the_coach_is_told_what_the_number_is():
    from app.services.video_analysis.llm_recommendations import _helmet_prompt_line

    line = _helmet_prompt_line({"helmet_above_back": 26.0})
    assert "+26.0% of torso length" in line and "NOT degrees" in line
    assert "do NOT judge it" in _helmet_prompt_line({"helmet_above_back": 26.0, "helmet_check": True})
    assert "not measured" in _helmet_prompt_line({})
