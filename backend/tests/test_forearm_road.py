"""Forearm tilt is an aero-bar measure, and off aero bars it is not scored.

Three hoods clips of one rider: two read -42 deg and were silently dropped
by the summary's plausibility bound, the third read -25, passed, and lost
five points to (5, 20) -- an aero band pasted onto the hoods. Pinned here:
on a road position the component is absent AND named in the coverage's
``excluded`` with the reason; on aero bars it is scored as before; and the
plan does not list it as "already in range" on the hoods.
"""
from __future__ import annotations

from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan
from app.services.video_analysis.biomechanics.technique_scorer import score_cycling

SUMMARY = {
    "frames_analyzed": 150, "knee_at_bdc": 140.0, "knee_at_tdc": 70.0,
    "trunk_angle_avg": 45.0, "elbow_angle_avg": 155.0, "shoulder_angle_avg": 90.0,
    "head_alignment_avg": 90.0, "pelvic_ratio": 2.5, "saddle_height_assessment": "optimal",
    "forearm_tilt_avg": -25.0,
}


def test_the_hoods_are_scored_on_eight_measures_and_say_eight():
    out = score_cycling(SUMMARY, {}, cycling_position="road_hoods")
    assert "forearm_tilt" not in out["component_scores"]
    cov = out["coverage"]
    assert cov["measures_total"] == 8 and cov["measures_scored"] == 8
    assert cov["missing"] == [] and cov["weight_covered"] == 1.0
    assert out["overall_score"] >= 95
    drops = score_cycling(SUMMARY, {}, cycling_position="road_drops")
    assert "forearm_tilt" not in drops["component_scores"]
    # The lottery, pinned: a -25 forearm on the hoods used to cost points.
    assert out["overall_score"] == score_cycling(
        {**SUMMARY, "forearm_tilt_avg": 12.0}, {}, cycling_position="road_hoods",
    )["overall_score"]


def test_aero_bars_still_score_it():
    out = score_cycling({**SUMMARY, "forearm_tilt_avg": 12.0}, {}, cycling_position="triathlon")
    assert out["component_scores"]["forearm_tilt"] == 100.0
    assert out["coverage"]["measures_total"] == 9


def test_the_plan_does_not_praise_a_hoods_forearm():
    plan = build_action_plan(
        position="road_hoods", angle_statistics={}, sport_specific_metrics=SUMMARY,
        technique_score=90, letter_grade="A", detected_issues=[],
    )
    assert not any(g.get("metric") == "forearm_tilt" for g in plan.good_metrics)
    aero = build_action_plan(
        position="triathlon", angle_statistics={},
        sport_specific_metrics={**SUMMARY, "forearm_tilt_avg": 12.0},
        technique_score=90, letter_grade="A", detected_issues=[],
    )
    assert any(g.get("metric") == "forearm_tilt" for g in aero.good_metrics)
