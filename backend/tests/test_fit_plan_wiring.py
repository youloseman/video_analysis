"""The cycling fit plan: built for rides, absent for runs, never free.

A rider's "what do I do now" is a part to move and a distance to move it, which
is a different shape from a runner's drills. These tests pin the two apart: the
keys must not collide, and the fit plan must sit behind the same paywall as the
drills without the gate having to be told about it.
"""
from __future__ import annotations

from app.services.result_gating import (
    ACCESS_FULL,
    ACCESS_TEASER,
    _UNLOCKS,
    gate_for_access,
)
from app.services.video_analysis.biomechanics.action_plan_builder import (
    action_plan_to_json,
    build_action_plan,
)


def _summary(knee_bdc: float = 148.0) -> dict:
    """Metrics for a rider whose saddle is a touch high."""
    return {
        "knee_at_bdc": knee_bdc,
        "knee_at_tdc": 72.0,
        "hip_angle_avg": 58.0,
        "trunk_angle_avg": 22.0,
        "elbow_angle_avg": 95.0,
        "shoulder_angle_avg": 88.0,
    }


def test_a_high_saddle_produces_an_adjustment_with_a_direction_and_an_amount():
    plan = action_plan_to_json(build_action_plan(
        position="triathlon", angle_statistics={}, sport_specific_metrics=_summary(152.0),
        technique_score=88, letter_grade="B", detected_issues=[],
    ))
    saddle = [d for d in plan["diagnostics"] if d["component"] == "saddle_height"]
    assert saddle, "a knee 3 deg past the top of its band should be called out"
    d = saddle[0]
    assert d["action"] == "lower_saddle"      # direction, not just "wrong"
    assert d["amount"]                        # and how far to move it
    assert d["reason"]                        # and why, so it can be argued with
    assert tuple(d["target_range"]) == (138, 145)


def test_a_knee_a_hair_past_the_band_is_not_yet_an_adjustment():
    """There is a tolerance on the boundary, and it is deliberate.

    148 deg against a 138-145 band is 3 deg over and does NOT produce advice;
    148.2 does. Telling somebody to take a 5 mm spanner to their bike because a
    2D estimate landed one tenth of a degree outside a reference range would be
    false precision, and this is where that line sits.
    """
    quiet = action_plan_to_json(build_action_plan(
        position="triathlon", angle_statistics={}, sport_specific_metrics=_summary(148.0),
        technique_score=90, letter_grade="A", detected_issues=[],
    ))
    assert not [d for d in quiet["diagnostics"] if d["component"] == "saddle_height"]


def test_a_saddle_inside_its_band_is_not_an_adjustment():
    plan = action_plan_to_json(build_action_plan(
        position="triathlon", angle_statistics={}, sport_specific_metrics=_summary(141.0),
        technique_score=95, letter_grade="A", detected_issues=[],
    ))
    assert not [d for d in plan["diagnostics"] if d["component"] == "saddle_height"]
    # ...and it says so, rather than going quiet about the part that is fine.
    assert plan["good_metrics"]


def test_the_two_plans_never_share_a_key():
    """training_plan is drills; fit_plan is adjustments. Same field = future bug."""
    plan = action_plan_to_json(build_action_plan(
        position="road_hoods", angle_statistics={}, sport_specific_metrics=_summary(),
        technique_score=80, letter_grade="B", detected_issues=[],
    ))
    assert "top_3_priorities" not in plan   # the runner's shape
    assert "diagnostics" in plan            # the rider's


def test_the_gate_withholds_the_fit_plan_from_a_free_rider():
    """The allowlist means a new paid field is withheld by default -- prove it."""
    result = {
        "status": "completed", "sport_type": "bike", "technique_score": 88,
        "letter_grade": "B", "fit_plan": {"diagnostics": [{"component": "saddle_height"}]},
        "angle_statistics": {"right_knee": {"mean": 140}},
    }
    free = gate_for_access(dict(result), ACCESS_TEASER)
    assert "fit_plan" not in free
    paid = gate_for_access(dict(result), ACCESS_FULL)
    assert paid["fit_plan"]["diagnostics"]


def test_the_upsell_names_the_fit_plan():
    """A locked rider should be told what they are missing, by name."""
    assert "fit" in _UNLOCKS
