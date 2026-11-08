"""Cadence is graded one-sided: a fast runner is not penalised for being fast.

IMG_5585 (an elite runner, ~200 spm, eye-counted) lost 13 points on the cadence
component against the old 170-190 band. Below 170 the penalty is unchanged.
"""

from app.services.video_analysis.biomechanics.sport_configs import (
    CADENCE_TYPICAL,
    RUNNING_REFERENCE,
    reference_bands,
)
from app.services.video_analysis.biomechanics.technique_scorer import score_in_range


def _cadence_component(spm: float) -> float:
    from app.services.video_analysis.biomechanics.technique_scorer import score_running

    out = score_running({"cadence_spm": spm}, {})
    return out["component_scores"]["cadence"]


def test_high_cadence_is_full_marks():
    for spm in (170.0, 185.0, 200.0, 203.2, 215.0):
        assert _cadence_component(spm) == 100.0, spm


def test_low_cadence_penalty_is_unchanged():
    for spm in (150.0, 160.0, 165.0):
        assert _cadence_component(spm) == score_in_range(spm, *CADENCE_TYPICAL)
        assert _cadence_component(spm) < 100.0


def test_served_band_has_no_190_ceiling():
    band = reference_bands("run")["cadence_spm"]
    assert band["lo"] == 170 and band["hi"] == RUNNING_REFERENCE["cadence_spm"][1] > 205


# --- elbow: the same one-sided treatment (2026-11-07) ----------------------
# Every elite clip we hold carries the elbow at 62-69 deg and lost 45-70 points
# on the component against 85-100. The fault is an arm carried too open.

def _elbow_component(mean: float) -> float:
    from app.services.video_analysis.biomechanics.technique_scorer import score_running

    return score_running({}, {"elbow": {"mean": mean}})["component_scores"]["elbow_swing"]


def test_tight_elite_elbow_carry_is_full_marks():
    for deg in (55.0, 62.5, 64.3, 67.7, 90.0, 100.0):
        assert _elbow_component(deg) == 100.0, deg


def test_open_elbow_penalty_is_unchanged():
    from app.services.video_analysis.biomechanics.sport_configs import ELBOW_TYPICAL

    for deg in (110.0, 125.0):
        assert _elbow_component(deg) == score_in_range(deg, *ELBOW_TYPICAL) < 100.0


def test_action_plan_does_not_flag_a_tight_elbow():
    from app.services.video_analysis.biomechanics.running_action_plan_builder import (
        build_running_action_plan,
    )

    plan = build_running_action_plan({"score": 80, "summary": {"elbow_mean": 64.0}})
    assert not [d for d in plan.diagnostics if d.metric_name == "elbow_angle"]


# --- knee at contact and ground contact: the same treatment (2026-11-08) ------
# An elite clip (IMG_5585, 91/A) was told to extend the lower leg before contact
# (153 deg vs 160-175) and to spend longer on the ground (167 ms vs 180-270).

def _knee_contact_component(knee_max: float) -> float:
    from app.services.video_analysis.biomechanics.technique_scorer import score_running

    return score_running({}, {"knee": {"p95": knee_max, "p05": 70.0}})["component_scores"]["knee_contact"]


def test_a_more_flexed_landing_is_full_marks_and_a_locked_knee_is_not():
    from app.services.video_analysis.biomechanics.sport_configs import KNEE_CONTACT_TYPICAL

    for deg in (145.0, 152.9, 160.0, 168.0, 175.0):
        assert _knee_contact_component(deg) == 100.0, deg
    assert _knee_contact_component(179.3) == score_in_range(179.3, *KNEE_CONTACT_TYPICAL) < 100.0


def test_served_contact_bands_are_one_sided():
    bands = reference_bands("run")
    assert bands["knee_max"]["lo"] == 140 and bands["knee_max"]["hi"] == 175
    assert bands["ground_contact_ms"]["lo"] == 100 and bands["ground_contact_ms"]["hi"] == 270


def test_the_plan_names_the_locked_knee_and_leaves_a_soft_landing_alone():
    import dataclasses
    import json

    from app.services.video_analysis.biomechanics.running_action_plan_builder import (
        build_running_action_plan,
    )

    plan = build_running_action_plan({"score": 80, "summary": {"knee_max": 179.3}})
    assert [x for x in plan.diagnostics if x.metric_name == "knee_contact"]
    txt = json.dumps(dataclasses.asdict(plan), default=str) if dataclasses.is_dataclass(plan) else str(plan)
    assert "near-locked" in txt and "Insufficient knee flexion" not in txt
    soft = build_running_action_plan({"score": 80, "summary": {"knee_max": 152.9}})
    assert not [x for x in soft.diagnostics if x.metric_name == "knee_contact"]
