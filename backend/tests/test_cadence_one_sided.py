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
