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
