"""Physically impossible angles are artefacts, and artefacts are counted, not averaged.

The score and the coach already read p05/p95, so a single mistracked frame
never moved the grade -- but the joint table printed the raw extremes, so
the same frame put "Knee min 8" in front of the reader with nothing marking
it. And a joint that lost every frame to the envelope looked identical to a
joint the camera never saw.
"""

from __future__ import annotations

import math

from app.services.video_analysis.biomechanics.base_analyzer import (
    ANGLE_ENVELOPES,
    ARTEFACT_FLAG_PCT,
    SportAnalyzer,
    angle_envelope,
    canonical_angle_name,
)
from app.services.video_analysis.biomechanics.confidence_scorer import (
    compute_analysis_confidence,
)
from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis


class _Stub(SportAnalyzer):
    def analyze_frame(self, *a, **k):  # pragma: no cover - not used
        raise NotImplementedError

    def compute_summary(self):  # pragma: no cover
        return {}

    def detect_issues(self):  # pragma: no cover
        return []


def _stats(**series: list[float]) -> dict:
    an = _Stub("run")
    n = max(len(v) for v in series.values())
    for i in range(n):
        an.add_frame_result(FrameAnalysis(
            timestamp_ms=i * 33.0,
            angles={k: (v[i] if i < len(v) else math.nan) for k, v in series.items()},
        ))
    return an.compute_angle_statistics()


# --------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------
def test_side_prefixes_share_the_joint_envelope():
    assert canonical_angle_name("left_knee") == "knee" and canonical_angle_name("knee") == "knee"
    assert angle_envelope("right_ankle") == ANGLE_ENVELOPES["ankle"]
    assert angle_envelope("shoulder") is None, "a 3-point shoulder angle is legitimately anywhere"


def test_the_envelope_is_looser_than_any_real_stroke():
    """Real extremes seen on the golden clips (tests/golden/*.json) must all
    sit inside: a sprinter's swing knee, a TT rider's closed hip, toe-off."""
    seen = {"knee": (60.7, 170.6), "hip": (41.2, 178.8), "elbow": (62.9, 145.7),
            "ankle": (52.2, 158.6), "trunk": (-6.8, 15.1), "trunk_angle": (17.2, 25.3),
            "forearm_tilt": (9.0, 18.4)}
    for joint, (lo, hi) in seen.items():
        env_lo, env_hi = ANGLE_ENVELOPES[joint]
        assert env_lo < lo and hi < env_hi, joint


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def test_one_impossible_frame_no_longer_sets_the_minimum():
    knee = [150.0, 140.0, 120.0, 100.0, 90.0, 100.0, 120.0, 140.0, 150.0, 8.0]
    s = _stats(knee=knee)["knee"]
    assert s["min"] == 90.0 and s["artefact_frames"] == 1 and s["valid_frames"] == 9
    assert s["artefact_pct"] == 10.0
    assert s["range"] == 60.0, "range is taken over the surviving frames too"


def test_a_frame_the_camera_lost_is_a_gap_not_an_artefact():
    s = _stats(knee=[150.0, math.nan, 120.0])["knee"]
    assert s["nan_frames"] == 1 and s["artefact_frames"] == 0
    assert s["nan_pct"] == round(100 / 3, 1)


def test_a_joint_lost_entirely_to_the_envelope_says_so():
    """Distinct from "never seen": mean is None either way, but the counts
    tell the table which sentence to print."""
    s = _stats(elbow=[5.0, 3.0, 12.0])["elbow"]
    assert s["mean"] is None
    assert s["valid_frames"] == 0 and s["artefact_frames"] == 3 and s["nan_frames"] == 0
    assert s["artefact_pct"] == 100.0 and s["nan_pct"] == 0.0


def test_joints_without_an_envelope_are_untouched():
    s = _stats(shoulder=[0.5, 179.5, 90.0])["shoulder"]
    assert s["min"] == 0.5 and s["max"] == 179.5 and s["artefact_frames"] == 0


def test_the_robust_extremes_are_computed_after_the_filter():
    knee = [100.0] * 50 + [160.0] * 50 + [5.0] * 3
    s = _stats(knee=knee)["knee"]
    assert s["p05"] == 100.0 and s["p95"] == 160.0 and s["min"] == 100.0


# --------------------------------------------------------------------------
# Summary-level plausibility (bike) -- never a silent drop
# --------------------------------------------------------------------------
def test_an_implausible_bike_mean_is_recorded_not_dropped():
    from app.services.video_analysis.biomechanics.cycling_analyzer import CyclingAnalyzer

    an = CyclingAnalyzer.__new__(CyclingAnalyzer)
    summary: dict = {}
    an._set_if_plausible(summary, "knee_at_bdc", 212.0)   # no knee opens past straight
    an._set_if_plausible(summary, "trunk_angle_avg", 41.0)
    assert "knee_at_bdc" not in summary, "downstream still reads absence as 'no measurement'"
    assert summary["outside_envelope"] == {"knee_at_bdc": 212.0}
    assert summary["trunk_angle_avg"] == 41.0


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
def _confidence(stats):
    return compute_analysis_confidence(
        angle_statistics=stats, frames_processed=400, butterworth_meta=None,
        analysis_warnings=[], landmark_quality={"overall_pct": 95.0, "confidence": "high"},
    )


def test_a_few_artefact_frames_do_not_touch_confidence():
    out = _confidence({"knee": {"nan_pct": 1.0, "artefact_pct": ARTEFACT_FLAG_PCT - 1, "valid_frames": 390}})
    assert out["level"] == "high" and out["factors"]["artefact_angles"] == []


def test_a_joint_with_many_artefact_frames_lowers_confidence_and_says_which():
    out = _confidence({
        "knee": {"nan_pct": 1.0, "artefact_pct": 12.0, "valid_frames": 350},
        "elbow": {"nan_pct": 1.0, "artefact_pct": 0.0, "valid_frames": 396},
    })
    assert out["level"] == "medium"
    assert out["factors"]["artefact_angles"] == ["knee"]
    assert "impossible" in out["explanation"] and "knee" in out["explanation"]


def test_old_results_without_the_field_still_score():
    out = _confidence({"knee": {"nan_pct": 1.0, "valid_frames": 396}})
    assert out["level"] == "high"
