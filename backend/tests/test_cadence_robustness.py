"""Cadence must survive the knee-valley method being fed a corrupted knee.

A backlit clip whose leg indices kept swapping fragmented the near-knee
valley train and published 124 spm for a ~165 spm run -- plausible enough
to pass every gate, wrong enough to drive the whole coaching plan. The
spectral estimator reads the averaged both-ankles bounce (immune to
left/right identity swaps) and overrides the knee method when the two
disagree.
"""

from __future__ import annotations

import math

from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis
from app.services.video_analysis.biomechanics.running_analyzer import RunningAnalyzer

FPS = 30.0
SPM = 165.0
STRIDE_HZ = SPM / 120.0     # per-leg stride frequency
DURATION_S = 10.0


def ankle_y(phi: float) -> float:
    """Per-leg vertical ankle trajectory: flat-ish stance, lifted swing."""
    swing = max(0.0, math.sin(phi))
    return 0.82 - 0.05 * swing ** 1.5


def build_analyzer(knee_fn) -> RunningAnalyzer:
    """Analyzer with fabricated frame_results: clean ankles, knee via knee_fn."""
    analyzer = RunningAnalyzer(fps=FPS)
    n = int(DURATION_S * FPS)
    for i in range(n):
        t = i / FPS
        phi = 2 * math.pi * STRIDE_HZ * t
        fr = FrameAnalysis(timestamp_ms=t * 1000.0)
        fr.angles["knee"] = knee_fn(phi, i)
        fr.extra_metrics["_norm_left_ankle_y"] = ankle_y(phi)
        fr.extra_metrics["_norm_right_ankle_y"] = ankle_y(phi + math.pi)
        analyzer.frame_results.append(fr)
    return analyzer


def clean_knee(phi: float, _i: int) -> float:
    return 130.0 + 35.0 * math.sin(phi + math.pi / 2)


def test_spectral_estimate_reads_the_true_cadence():
    analyzer = build_analyzer(clean_knee)
    spm = analyzer._cadence_spectral()
    assert abs(spm - SPM) < 5.0, spm


def test_agreeing_methods_keep_the_knee_estimate():
    analyzer = build_analyzer(clean_knee)
    cadence = analyzer._compute_cadence()
    assert abs(cadence - SPM) < 8.0, cadence


def test_corrupted_knee_train_is_overridden_by_spectral():
    """Every 3rd stride's valley is erased -- the classic swap artifact.

    The surviving valleys still yield in-gate intervals, so the knee method
    returns a plausible-but-low cadence on its own; the cross-check must
    hand the answer to the swap-immune spectral reading instead.
    """
    def corrupted_knee(phi: float, i: int) -> float:
        stride_no = int(phi / (2 * math.pi))
        if stride_no % 3 == 2:
            return 140.0  # valley erased: sits above the series mean
        return clean_knee(phi, i)

    analyzer = build_analyzer(corrupted_knee)
    knee_only = analyzer._compute_cadence_from_strides(
        analyzer._detect_steps_from_knee_angles()
    )
    cadence = analyzer._compute_cadence()
    assert abs(cadence - SPM) < 6.0, (knee_only, cadence)


def test_spectral_refuses_flat_or_short_signals():
    analyzer = RunningAnalyzer(fps=FPS)
    for i in range(60):
        fr = FrameAnalysis(timestamp_ms=i / FPS * 1000.0)
        fr.extra_metrics["_norm_left_ankle_y"] = 0.8   # no rhythm at all
        fr.extra_metrics["_norm_right_ankle_y"] = 0.8
        analyzer.frame_results.append(fr)
    assert analyzer._cadence_spectral() == 0.0

    short = build_analyzer(clean_knee)
    short.frame_results = short.frame_results[:10]
    assert short._cadence_spectral() == 0.0


# ---------------------------------------------------------------------------
# Body scale: one bad opening frame must not poison meter conversions
# ---------------------------------------------------------------------------
class _LM:
    def __init__(self, y: float):
        self.x = 0.5
        self.y = y
        self.z = 0.0
        self.visibility = 0.95


def _frame_landmarks(torso_norm: float):
    """33 landmarks with shoulders/hips spanning torso_norm vertically."""
    lms = [_LM(0.5) for _ in range(33)]
    for i in (11, 12):
        lms[i] = _LM(0.4)
    for i in (23, 24):
        lms[i] = _LM(0.4 + torso_norm)
    return lms


def test_body_scale_uses_the_median_not_the_first_frame():
    analyzer = RunningAnalyzer(fps=FPS)
    # Garbage opening detection: tiny torso -> huge meters-per-unit scale
    # (this is what once doubled vertical oscillation to 17 cm).
    analyzer._estimate_body_scale(_frame_landmarks(0.05))
    for _ in range(30):
        analyzer._estimate_body_scale(_frame_landmarks(0.20))
    analyzer._lock_body_scale()  # first consumer (VOSC) locks lazily
    assert analyzer._body_scale_estimated
    assert abs(analyzer._pixel_to_meter - 0.45 / 0.20) < 0.05


def test_body_scale_locks_lazily_for_short_clips():
    analyzer = RunningAnalyzer(fps=FPS)
    for _ in range(5):
        analyzer._estimate_body_scale(_frame_landmarks(0.18))
    assert not analyzer._body_scale_estimated
    analyzer._lock_body_scale()
    assert analyzer._body_scale_estimated
    assert abs(analyzer._pixel_to_meter - 0.45 / 0.18) < 0.05
