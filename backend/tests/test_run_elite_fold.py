"""An elite runner must not be marked down for running like one.

IMG_5455 (2026-11-01): a Kenyan elite filmed side-on in 4x slow motion scored
84/B. Two causes, both pinned here:

  1. The swing-knee band was two-sided (80-100). His knee folded to 48 deg --
     more fold, the GOOD direction, deepening with speed -- and the component
     scored 24/100, which the worst-component pull turned into -8 points.
  2. On a slow-motion run the landmark filter kept its normal-speed cutoff,
     so it sat ~20x above the stride and passed every frame's re-detection
     jitter: the skeleton visibly shook. Bike already scaled its cutoff to
     the clip's own period; run did not.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

from app.services.video_analysis.biomechanics.landmark_stabilizer import (
    stabilize_landmarks,
)
from app.services.video_analysis.biomechanics.sport_configs import reference_bands
from app.services.video_analysis.biomechanics.technique_scorer import score_running


def _summary():
    return {
        "trunk_lean_avg": 4.0,
        "cadence_spm": 180.0,
        "vertical_oscillation_m": 0.08,
        "overstride_ratio": 0.10,
    }


def _knee(p05: float) -> dict:
    return {"knee": {"p95": 165.0, "p05": p05}, "elbow": {"mean": 90.0}}


def test_a_deep_swing_fold_is_not_a_fault():
    elite = score_running(_summary(), _knee(48.0))
    assert elite["component_scores"]["knee_swing"] == 100.0


def test_a_knee_that_does_not_fold_still_is():
    stiff = score_running(_summary(), _knee(125.0))
    assert stiff["component_scores"]["knee_swing"] < 60.0


def test_the_served_band_says_the_same_thing():
    band = reference_bands("run")["knee_min"]
    assert band["hi"] == 100
    assert band["lo"] <= 48          # the elite's fold reads "in range" on the tile


def _lm(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)


def _running(n: int, stride_frames: float) -> list[dict]:
    """Two ankles bouncing in antiphase, one bounce per stride."""
    frames = []
    for i in range(n):
        ph = 2 * math.pi * i / stride_frames
        lms = [_lm(0.5, 0.4) for _ in range(33)]
        for idx, off in ((27, 0.0), (28, math.pi)):
            x = 0.5 + 0.08 * math.cos(ph + off)
            y = 0.8 - 0.05 * max(0.0, math.sin(ph + off))
            lms[idx] = _lm(x, y)
            lms[idx - 2] = _lm(x, y - 0.15)            # knee rides above
            lms[idx + 2] = _lm(x - 0.02, y + 0.01)     # heel
            lms[idx + 4] = _lm(x + 0.03, y + 0.01)     # toe
        frames.append({
            "normalized_landmarks": lms,
            "world_landmarks": [_lm(p.x, p.y) for p in lms],
            "timestamp_ms": i * 1000.0 / 30.0,
        })
    return frames


def test_slow_motion_run_filter_follows_the_stride():
    ctx: dict = {}
    # 4x slow motion: a 0.7 s stride spread over ~84 frames at 30 fps
    stabilize_landmarks(_running(360, 84.0), "run", None, fps=30.0, context=ctx)
    assert ctx["butterworth_meta"]["actual_cutoff_hz"] < 3.0


def test_normal_speed_run_keeps_the_shipped_cutoff():
    ctx: dict = {}
    stabilize_landmarks(_running(180, 21.0), "run", None, fps=30.0, context=ctx)
    assert ctx["butterworth_meta"]["actual_cutoff_hz"] == 6.0
