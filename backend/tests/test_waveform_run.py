"""The running waveform comparison must actually run.

The near-side filter matched key PREFIXES, run keys are unprefixed (already
near-side by construction), so the joint map came back empty on every running
clip with a camera side -- the comparison never reached the reference file,
8% of the running rubric was silently unearnable, and score_coverage capped
at 0.92 on perfect footage. The coverage test stubbed a similarity score
production could not produce, so the whole chain stayed green.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.services.video_analysis.biomechanics.waveform_comparator import (
    SPORT_COMPARE_JOINTS,
    _get_compare_joints,
    compute_waveform_comparison,
)

_REF = (
    Path(__file__).resolve().parents[1]
    / "app" / "services" / "video_analysis" / "biomechanics"
    / "reference_data" / "running_reference.json"
)


def test_unprefixed_run_keys_survive_the_side_filter():
    assert _get_compare_joints("run", "left") == SPORT_COMPARE_JOINTS["run"]
    assert _get_compare_joints("run", "right") == SPORT_COMPARE_JOINTS["run"]
    # Bike keys carry their side and must still be filtered by it.
    bike_left = _get_compare_joints("bike", "left")
    assert bike_left and all(k.startswith("left_") for k in bike_left)


def test_a_running_clip_with_a_camera_side_gets_a_similarity_score():
    """End to end against the real reference file, not a stub: an athlete
    whose knee traces the reference mean must score, not silently skip."""
    ref = json.loads(_REF.read_text())
    knee_mean = np.array(ref["joints"]["knee"]["mean"], dtype=float)

    angle_history = {
        "knee": list(knee_mean),
        "hip": list(np.array(ref["joints"]["hip"]["mean"], dtype=float)),
    }
    timestamps = list(np.linspace(0.0, 0.7, len(knee_mean)))

    out = compute_waveform_comparison(
        angle_history, timestamps, "run", phase_data=None, camera_side="left",
    )

    assert out["comparisons"], "the comparison must not skip on a run clip"
    assert out["overall_similarity_score"] is not None
    assert out["overall_similarity_score"] > 80.0
