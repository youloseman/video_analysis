"""BDC/TDC peak-picking must stand on measured frames at the series' own rate.

Two regressions this pins, both found by the 2026-08 audit before they could
ship in the knee-extremes change:

* The angle Butterworth interpolates short NaN gaps IN PLACE, so the filtered
  series' own NaN mask can no longer see them -- and the identity gate's
  exclusions cluster exactly at TDC, where the valleys are found. An extremum
  standing on an interpolated stretch is a number measured off a prediction.

* ``find_peaks`` spacing was computed from the nominal video fps, but adaptive
  sampling decimates long clips. At stride 4 the demanded spacing spanned two
  real revolutions: most strokes suppressed, only the tallest peaks kept,
  BDC biased up -- toward a false "saddle too high".
"""
from __future__ import annotations

import math

import numpy as np

from app.services.video_analysis.biomechanics.cycling_analyzer import CyclingAnalyzer


def _pedal_series(n: int, period: float, lo=65.0, hi=145.0, phase=0.0):
    mid, amp = (hi + lo) / 2, (hi - lo) / 2
    return [
        mid + amp * math.sin(2 * math.pi * (i / period) + phase)
        for i in range(n)
    ]


def _analyzer(fps: float, series: list[float], timestamps: list[float]):
    an = CyclingAnalyzer(fps=fps, frame_aspect=9 / 16)
    an._near_side = "right"
    an.camera_side = "right"
    an.right_knee_angles = list(series)
    an.left_knee_angles = [float("nan")] * len(series)
    an.angle_history["right_knee"] = list(series)
    an.angle_timestamps = list(timestamps)
    return an


def test_a_valley_on_a_gate_excluded_frame_is_not_measured():
    """The filtered series carries a plausible interpolated valley where the
    raw accumulator says nothing was measured. The raw mask must win."""
    period = 40.0
    n = 200
    series = _pedal_series(n, period, phase=-math.pi / 2)  # valleys at k*40
    an = _analyzer(60.0, series, [i / 60.0 for i in range(n)])

    # Gate excluded the frames around the SECOND valley: raw NaN there, but
    # the "filtered" series still carries smooth (interpolated) values.
    poisoned_valley = 40
    for k in range(poisoned_valley - 2, poisoned_valley + 3):
        an.right_knee_angles[k] = float("nan")

    arr = np.array(an.angle_history["right_knee"], dtype=float)
    raw_valid = ~np.isnan(np.array(an.right_knee_angles, dtype=float))
    out = an._bdc_tdc_from_peaks(arr, raw_valid=raw_valid)

    assert out is not None
    assert poisoned_valley not in out["bdc_indices"], "bdc unaffected here"
    # The poisoned valley must not appear among the *used* extremes: with the
    # mask it is discarded; without it, it was counted as measured.
    tdc_like = [i for i in range(n) if abs(arr[i] - 65.0) < 1.0]
    used_strokes = out["n_strokes"]
    out_unmasked = an._bdc_tdc_from_peaks(arr, raw_valid=None)
    assert out_unmasked["n_strokes"] > used_strokes, (
        f"masking must remove the poisoned extreme "
        f"({out_unmasked['n_strokes']} vs {used_strokes}; "
        f"valleys at {tdc_like[:6]}...)"
    )


def test_peak_spacing_follows_the_effective_fps_not_the_container():
    """A decimated series (stride 4: 15 analysed frames a second on a 60 fps
    clip) still has one countable stroke per revolution."""
    n = 150
    eff_fps = 15.0
    period = 10.0                      # 90 rpm at 15 analysed fps
    series = _pedal_series(n, period)
    an = _analyzer(60.0, series, [i / eff_fps for i in range(n)])

    result = an._get_bdc_tdc_angles()

    diag = an._bdc_tdc_diag.get("right", {})
    assert diag.get("method") == "peaks"
    # ~15 revolutions in the series; nominal-fps spacing (24 frames) would
    # have kept at most 6.
    assert diag.get("pedal_strokes", 0) >= 12
    assert abs(result["right_knee_at_bdc"] - 145.0) < 2.0
    assert abs(result["right_knee_at_tdc"] - 65.0) < 2.0
