"""Which knee series the reported BDC/TDC are read off.

The rider watches the overlay video print a knee angle on every frame and then
reads a single "knee at TDC" number in the report. Those two have to be the same
measurement. They were not: the overlay prints ``angle_history``, which the
advanced pipeline low-passes in place, while BDC/TDC were picked off the raw
per-frame accumulator -- so the report could claim a maximum flexion deeper than
any value the video ever showed, which reads as an invented number.

Peak-picking is the worst place for that gap: a valley of an unfiltered signal
is the movement plus whatever noise pointed down on that frame, and BDC is what
the saddle-height verdict is read from.
"""
from __future__ import annotations

import math

import numpy as np

from app.services.video_analysis.biomechanics.cycling_analyzer import CyclingAnalyzer

FPS = 60.0
STROKE_FRAMES = 40          # ~90 rpm at 60 fps
TDC, BDC = 66.0, 145.0


def _pedal_strokes(n_strokes: int = 8, tdc: float = TDC, bdc: float = BDC) -> list[float]:
    """A clean knee trace: one flexion valley + one extension peak per rev."""
    mid, amp = (bdc + tdc) / 2, (bdc - tdc) / 2
    return [
        mid - amp * math.cos(2 * math.pi * (i / STROKE_FRAMES) + math.pi)
        for i in range(n_strokes * STROKE_FRAMES)
    ]


def _spiked(series: list[float], depth: float = 9.0) -> list[float]:
    """The same trace with a downward noise spike on each valley.

    Stands in for what the unfiltered signal carries: the low-pass pass the
    pipeline runs before compute_summary is exactly what removes these.
    """
    out = list(series)
    for i in range(len(out)):
        if i % STROKE_FRAMES == STROKE_FRAMES // 2:
            out[i] -= depth
    return out


def _analyzer(*, filtered: list[float] | None, raw: list[float], side: str = "left") -> CyclingAnalyzer:
    a = object.__new__(CyclingAnalyzer)
    a.fps = FPS
    a.angle_history = {f"{side}_knee": list(filtered)} if filtered else {}
    a.left_knee_angles = list(raw) if side == "left" else []
    a.right_knee_angles = list(raw) if side == "right" else []
    a._bdc_tdc_diag = {}
    a._analyzer_warnings = []
    # get_effective_fps falls back to self.fps on an empty timestamp list --
    # __init__ normally creates this, and these analyzers bypass __init__.
    a.angle_timestamps = []
    return a


def test_extremes_come_from_the_series_the_overlay_prints():
    clean = _pedal_strokes()
    a = _analyzer(filtered=clean, raw=_spiked(clean))

    got = a._get_bdc_tdc_angles()

    assert got["left_knee_at_tdc"] == np.float64(min(clean)) or abs(
        got["left_knee_at_tdc"] - min(clean)
    ) < 0.6
    assert abs(got["left_knee_at_bdc"] - max(clean)) < 0.6


def test_reported_tdc_is_never_deeper_than_anything_the_video_showed():
    """The complaint that started this: report 54 deg, overlay never below 57."""
    clean = _pedal_strokes()
    a = _analyzer(filtered=clean, raw=_spiked(clean, depth=12.0))

    tdc = a._get_bdc_tdc_angles()["left_knee_at_tdc"]

    assert tdc >= min(clean) - 0.1, "reported flexion deeper than the displayed minimum"


def test_the_raw_accumulator_still_covers_a_clip_the_filter_never_ran_on():
    """Too few frames for the advanced pipeline -> angle_history is empty."""
    clean = _pedal_strokes()
    a = _analyzer(filtered=None, raw=clean)

    got = a._get_bdc_tdc_angles()

    assert abs(got["left_knee_at_tdc"] - min(clean)) < 0.6
    assert abs(got["left_knee_at_bdc"] - max(clean)) < 0.6


def test_a_series_too_short_to_mean_anything_is_left_alone():
    a = _analyzer(filtered=[70.0, 71.0], raw=[70.0, 71.0])
    assert a._get_bdc_tdc_angles() == {}


def test_both_sides_are_read_off_their_own_filtered_series():
    clean = _pedal_strokes()
    a = object.__new__(CyclingAnalyzer)
    a.fps = FPS
    a.angle_history = {
        "left_knee": clean,
        "right_knee": _pedal_strokes(tdc=TDC + 6, bdc=BDC - 4),
    }
    a.left_knee_angles = []
    a.right_knee_angles = []
    a._bdc_tdc_diag = {}
    a._analyzer_warnings = []
    a.angle_timestamps = []

    got = a._get_bdc_tdc_angles()

    assert abs(got["left_knee_at_tdc"] - TDC) < 0.6
    assert abs(got["right_knee_at_tdc"] - (TDC + 6)) < 0.6
