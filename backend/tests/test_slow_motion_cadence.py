"""Reading a real cadence out of a slow-motion clip, or refusing to.

Phones shoot slo-mo at 120 or 240 fps and store it at 30, so the stored
timeline runs 4x or 8x too slow and every time-derived metric with it. Today
that clip reports "cadence n/a" and withholds ground contact and flight -- the
honest answer, but a poor one when the answer is actually recoverable.

It is recoverable because running cadence is tightly bounded. Ankles bouncing
at 22 spm cannot be a real rhythm; multiply by 8 and you get 176, which is
exactly a runner, while 4x would give 88, which is nobody. When one factor
fits and the others do not, the answer is determined rather than guessed.

The refusals matter as much as the recoveries: 2x slow motion maps a normal
stride onto ~85 spm, which is also a plausible (if slow) real rhythm, and
nothing in the ankle trace separates them. Doubling a real number is worse
than admitting we cannot tell, so that case stays unanswered.
"""

from __future__ import annotations

import math

from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis
from app.services.video_analysis.biomechanics.running_analyzer import RunningAnalyzer

STORED_FPS = 30.0
DURATION_S = 12.0


def ankle_y(phi: float) -> float:
    return 0.82 - 0.05 * max(0.0, math.sin(phi)) ** 1.5


def analyzer_at(real_spm: float, slowdown: float = 1.0) -> RunningAnalyzer:
    """Frames as a phone would store them: real motion divided by slowdown."""
    analyzer = RunningAnalyzer(fps=STORED_FPS)
    stride_hz = (real_spm / 120.0) / slowdown
    n = int(DURATION_S * STORED_FPS)
    for i in range(n):
        t = i / STORED_FPS
        phi = 2 * math.pi * stride_hz * t
        fr = FrameAnalysis(timestamp_ms=t * 1000.0)
        fr.angles["knee"] = 130.0 + 35.0 * math.sin(phi + math.pi / 2)
        fr.extra_metrics["_norm_left_ankle_y"] = ankle_y(phi)
        fr.extra_metrics["_norm_right_ankle_y"] = ankle_y(phi + math.pi)
        analyzer.frame_results.append(fr)
    return analyzer


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------
def test_eight_times_slow_motion_is_recognised_and_undone():
    analyzer = analyzer_at(176.0, slowdown=8.0)
    cadence = analyzer._compute_cadence()
    assert analyzer._slowmo_factor == 8
    assert abs(cadence - 176.0) < 10.0, cadence


def test_four_times_slow_motion_is_recognised_and_undone():
    analyzer = analyzer_at(180.0, slowdown=4.0)
    cadence = analyzer._compute_cadence()
    assert analyzer._slowmo_factor == 4
    assert abs(cadence - 180.0) < 10.0, cadence


def test_the_recovered_factor_also_rescales_durations():
    """Ground contact and flight read time off the same stretched timeline,
    so correcting the frame spacing has to correct them too."""
    analyzer = analyzer_at(176.0, slowdown=8.0)
    stretched = analyzer._median_frame_spacing_ms()
    analyzer._compute_cadence()
    assert analyzer._slowmo_factor == 8
    assert analyzer._median_frame_spacing_ms() == stretched / 8


def test_the_athlete_is_told_the_clip_was_rescaled():
    analyzer = analyzer_at(176.0, slowdown=8.0)
    analyzer._compute_cadence()
    assert any("slow motion" in w for w in analyzer._analyzer_warnings)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_a_normal_speed_clip_is_never_rescaled():
    analyzer = analyzer_at(172.0)
    cadence = analyzer._compute_cadence()
    assert analyzer._slowmo_factor is None
    assert abs(cadence - 172.0) < 8.0, cadence
    assert analyzer._median_frame_spacing_ms() > 0


def test_two_times_slow_motion_is_refused_rather_than_doubled():
    """~85 spm is equally consistent with a shuffle and with 2x slow motion.
    Guessing would double a real measurement."""
    analyzer = analyzer_at(170.0, slowdown=2.0)
    cadence = analyzer._compute_cadence()
    assert analyzer._slowmo_factor is None
    assert cadence == 0.0


def test_a_rhythm_no_factor_can_rescue_is_left_alone():
    """9 spm times 4 or 8 still lands nowhere near a running cadence."""
    analyzer = analyzer_at(176.0, slowdown=20.0)
    assert analyzer._compute_cadence() == 0.0
    assert analyzer._slowmo_factor is None


def test_a_flat_signal_produces_no_slow_motion_verdict():
    analyzer = RunningAnalyzer(fps=STORED_FPS)
    for i in range(int(DURATION_S * STORED_FPS)):
        fr = FrameAnalysis(timestamp_ms=i / STORED_FPS * 1000.0)
        fr.extra_metrics["_norm_left_ankle_y"] = 0.8
        fr.extra_metrics["_norm_right_ankle_y"] = 0.8
        analyzer.frame_results.append(fr)
    assert analyzer._compute_cadence() == 0.0
    assert analyzer._slowmo_factor is None


def test_the_summary_says_the_time_base_was_inferred():
    analyzer = analyzer_at(176.0, slowdown=8.0)
    analyzer.camera_side = "left"
    summary = analyzer.compute_summary()
    assert summary.get("slow_motion_factor") == 8
    assert summary.get("time_base_inferred") is True
