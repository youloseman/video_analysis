"""Slow motion read from the frames, before anything reads time.

The earlier path was the last resort after every other cadence method, on a spectrum
that needs several steps. A runner crossing the frame in 8x slow motion is about a
second of running, so a clip like that came back with no cadence at all, or with a
false one (136 spm for a run at about 192). These pin the reading on the shapes that
matter: one whole step in a slow clip, a normal clip staying normal, and a tracking
hiccup at normal speed not being taken for slow motion.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.video_analysis.biomechanics.running_analyzer import RunningAnalyzer


def lm(x, y=0.5):
    return SimpleNamespace(x=x, y=y, visibility=0.9)


def analyzer(gaps: list[float], fps: float, bounce_per_step: float | None = None) -> RunningAnalyzer:
    """An analyzer holding only what the reading looks at: where the two ankles are."""
    a = RunningAnalyzer(fps=fps)
    a.frame_results = []
    for i, g in enumerate(gaps):
        marks = [lm(0.5) for _ in range(33)]
        y = 0.8 + (0.02 * math.sin(2 * math.pi * i / bounce_per_step) if bounce_per_step else 0.0)
        marks[27], marks[28] = lm(0.5 + g, y), lm(0.5 - g, y)
        a.frame_results.append(SimpleNamespace(
            timestamp_ms=i / fps * 1000.0,
            extra_metrics={"_norm_landmarks": marks, "_norm_left_ankle_y": y, "_norm_right_ankle_y": y},
        ))
    return a


def steps(frames: int, per_step: float, phase: float = 0.0) -> list[float]:
    """The signed gap between the feet: a sine with a period of two steps."""
    return [0.12 * math.sin(math.pi * (i + phase) / per_step) for i in range(frames)]


def test_a_second_of_running_in_8x_is_read_from_one_whole_step():
    a = analyzer(steps(193, 75, phase=-28.5), fps=30.0)
    a._detect_slow_motion()
    assert a._slowmo_capture_fps == 240
    assert a._slowmo_cadence == pytest.approx(192, abs=3)
    assert a._slowmo_factor == 8


def test_a_clock_that_lies_does_not_move_the_answer():
    """An iPhone edit list reported 26.65 fps for a clip that plays at 30. A frame of
    slow motion is 1/240 s whatever the container says."""
    a = analyzer(steps(193, 75, phase=-28.5), fps=26.65)
    a._detect_slow_motion()
    assert a._slowmo_capture_fps == 240
    assert a._slowmo_cadence == pytest.approx(192, abs=3)
    assert a._slowmo_factor == pytest.approx(240 / 26.65)
    # A frame is then 1/240 s, whatever the timestamps were built from.
    assert a._median_frame_spacing_ms() == pytest.approx(1000 / 240)


def test_4x_is_told_from_8x_by_what_a_runner_can_do():
    a = analyzer(steps(400, 42), fps=30.0)
    a._detect_slow_motion()
    assert a._slowmo_capture_fps == 120
    assert a._slowmo_cadence == pytest.approx(171, abs=3)


@pytest.mark.parametrize("fps,per_step", [(30.0, 10.5), (60.0, 21.0)])
def test_a_normal_clip_stays_normal(fps, per_step):
    a = analyzer(steps(300, per_step), fps=fps)
    a._detect_slow_motion()
    assert a._slowmo_capture_fps is None and a._slowmo_factor is None


def test_half_a_cadence_is_left_alone():
    """85 spm in the file is 2x slow motion or a real shuffle, and nothing here can
    tell them apart."""
    a = analyzer(steps(300, 21.0), fps=30.0)
    a._detect_slow_motion()
    assert a._slowmo_factor is None


def test_a_hiccup_at_normal_speed_is_not_slow_motion():
    """Two far-apart bumps over steady slow motion of the feet: the gap alone reads
    as 8x, and the ankles bouncing at 171 spm in the file's own time say it is not."""
    def bump(i: int, c: int) -> float:
        return 0.15 * math.exp(-((i - c) / 10.0) ** 2)

    gaps = [0.01 * math.sin(i / 30) + bump(i, 40) + bump(i, 115) for i in range(193)]
    a = analyzer(gaps, fps=30.0, bounce_per_step=10.5)
    a._detect_slow_motion()
    assert a._slowmo_factor is None


@pytest.mark.parametrize("motion,slow", [(0.025, True), (0.061, True), (0.081, False), (0.2, False)])
def test_how_much_the_feet_move_per_frame_decides(motion, slow):
    """The measured split: 2.4-6.1% of their range per frame on real slow clips, 8%+
    on normal ones. A rhythm that looks slow with feet moving this fast is a hiccup."""
    a = analyzer(steps(193, 75), fps=30.0)
    a._step_frames_from_ankle_gap = lambda: (75.0, 2, motion)
    a._detect_slow_motion()
    assert bool(a._slowmo_factor) is slow


def test_feet_that_never_separate_say_nothing():
    a = analyzer([0.005 * math.sin(i / 20) for i in range(200)], fps=30.0)
    a._detect_slow_motion()
    assert a._slowmo_factor is None


def test_a_slow_clip_gets_its_cadence_before_the_other_methods_can_guess():
    a = analyzer(steps(193, 75, phase=-28.5), fps=30.0)
    # Stands in for the method that once accepted 136 spm off this kind of clip.
    a._detect_steps_from_knee_angles = lambda: pytest.fail("the knee method must not run")
    assert a._compute_cadence() == pytest.approx(192, abs=3)
    assert a._slowmo_capture_fps == 240


def test_the_edge_of_the_recording_is_not_a_landing():
    a = RunningAnalyzer(fps=30.0)
    a._contact_frame_indices = lambda min_run=3: [0, 40, 80]
    assert a._landing_frame_indices() == [40, 80]
