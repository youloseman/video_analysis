"""Stance is where the foot is planted -- not merely where it is not moving down.

The shipped detector read ``ankle_y_velocity < 0.001``, which accepts every
NEGATIVE velocity, so an ankle travelling upward -- the whole heel-recovery
phase -- counted as ground contact. On a cleanly tracked clip it called 95% of
the cycle stance, against 30-40% for real running, and the plausibility gates
downstream hid the damage by withholding the metrics it poisoned.

These tests build a stride whose duty factor is known by construction and check
the whole-clip pass recovers it, at two frame rates and under slow motion.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis
from app.services.video_analysis.biomechanics.running_analyzer import (
    GROUND_CONTACT_PHASES,
    RunningAnalyzer,
)

DUTY = 0.35          # share of each leg's cycle spent on the ground
CADENCE = 168.0      # steps per minute


def _lm(x, y, vis=0.9):
    return SimpleNamespace(x=float(x), y=float(y), z=0.0, visibility=vis)


def _foot(phase: float) -> tuple[float, float]:
    """One foot's path, hip-relative, over its cycle ``phase`` in [0, 1).

    Stance: planted, so the hips pass over it and it slides straight back at a
    constant rate, low to the ground. Swing: returns forward over the rest of
    the cycle, lifting as it goes.
    """
    if phase < DUTY:
        t = phase / DUTY
        return (0.25 - 0.5 * t, 0.42)                    # back fast, stays low
    t = (phase - DUTY) / (1.0 - DUTY)
    return (-0.25 + 0.5 * t, 0.42 - 0.16 * math.sin(math.pi * t))


def build(seconds=6.0, fps=60.0, slow=1.0, cadence=CADENCE):
    """A runner whose feet do exactly the above, half a cycle apart.

    ``slow`` stretches the TIMELINE without changing the movement, which is
    what a slow-motion clip is: at 8x every per-frame velocity is an eighth of
    its real value, and an absolute velocity threshold silently stops meaning
    anything.
    """
    analyzer = RunningAnalyzer(fps=fps)
    analyzer.camera_side = "left"
    n = int(seconds * fps)
    stride_s = 120.0 / cadence * slow          # one leg's cycle, on the timeline
    for i in range(n):
        phase = (i / fps / stride_s) % 1.0
        lx, ly = _foot(phase)
        rx, ry = _foot((phase + 0.5) % 1.0)
        hip_x, hip_y = 0.5, 0.45
        lms = [_lm(hip_x, hip_y - 0.25) for _ in range(33)]
        lms[11] = _lm(hip_x, hip_y - 0.22)
        lms[12] = _lm(hip_x, hip_y - 0.22)
        lms[23] = _lm(hip_x - 0.01, hip_y)
        lms[24] = _lm(hip_x + 0.01, hip_y)
        lms[27] = _lm(hip_x + lx, hip_y + ly)          # left ankle
        lms[28] = _lm(hip_x + rx, hip_y + ry)          # right ankle
        # Toes ahead of heels: the direction-of-travel cue.
        lms[29], lms[31] = _lm(hip_x + lx - 0.02, hip_y + ly), _lm(hip_x + lx + 0.03, hip_y + ly)
        lms[30], lms[32] = _lm(hip_x + rx - 0.02, hip_y + ry), _lm(hip_x + rx + 0.03, hip_y + ry)

        fr = FrameAnalysis(timestamp_ms=i / fps * 1000.0)
        fr.angles["knee"] = 150.0 if phase < DUTY else 95.0
        fr.extra_metrics["gait_phase"] = "unknown"
        fr.extra_metrics["_norm_landmarks"] = lms
        analyzer.frame_results.append(fr)
        analyzer.angle_timestamps.append(i / fps)
    return analyzer


def stance_share(analyzer) -> float:
    return float(np.mean([
        fr.extra_metrics["gait_phase"] in GROUND_CONTACT_PHASES
        for fr in analyzer.frame_results
    ]))


# --- the bug it replaces ---------------------------------------------------

def test_an_ankle_travelling_upward_is_not_ground_contact():
    """The whole defect in one line: a NEGATIVE velocity passed `< 0.001`, so
    heel recovery counted as stance."""
    analyzer = RunningAnalyzer(fps=60.0)
    lifting = analyzer.detect_gait_phase(
        knee_angle=150.0, ankle_y=0.7, hip_y=0.45, foot_y=0.7,
        ankle_y_velocity=-0.02,          # rising fast
    )
    assert lifting.value not in GROUND_CONTACT_PHASES


# --- the replacement -------------------------------------------------------

def test_the_whole_clip_pass_recovers_the_duty_factor():
    """Feet built to be down 35% of each cycle each; with two legs alternating,
    SOME foot is down about 70% of the time."""
    analyzer = build()
    meta = analyzer.recompute_gait_phases()
    assert meta["method"] == "lower_ankle_fore_aft"
    assert stance_share(analyzer) == pytest.approx(2 * DUTY, abs=0.12)


@pytest.mark.parametrize("fps", [30.0, 60.0, 120.0])
def test_the_answer_does_not_depend_on_frame_rate(fps):
    """The old threshold was an absolute per-frame distance, so it meant
    different things at every frame rate."""
    analyzer = build(fps=fps)
    analyzer.recompute_gait_phases()
    assert stance_share(analyzer) == pytest.approx(2 * DUTY, abs=0.12)


@pytest.mark.parametrize("slow", [1.0, 4.0, 8.0])
def test_slow_motion_does_not_change_the_answer(slow):
    """At 8x every per-frame velocity is an eighth of its real value. A clip
    like that is what showed 95% of the cycle called stance."""
    analyzer = build(seconds=6.0 * slow, slow=slow)
    analyzer.recompute_gait_phases()
    assert stance_share(analyzer) == pytest.approx(2 * DUTY, abs=0.12)


def test_a_faster_cadence_gives_more_contacts_not_a_different_duty():
    slow_runner = build(cadence=150.0)
    fast_runner = build(cadence=190.0)
    slow_runner.recompute_gait_phases()
    fast_runner.recompute_gait_phases()
    assert len(fast_runner.stance_runs()) > len(slow_runner.stance_runs())
    assert stance_share(fast_runner) == pytest.approx(
        stance_share(slow_runner), abs=0.12)


def test_a_clip_too_short_to_hold_a_cycle_is_left_alone():
    analyzer = build(seconds=0.2)
    assert analyzer.recompute_gait_phases() is None


def test_frames_with_no_landmarks_do_not_crash_it():
    analyzer = build()
    for fr in analyzer.frame_results[10:20]:
        fr.extra_metrics.pop("_norm_landmarks")
    assert analyzer.recompute_gait_phases() is not None


# --- near-leg contacts -----------------------------------------------------

def test_contacts_alternate_feet_but_near_leg_contacts_do_not():
    """stance_runs reports every footfall -- that is what ground contact and
    flight time are about. The near-leg filter halves them, because overstride
    and foot strike read the near ankle and it is mid-swing on the others."""
    analyzer = build()
    analyzer.recompute_gait_phases()
    all_contacts = analyzer.stance_runs()
    near_contacts = analyzer._contact_frame_indices()
    assert len(all_contacts) >= 6
    assert 0.3 <= len(near_contacts) / len(all_contacts) <= 0.7


def test_the_near_leg_contacts_are_the_ones_with_the_near_foot_down():
    analyzer = build()          # camera_side "left" -> ankle 27
    analyzer.recompute_gait_phases()
    depths = []
    for i in analyzer._contact_frame_indices():
        lms = analyzer.frame_results[i].extra_metrics["_norm_landmarks"]
        depths.append(lms[27].y - (lms[23].y + lms[24].y) / 2)
    # Planted: the built foot sits at 0.42 below the hips during stance and
    # lifts to 0.26 at the top of the swing.
    assert float(np.median(depths)) > 0.36


def test_nothing_matching_falls_back_to_every_contact_rather_than_none():
    """Better to measure the wrong leg loudly than to silently measure
    nothing -- the near-side conflict check upstream is what withholds it."""
    analyzer = build()
    analyzer.recompute_gait_phases()
    for fr in analyzer.frame_results:
        fr.extra_metrics["_near_foot_depth"] = float("nan")
    assert len(analyzer._contact_frame_indices()) == len(analyzer.stance_runs())


# --- the gate that reads it ------------------------------------------------

def test_the_kinogram_band_matches_what_the_phases_now_mean():
    """The phases describe ANY footfall, so the share is about twice the duty
    factor. A band written for one leg would refuse every real clip."""
    from app.services.video_analysis.kinogram import STANCE_FRACTION_RANGE

    lo, hi = STANCE_FRACTION_RANGE
    assert lo <= 2 * 0.22 <= hi          # sprinting
    assert lo <= 2 * 0.40 <= hi          # easy distance running
