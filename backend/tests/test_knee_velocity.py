"""Angular speed was already being computed and thrown away. Now it is reported.

Two things have to hold. It has to be a robust reading -- differentiating turns
one mistracked frame into a huge spurious velocity, so a naive max would be set
by the worst frame in the clip, always in the alarming direction. And it has to
stay UNGRADED, because the published figures are for a stated running speed this
pipeline cannot measure and every low-pass attenuates a derivative.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.video_analysis.biomechanics.phase_portrait import (
    compute_phase_portraits,
    peak_velocities,
    summary_peak_velocities,
)

FPS = 50.0
SECONDS = 6.0


def knee_signal(hz=1.4, amp=40.0, seconds=SECONDS, fps=FPS):
    """A knee sweeping open and closed, plus its timestamps."""
    t = np.arange(0, seconds, 1.0 / fps)
    return 120.0 + amp * np.sin(2 * np.pi * hz * t), t


# --- the reading itself ----------------------------------------------------

def test_a_sinusoid_reports_close_to_its_analytic_peak_speed():
    """d/dt of A*sin(2*pi*f*t) peaks at A*2*pi*f. The p95 sits just under it,
    which is the price of being robust and is worth paying."""
    angles, t = knee_signal(hz=1.4, amp=40.0)
    velocity = np.gradient(angles, t)
    analytic = 40.0 * 2 * np.pi * 1.4
    ext, flex = peak_velocities(velocity)
    assert ext == pytest.approx(analytic, rel=0.15)
    assert flex == pytest.approx(analytic, rel=0.15)


def test_both_directions_come_back_positive():
    """'How fast' is the question; the direction is already in the name."""
    angles, t = knee_signal()
    ext, flex = peak_velocities(np.gradient(angles, t))
    assert ext > 0 and flex > 0


def test_one_mistracked_frame_does_not_set_the_peak():
    """The reason for percentiles rather than max: a single-frame position
    error becomes an enormous velocity spike when differentiated."""
    angles, t = knee_signal()
    velocity = np.gradient(angles, t)
    clean_ext, clean_flex = peak_velocities(velocity)
    velocity[137] = 9000.0
    velocity[138] = -9000.0
    ext, flex = peak_velocities(velocity)
    assert ext == pytest.approx(clean_ext, rel=0.05)
    assert flex == pytest.approx(clean_flex, rel=0.05)


def test_a_faster_movement_reads_faster():
    slow, t1 = knee_signal(hz=1.2)
    fast, t2 = knee_signal(hz=2.4)
    slow_ext, _ = peak_velocities(np.gradient(slow, t1))
    fast_ext, _ = peak_velocities(np.gradient(fast, t2))
    assert fast_ext > slow_ext * 1.5


def test_gaps_are_ignored_rather_than_counted_as_stillness():
    angles, t = knee_signal()
    velocity = np.gradient(angles, t)
    clean_ext, _ = peak_velocities(velocity)
    velocity[50:90] = np.nan
    ext, _ = peak_velocities(velocity)
    assert ext == pytest.approx(clean_ext, rel=0.10)


def test_too_little_signal_says_nothing_rather_than_zero():
    assert peak_velocities(np.full(5, 100.0)) == (None, None)
    assert peak_velocities(np.full(200, np.nan)) == (None, None)


def test_a_motionless_joint_reads_zero_not_none():
    """Still is a measurement; missing is not. They must not look alike."""
    ext, flex = peak_velocities(np.zeros(200))
    assert ext == 0.0 and flex == 0.0


# --- reaching the portrait -------------------------------------------------

def portraits(sport="run", joint="knee"):
    angles, t = knee_signal()
    return compute_phase_portraits(
        {joint: list(angles)}, list(t), sport, camera_side="left",
    )


def test_the_portrait_now_carries_the_peaks():
    data = portraits()["joints"]["knee"]
    assert data["peak_extension_velocity"] > 0
    assert data["peak_flexion_velocity"] > 0
    # The old range reading stays: it is what the chart's axis uses.
    assert data["velocity_range"] > 0


# --- reaching the summary --------------------------------------------------

def test_running_lifts_the_near_knee():
    out = summary_peak_velocities(portraits(), "run", "left")
    assert out["knee_extension_velocity_dps"] > 0
    assert out["knee_flexion_velocity_dps"] > 0


def test_cycling_lifts_the_knee_on_the_side_facing_the_camera():
    data = portraits(sport="bike", joint="right_knee")
    assert summary_peak_velocities(data, "bike", "right")
    assert summary_peak_velocities(data, "bike", "left") == {}


def test_the_reading_is_flagged_as_a_self_comparison():
    """Published peaks are for a stated running speed we do not measure, and a
    low-pass attenuates any derivative. The flag is what stops the UI and the
    coach prompt from treating it as a benchmark."""
    assert summary_peak_velocities(portraits(), "run", "left")[
        "knee_velocity_relative"] is True


def test_no_band_is_published_for_it():
    """A guard against a later well-meaning commit adding one."""
    from app.services.video_analysis.biomechanics.sport_configs import (
        RUNNING_REFERENCE,
    )

    assert not [k for k in RUNNING_REFERENCE if "velocity" in k]


def test_a_missing_portrait_yields_no_metric_rather_than_zero():
    assert summary_peak_velocities(None, "run", "left") == {}
    assert summary_peak_velocities({}, "run", "left") == {}
    assert summary_peak_velocities({"joints": {}}, "run", "left") == {}


def test_a_joint_the_portrait_skipped_yields_no_metric():
    """Joints with no detectable cycles are dropped from the portrait, and the
    summary must follow rather than invent a value."""
    assert summary_peak_velocities(
        {"joints": {"hip": {"peak_extension_velocity": 400.0}}}, "run", "left",
    ) == {}


# --- what the coach is told ------------------------------------------------

def test_the_coach_is_told_not_to_benchmark_the_number():
    from app.services.video_analysis.llm_recommendations import build_metrics_block

    block = build_metrics_block(
        "run", 78, "B", None, [], {},
        {"knee_extension_velocity_dps": 610.0,
         "knee_flexion_velocity_dps": 540.0},
    )
    assert "610" in block and "540" in block
    assert "RELATIVE MEASURE" in block
    assert "do NOT compare these to published values" in block


def test_the_coach_line_degrades_when_the_metric_is_absent():
    from app.services.video_analysis.llm_recommendations import build_metrics_block

    block = build_metrics_block("run", 78, "B", None, [], {}, {})
    assert "Knee angular speed: n/a" in block
