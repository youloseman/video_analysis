"""Angle formulas must not care which way the rider faces.

A side-view clip can be filmed from either side of the bike, and nothing in the
product tells the athlete which one to pick. So every angle has to read the same
for a posture and its mirror image -- otherwise half of all analyses report
something else.

Forearm tilt used to fail exactly that: measured off the horizontal with a
signed dx, a rider facing left came back as `180 - tilt`. At ~168 deg against a
5-25 deg reference the report showed a physically impossible "out of range",
and the summary metric was quietly dropped for falling outside its plausibility
envelope -- taking its share of the technique score with it.

Pure math, no ML stack: these formulas take plain landmark objects.
"""

from __future__ import annotations

import math

import pytest

from app.services.video_analysis.biomechanics.angle_calculator import (
    calculate_forearm_tilt_2d,
    calculate_segment_to_vertical_from_points,
    calculate_shank_foot_angle_2d,
)


class LM:
    """Minimal stand-in for a MediaPipe landmark."""

    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x = x
        self.y = y
        self.visibility = visibility


def forearm(elbow: LM, wrist: LM) -> float:
    return calculate_forearm_tilt_2d([elbow, wrist], 0, 1)[0]


def mirrored(p: LM) -> LM:
    """The same point in a horizontally flipped frame."""
    return LM(1.0 - p.x, p.y, p.visibility)


# --------------------------------------------------------------------------
# Facing direction
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("dx", "dy", "expected"),
    [
        (0.20, -0.05, 14.0),    # wrist forward and above the elbow
        (0.20, 0.05, -14.0),    # wrist forward and below it
        (0.20, 0.0, 0.0),       # dead level
        (0.0, -0.10, 90.0),     # straight up
    ],
)
def test_tilt_reads_the_same_facing_either_way(dx, dy, expected):
    elbow = LM(0.40, 0.50)
    wrist = LM(elbow.x + dx, elbow.y + dy)

    facing_right = forearm(elbow, wrist)
    facing_left = forearm(mirrored(elbow), mirrored(wrist))

    assert facing_right == pytest.approx(expected, abs=0.6)
    assert facing_left == pytest.approx(facing_right, abs=0.05)


def test_a_realistic_aero_posture_stays_inside_the_reference_range():
    """The regression that started this: 5-25 deg is the reference band, and
    the mirrored rider used to land near 168."""
    elbow, wrist = LM(0.40, 0.50), LM(0.62, 0.44)

    for e, w in ((elbow, wrist), (mirrored(elbow), mirrored(wrist))):
        tilt = forearm(e, w)
        assert 5.0 <= tilt <= 25.0, tilt


def test_tilt_never_leaves_the_quarter_turn_either_side():
    """Whatever the geometry, the answer is a tilt off horizontal -- so it
    cannot exceed a right angle, which is what put 168 out of bounds."""
    for i in range(-9, 10):
        for j in range(-9, 10):
            if i == j == 0:
                continue
            tilt = forearm(LM(0.5, 0.5), LM(0.5 + i / 20, 0.5 + j / 20))
            assert -90.0 <= tilt <= 90.0, (i, j, tilt)


# --------------------------------------------------------------------------
# The neighbours that already got this right -- pinned so they stay that way
# --------------------------------------------------------------------------
def test_trunk_angle_is_mirror_stable_too():
    top, bottom = LM(0.45, 0.30), LM(0.55, 0.62)
    a = calculate_segment_to_vertical_from_points(top, bottom)
    b = calculate_segment_to_vertical_from_points(mirrored(top), mirrored(bottom))
    assert a == pytest.approx(b, abs=0.05)


def test_low_visibility_still_refuses_to_answer():
    """The fix must not turn an unmeasurable frame into a confident zero."""
    tilt, _vis = calculate_forearm_tilt_2d([LM(0.4, 0.5, 0.1), LM(0.6, 0.45)], 0, 1)
    assert math.isnan(tilt)


# --------------------------------------------------------------------------
# Run ankle: shank axis vs foot axis
#
# The old form measured a vertex angle knee-ankle-HEEL: the "foot" ray
# pointed into the shoe sole, the value ran ~25-40 deg high against the
# reference band, and it inherited every pixel of BlazePose's habit of
# parking the ankle landmark high on the shin above bulky shoes. The axis
# form (ankle->knee vs heel->toe) is the standard 2D gait convention:
# neutral reads 90, and the ankle point only contributes a direction.
# --------------------------------------------------------------------------
def ankle_axis(knee: LM, ankle_pt: LM, heel: LM, toe: LM) -> float:
    return calculate_shank_foot_angle_2d([knee, ankle_pt, heel, toe], 0, 1, 2, 3)[0]


def test_neutral_stance_reads_a_right_angle():
    """Foot flat, shank vertical -- the textbook 90 deg."""
    a = ankle_axis(LM(0.5, 0.30), LM(0.5, 0.70), LM(0.47, 0.78), LM(0.62, 0.78))
    assert a == pytest.approx(90.0, abs=1e-6)


def test_plantarflexion_reads_above_ninety_dorsiflexion_below():
    """Toe-off opens the angle, midstance dorsiflexion closes it --
    the direction of change the old heel-based form had inverted."""
    toe_off = ankle_axis(
        LM(0.58, 0.35), LM(0.50, 0.70), LM(0.45, 0.72), LM(0.58, 0.80),
    )
    dorsiflexed = ankle_axis(
        LM(0.62, 0.36), LM(0.50, 0.70), LM(0.47, 0.78), LM(0.62, 0.78),
    )
    assert toe_off > 95.0, toe_off
    assert dorsiflexed < 85.0, dorsiflexed


def test_ankle_reads_the_same_facing_either_way():
    pts = (LM(0.58, 0.35), LM(0.50, 0.70), LM(0.45, 0.72), LM(0.58, 0.80))
    assert ankle_axis(*pts) == pytest.approx(
        ankle_axis(*(mirrored(p) for p in pts)), abs=0.05,
    )


def test_ankle_landmark_drifting_up_the_shin_barely_moves_the_reading():
    """The reason this is an axis, not a vertex: BlazePose parks the ankle
    point high on the shin above running shoes. Drift ALONG the shank
    changes nothing at all; a straight-up drift on a tilted shank moves
    the reading by a few degrees, not the ~15-20 a vertex angle absorbs."""
    # Vertical shank: an upward drift IS along the axis -- exactly invariant.
    on_axis = ankle_axis(LM(0.5, 0.30), LM(0.5, 0.62), LM(0.47, 0.78), LM(0.62, 0.78))
    assert on_axis == pytest.approx(90.0, abs=1e-6)

    # Tilted shank, ankle lifted ~25% of shank length straight up.
    true_pos = ankle_axis(
        LM(0.62, 0.36), LM(0.50, 0.70), LM(0.47, 0.78), LM(0.62, 0.78),
    )
    drifted = ankle_axis(
        LM(0.62, 0.36), LM(0.50, 0.62), LM(0.47, 0.78), LM(0.62, 0.78),
    )
    assert abs(drifted - true_pos) < 6.0, (true_pos, drifted)


def test_ankle_low_visibility_refuses_to_answer():
    a = ankle_axis(
        LM(0.5, 0.30), LM(0.5, 0.70), LM(0.47, 0.78), LM(0.62, 0.78, 0.2),
    )
    assert math.isnan(a)
