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
