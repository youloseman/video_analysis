"""The bike trunk angle is the shoulder-hip line, and nothing on the head moves it.

Until 2026-11-17 a clip's trunk was read from a 0.7 shoulder / 0.3 ear blend
(an attempt at C7). The ear is on the head, so the trunk followed the head:
one rider with one back read 10.6 deg head down and 16.5 head up, and a
time-trial rider whose marker-to-marker back angle was 20 deg read 8.9 and was
told he sat "outside the whole pro field" -- a field measured, like every
photo, on the plain shoulder-hip line.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.video_analysis.biomechanics import cycling_analyzer as CA

ASPECT = 16 / 9


def _pose(trunk_deg: float, ear_dx: float = 0.05, ear_dy: float = -0.04):
    """33 normalized landmarks of a rider facing right, filmed from the right.

    The near hip sits at a fixed point and the near shoulder ``trunk_deg``
    above horizontal from it, in true image proportions; the ear is placed
    relative to the shoulder (image units), so a test can move the head alone.
    """
    lm = [SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.95) for _ in range(33)]
    hip = (0.40, 0.55)
    length = 0.28  # in height units
    dx = length * math.cos(math.radians(trunk_deg)) / ASPECT
    dy = -length * math.sin(math.radians(trunk_deg))
    sh = (hip[0] + dx, hip[1] + dy)
    for i in (23, 24):
        lm[i].x, lm[i].y = hip
    for i in (11, 12):
        lm[i].x, lm[i].y = sh
    for i in (7, 8):
        lm[i].x, lm[i].y = sh[0] + ear_dx, sh[1] + ear_dy
    # A leg under the hip so the frame is a plausible rider.
    for i in (25, 26):
        lm[i].x, lm[i].y = 0.47, 0.72
    for i in (27, 28):
        lm[i].x, lm[i].y = 0.44, 0.90
    return lm


def _trunk(landmarks) -> float:
    a = CA.CyclingAnalyzer(fps=30.0, cycling_position="tt_aero", frame_aspect=ASPECT)
    a._near_side = "right"
    return a.analyze_frame(landmarks, landmarks, 0.0).angles["trunk_angle"]


@pytest.mark.parametrize("deg", [9.0, 15.0, 20.0, 45.0])
def test_it_is_the_shoulder_hip_line(deg):
    assert _trunk(_pose(deg)) == pytest.approx(deg, abs=0.05)


def test_the_head_does_not_move_the_trunk():
    """Head up (ear above and ahead of the shoulder) and head down (ear below
    it, as on aero bars) are one back and one number."""
    up = _trunk(_pose(15.0, ear_dx=0.05, ear_dy=-0.06))
    down = _trunk(_pose(15.0, ear_dx=0.05, ear_dy=+0.06))
    assert up == pytest.approx(down, abs=1e-9)
    assert up == pytest.approx(15.0, abs=0.05)


def test_a_hidden_ear_changes_nothing():
    pose = _pose(18.0)
    for i in (7, 8):
        pose[i].visibility = 0.0
    assert _trunk(pose) == pytest.approx(18.0, abs=0.05)


def test_a_clip_and_a_photo_read_the_same_line():
    """The photo's trunk (photo_analyzer) is 90 minus the shoulder-hip segment
    against vertical, on the image landmarks in true proportions."""
    from app.services.video_analysis.biomechanics.angle_calculator import calculate_segment_to_vertical
    pose = _pose(13.0, ear_dy=+0.08)
    image = [CA._image_plane(p, ASPECT) for p in pose]
    photo = 90.0 - calculate_segment_to_vertical(image, 12, 24)
    assert _trunk(pose) == pytest.approx(photo, abs=1e-9)


def test_the_result_says_which_formula_it_used():
    a = CA.CyclingAnalyzer(fps=30.0, cycling_position="tt_aero", frame_aspect=ASPECT)
    a._near_side = "right"
    pose = _pose(20.0)
    for i in range(30):
        a.frame_results.append(a.analyze_frame(pose, pose, i * 33.3))
    method = a.compute_summary()["diagnostics"]["trunk_angle_method"]
    assert method["method"] == CA.TRUNK_METHOD == "shoulder_hip"
    assert "avg_ear_visibility" not in method
