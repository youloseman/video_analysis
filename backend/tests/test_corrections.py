"""An athlete moving a joint point: what is allowed, and what it does.

A correction is one constant offset per landmark, applied to every frame. The
tests pin the three things that make that safe: only the joints the report
reads can be moved, the frames' own geometry gets a say before anything is
applied, and a point we could not measure is not made measurable by an
offset.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from app.services.video_analysis.biomechanics.corrections import (
    DRAGGABLE_LANDMARKS,
    MAX_OFFSET,
    apply_corrections,
    check_plausibility,
    normalize_corrections,
)


def _lm(x, y):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=1.0)


def _rider_frame(i: int) -> dict:
    """A right-side rider with a realistic thigh (hip 23/24 -> knee 25/26)."""
    lms = [_lm(0.5, 0.5) for _ in range(33)]
    bob = 0.002 * math.sin(i / 5)
    lms[24] = _lm(0.30, 0.40 + bob)          # right hip
    lms[26] = _lm(0.36, 0.58 + bob)          # right knee
    lms[28] = _lm(0.32, 0.76 + bob)          # right ankle
    lms[30] = _lm(0.30, 0.78 + bob)          # right heel
    lms[32] = _lm(0.37, 0.79 + bob)          # right toe
    lms[12] = _lm(0.50, 0.30)                # right shoulder
    lms[14] = _lm(0.56, 0.36)
    lms[16] = _lm(0.64, 0.36)
    lms[8] = _lm(0.58, 0.27)
    return {
        "normalized_landmarks": lms,
        "world_landmarks": [_lm(0.0, 0.0) for _ in range(33)],
        "timestamp_ms": i * 33.0, "frame_idx": i,
        "frame_width": 1080, "frame_height": 1920,
    }


FRAMES = [_rider_frame(i) for i in range(60)]


# --- normalize -------------------------------------------------------------

def test_only_the_near_side_joints_may_move():
    with pytest.raises(ValueError, match="right-side joints are measured"):
        normalize_corrections([{"landmark": 25, "dx": 0.01, "dy": 0.0}], "right")


def test_the_draggable_set_is_the_measured_joints_plus_the_ear():
    assert DRAGGABLE_LANDMARKS["right"] == (8, 12, 14, 16, 24, 26, 28, 30, 32)
    assert DRAGGABLE_LANDMARKS["left"] == (7, 11, 13, 15, 23, 25, 27, 29, 31)


def test_a_face_or_hand_point_is_not_a_joint():
    with pytest.raises(ValueError):
        normalize_corrections([{"landmark": 0, "dx": 0.01, "dy": 0.0}], "right")
    with pytest.raises(ValueError):
        normalize_corrections([{"landmark": 20, "dx": 0.01, "dy": 0.0}], "right")


def test_a_move_across_a_quarter_of_the_frame_is_not_a_correction():
    with pytest.raises(ValueError, match="quarter of the frame"):
        normalize_corrections(
            [{"landmark": 24, "dx": MAX_OFFSET + 0.01, "dy": 0.0}], "right",
        )


def test_a_non_number_is_refused():
    with pytest.raises(ValueError):
        normalize_corrections([{"landmark": 24, "dx": "0.1", "dy": 0.0}], "right")
    with pytest.raises(ValueError):
        normalize_corrections([{"landmark": 24, "dx": math.nan, "dy": 0.0}], "right")
    with pytest.raises(ValueError):
        normalize_corrections([{"landmark": True, "dx": 0.1, "dy": 0.0}], "right")


def test_two_edits_to_one_joint_add_up():
    """'I moved it, then nudged it a bit more' is one offset, not two."""
    out = normalize_corrections([
        {"landmark": 24, "dx": 0.01, "dy": -0.02, "frame_idx": 10},
        {"landmark": 24, "dx": 0.005, "dy": 0.0, "frame_idx": 42},
    ], "right")
    assert out == [{"landmark": 24, "dx": 0.015, "dy": -0.02, "frame_idx": 42}]


def test_an_edit_that_cancels_itself_out_is_dropped():
    out = normalize_corrections([
        {"landmark": 24, "dx": 0.01, "dy": 0.0},
        {"landmark": 24, "dx": -0.01, "dy": 0.0},
    ], "right")
    assert out == []


def test_nothing_in_is_nothing_out():
    assert normalize_corrections(None, "left") == []
    assert normalize_corrections([], "left") == []


# --- apply -----------------------------------------------------------------

def test_the_offset_lands_on_every_frame_and_on_nothing_else():
    frames = [_rider_frame(i) for i in range(30)]
    before_knee = [(f["normalized_landmarks"][26].x, f["normalized_landmarks"][26].y) for f in frames]
    before_hip = [(f["normalized_landmarks"][24].x, f["normalized_landmarks"][24].y) for f in frames]

    touched = apply_corrections(
        frames, [{"landmark": 24, "dx": 0.01, "dy": -0.03}], "bike",
    )

    assert touched == {24: 30}
    for f, (kx, ky), (hx, hy) in zip(frames, before_knee, before_hip):
        assert (f["normalized_landmarks"][26].x, f["normalized_landmarks"][26].y) == (kx, ky)
        assert f["normalized_landmarks"][24].x == pytest.approx(hx + 0.01)
        assert f["normalized_landmarks"][24].y == pytest.approx(hy - 0.03)


def test_a_gated_frame_stays_gated():
    frames = [_rider_frame(i) for i in range(10)]
    frames[3]["normalized_landmarks"][24] = SimpleNamespace(
        x=math.nan, y=math.nan, z=math.nan, visibility=0.2,
    )
    touched = apply_corrections(frames, [{"landmark": 24, "dx": 0.01, "dy": 0.0}], "bike")
    assert touched == {24: 9}
    assert math.isnan(frames[3]["normalized_landmarks"][24].x)


def test_the_world_skeleton_is_left_alone():
    """It decides the camera side and carries the visibility the quality
    metrics read; a correction changes neither."""
    frames = [_rider_frame(i) for i in range(5)]
    apply_corrections(frames, [{"landmark": 24, "dx": 0.05, "dy": 0.05}], "bike")
    for f in frames:
        assert (f["world_landmarks"][24].x, f["world_landmarks"][24].y) == (0.0, 0.0)


def test_running_clips_are_refused_rather_than_silently_unchanged():
    """The run analyzer reads angles off world landmarks: an image-space
    offset would move the skeleton on screen and change nothing in the
    report, which is the one outcome worse than saying no."""
    with pytest.raises(ValueError, match="cycling clips only"):
        apply_corrections(FRAMES, [{"landmark": 24, "dx": 0.01, "dy": 0.0}], "run")


# --- plausibility ----------------------------------------------------------

def test_a_small_nudge_raises_no_eyebrows():
    assert check_plausibility(
        FRAMES, [{"landmark": 26, "dx": 0.004, "dy": 0.003}], "right", aspect=0.5625,
    ) == []


def test_a_knee_dragged_far_off_the_thigh_is_flagged_on_both_its_bones():
    warnings = check_plausibility(
        FRAMES, [{"landmark": 26, "dx": 0.0, "dy": -0.12}], "right", aspect=0.5625,
    )
    segments = {w["segment"] for w in warnings}
    assert "hip-knee" in segments
    assert "knee-ankle" in segments
    assert all(w["landmark"] == 26 for w in warnings)
    assert all("%" in w["message"] for w in warnings)


def test_the_athlete_may_still_be_right():
    """A warning is advice, not a refusal: it comes back as data, nothing raises."""
    warnings = check_plausibility(
        FRAMES, [{"landmark": 24, "dx": 0.0, "dy": -0.10}], "right", aspect=0.5625,
    )
    assert warnings and isinstance(warnings, list)


def test_pushing_a_point_off_screen_is_refused():
    with pytest.raises(ValueError, match="outside the picture"):
        check_plausibility(
            FRAMES, [{"landmark": 32, "dx": 0.0, "dy": 0.24}], "right",
        )


def test_lengths_are_read_with_the_frame_aspect():
    """Normalized x spans the width and y the height; on a portrait clip a
    horizontal move is worth far fewer pixels than a vertical one."""
    portrait = check_plausibility(
        FRAMES, [{"landmark": 26, "dx": 0.05, "dy": 0.0}], "right", aspect=0.5625,
    )
    square = check_plausibility(
        FRAMES, [{"landmark": 26, "dx": 0.05, "dy": 0.0}], "right", aspect=1.0,
    )
    assert len(square) >= len(portrait)
