"""How heavy the skeleton is drawn, given that nobody sees it at drawn size.

Both renderers hand their output on to a resize: the overlay video is drawn on
the display crop and encoded down to a ~720p long edge, the keyframe still is
drawn full-size and shrunk to 720 wide. Weighting the stroke against the surface
being drawn on therefore describes a frame nobody looks at -- and it is how a
display crop narrower than the 900 px design canvas pinned the skeleton to its
minimum weight and the encoder then shaved it further, leaving the video looking
like hairlines next to the glowing still of the same rider.
"""
from __future__ import annotations

import pytest

from app.services.video_analysis.overlay_style import (
    SKELETON_DOT_PX,
    SKELETON_LINE_PX,
    skeleton_weights,
)


def _delivered(width: int, height: int, out_scale: float) -> tuple[float, float]:
    line_w, dot_r = skeleton_weights(width, height, out_scale=out_scale)
    return line_w * out_scale, dot_r * out_scale


@pytest.mark.parametrize(
    ("width", "height", "out_scale"),
    [
        (810, 1450, 1280 / 1450),    # portrait clip with a display crop
        (1072, 1920, 1280 / 1920),   # portrait clip, no crop
        (1920, 1080, 1280 / 1920),   # landscape clip
        (720, 1280, 1.0),            # already at the delivery size
    ],
)
def test_the_delivered_stroke_lands_on_target_however_far_it_is_scaled(
    width, height, out_scale,
):
    line_px, dot_px = _delivered(width, height, out_scale)
    assert SKELETON_LINE_PX * 0.6 <= line_px <= SKELETON_LINE_PX * 1.7
    assert SKELETON_DOT_PX * 0.6 <= dot_px <= SKELETON_DOT_PX * 1.7


def test_a_display_crop_no_longer_thins_the_skeleton():
    """The crop shrinks the drawing surface; the rider still sees the same line.

    810x1450 is the crop the repo's bike clip produces from a 1072x1920 source.
    """
    cropped = _delivered(810, 1450, 1280 / 1450)
    uncropped = _delivered(1072, 1920, 1280 / 1920)
    assert abs(cropped[0] - uncropped[0]) < 1.0
    assert abs(cropped[1] - uncropped[1]) < 1.5


def test_it_is_heavier_than_the_hairline_it_replaced():
    """The old rule bottomed out at a 2 px line and a 3 px dot before the encode."""
    line_px, dot_px = _delivered(810, 1450, 1280 / 1450)
    assert line_px > 2 * (1280 / 1450)
    assert dot_px > 3 * (1280 / 1450)


def test_a_missing_or_nonsense_scale_is_treated_as_no_resize():
    assert skeleton_weights(1280, 720, out_scale=0) == skeleton_weights(1280, 720)
    assert skeleton_weights(1280, 720, out_scale=4.0) == skeleton_weights(1280, 720)


def test_a_tiny_frame_still_gets_a_visible_skeleton():
    line_w, dot_r = skeleton_weights(160, 90, out_scale=1.0)
    assert line_w >= 2
    assert dot_r >= 3
