"""Helmet above the top of the back, read off the person silhouette.

Replaces the photo "Head position" score, which put the ear under a line
through the hip and shoulder joint centres -- a line inside the body -- and so
gave 100/100 to every aero rider, all fifteen Kona 2026 pros included. A
reader called it meaningless and was right (2026-10-11).

These pin the geometry on hand-drawn masks: the reference is level (the
wind is), so the same helmet gap reads the same over a climbing back and a
flat one; the number is a share of torso length (so it does not depend on
image size); and a silhouette that cannot carry the reading gives none
rather than a wrong one.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.video_analysis.biomechanics.head_silhouette import (
    CONTOUR_COLUMNS,
    helmet_above_back,
    top_contour,
)

W, H = 1000, 600
SHOULDER, HIP, NOSE = (0.62, 0.40), (0.30, 0.52), (0.80, 0.42)


def _rider(helmet_top_px: int, size=(H, W), back_lift_px: int = 30) -> np.ndarray:
    """A right-facing rider: a back running straight from hip to shoulder,
    ``back_lift_px`` above the joint line, and a round helmet in front."""
    h, w = size
    m = np.zeros((h, w), bool)
    sx, sy = SHOULDER[0] * w, SHOULDER[1] * h
    hx, hy = HIP[0] * w, HIP[1] * h
    for x in range(int(hx - 20 * w / W), int(sx + 40 * w / W)):
        y_back = hy + (x - hx) * (sy - hy) / (sx - hx) - back_lift_px * h / H
        m[int(y_back):int(hy) + 80, x] = True
    cx, r = int(0.72 * w), int(0.07 * w)
    yy, xx = np.ogrid[:h, :w]
    m |= (xx - cx) ** 2 + (yy - (helmet_top_px * h / H + r)) ** 2 <= r ** 2
    return m


def _read(mask: np.ndarray):
    h, w = mask.shape
    return helmet_above_back(top_contour(mask), SHOULDER, HIP, NOSE, w / h)


def test_the_contour_is_the_first_person_pixel_per_column():
    m = np.zeros((100, 512), bool)
    m[40:, 100:200] = True
    edge = top_contour(m)
    assert len(edge) == CONTOUR_COLUMNS
    assert edge[150] == 0.4 and edge[50] is None


def test_a_helmet_over_the_back_reads_positive_and_a_tucked_one_about_zero():
    """Tucked under the line, the highest thing in front of the shoulder is
    the top of the shoulder itself, which sits on the line: about zero. It
    reads below zero only when the outline dips in front of the shoulder, as
    it did for one Kona pro (-3 %)."""
    up = _read(_rider(helmet_top_px=150))
    down = _read(_rider(helmet_top_px=240))
    assert up is not None and down is not None
    assert up["value"] > 5 and down["value"] <= 1
    assert up["fit_rms_pct"] < 1.0


def test_it_is_a_share_of_the_torso_not_pixels():
    small = _read(_rider(helmet_top_px=150))
    big = _read(_rider(helmet_top_px=150, size=(2 * H, 2 * W)))
    assert big["value"] == pytest.approx(small["value"], abs=0.6)


def test_no_silhouette_no_reading():
    assert helmet_above_back(None, SHOULDER, HIP, NOSE, W / H) is None
    assert _read(np.zeros((H, W), bool)) is None


def test_a_helmet_cut_off_by_the_frame_is_not_read():
    """The top of the silhouette at the top edge means the helmet goes on
    outside the picture: its real height is unknown."""
    assert _read(_rider(helmet_top_px=-40)) is None


def test_a_back_that_is_not_a_straight_edge_gives_no_reading():
    m = _rider(helmet_top_px=150)
    rng = np.random.default_rng(1)
    for x in range(int(HIP[0] * W), int(SHOULDER[0] * W)):
        m[: int(rng.integers(0, int(0.45 * H))), x] = rng.random() < 0.5
    assert _read(m) is None


def _level_case(climb_px: int, gap_px: int = 60):
    """A back from x=280 to x=640 that climbs ``climb_px`` toward the
    shoulder, joint points 30 px under its surface, and a helmet whose top is
    ``gap_px`` above the back's highest point. Returns (reading, expected %)."""
    m = np.zeros((H, W), bool)
    y0 = 330
    for x in range(260, 680):
        t = min(1.0, max(0.0, (x - 280) / 360))
        m[int(y0 - climb_px * t):520, x] = True
    back_top = y0 - climb_px
    r = 70
    yy, xx = np.ogrid[:H, :W]
    m |= (xx - 730) ** 2 + (yy - (back_top - gap_px + r)) ** 2 <= r ** 2
    hip = (280 / W, (y0 + 30) / H)
    shoulder = (640 / W, (back_top + 30) / H)
    torso_px = float(np.hypot(360, climb_px))
    got = helmet_above_back(top_contour(m), shoulder, hip, (790 / W, (back_top + 40) / H), W / H)
    return got, gap_px / torso_px * 100


@pytest.mark.parametrize("climb_px", [0, 60, 120])
def test_the_same_helmet_gap_reads_the_same_over_any_back(climb_px):
    """Artur, 2026-10-11: a line carried forward ALONG the back rose into the
    helmet of a rider whose back climbs to the shoulders and stayed low for a
    flat one -- -3 % against +16 % for helmets at much the same height. The
    wind is level, so the reference is a level line through the top of the
    back."""
    got, expected = _level_case(climb_px)
    assert got is not None
    assert got["value"] == pytest.approx(expected, abs=1.5)
    (x0, y0), (x1, y1) = got["line"]
    assert y0 == y1, "the reference line is level"


def test_a_flush_helmet_tail_asks_for_a_check_and_a_placed_point_clears_it():
    """A helmet whose tail lies on the back leaves no step in the outline:
    the reading flags itself, and once the athlete has placed the points
    the flag is gone and the number is theirs."""
    from app.services.video_analysis.biomechanics.head_silhouette import (
        BACK_TOP_POINT,
        apply_point_corrections,
    )
    m = _rider(helmet_top_px=150)
    # fill the gap between the back and the helmet: the tail lies flush
    for x in range(int(0.60 * W), int(0.68 * W)):
        col = np.nonzero(m[:, x])[0]
        top = col.min() if col.size else H
        m[min(top, 205):, x] = True
    got = _read(m)
    assert got is not None and got["check"] is True
    fixed = apply_point_corrections(got, {BACK_TOP_POINT: (0.0, 0.02)})
    assert fixed["check"] is False and fixed["moved"] == [BACK_TOP_POINT]
    assert fixed["value"] > got["value"] and fixed["auto_value"] == got["value"]
