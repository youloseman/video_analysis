"""Helmet above the back line, read off the person silhouette.

Replaces the photo "Head position" score, which put the ear under a line
through the hip and shoulder joint centres -- a line inside the body -- and so
gave 100/100 to every aero rider, all fifteen Kona 2026 pros included. A
reader called it meaningless and was right (2026-10-11).

These pin the geometry on hand-drawn masks: a helmet over the back reads
positive, one tucked under it reads negative, the number is a share of torso
length (so it does not depend on image size), and a silhouette that cannot
carry the reading gives none rather than a wrong one.
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
    for x in range(int(hx) - 20, int(sx) + 40):
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
