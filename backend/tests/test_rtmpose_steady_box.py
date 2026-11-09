"""The steady crop box for RTMPose on the bike (rtmpose_refiner.STEADY_BOX_SPORTS).

Off since 2026-11-09: it moved the TT reference clip's knee 1.3 deg and flipped
its saddle verdict. Kept, tested, for when a labelled set can judge it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.video_analysis.detectors import rtmpose_refiner as RR


def _refiner(monkeypatch, on: bool) -> RR.RTMPoseRefiner:
    monkeypatch.setattr(RR, "STEADY_BOX_SPORTS", frozenset({"bike"}) if on else frozenset())
    inner = SimpleNamespace(config=SimpleNamespace(sport="bike"))
    return RR.RTMPoseRefiner(inner, session=None)


def _points(foot_y: float, dx: float = 0.0) -> np.ndarray:
    p = np.full((33, 2), np.nan)
    p[[11, 12, 23, 24]] = [[300 + dx, 200], [310 + dx, 205], [380 + dx, 330], [390 + dx, 335]]
    p[25] = [420 + dx, 420]
    p[27] = [400 + dx, foot_y]
    return p


def _box(r: RR.RTMPoseRefiner, p: np.ndarray, ms: int) -> np.ndarray:
    return np.array(r._box(p, np.isfinite(p).all(1), 1280, 720, ms))


def test_off_the_box_rides_the_pedalling_foot(monkeypatch):
    r = _refiner(monkeypatch, on=False)
    heights = [np.ptp(_box(r, _points(500 + 60 * np.sin(i / 3)), i * 33)[[1, 3]]) for i in range(60)]
    assert np.ptp(heights) > 80


def test_on_the_box_holds_still_after_a_revolution(monkeypatch):
    r = _refiner(monkeypatch, on=True)
    boxes = [_box(r, _points(500 + 60 * np.sin(i / 3)), i * 33) for i in range(60)]
    late = np.array(boxes[31:])
    assert np.ptp(late[:, 3]) < 1.0, "bottom edge steady"


def test_on_the_box_follows_a_rider_crossing_the_frame(monkeypatch):
    r = _refiner(monkeypatch, on=True)
    first = _box(r, _points(500), 0)
    for i in range(1, 30):
        last = _box(r, _points(500, dx=10.0 * i), i * 33)
    assert abs((last[0] - first[0]) - 290) < 1.0, "moves with the torso"
    assert abs((last[2] - last[0]) - (first[2] - first[0])) < 1.0, "same width"
