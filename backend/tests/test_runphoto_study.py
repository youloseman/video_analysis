"""The running study's readings use the product's conventions.

``compare`` is only a measurement of the detector if the hand-checked points
are read the way photo_analyzer reads the model's: same overstride, same signs,
same foot-strike thresholds. These pin that on a drawn runner facing right.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "field_study"))
from runphoto import readings  # noqa: E402

from app.services.video_analysis import photo_analyzer as PA  # noqa: E402

# A runner at touchdown, facing right (+x), image y down: the foot lands
# ahead of the hip, shin leaning back, heel below the toe.
LANDING = {"S": (505, 100), "E": (530, 170), "W": (575, 160), "H": (500, 260),
           "K": (560, 360), "A": (600, 460), "L": (590, 475), "F": (640, 465)}


def test_overstride_shin_and_foot_read_like_the_product():
    r = readings(LANDING, facing=1)
    leg = ((60 ** 2 + 100 ** 2) ** 0.5) + ((40 ** 2 + 100 ** 2) ** 0.5)
    assert r["overstride"] == pytest.approx(100 / leg)
    assert r["shin"] < 0                      # ankle ahead of the knee: reaching
    assert r["trunk"] > 0                     # shoulder ahead of the hip
    assert r["strike"] == "heel" and r["foot"] > PA._FOOT_STRIKE_HEEL_ABOVE_DEG


def test_the_same_runner_facing_left_reads_the_same():
    mirrored = {k: (1000 - x, y) for k, (x, y) in LANDING.items()}
    a, b = readings(LANDING, 1), readings(mirrored, -1)
    for k in ("knee", "trunk", "shin", "overstride", "foot"):
        assert a[k] == pytest.approx(b[k])


def test_foot_strike_thresholds_are_the_products():
    flat = dict(LANDING, L=(590, 470), F=(640, 470))
    assert readings(flat, 1)["strike"] == PA._foot_strike_from_ground_angle(0.0) == "midfoot"
