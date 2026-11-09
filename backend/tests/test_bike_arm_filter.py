"""Bike arms are filtered at half the legs' cutoff (2026-11-09, IMG_5572).

RTMPose re-places every joint on every frame; on an elbow resting on the pads
its scatter was what the overlay drew. The same noise on an arm and on a leg
must now come out quieter on the arm, and only on the bike.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.services.video_analysis.biomechanics.landmark_stabilizer import (
    BIKE_ARM_LANDMARKS,
    _apply_butterworth_landmarks,
)

ELBOW, KNEE = 13, 25


def _frames(n: int = 150, seed: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.004, size=(n, 33, 2))
    out = []
    for f in range(n):
        lms = [SimpleNamespace(x=0.5 + noise[f, i, 0], y=0.5 + noise[f, i, 1], z=0.0, visibility=1.0)
               for i in range(33)]
        out.append({"normalized_landmarks": lms,
                    "world_landmarks": [SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0) for _ in range(33)]})
    return out


def _jitter(frames: list[dict], i: int) -> float:
    x = np.array([fr["normalized_landmarks"][i].x for fr in frames])
    return float(np.median(np.abs(np.diff(x, 2))))


def test_the_arm_is_smoothed_harder_than_the_leg_on_the_bike():
    frames = _frames()
    _, meta = _apply_butterworth_landmarks(frames, "bike", None, 30.0)
    assert meta["arm_cutoff_hz"] == meta["actual_cutoff_hz"] / 2
    assert _jitter(frames, ELBOW) < 0.5 * _jitter(frames, KNEE)


def test_the_run_keeps_one_cutoff():
    frames = _frames()
    _, meta = _apply_butterworth_landmarks(frames, "run", None, 30.0)
    assert "arm_cutoff_hz" not in meta
    ratio = _jitter(frames, ELBOW) / _jitter(frames, KNEE)
    assert 0.6 < ratio < 1.6


def test_the_arm_set_is_shoulders_to_hands():
    assert BIKE_ARM_LANDMARKS == frozenset(range(11, 23))
