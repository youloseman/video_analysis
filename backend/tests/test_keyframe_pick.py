"""The keyframe still must be drawn from a frame whose near leg draws.

The results page leads with this image; a frame whose near leg was
identity-gated renders with the leg missing or misplaced, and mean landmark
visibility cannot see that (the gate deliberately leaves visibility alone).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from app.services.video_analysis.video_visualizer import VideoVisualizer


def _lm(x=0.5, y=0.5, vis=0.9):
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis)


def _frame(leg_nan=False, filled=False, vis=0.9):
    lms = [_lm(vis=vis) for _ in range(33)]
    if leg_nan:
        for i in (24, 26, 28, 30, 32):
            lms[i] = _lm(x=math.nan, y=math.nan, vis=vis)
    fd = {"normalized_landmarks": lms}
    if filled:
        fd["leg_gate_filled"] = {"right"}
    return fd


def _viz(frames):
    v = VideoVisualizer.__new__(VideoVisualizer)
    v.frame_data_list = frames
    v.camera_side = "right"
    return v


def test_a_gated_or_naned_leg_frame_never_wins():
    frames = [_frame() for _ in range(100)]
    # Poison the central band with high-visibility but undrawable frames:
    # NaN legs and gate-filled predictions, both shinier than the honest ones.
    for k in range(30, 71):
        frames[k] = _frame(leg_nan=(k % 2 == 0), filled=(k % 2 == 1), vis=1.0)
    frames[48] = _frame(vis=0.7)          # the one honest frame in the band

    picked = _viz(frames)._pick_keyframe_idx()

    fd = frames[picked]
    assert not fd.get("leg_gate_filled")
    lm = fd["normalized_landmarks"][28]
    assert not math.isnan(lm.x)


def test_all_undrawable_still_returns_a_frame():
    frames = [_frame(leg_nan=True) for _ in range(50)]
    picked = _viz(frames)._pick_keyframe_idx()
    assert 0 <= picked < 50
