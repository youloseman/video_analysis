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
    v.sport_type = "bike"
    v.summary = {}
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


def _leg_frame(knee_deg: float, ankle_drop: float = 0.0):
    """A right-side frame whose near knee reads ``knee_deg``: hip above the
    knee, the shin swung to the angle; ``ankle_drop`` lowers the ankle."""
    lms = [_lm() for _ in range(33)]
    hip, knee = (0.5, 0.3), (0.55, 0.5)
    thigh = math.atan2(hip[1] - knee[1], hip[0] - knee[0])
    shin = thigh - math.radians(knee_deg)
    ax, ay = knee[0] + 0.2 * math.cos(shin), knee[1] + 0.2 * math.sin(shin)
    ay += ankle_drop
    for i, (x, y) in ((24, hip), (26, knee), (28, (ax, ay)), (30, (ax - 0.02, ay)), (32, (ax + 0.04, ay))):
        lms[i] = _lm(x=x, y=y)
    return {"normalized_landmarks": lms, "frame_width": 1000, "frame_height": 1000}


def test_the_bike_keyframe_agrees_with_the_reported_knee():
    """A subscriber's report (2026-10-14) led with the leg drawn between his
    two legs: the lowest-ankle frame was the one where MediaPipe misplaced the
    ankle, knee 133 on the picture against knee_at_bdc 139.7 in the table.
    Among the bottom-of-stroke frames, the one that reads the reported knee
    is drawn."""
    frames = [_leg_frame(120.0) for _ in range(100)]
    for k in range(40, 60):
        frames[k] = _leg_frame(139.5)
    frames[50] = _leg_frame(133.0, ankle_drop=0.05)    # the lowest, and wrong
    v = _viz(frames)
    v.summary = {"knee_at_bdc": 139.7}
    picked = v._pick_keyframe_idx()
    assert picked != 50
    assert abs(v._near_knee_deg(picked) - 139.7) <= 3.0
    # Without a reported knee, the old rule stands: the lowest ankle.
    v.summary = {}
    assert v._pick_keyframe_idx() == 50


def test_the_keyframe_agrees_with_the_knee_it_prints():
    """The label on the picture comes from the analyzer's filtered series, the
    one knee_at_bdc is read from -- not from the frame's landmarks. A Coach
    Notes client (2026-10-14) got a frame whose landmarks read 142.3 (within
    3 of 144.7) while its label printed 137.8."""
    frames = [_leg_frame(120.0) for _ in range(100)]
    for k in range(40, 60):
        frames[k] = _leg_frame(142.0, ankle_drop=0.05)     # the bottom of the stroke
    v = _viz(frames)
    v.summary = {"knee_at_bdc": 144.7}
    series = [120.0] * 100
    for k in range(40, 60):
        series[k] = 137.8                 # landmarks agree, the printed value does not
    series[47] = 144.5                    # the one frame that prints the report's knee
    v.analyzer = SimpleNamespace(angle_history={"right_knee": series})
    picked = v._pick_keyframe_idx()
    assert picked == 47
    assert abs(v._near_knee_deg(picked) - 144.7) <= 3.0
