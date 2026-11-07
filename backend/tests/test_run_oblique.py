"""A run filmed well off side-on is not graded (camera_view.RUN_OBLIQUE_YAW_DEG).

Calibrated on Artur's treadmill clips at 0/15/30 deg: the depth ratio cannot
see a 30-40 deg oblique, the torso yaw can. Before this, the quality gate caught
those clips only because MediaPipe swapped the legs on them.
"""

from __future__ import annotations

import inspect
import math

from app.services.video_analysis.biomechanics.camera_view import (
    MIN_FRAMES,
    RUN_OBLIQUE_YAW_DEG,
    _torso_yaw_deg,
    detect_camera_view,
    run_camera_oblique,
)
from app.services.video_analysis.capture_report import build_capture_report


class LM:
    def __init__(self, x, y, z, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def turned(yaw_deg: float, width: float = 0.4, vis: float = 1.0):
    """A torso side-on to the camera (left/right split in depth), turned by yaw."""
    a = math.radians(yaw_deg)
    dx, dz = width / 2 * math.sin(a), width / 2 * math.cos(a)
    pts = [LM(0.0, 0.0, 0.0, vis) for _ in range(33)]
    pts[11], pts[12] = LM(-dx, -0.3, dz, vis), LM(dx, -0.3, -dz, vis)
    pts[23], pts[24] = LM(-dx, 0.3, dz, vis), LM(dx, 0.3, -dz, vis)
    return pts


def frames(yaw, n=60, **kw):
    return [{"world_landmarks": turned(yaw, **kw)} for _ in range(n)]


def test_the_yaw_reads_the_turn_and_ignores_scale():
    for yaw in (0, 10, 27, 40, 90):
        assert abs(_torso_yaw_deg(turned(yaw)) - yaw) < 1e-6
        assert abs(_torso_yaw_deg(turned(yaw, width=0.25)) - yaw) < 1e-6


def test_the_view_reports_the_median_yaw():
    clip = frames(10, 40) + frames(40, 20)
    assert detect_camera_view(clip)["yaw_deg"] == 10.0


def test_a_hidden_torso_and_too_few_frames_give_no_yaw():
    assert detect_camera_view(frames(40, vis=0.2))["yaw_deg"] is None
    assert detect_camera_view(frames(40, MIN_FRAMES - 1))["yaw_deg"] is None


def test_oblique_starts_at_the_bar():
    assert run_camera_oblique({"yaw_deg": RUN_OBLIQUE_YAW_DEG})
    assert not run_camera_oblique({"yaw_deg": RUN_OBLIQUE_YAW_DEG - 0.1})
    assert not run_camera_oblique({"yaw_deg": None}) and not run_camera_oblique(None)


def test_the_depth_ratio_alone_would_call_a_35_deg_clip_side_on():
    # Why the yaw exists: cos(35) keeps the ratio far above its 0.35 bar.
    view = detect_camera_view(frames(35))
    assert view["view"] == "side" and run_camera_oblique(view)


def _row(view, sport):
    report = build_capture_report(sport_type=sport, camera_view=view)
    return next(c for c in report["checks"] if c["id"] == "camera_view")


def test_the_capture_report_says_oblique_when_the_score_is_withheld():
    view = detect_camera_view(frames(35))
    row = _row(view, "run")
    assert row["status"] == "bad" and "off side-on" in row["measured"] and row["action"]
    # Bike is not calibrated for this: the row stays as it was.
    assert _row(view, "bike")["status"] == "good"
    assert _row(detect_camera_view(frames(12)), "run")["status"] == "good"


def test_the_runner_withholds_an_oblique_run_after_the_gate_and_the_position():
    from app.services.video_analysis.runner import analyze_from_frames

    src = inspect.getsource(analyze_from_frames)
    i_view = src.index("camera_view_result = detect_camera_view(raw_frame_data)")
    i_withhold = src.index('"reason": "camera_oblique"')
    assert i_view < src.index("score_withheld: dict[str, Any] | None = None") < i_withhold
    assert src.index('"reason": "quality_gate"') < src.index('"reason": "position_mismatch"') < i_withhold
    assert 'sport_type == "run" and run_camera_oblique(camera_view_result)' in src
