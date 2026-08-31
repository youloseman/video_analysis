"""Was the clip filmed from the side? The pipeline assumed so and never asked.

Every angle in the side-view path assumes the camera is perpendicular to the
movement. Filmed from the front or behind, a knee angle is a projection of a
movement happening mostly toward the lens -- not a smaller version of the
truth, a different quantity wearing the same units. ``camera_view`` is
hardcoded to None ("side") in the runner, so a clip shot from behind was
measured, scored and reported exactly like a good one. The photo path has
caught this since it was written; the video path never got the equivalent.

CALIBRATION, HONESTLY. The positive verdict is evidenced: both reference clips
read a depth ratio of ~0.89 against a 0.35 threshold, on every frame. The
negative verdict is NOT -- no clip filmed from behind exists to calibrate it,
and a synthetically rotated body is not the same thing. So the detector may
confirm a side view and must not accuse anyone of the opposite, and these
tests pin that asymmetry as deliberate rather than let a future edit "finish"
it.
"""

from __future__ import annotations

import math

import pytest

from app.services.video_analysis.biomechanics.camera_view import (
    MIN_FRAMES,
    SIDE_VIEW_MIN_RATIO,
    UNCERTAIN_MIN_RATIO,
    _torso_depth_ratio,
    detect_camera_view,
)
from app.services.video_analysis.capture_report import build_capture_report


class LM:
    def __init__(self, x, y, z, visibility=1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def body(depth_gap: float, *, width: float = 0.4, visibility: float = 1.0):
    """33 landmarks with the two sides of the torso at a chosen depth apart.

    depth_gap is in the same units as width, so the ratio the detector
    computes is exactly depth_gap / width.
    """
    pts = [LM(0.0, 0.0, 0.0, visibility) for _ in range(33)]
    pts[11] = LM(-width / 2, -0.3, +depth_gap / 2, visibility)   # left shoulder
    pts[12] = LM(+width / 2, -0.3, -depth_gap / 2, visibility)   # right shoulder
    pts[23] = LM(-width / 2, +0.3, +depth_gap / 2, visibility)   # left hip
    pts[24] = LM(+width / 2, +0.3, -depth_gap / 2, visibility)   # right hip
    return pts


def clip(depth_gap: float, n: int = 60, **kw):
    return [{"world_landmarks": body(depth_gap, **kw)} for _ in range(n)]


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

def test_the_ratio_is_scaled_by_the_body_that_produced_it():
    """World-landmark z is in metres, so a raw gap would read a child and an
    adult differently. Dividing by the shoulder separation makes it a
    proportion of the body it was measured on.

    The divisor is the 3D distance -- the rider's actual shoulder breadth --
    not its x projection, which is what keeps the ratio bounded in [0, 1]:
    all-separation-in-depth is 1 (square on), none is 0 (facing the lens).
    Real side-on clips land at 0.89, which is that scale behaving.
    """
    gap, width = 0.2, 0.4
    breadth = math.hypot(width, gap)          # shoulders differ in x AND z
    assert _torso_depth_ratio(body(gap, width=width)) == pytest.approx(
        gap / breadth, abs=1e-6,
    )
    # The same pose on a body twice the size reads the same.
    assert _torso_depth_ratio(body(gap, width=width)) == pytest.approx(
        _torso_depth_ratio(body(gap * 2, width=width * 2)), abs=1e-6,
    )


def test_a_side_view_reads_high():
    got = detect_camera_view(clip(0.4, n=100))     # ratio 1.0
    assert got["view"] == "side"
    assert got["frames"] == 100


def test_a_face_on_body_reads_not_side():
    """Both sides of the torso at the same depth is what facing the lens looks
    like."""
    got = detect_camera_view(clip(0.01))           # ratio 0.025
    assert got["view"] == "not_side"


def test_the_band_between_the_thresholds_is_unknown():
    """Neither confident answer. Saying nothing is the correct output."""
    mid = (SIDE_VIEW_MIN_RATIO + UNCERTAIN_MIN_RATIO) / 2
    got = detect_camera_view(clip(mid * 0.4))
    assert got["view"] == "unknown"


def test_the_median_decides_not_a_few_frames():
    """A runner crossing the frame turns slightly at the edges. A handful of
    off-axis frames must not flip a clip that is plainly side on."""
    frames = clip(0.4, n=90) + clip(0.01, n=10)
    assert detect_camera_view(frames)["view"] == "side"


# --------------------------------------------------------------------------
# refusing to guess
# --------------------------------------------------------------------------

def test_too_few_frames_is_unknown_not_a_verdict():
    got = detect_camera_view(clip(0.4, n=MIN_FRAMES - 1))
    assert got["view"] == "unknown"
    assert got["ratio"] is None
    assert "too few frames" in got["reason"]


def test_no_frames_at_all_is_unknown():
    for empty in ([], None, [{}], [{"world_landmarks": None}]):
        assert detect_camera_view(empty)["view"] == "unknown"


def test_invisible_torso_frames_are_not_evidence():
    """A joint the model could not see says nothing about the camera angle."""
    assert math.isnan(_torso_depth_ratio(body(0.4, visibility=0.1)))
    assert detect_camera_view(clip(0.4, visibility=0.1))["view"] == "unknown"


def test_coincident_shoulders_are_refused_not_divided_by():
    """A pose collapsed to a point has no ruler to scale against. The divisor
    is the 3D shoulder separation, so both the width AND the depth gap have to
    vanish before it does."""
    assert math.isnan(_torso_depth_ratio(body(0.0, width=0.0)))


def test_non_finite_coordinates_are_refused():
    pts = body(0.4)
    pts[11] = LM(float("nan"), 0.0, 0.0)
    assert math.isnan(_torso_depth_ratio(pts))


def test_a_short_landmark_list_does_not_raise():
    assert math.isnan(_torso_depth_ratio([LM(0, 0, 0)] * 5))


# --------------------------------------------------------------------------
# what the athlete is told -- and what they are deliberately NOT told
# --------------------------------------------------------------------------

def _row(view):
    report = build_capture_report(sport_type="run", camera_view=view)
    return next((c for c in report["checks"] if c["id"] == "camera_view"), None)


def test_a_confirmed_side_view_is_reported_as_good():
    """The assumption the whole report rests on, stated instead of implied."""
    row = _row({"view": "side", "ratio": 0.9, "frames": 300, "reason": None})
    assert row["status"] == "good"
    assert "side on" in row["measured"]


def test_an_unconfirmed_view_never_warns():
    """THE point of this file. The negative verdict is uncalibrated -- no clip
    filmed from behind exists -- so it must not become an accusation. Tuning a
    threshold against a synthetically rotated body and then telling an athlete
    they misfilmed is exactly what this codebase refuses everywhere else."""
    for view in ("not_side", "unknown"):
        row = _row({"view": view, "ratio": 0.1, "frames": 300, "reason": None})
        assert row["status"] == "unknown", f"{view} must not warn until calibrated"
        assert row["action"] == "", "no action can be recommended on this evidence"


def test_the_row_never_counts_as_a_problem():
    """A 'poor' capture verdict sends people off to re-film. This check has not
    earned the right to do that."""
    for view in ("side", "not_side", "unknown"):
        report = build_capture_report(
            sport_type="run",
            camera_view={"view": view, "ratio": 0.5, "frames": 300, "reason": None},
        )
        problems = [c["id"] for c in report["checks"]
                    if c["status"] in ("bad", "warn")]
        assert "camera_view" not in problems


def test_no_view_no_row():
    """A clip the detector could not read adds nothing to the report rather
    than an empty row."""
    assert _row(None) is None
    assert _row({"view": None}) is None


def test_the_runner_measures_it():
    """It is computed from the frames rather than left hardcoded to None."""
    import inspect

    from app.services.video_analysis.runner import analyze_from_frames

    src = inspect.getsource(analyze_from_frames)
    assert "detect_camera_view(raw_frame_data)" in src
    assert 'summary["camera_view"]' in src
