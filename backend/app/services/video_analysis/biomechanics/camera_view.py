"""Was this clip actually filmed from the side?

Every angle in the side-view pipeline assumes the camera is perpendicular to
the plane of movement. Filmed from the front or behind, a knee angle is a
projection of a movement happening mostly toward the lens, and the number that
comes out is not a smaller version of the truth -- it is a different quantity
wearing the same units. The pipeline had no way to notice: ``camera_view`` is
hardcoded to None ("side") in ``runner``, so a clip shot from behind was
measured, scored and reported exactly like a good one.

The photo path has caught this since it was written -- one frame, the depth gap
between the left and right shoulder-hip pairs -- and the video path never got
the equivalent. This is that check, done better because a video has hundreds of
frames instead of one.

HOW. MediaPipe's world landmarks put z along the camera axis. Side on, the two
sides of the body sit at clearly different depths, because one is nearer the
lens than the other. Face on or from behind, they sit at nearly the same depth.
So the depth gap between (left shoulder, left hip) and (right shoulder, right
hip), measured against the body's own scale, separates the two.

Scale matters: z is in metres for world landmarks, so a raw threshold would
read a child and an adult differently. The gap is divided by shoulder width,
which makes it a proportion of the body it was measured on.

WHAT IT DOES NOT DO. It cannot tell front from back -- both put the shoulders
at the same depth -- and it does not try. The distinction that matters here is
"the side-view maths applies" versus "it does not", and for that the two are
the same answer. It also does not refuse the analysis: an athlete who filmed
the wrong way still gets their clip measured, with the report saying plainly
that the angles are projections of a movement the camera could not see.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24

# Depth gap between the two sides of the torso as a fraction of shoulder
# breadth, which bounds it in [0, 1]: all the separation in depth is 1 (square
# on), none of it is 0 (facing the lens).
#
# The two thresholds are calibrated very differently, and the asymmetry is the
# point.
#
# SIDE_VIEW_MIN_RATIO is measured. Both reference clips -- one bike, one run --
# read 0.896 and 0.890, on every one of their 503 and 362 frames. The bar sits
# at 0.35 rather than anywhere near that, because a clip only has to be
# unambiguously side-on to clear it and the cost of a false "not side on" is an
# athlete told to re-film something that was fine.
#
# UNCERTAIN_MIN_RATIO is NOT measured. No clip filmed from behind exists in the
# repo, so the only evidence below the bar is a synthetically rotated body,
# which read ~0.20 -- and a rotated set of world landmarks is not what
# MediaPipe would actually produce for a person facing the camera. The value is
# a placeholder that keeps that region out of the confident band; nothing
# user-facing may act on a "not_side" verdict until a real off-axis clip
# calibrates it. See capture_report._camera_view_check.
SIDE_VIEW_MIN_RATIO = 0.35
UNCERTAIN_MIN_RATIO = 0.18

# Frames with a readable torso needed before the verdict means anything.
MIN_FRAMES = 20

# How far the torso is turned from square-on to the lens, in degrees: the angle
# of the hip line and of the shoulder line in the world landmarks' x-z plane
# (0 = the line points straight at the camera, i.e. filmed exactly side-on;
# 90 = face-on or from behind), averaged, median over the clip.
#
# The depth ratio above cannot see a 30 deg oblique: it goes as cos(yaw) and
# Artur's 30 deg treadmill clip read 0.73 against 0.66-0.94 for clean side
# views. This one goes as the angle itself. Calibrated 2026-11-07 on Artur's
# treadmill set filmed at 0 / 15 / 30 deg either side (upload/photo_run) and 19
# side-on run clips (golden, Check, the elite IMG_5455, stadium, slow motion):
#
#   side-on clips      4.0 - 17.6 deg   (worst: Run00000851 17.6, IMG_5455 17.5)
#   "15_right"         21.5             "30_left"   18.7
#   "30_right"         34.3 / 34.6 (HD)
#
# The set fits measured ~= 0.8 x true with its "0" itself ~10 deg off, so
# 30_left is really ~20 deg and 30_right ~40 deg. Up to ~20 deg a knee angle
# moves <= 2 deg (simulation, flapp-no-oblique-correction), so the bar sits
# between: 9.5 deg above the worst side-on clip, 7 below the 40 deg one.
# MediaPipe supplies z and the world landmarks whether or not RTMPose refines
# x/y, so the reading is the same under either detector.
RUN_OBLIQUE_YAW_DEG = 27.0

# Landmark visibility below which a frame is not evidence of anything.
MIN_VISIBILITY = 0.5


def _torso_depth_ratio(landmarks: Any) -> float:
    """|depth(left side) - depth(right side)| / shoulder width, or NaN."""
    try:
        pts = {
            i: landmarks[i]
            for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        }
    except (IndexError, KeyError, TypeError):
        return math.nan
    for lm in pts.values():
        vis = getattr(lm, "visibility", 1.0)
        if vis is not None and vis < MIN_VISIBILITY:
            return math.nan
        for axis in ("x", "y", "z"):
            v = getattr(lm, axis, None)
            if v is None or not math.isfinite(float(v)):
                return math.nan

    left_z = (float(pts[L_SHOULDER].z) + float(pts[L_HIP].z)) / 2.0
    right_z = (float(pts[R_SHOULDER].z) + float(pts[R_HIP].z)) / 2.0
    # Shoulder width in the same units, as the body's own ruler. Taken in 3D so
    # it does not itself collapse when the athlete turns.
    width = math.dist(
        (float(pts[L_SHOULDER].x), float(pts[L_SHOULDER].y), float(pts[L_SHOULDER].z)),
        (float(pts[R_SHOULDER].x), float(pts[R_SHOULDER].y), float(pts[R_SHOULDER].z)),
    )
    if width < 1e-6:
        return math.nan
    return abs(left_z - right_z) / width


def _torso_yaw_deg(landmarks: Any) -> float:
    """Mean angle of the hip and shoulder lines off the camera axis, or NaN."""
    try:
        pts = [landmarks[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)]
    except (IndexError, KeyError, TypeError):
        return math.nan
    for lm in pts:
        vis = getattr(lm, "visibility", 1.0)
        if vis is not None and vis < MIN_VISIBILITY:
            return math.nan
        for axis in ("x", "z"):
            v = getattr(lm, axis, None)
            if v is None or not math.isfinite(float(v)):
                return math.nan
    ls, rs, lh, rh = pts
    sh = math.degrees(math.atan2(abs(float(ls.x) - float(rs.x)), abs(float(ls.z) - float(rs.z))))
    hp = math.degrees(math.atan2(abs(float(lh.x) - float(rh.x)), abs(float(lh.z) - float(rh.z))))
    return (sh + hp) / 2.0


def run_camera_oblique(view: dict[str, Any] | None) -> bool:
    """A run clip filmed far enough off side-on that its angles are not graded."""
    yaw = (view or {}).get("yaw_deg")
    return yaw is not None and yaw >= RUN_OBLIQUE_YAW_DEG


def detect_camera_view(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Judge the camera's angle to the movement from the whole clip.

    Returns ``view`` ("side" | "not_side" | "unknown"), the median ratio it was
    decided on, and how many frames were readable. ``unknown`` whenever there
    is not enough evidence -- an absent verdict, never a guessed one.
    """
    ratios = []
    yaws = []
    for frame in frames or []:
        lms = frame.get("world_landmarks") or frame.get("normalized_landmarks")
        if not lms:
            continue
        r = _torso_depth_ratio(lms)
        if not math.isnan(r):
            ratios.append(r)
        # World landmarks only: normalised x and z are in different units.
        if frame.get("world_landmarks"):
            y = _torso_yaw_deg(frame["world_landmarks"])
            if not math.isnan(y):
                yaws.append(y)
    yaw_deg = round(float(np.median(yaws)), 1) if len(yaws) >= MIN_FRAMES else None

    if len(ratios) < MIN_FRAMES:
        return {
            "view": "unknown", "ratio": None, "frames": len(ratios),
            "yaw_deg": yaw_deg,
            "reason": "too few frames with a readable torso",
        }

    # Median, not mean: a runner passing through the frame turns slightly, and
    # a handful of frames near the edges should not decide the clip.
    ratio = float(np.median(ratios))
    if ratio >= SIDE_VIEW_MIN_RATIO:
        view = "side"
    elif ratio < UNCERTAIN_MIN_RATIO:
        view = "not_side"
    else:
        view = "unknown"
    return {
        "view": view,
        "ratio": round(ratio, 3),
        "frames": len(ratios),
        "yaw_deg": yaw_deg,
        "reason": None,
    }
