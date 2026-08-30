"""An athlete's own adjustments to where the joint points sit. Bike only.

MediaPipe puts a joint in the wrong place for reasons the athlete can see and
the model cannot: black shorts merge the hip with the saddle, the foot cluster
slides off the shoe onto the pedal, a jersey seam reads as a shoulder. Three
automatic repairs for the leg-identity half of this were measured and each made
things worse (see ``landmark_stabilizer``); the person looking at the picture
does better. So this lets them say where the joint is.

The model of a correction is deliberately narrow. One correction is a constant
offset ``(dx, dy)`` for one landmark, in normalized image coordinates, applied
to EVERY frame. That is the shape of the errors it exists for: the cause (dark
fabric, the shape of a shoe) is the same on every frame, so the point sits the
same distance from the true joint on every frame, and MediaPipe keeps tracking
it -- the offset rides along. What it does not do is move a point on one frame
only, which would fix one picture and leave every measurement, all of which are
whole-clip statistics, exactly where it was.

Bike only, for now. The bike analyzer reads every angle off the image-plane
landmarks (the same points the athlete drags), so an image-space offset flows
straight into the numbers. The run analyzer reads angles off MediaPipe's WORLD
landmarks, which an image-space drag cannot reach without a mapping that has
not been validated -- applying a correction there would move the skeleton on
screen and change nothing in the report, which is worse than refusing.

Every correction is validated against the frames it will be applied to, and
the report is told about it (``runner.analyze_from_frames`` echoes them). An
adjusted measurement must never pass for an automatic one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# The points an athlete may move, per near side: the joints the bike angles
# are read from, plus the ear (blended toward C7 for the trunk angle). Nothing
# on the far side -- it is neither measured nor drawn on a side view -- and
# nothing on the face or hands.
DRAGGABLE_LANDMARKS: dict[str, tuple[int, ...]] = {
    "left": (7, 11, 13, 15, 23, 25, 27, 29, 31),
    "right": (8, 12, 14, 16, 24, 26, 28, 30, 32),
}

LANDMARK_NAMES: dict[int, str] = {
    7: "left ear", 8: "right ear",
    11: "left shoulder", 12: "right shoulder",
    13: "left elbow", 14: "right elbow",
    15: "left wrist", 16: "right wrist",
    23: "left hip", 24: "right hip",
    25: "left knee", 26: "right knee",
    27: "left ankle", 28: "right ankle",
    29: "left heel", 30: "right heel",
    31: "left foot index", 32: "right foot index",
}

# A joint cannot be a quarter of the frame away from where the model put it;
# past this the input is a slip or an attempt to draw a different rider.
MAX_OFFSET = 0.25
# A correction that pushes the point outside the picture on more than this
# share of frames is refused: those frames would go from "measured wrong" to
# "measured off-screen", which no one asked for.
OUT_OF_FRAME_MAX_SHARE = 0.05
# A segment whose median length changes by more than this after a correction
# earns a warning. The athlete may well be right -- a hip 25% further from the
# knee than the model thought is exactly what black shorts produce -- but
# they should hear that every other frame disagrees before they commit.
SEGMENT_CHANGE_WARN = 0.25

# The bones each draggable point belongs to, left side (right = index + 1).
_SEGMENTS_LEFT: tuple[tuple[int, int, str], ...] = (
    (7, 11, "ear-shoulder"),
    (11, 13, "shoulder-elbow"),
    (11, 23, "shoulder-hip"),
    (13, 15, "elbow-wrist"),
    (23, 25, "hip-knee"),
    (25, 27, "knee-ankle"),
    (27, 29, "ankle-heel"),
    (27, 31, "ankle-toe"),
    (29, 31, "heel-toe"),
)


def _segments(side: str) -> tuple[tuple[int, int, str], ...]:
    if side == "left":
        return _SEGMENTS_LEFT
    return tuple((a + 1, b + 1, name) for a, b, name in _SEGMENTS_LEFT)


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def normalize_corrections(
    raw: list[dict[str, Any]] | None, camera_side: str,
) -> list[dict[str, Any]]:
    """Validate and merge what the client sent into one entry per landmark.

    Merging sums: a second round of adjustment to the same joint is applied on
    top of the first, which is what "I moved it, then nudged it a bit more"
    means. Raises ``ValueError`` with a message meant for the athlete.
    """
    if camera_side not in DRAGGABLE_LANDMARKS:
        raise ValueError("camera side must be 'left' or 'right'")
    allowed = DRAGGABLE_LANDMARKS[camera_side]
    merged: dict[int, dict[str, Any]] = {}
    for entry in raw or []:
        if not isinstance(entry, dict):
            raise ValueError("each correction must be an object")
        idx = entry.get("landmark")
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise ValueError("correction.landmark must be a MediaPipe index")
        if idx not in allowed:
            name = LANDMARK_NAMES.get(idx, f"landmark {idx}")
            raise ValueError(
                f"{name} cannot be adjusted on a clip filmed from the "
                f"{camera_side}: only the {camera_side}-side joints are measured."
            )
        dx, dy = entry.get("dx"), entry.get("dy")
        if not _finite(dx) or not _finite(dy):
            raise ValueError("correction.dx and .dy must be finite numbers")
        dx, dy = float(dx), float(dy)
        if abs(dx) > MAX_OFFSET or abs(dy) > MAX_OFFSET:
            raise ValueError(
                f"{LANDMARK_NAMES[idx]} moved more than a quarter of the frame; "
                "that is further than a joint can be from where the model put it."
            )
        frame_idx = entry.get("frame_idx")
        cur = merged.setdefault(idx, {"landmark": idx, "dx": 0.0, "dy": 0.0})
        cur["dx"] = round(cur["dx"] + dx, 6)
        cur["dy"] = round(cur["dy"] + dy, 6)
        if isinstance(frame_idx, int) and not isinstance(frame_idx, bool):
            # Where the athlete made the LAST edit -- an audit detail today,
            # the anchor for per-frame interpolation if that is ever built.
            cur["frame_idx"] = frame_idx
    # A correction that sums to nothing is not a correction.
    return [c for c in merged.values() if c["dx"] != 0.0 or c["dy"] != 0.0]


def _point(frame: dict[str, Any], idx: int) -> tuple[float, float] | None:
    lm = frame["normalized_landmarks"][idx]
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if not _finite(x) or not _finite(y):
        return None
    return (float(x), float(y))


def check_plausibility(
    frames: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    camera_side: str,
    *,
    aspect: float = 1.0,
    min_samples: int = 5,
) -> list[dict[str, Any]]:
    """What the clip's own geometry says about each correction, BEFORE applying.

    For every bone the moved point belongs to, compares the median length
    across the clip with and without the offset. Returns a warning per bone
    that changes by more than ``SEGMENT_CHANGE_WARN``; raises ``ValueError``
    for a correction that pushes the point out of the frame on more than
    ``OUT_OF_FRAME_MAX_SHARE`` of frames.

    ``aspect`` is frame width over height: normalized x spans the width and y
    the height, so a length read without it is anisotropic.

    ``min_samples``: how many frames a bone must be measured on before its
    median means anything. A clip wants a handful; a single photo IS one
    sample and passes 1 -- there the comparison is simply before vs after.
    """
    offsets = {c["landmark"]: (c["dx"], c["dy"]) for c in corrections}
    warnings: list[dict[str, Any]] = []
    if not frames or not offsets:
        return warnings

    def _shifted(frame: dict[str, Any], idx: int) -> tuple[float, float] | None:
        p = _point(frame, idx)
        if p is None:
            return None
        dx, dy = offsets.get(idx, (0.0, 0.0))
        return (p[0] + dx, p[1] + dy)

    def _length(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])

    for idx, (dx, dy) in offsets.items():
        pts = [p for p in (_point(f, idx) for f in frames) if p is not None]
        if not pts:
            continue
        out = sum(
            1 for (x, y) in pts
            if not (0.0 <= x + dx <= 1.0 and 0.0 <= y + dy <= 1.0)
        )
        if out / len(pts) > OUT_OF_FRAME_MAX_SHARE:
            raise ValueError(
                f"That puts the {LANDMARK_NAMES[idx]} outside the picture on "
                f"{round(out / len(pts) * 100)}% of frames."
            )

    seen: set[tuple[int, int]] = set()
    for a, b, name in _segments(camera_side):
        if a not in offsets and b not in offsets:
            continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        before, after = [], []
        for f in frames:
            pa, pb = _point(f, a), _point(f, b)
            if pa is None or pb is None:
                continue
            before.append(_length(pa, pb))
            after.append(_length(_shifted(f, a), _shifted(f, b)))  # type: ignore[arg-type]
        if len(before) < max(1, min_samples):
            continue
        med_before = float(np.median(before))
        med_after = float(np.median(after))
        if med_before <= 1e-6:
            continue
        change = (med_after - med_before) / med_before
        if abs(change) > SEGMENT_CHANGE_WARN:
            moved = a if a in offsets else b
            warnings.append({
                "landmark": moved,
                "segment": name,
                "change_pct": round(change * 100, 1),
                "message": (
                    f"Moving the {LANDMARK_NAMES[moved]} there makes the "
                    f"{name} segment {abs(round(change * 100))}% "
                    f"{'longer' if change > 0 else 'shorter'} than the model "
                    "measured it. The model may simply have been wrong -- or "
                    "this is a slip. Worth a second look before applying."
                ),
            })
    return warnings


def apply_corrections(
    frames: list[dict[str, Any]],
    corrections: list[dict[str, Any]],
    sport_type: str,
) -> dict[int, int]:
    """Shift the corrected landmarks on every frame, in place.

    Only the normalized (image-plane) landmarks move: on a bike clip those are
    what every angle is read from and what the overlay draws. The world
    landmarks are left alone -- they decide the camera side (by depth) and
    carry the visibility the quality metrics read, and a correction changes
    neither of those. Frames where the point is gated (NaN) stay gated: a
    point we could not measure is not made measurable by an offset.

    Returns ``{landmark: frames moved}``. Raises ``ValueError`` for a sport the
    offset would not reach the measurements of.
    """
    if sport_type != "bike":
        raise ValueError(
            "Joint corrections are available for cycling clips only: the "
            "running analysis measures from a different set of points that an "
            "on-screen adjustment does not reach yet."
        )
    touched: dict[int, int] = {}
    for c in corrections:
        idx, dx, dy = int(c["landmark"]), float(c["dx"]), float(c["dy"])
        n = 0
        for frame in frames:
            lms = frame.get("normalized_landmarks") or []
            if idx >= len(lms):
                continue
            lm = lms[idx]
            if not _finite(getattr(lm, "x", None)) or not _finite(getattr(lm, "y", None)):
                continue
            lm.x = float(lm.x) + dx
            lm.y = float(lm.y) + dy
            n += 1
        touched[idx] = n
    return touched


__all__ = [
    "DRAGGABLE_LANDMARKS",
    "LANDMARK_NAMES",
    "MAX_OFFSET",
    "OUT_OF_FRAME_MAX_SHARE",
    "SEGMENT_CHANGE_WARN",
    "apply_corrections",
    "check_plausibility",
    "normalize_corrections",
]
