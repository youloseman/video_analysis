"""Single-photo body position analysis with scoring and annotated thumbnail.

Unlike video analysis (multi-frame, temporal, background task, DB storage),
photo analysis is:
- Single frame (static_image_mode=True in MediaPipe)
- Synchronous (returns JSON directly)
- No DB storage (instant feedback)
- Reuses the same angle calculation functions from the video pipeline
"""

import base64
import io
import math
import time
from typing import Any

import numpy as np
import structlog
from PIL import Image, ImageOps

from app.services.video_analysis import overlay_style

from app.services.video_analysis.biomechanics.angle_calculator import (
    MIN_LANDMARK_VISIBILITY,
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_body_rotation,
    calculate_forearm_tilt_2d,
    calculate_head_alignment_2d,
    calculate_segment_to_vertical,
)
from app.services.video_analysis.biomechanics.cycling_positions import (
    get_cycling_reference,
    get_position_label,
)
from app.services.video_analysis.pipeline import _draw_dashed_line
from app.services.video_analysis.biomechanics.sport_configs import (
    RUNNING_REFERENCE,
    SWIMMING_REFERENCE,
)
from app.core.config import settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Angle landmark triplets per sport
# ---------------------------------------------------------------------------

# Running: near-side only, unprefixed keys (matches running_analyzer.py)
RUNNING_PHOTO_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "left": {
        "knee": (23, 25, 27),   # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        "hip": (11, 23, 25),    # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
        "ankle": (25, 27, 29),  # LEFT_KNEE, LEFT_ANKLE, LEFT_HEEL
        "elbow": (11, 13, 15),  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    },
    "right": {
        "knee": (24, 26, 28),
        "hip": (12, 24, 26),
        "ankle": (26, 28, 30),
        "elbow": (12, 14, 16),
    },
}

RUNNING_TRUNK_LANDMARKS: dict[str, tuple[int, int]] = {
    "left": (11, 23),
    "right": (12, 24),
}

# Cycling: near-side only, unprefixed keys (matches cycling_analyzer.py)
CYCLING_PHOTO_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "left": {
        "knee": (23, 25, 27),
        "hip": (11, 23, 25),
        "ankle": (25, 27, 31),   # KNEE, ANKLE, FOOT_INDEX
        "elbow": (11, 13, 15),
        "shoulder": (13, 11, 23),  # ELBOW -> SHOULDER -> HIP
    },
    "right": {
        "knee": (24, 26, 28),
        "hip": (12, 24, 26),
        "ankle": (26, 28, 32),
        "elbow": (12, 14, 16),
        "shoulder": (14, 12, 24),  # ELBOW -> SHOULDER -> HIP
    },
}

# Swimming: both sides, prefixed keys, uses 3D angles
SWIMMING_PHOTO_ANGLES: dict[str, tuple[int, int, int]] = {
    "left_shoulder": (23, 11, 13),   # HIP -> SHOULDER -> ELBOW
    "right_shoulder": (24, 12, 14),
    "left_elbow": (11, 13, 15),      # SHOULDER -> ELBOW -> WRIST
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),       # HIP -> KNEE -> ANKLE
    "right_knee": (24, 26, 28),
}

# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

_ANGLE_LABELS: dict[str, dict[str, str]] = {
    "run": {
        "knee": "Knee Angle",
        "hip": "Hip Angle",
        "ankle": "Ankle Angle",
        "elbow": "Elbow Angle",
        "trunk": "Trunk Lean",
    },
    "bike": {
        "knee": "Knee Angle",
        "hip": "Hip Angle",
        "ankle": "Ankle Angle",
        "elbow": "Elbow Angle",
        "trunk": "Trunk Angle",
        "shoulder": "Shoulder Angle",
        "forearm_tilt": "Forearm Tilt",
        "head_alignment": "Head Tuck",
        "pelvic_ratio": "Pelvic Rotation",
    },
    "swim": {
        "left_shoulder": "Left Shoulder",
        "right_shoulder": "Right Shoulder",
        "left_elbow": "Left Elbow",
        "right_elbow": "Right Elbow",
        "left_knee": "Left Knee",
        "right_knee": "Right Knee",
        "body_rotation": "Body Rotation",
        "streamline": "Streamline",
    },
}

# ---------------------------------------------------------------------------
# Optimal ranges per sport
# ---------------------------------------------------------------------------


def _get_running_optimal_ranges() -> dict[str, tuple[float, float]]:
    """Optimal ranges for running photo (generic, not gait-phase-specific)."""
    return {
        "knee": (80, 175),   # Wide -- photo captures one unknown gait instant
        "hip": (150, 180),
        "trunk": RUNNING_REFERENCE["trunk_lean"],       # (4, 8)
        "elbow": RUNNING_REFERENCE["elbow_angle"],      # (85, 100)
        "ankle": (90, 120),
    }


def _get_cycling_optimal_ranges(
    cycling_position: str | None,
    pedal_phase: str = "near_bdc",
) -> dict[str, tuple[float, float]]:
    """Position-dependent cycling ranges with pedal-phase-aware knee range."""
    ref = get_cycling_reference(cycling_position)
    # A single photo cannot verify the crank is truly at TDC/BDC -- the pedal
    # phase is a coarse guess from the knee angle alone. So widen the band on
    # both sides (not just -10 on the min) to avoid faulting a rider for a knee
    # angle caught slightly off the assumed dead-centre.
    if pedal_phase == "near_bdc":
        knee_range = (ref["knee_at_bdc"][0] - 12, ref["knee_at_bdc"][1] + 8)
    elif pedal_phase == "near_tdc":
        knee_range = (ref["knee_at_tdc"][0] - 12, ref["knee_at_tdc"][1] + 8)
    else:
        # Mid-stroke: very wide range -- cannot reliably score
        knee_range = (60, 155)
    return {
        "knee": knee_range,
        "hip": ref["hip_angle_max"],
        "trunk": ref["trunk_angle"],
        "elbow": ref["elbow_angle"],
        "ankle": (70, 110),
        "shoulder": ref["shoulder_angle"],
        "forearm_tilt": ref["forearm_tilt"],
        "head_alignment": ref["head_alignment"],
        "pelvic_ratio": ref["pelvic_ratio"],
    }


def _get_swimming_optimal_ranges() -> dict[str, tuple[float, float]]:
    return {
        "left_shoulder": (120, 180),
        "right_shoulder": (120, 180),
        "left_elbow": SWIMMING_REFERENCE["elbow_at_catch"],   # (90, 120)
        "right_elbow": SWIMMING_REFERENCE["elbow_at_catch"],
        "left_knee": (150, 180),
        "right_knee": (150, 180),
        "body_rotation": SWIMMING_REFERENCE["body_rotation"],  # (40, 60)
        "streamline": SWIMMING_REFERENCE["streamline"],        # (0, 10)
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

PHOTO_RUNNING_WEIGHTS = {
    "knee": 0.30, "trunk": 0.25, "elbow": 0.20, "hip": 0.15, "ankle": 0.10,
}
# Hip and ankle sweep through a huge range across the pedal stroke (open at the
# bottom, closed at the top), exactly like the knee -- but unlike the knee we
# have no per-phase reference band for them, so a single-photo value can't be
# scored against the stroke-*average* band without faulting good positions (a
# near-BDC hip reads ~95deg, well above the average band). They are therefore
# measured + shown but NOT scored or judged from a still. (See the interactive
# pedal-phase picker for the follow-up that scores them against the right band.)
_PHASE_DEPENDENT_BIKE = frozenset({"hip", "ankle"})

PHOTO_CYCLING_WEIGHTS = {
    "knee": 0.24, "trunk": 0.20, "shoulder": 0.13, "head_alignment": 0.11,
    "elbow": 0.11, "pelvic_ratio": 0.08, "forearm_tilt": 0.05,
}
# Near TDC/BDC from a single photo: the pedal phase is only a guess, so trust
# the knee less than a true motion-capture dead-centre -- shift its weight to
# the pedal-phase-independent trunk/shoulder/head metrics.
PHOTO_CYCLING_WEIGHTS_PHASED = {
    "knee": 0.14, "trunk": 0.24, "shoulder": 0.14, "head_alignment": 0.12,
    "elbow": 0.11, "pelvic_ratio": 0.08, "forearm_tilt": 0.05,
}
# Mid-stroke: knee unreliable, redistribute weight to the phase-independent set.
PHOTO_CYCLING_WEIGHTS_MIDSTROKE = {
    "knee": 0.06, "trunk": 0.27, "shoulder": 0.15, "head_alignment": 0.13,
    "elbow": 0.12, "pelvic_ratio": 0.08, "forearm_tilt": 0.05,
}
PHOTO_SWIMMING_WEIGHTS = {
    "left_elbow": 0.20, "right_elbow": 0.10,
    "body_rotation": 0.25, "streamline": 0.25,
    "left_shoulder": 0.10, "right_shoulder": 0.10,
}

_PHOTO_WEIGHTS = {
    "run": PHOTO_RUNNING_WEIGHTS,
    "bike": PHOTO_CYCLING_WEIGHTS,
    "swim": PHOTO_SWIMMING_WEIGHTS,
}


def _score_single_angle(
    value: float, optimal_min: float, optimal_max: float,
    low_tolerance: float = 1.0, high_tolerance: float = 1.0,
) -> int | None:
    """Score one angle 0-100 by distance from the optimal range.

    Linear falloff (not a step function) so a value 0.1 deg out of range
    doesn't cliff-drop 20 points -- the score degrades smoothly at
    ``FALLOFF_PER_DEG`` points per degree outside the band, floored at 20.

    ``low_tolerance`` / ``high_tolerance`` scale the penalty on each side
    (values > 1.0 = more lenient). Used to make a metric asymmetric, e.g.
    a deep aero trunk being *below* the optimal min is far less of a fault
    than being above it, so we forgive the low side more gently.
    """
    if math.isnan(value):
        return None

    if optimal_min <= value <= optimal_max:
        return 100

    FALLOFF_PER_DEG = 2.0  # points lost per degree outside the range
    FLOOR = 20

    if value < optimal_min:
        distance = (optimal_min - value) / max(low_tolerance, 0.1)
    else:
        distance = (value - optimal_max) / max(high_tolerance, 0.1)

    score = 100.0 - distance * FALLOFF_PER_DEG
    return int(round(max(FLOOR, min(100.0, score))))


def _assign_photo_grade(score: int) -> str:
    """Convert 0-100 score to grade label."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Needs Work"


# Asymmetric scoring tolerances for aggressive aero positions. A trunk or hip
# angle *below* the optimal min means a flatter back / more closed hip -- which
# is the aerodynamic GOAL, not a fault -- so penalize the low side ~3x more
# gently. Being above the max (too upright / too open = draggy) keeps the full
# penalty. Format: angle_key -> (low_tolerance, high_tolerance).
_AERO_TOLERANCES = {
    "trunk": (3.0, 1.0),   # flat aero back below min: forgiven; too upright: full penalty
    "hip":   (2.5, 1.0),   # closed aero hip below min: mostly forgiven
}
_AERO_POSITIONS = {"tt_aero", "triathlon"}


def _score_photo_angles(
    angles: dict[str, float],
    optimal_ranges: dict[str, tuple[float, float]],
    sport: str,
    weights_override: dict[str, float] | None = None,
    cycling_position: str | None = None,
) -> dict[str, Any]:
    """Score all angles and compute weighted overall score."""
    weights = weights_override or _PHOTO_WEIGHTS.get(sport, PHOTO_RUNNING_WEIGHTS)
    is_aero = sport == "bike" and cycling_position in _AERO_POSITIONS
    per_angle: dict[str, dict[str, Any]] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for angle_key, weight in weights.items():
        value = angles.get(angle_key)
        if value is None or angle_key not in optimal_ranges:
            continue

        opt_min, opt_max = optimal_ranges[angle_key]
        low_tol, high_tol = _AERO_TOLERANCES.get(angle_key, (1.0, 1.0)) if is_aero else (1.0, 1.0)
        score = _score_single_angle(value, opt_min, opt_max, low_tol, high_tol)
        if score is None:
            continue

        per_angle[angle_key] = {
            "score": score,
            "weight": weight,
            "weighted": round(score * weight, 1),
        }
        weighted_sum += score * weight
        total_weight += weight

    if total_weight > 0:
        overall = int(round(weighted_sum / total_weight))
    else:
        overall = 50

    overall = max(0, min(100, overall))

    return {
        "overall_score": overall,
        "grade": _assign_photo_grade(overall),
        "per_angle": per_angle,
    }


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

# Skeleton connections (from pipeline.py)
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 17), (16, 18), (15, 19), (16, 20),
    (15, 21), (16, 22), (17, 19), (18, 20),
    (19, 21), (20, 22),
    (27, 29), (28, 30), (27, 31), (28, 32),
    (29, 31), (30, 32),
]

_LEFT_SIDE = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_SIDE = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
_MIDLINE = {0, 9, 10}
_VIS_THRESHOLD = 0.5

# Label positioning: landmark index + offset direction per angle key
_LABEL_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "run": {
        "left": {
            "knee":  {"idx": 25, "offset_dir": "left"},
            "hip":   {"idx": 23, "offset_dir": "up-left"},
            "trunk": {"idx": 11, "offset_dir": "up"},
            "elbow": {"idx": 13, "offset_dir": "up-left"},
            "ankle": {"idx": 27, "offset_dir": "down-left"},
        },
        "right": {
            "knee":  {"idx": 26, "offset_dir": "right"},
            "hip":   {"idx": 24, "offset_dir": "up-right"},
            "trunk": {"idx": 12, "offset_dir": "up"},
            "elbow": {"idx": 14, "offset_dir": "up-right"},
            "ankle": {"idx": 28, "offset_dir": "down-right"},
        },
    },
    "bike": {
        "left": {
            "knee":         {"idx": 25, "offset_dir": "left"},
            "hip":          {"idx": 23, "offset_dir": "up-left"},
            "trunk":        {"idx": 11, "offset_dir": "up"},
            "elbow":        {"idx": 13, "offset_dir": "up-left"},
            "ankle":        {"idx": 27, "offset_dir": "down-left"},
            "shoulder":     {"idx": 11, "offset_dir": "right"},
            "forearm_tilt": {"idx": 15, "offset_dir": "down-right"},
        },
        "right": {
            "knee":         {"idx": 26, "offset_dir": "right"},
            "hip":          {"idx": 24, "offset_dir": "up-right"},
            "trunk":        {"idx": 12, "offset_dir": "up"},
            "elbow":        {"idx": 14, "offset_dir": "up-right"},
            "ankle":        {"idx": 28, "offset_dir": "down-right"},
            "shoulder":     {"idx": 12, "offset_dir": "left"},
            "forearm_tilt": {"idx": 16, "offset_dir": "down-left"},
        },
    },
    "swim": {
        "any": {
            "left_shoulder":  {"idx": 11, "offset_dir": "up"},
            "right_shoulder": {"idx": 12, "offset_dir": "up-right"},
            "left_elbow":     {"idx": 13, "offset_dir": "left"},
            "right_elbow":    {"idx": 14, "offset_dir": "right"},
            "left_knee":      {"idx": 25, "offset_dir": "down-left"},
            "right_knee":     {"idx": 26, "offset_dir": "down-right"},
        },
    },
}


def _is_near_side(idx: int, camera_side: str | None) -> bool:
    """Check if a landmark index belongs to the near side."""
    if idx in _MIDLINE:
        return True
    if camera_side == "left":
        return idx in _LEFT_SIDE
    elif camera_side == "right":
        return idx in _RIGHT_SIDE
    return True


def _score_to_color(score: int | None) -> tuple[int, int, int]:
    """Map score to BGR color for thumbnail drawing."""
    if score is None:
        return (180, 180, 180)  # Gray for N/A
    if score >= 80:
        return (0, 220, 0)     # Green
    elif score >= 60:
        return (0, 200, 255)   # Orange
    else:
        return (0, 0, 255)     # Red


def _draw_arc(
    cv2_mod, frame, pixel_coords: list[tuple[int, int, float]],
    prox_idx: int, vert_idx: int, dist_idx: int,
    color: tuple[int, int, int], body_height_px: float,
) -> None:
    """Draw angle arc at joint vertex (same algorithm as pipeline._draw_angle_arc)."""
    max_idx = max(prox_idx, vert_idx, dist_idx)
    if max_idx >= len(pixel_coords):
        return

    px, py, pv = pixel_coords[prox_idx]
    vx, vy, vv = pixel_coords[vert_idx]
    dx, dy, dv = pixel_coords[dist_idx]

    if min(pv, vv, dv) < _VIS_THRESHOLD:
        return

    v1x, v1y = px - vx, py - vy
    v2x, v2y = dx - vx, dy - vy

    ang1 = math.degrees(math.atan2(v1y, v1x))
    ang2 = math.degrees(math.atan2(v2y, v2x))

    start = min(ang1, ang2)
    end = max(ang1, ang2)
    if end - start > 180:
        start, end = end, start + 360

    radius = max(12, int(body_height_px * 0.08))
    cv2_mod.ellipse(
        frame, (vx, vy), (radius, radius), 0,
        start, end, color, 1, cv2_mod.LINE_AA,
    )


def _generate_photo_thumbnail(
    cv2_mod, image, normalized_landmarks,
    angles: dict[str, float],
    score_data: dict[str, Any],
    camera_side: str,
    sport: str,
    arc_triplets: dict[str, tuple[int, int, int]],
    cycling_position: str | None = None,
    hide_angle_values: bool = False,
    optimal_ranges: dict[str, tuple[float, float]] | None = None,
) -> bytes:
    """Generate annotated thumbnail with skeleton, angle labels, and score badge.

    Returns PNG image as bytes. ``hide_angle_values`` (free-tier teaser): keep
    the skeleton + arcs but mask the numeric angle labels and burn a watermark.
    """
    frame = image.copy()
    h, w = frame.shape[:2]
    optimal_ranges = optimal_ranges or {}

    # Convert normalized landmarks to pixel coordinates
    pixel_coords: list[tuple[int, int, float]] = []
    for lm in normalized_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        vis = getattr(lm, "visibility", 1.0)
        pixel_coords.append((px, py, float(vis)))

    # --- 1. DRAW SKELETON (neon + soft glow, shared style) ---
    side_filter = camera_side if sport in ("run", "bike") else None

    _segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for start_idx, end_idx in _POSE_CONNECTIONS:
        if start_idx >= len(pixel_coords) or end_idx >= len(pixel_coords):
            continue
        sx, sy, sv = pixel_coords[start_idx]
        ex, ey, ev = pixel_coords[end_idx]
        if sv < _VIS_THRESHOLD or ev < _VIS_THRESHOLD:
            continue
        if side_filter and not (
            _is_near_side(start_idx, side_filter)
            and _is_near_side(end_idx, side_filter)
        ):
            continue
        _segments.append(((sx, sy), (ex, ey)))

    _dots: list[tuple[int, int]] = []
    for i, (px, py, vis) in enumerate(pixel_coords):
        if vis < _VIS_THRESHOLD:
            continue
        if side_filter and not _is_near_side(i, side_filter):
            continue
        _dots.append((px, py))

    # scale the skeleton weight to the image so it reads on both small and large photos
    _sk = max(1.0, min(2.4, w / 900))
    overlay_style.draw_glow_skeleton(
        cv2_mod, frame, _segments, _dots,
        glow=True, line_w=max(2, int(2 * _sk)), dot_r=max(3, int(4 * _sk)),
    )
    chips = overlay_style.ChipLayer(frame)

    # Header first: it reserves the top strip so no metric chip lands under it.
    _sport_labels = {"run": "RUN", "bike": "BIKE", "swim": "SWIM"}
    _overall = score_data.get("overall_score", 0)
    _grade = score_data.get("grade", "")
    _hdr_status = "good" if _overall >= 75 else "warn" if _overall >= 60 else "bad"
    _title = (
        "AERODYNAMIC PROFILE"
        if (sport == "bike" and cycling_position in ("tt_aero", "triathlon"))
        else ("CYCLING PROFILE" if sport == "bike" else "RUNNING PROFILE")
    )
    _pad = int(max(10, h * 0.018))
    chips.header((_pad, _pad), _sport_labels.get(sport, sport.upper()),
                 f"{_overall}/100", _grade, _hdr_status,
                 right_text=_title, frame_w=w, scale=_sk)

    # --- 2. ANGLE ARCS + LABELS ---
    if len(pixel_coords) > 25:
        # Body size reference for adaptive scaling
        s11x, s11y, _ = pixel_coords[11]
        h23x, h23y, _ = pixel_coords[23]
        body_height_px = max(50.0, abs(s11y - h23y))
        offset_px = max(70, int(body_height_px * 0.5))

        offset_vectors = {
            "left":       (-offset_px, 0),
            "right":      (offset_px, 0),
            "up":         (0, -offset_px),
            "down":       (0, offset_px),
            "up-left":    (-offset_px, -int(offset_px * 0.7)),
            "up-right":   (offset_px, -int(offset_px * 0.7)),
            "down-left":  (-offset_px, int(offset_px * 0.7)),
            "down-right": (offset_px, int(offset_px * 0.7)),
        }

        per_angle = score_data.get("per_angle", {})
        labels = _ANGLE_LABELS.get(sport, {})

        # Get label configs for this sport + camera side
        if sport == "swim":
            label_cfg = _LABEL_CONFIGS.get("swim", {}).get("any", {})
        else:
            label_cfg = _LABEL_CONFIGS.get(sport, {}).get(camera_side, {})

        for angle_key, angle_value in angles.items():
            if math.isnan(angle_value):
                continue
            cfg = label_cfg.get(angle_key)
            if not cfg:
                continue

            lm_idx = cfg["idx"]
            if lm_idx >= len(pixel_coords):
                continue
            _, _, lm_vis = pixel_coords[lm_idx]
            if lm_vis < _VIS_THRESHOLD:
                continue

            # Color based on score
            angle_score_data = per_angle.get(angle_key)
            score_val = angle_score_data["score"] if angle_score_data else None
            color = _score_to_color(score_val)

            # Draw arc at joint
            triplet = arc_triplets.get(angle_key)
            if triplet:
                _draw_arc(
                    cv2_mod, frame, pixel_coords,
                    *triplet, color, body_height_px,
                )

            # Leader line + neon chip (label + big coloured value)
            jx, jy, _ = pixel_coords[lm_idx]
            dx, dy = offset_vectors.get(cfg["offset_dir"], (offset_px, 0))
            # chips on the left of the joint are right-aligned so they hug the leader
            align = "right" if dx < 0 else "left"
            lx = max(5, min(w - 5, jx + dx))
            ly = max(20, min(h - 20, jy + dy))

            label_name = labels.get(angle_key, angle_key.replace("_", " ").title()).upper()
            if hide_angle_values:
                value_txt, status = "LOCKED", "muted"
            else:
                if angle_key == "pelvic_ratio":
                    value_txt = f"{angle_value:.1f}x"
                elif angle_key == "head_alignment":
                    value_txt = f"{angle_value:.0f}/100"
                else:
                    value_txt = f"{angle_value:.0f}°"
                # colour by the same zone rule the score/UI use; ratios need a
                # smaller margin floor than the degree default. Phase-dependent
                # joints (bike hip/ankle) aren't judged from a still -> neutral.
                if sport == "bike" and angle_key in _PHASE_DEPENDENT_BIKE:
                    status = "muted"
                else:
                    opt = optimal_ranges.get(angle_key)
                    floor = 0.3 if angle_key == "pelvic_ratio" else 3.0
                    status = (
                        overlay_style.status_for(angle_value, *opt, min_margin=floor)
                        if opt else "muted"
                    )

            status_rgb = overlay_style.STATUS_COLORS.get(status, overlay_style.INK_SOFT)
            overlay_style.draw_leader(cv2_mod, frame, (jx, jy), (lx, ly), status_rgb)
            chips.metric_chip((lx, ly), label_name, value_txt, status,
                              scale=_sk, align=align)

    # --- 2b. HEAD ALIGNMENT + PELVIC RATIO overlays (bike only) ---
    if sport == "bike" and len(pixel_coords) > 25:
        s11x, s11y, _ = pixel_coords[11]
        h23x, h23y, _ = pixel_coords[23]
        bh_px = max(50.0, abs(s11y - h23y))

        # Head alignment: dashed back-line + score near ear
        head_val = angles.get("head_alignment")
        if head_val is not None and not math.isnan(head_val) and head_val > 0 and bh_px > 40:
            if camera_side == "left":
                sh_i, hp_i, ear_i = 11, 23, 7
            else:
                sh_i, hp_i, ear_i = 12, 24, 8

            shx, shy, shv = pixel_coords[sh_i]
            hpx, hpy, hpv = pixel_coords[hp_i]
            eax, eay, eav = pixel_coords[ear_i]

            if shv >= _VIS_THRESHOLD and hpv >= _VIS_THRESHOLD:
                _draw_dashed_line(cv2_mod, frame, (hpx, hpy), (shx, shy), (180, 180, 255), 1)
                dx_ext = shx - hpx
                dy_ext = shy - hpy
                ext_x = shx + int(dx_ext * 0.5)
                ext_y = shy + int(dy_ext * 0.5)
                _draw_dashed_line(cv2_mod, frame, (shx, shy), (ext_x, ext_y), (180, 180, 255), 1)

                if eav >= _VIS_THRESHOLD:
                    if hide_angle_values:
                        h_status, h_text = "muted", "LOCKED"
                    else:
                        h_status = "good" if head_val >= 75 else ("warn" if head_val >= 50 else "bad")
                        h_text = f"{head_val:.0f}/100"
                    hlx = max(5, min(w - 5, eax + int(28 * _sk)))
                    hly = max(20, min(h - 20, eay - int(26 * _sk)))
                    overlay_style.draw_leader(
                        cv2_mod, frame, (eax, eay), (hlx, hly),
                        overlay_style.STATUS_COLORS.get(h_status, overlay_style.INK_SOFT),
                    )
                    chips.metric_chip((hlx, hly), "HEAD POSITION", h_text, h_status, scale=_sk)

        # Pelvic ratio: small label near hip
        pelvic_val = angles.get("pelvic_ratio")
        if pelvic_val is not None and not math.isnan(pelvic_val) and pelvic_val > 0 and bh_px > 40:
            ref_p = get_cycling_reference(cycling_position)
            p_min, p_max = ref_p["pelvic_ratio"]
            if hide_angle_values:
                p_status, p_text = "muted", "LOCKED"
            else:
                # ratio, not degrees -> small margin floor
                p_status = overlay_style.status_for(pelvic_val, p_min, p_max, min_margin=0.3)
                p_text = f"{pelvic_val:.1f}x"
            hp_i2 = 23 if camera_side == "left" else 24
            hpx2, hpy2, _ = pixel_coords[hp_i2]
            off_px = max(50, int(bh_px * 0.35))
            plx = max(5, min(w - 5, hpx2 - int(off_px * 0.5)))
            ply = max(20, min(h - 20, hpy2 + int(off_px * 0.6)))
            overlay_style.draw_leader(
                cv2_mod, frame, (hpx2, hpy2), (plx, ply),
                overlay_style.STATUS_COLORS.get(p_status, overlay_style.INK_SOFT),
            )
            chips.metric_chip((plx, ply), "PELVIC TILT", p_text, p_status,
                              scale=_sk, align="right")

    # --- 3. BRANDING, then paint the header + every chip in one PIL pass ---
    chips.brand((w - _pad, h - _pad), "FLAPP",
                "FREE" if hide_angle_values else "", scale=_sk)

    frame = chips.flush()

    # Encode to PNG
    success, buf = cv2_mod.imencode(".png", frame, [cv2_mod.IMWRITE_PNG_COMPRESSION, 6])
    if not success:
        raise ValueError("Failed to encode thumbnail image")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_camera_side(world_landmarks) -> str:
    """Detect which body side faces the camera (single frame).

    Same logic as base_analyzer.detect_camera_side():
    Z increases AWAY from camera; smaller Z = closer.
    """
    left_z = (world_landmarks[11].z + world_landmarks[23].z) / 2
    right_z = (world_landmarks[12].z + world_landmarks[24].z) / 2
    return "left" if left_z < right_z else "right"


def _classify_angle_status(
    value: float, optimal_min: float, optimal_max: float,
) -> str:
    """Classify angle measurement: optimal / acceptable / needs_work."""
    if math.isnan(value):
        return "insufficient_visibility"

    if optimal_min <= value <= optimal_max:
        return "optimal"

    range_width = optimal_max - optimal_min
    margin = max(range_width * 0.10, 3.0)  # At least 3 deg margin

    if (optimal_min - margin) <= value <= (optimal_max + margin):
        return "acceptable"

    return "needs_work"


def _auto_detect_cycling_position(trunk_angle: float) -> str:
    """Guess cycling position from trunk angle."""
    if math.isnan(trunk_angle):
        return "road_hoods"

    if trunk_angle < 22:
        return "tt_aero"
    elif trunk_angle < 33:
        return "road_drops"
    elif trunk_angle < 46:
        return "road_hoods"
    else:
        return "casual"


def _estimate_pedal_phase(knee_angle: float) -> str:
    """Estimate pedal phase from single-photo knee angle.

    Returns: 'near_bdc', 'near_tdc', or 'mid_stroke'.
    BDC = leg extended (high angle), TDC = leg flexed (low angle).
    """
    if math.isnan(knee_angle):
        return "mid_stroke"
    if knee_angle >= 125:
        return "near_bdc"
    elif knee_angle <= 85:
        return "near_tdc"
    else:
        return "mid_stroke"


def _safe_round(value: float, decimals: int = 1) -> float:
    """Round a value, returning NaN as-is."""
    if math.isnan(value):
        return value
    return round(value, decimals)


def _build_arc_triplets(
    sport: str, camera_side: str,
) -> dict[str, tuple[int, int, int]]:
    """Build arc triplet dict for thumbnail drawing."""
    if sport == "run":
        return dict(RUNNING_PHOTO_ANGLES[camera_side])
    elif sport == "bike":
        return dict(CYCLING_PHOTO_ANGLES[camera_side])
    elif sport == "swim":
        return dict(SWIMMING_PHOTO_ANGLES)
    return {}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def _preprocess_image_bytes(image_bytes: bytes) -> bytes:
    """Preprocess image: convert HEIC/HEIF to JPEG, apply EXIF orientation.

    Returns processed bytes suitable for cv2.imdecode.
    """
    # Check for HEIC/HEIF (ftyp box at offset 4)
    is_heic = (
        len(image_bytes) > 12
        and image_bytes[4:8] == b"ftyp"
    )

    if is_heic:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            logger.warning("HEIC_UNSUPPORTED", reason="pillow-heif not installed")
            return image_bytes

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    # For all other formats: apply EXIF auto-orientation if needed
    try:
        img = Image.open(io.BytesIO(image_bytes))
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            buf = io.BytesIO()
            fmt = img.format or "JPEG"
            if fmt.upper() == "MPO":
                fmt = "JPEG"
            transposed.save(buf, format=fmt, quality=95)
            return buf.getvalue()
    except Exception:
        logger.debug("Image EXIF auto-orient failed, using original", exc_info=True)

    return image_bytes


def analyze_photo(
    image_bytes: bytes,
    sport: str,
    cycling_position: str | None = None,
    hide_angle_values: bool = False,
) -> dict[str, Any]:
    """Analyze a single photo for body position feedback.

    This is a BLOCKING function (MediaPipe CPU work).
    The API endpoint wraps it in asyncio.to_thread().

    Args:
        image_bytes: Raw image file bytes (JPEG/PNG/WebP)
        sport: "run", "bike", or "swim"
        cycling_position: Optional cycling position for bike

    Returns:
        Dict matching PhotoAnalysisResponse schema.

    Raises:
        ValueError: If image invalid or no pose detected.
    """
    import cv2
    import mediapipe as mp

    start_time = time.time()

    # 0. Preprocess: HEIC conversion + EXIF auto-orientation
    image_bytes = _preprocess_image_bytes(image_bytes)

    # 1. Decode image
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Could not decode image. Ensure it is a valid JPEG, PNG, WebP, or HEIC file."
        )

    warnings: list[str] = []

    # Warning: low resolution
    h_img, w_img = image.shape[:2]
    if min(h_img, w_img) < 480:
        warnings.append(
            "Low resolution image detected. "
            "Higher resolution photos provide more precise measurements."
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. Run MediaPipe Pose — Tasks API (works with mediapipe >=0.10.14)
    wl = None
    nl = None

    try:
        from pathlib import Path as _Path
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        _model_paths = [
            settings.model_path,  # Flapp: backend/models/ (see core/config.py)
            _Path(__file__).parent / "models" / "pose_landmarker_heavy.task",
            _Path("/app/models/pose_landmarker_heavy.task"),
            _Path("pose_landmarker_heavy.task"),
        ]
        _model_path = None
        for _p in _model_paths:
            if _p.exists():
                _model_path = str(_p)
                break

        if _model_path:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=_model_path),
                running_mode=RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            landmarker = PoseLandmarker.create_from_options(options)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)
            landmarker.close()

            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                nl = result.pose_landmarks[0]
    except Exception as tasks_err:
        logger.warning("Tasks API failed for photo, trying legacy", err=str(tasks_err))

    # Fallback: Legacy Solutions API (mediapipe < 0.10.31)
    if wl is None:
        try:
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            ) as pose:
                result = pose.process(image_rgb)
            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks.landmark
                nl = result.pose_landmarks.landmark
        except AttributeError:
            pass  # mp.solutions removed in mediapipe >= 0.10.31

    if wl is None:
        raise ValueError(
            "No body pose detected in the image. "
            "Ensure the full body is visible and well-lit."
        )

    # 3. Detect camera side
    camera_side = _detect_camera_side(wl)

    # Warning: front/back view (not ideal side view)
    left_z = (wl[11].z + wl[23].z) / 2   # left shoulder + hip
    right_z = (wl[12].z + wl[24].z) / 2   # right shoulder + hip
    if abs(left_z - right_z) < 0.05:
        warnings.append(
            "Side view works best for position analysis. "
            "The photo appears to be from the front or back, "
            "which may give less accurate angle measurements."
        )

    # Warning: poor landmark visibility (bad lighting / occlusion)
    key_indices = [11, 12, 23, 24, 25, 26, 27, 28]
    avg_vis = sum(nl[i].visibility for i in key_indices) / len(key_indices)
    if avg_vis < 0.5:
        warnings.append(
            "Image quality appears low. "
            "Try a clearer, well-lit photo for more accurate results."
        )

    # 4. Compute angles (sport-specific)
    angles: dict[str, float] = {}
    pedal_phase: str | None = None  # Only set for cycling

    if sport == "run":
        for name, (a, b, c) in RUNNING_PHOTO_ANGLES[camera_side].items():
            val, _ = calculate_angle_2d(wl, a, b, c)
            angles[name] = _safe_round(val)

        sh_idx, hp_idx = RUNNING_TRUNK_LANDMARKS[camera_side]
        trunk = calculate_segment_to_vertical(wl, sh_idx, hp_idx)
        angles["trunk"] = _safe_round(trunk)

        optimal_ranges = _get_running_optimal_ranges()

    elif sport == "bike":
        for name, (a, b, c) in CYCLING_PHOTO_ANGLES[camera_side].items():
            val, _ = calculate_angle_2d(wl, a, b, c)
            angles[name] = _safe_round(val)

        trunk_from_vert = calculate_segment_to_vertical(wl, 11, 23)
        trunk = 90.0 - trunk_from_vert  # Convert to from-horizontal (bike fitting convention)
        angles["trunk"] = _safe_round(trunk)

        # Forearm tilt (atan2-based, not standard 3-point angle)
        if camera_side == "left":
            forearm_tilt, _ = calculate_forearm_tilt_2d(wl, 13, 15)
        else:
            forearm_tilt, _ = calculate_forearm_tilt_2d(wl, 14, 16)
        angles["forearm_tilt"] = _safe_round(forearm_tilt)

        # Head alignment (score 0-100)
        if camera_side == "left":
            head_score, _ = calculate_head_alignment_2d(wl, 11, 23, 7)
        else:
            head_score, _ = calculate_head_alignment_2d(wl, 12, 24, 8)
        angles["head_alignment"] = _safe_round(head_score)

        # Pelvic ratio (derived: hip / trunk)
        hip_val = angles.get("hip")
        trunk_val = angles.get("trunk")
        if (hip_val is not None and trunk_val is not None
                and trunk_val >= 5
                and not math.isnan(hip_val) and not math.isnan(trunk_val)):
            angles["pelvic_ratio"] = round(hip_val / trunk_val, 2)

        # Pedal phase estimation from knee angle
        knee_val = angles.get("knee")
        pedal_phase = _estimate_pedal_phase(knee_val) if knee_val is not None else "mid_stroke"

        if not cycling_position:
            cycling_position = _auto_detect_cycling_position(trunk)
            logger.info(
                "PHOTO_CYCLING_AUTODETECT",
                detected_position=cycling_position,
                trunk_angle=trunk,
            )

        optimal_ranges = _get_cycling_optimal_ranges(cycling_position, pedal_phase)

    elif sport == "swim":
        for name, (a, b, c) in SWIMMING_PHOTO_ANGLES.items():
            val, _ = calculate_angle_3d(wl, a, b, c)
            angles[name] = _safe_round(val)

        body_rot = abs(calculate_body_rotation(wl))
        angles["body_rotation"] = _safe_round(body_rot)

        streamline = calculate_segment_to_vertical(wl, 11, 27)
        angles["streamline"] = _safe_round(streamline)

        optimal_ranges = _get_swimming_optimal_ranges()

    else:
        raise ValueError(
            f"Unsupported sport: {sport}. Must be run, bike, or swim."
        )

    # Warning: partial body (some angles are NaN)
    nan_count = sum(1 for v in angles.values() if math.isnan(v))
    if 0 < nan_count < len(angles):
        measured = len(angles) - nan_count
        warnings.append(
            f"Partial body detected: {measured} of {len(angles)} angles could be measured. "
            "Ensure full body is visible for complete analysis."
        )

    # 5. Build angles_with_context
    labels = _ANGLE_LABELS.get(sport, {})
    angles_with_context: dict[str, dict[str, Any]] = {}

    for angle_name, angle_value in angles.items():
        if angle_name not in optimal_ranges:
            continue
        lbl = labels.get(angle_name, angle_name.replace("_", " ").title())
        # Pedal-phase-dependent joints (bike): measured + shown, but no verdict
        # from a single still -- the value is only meaningful once the crank
        # position is known.
        if sport == "bike" and angle_name in _PHASE_DEPENDENT_BIKE:
            angles_with_context[angle_name] = {
                "value": angle_value,
                "optimal_min": None,
                "optimal_max": None,
                "status": "phase_dependent",
                "note": ("Depends on the pedal position — not scored from a "
                         "single photo."),
                "label": lbl,
            }
            continue
        opt_min, opt_max = optimal_ranges[angle_name]
        angles_with_context[angle_name] = {
            "value": angle_value,
            "optimal_min": opt_min,
            "optimal_max": opt_max,
            "status": _classify_angle_status(angle_value, opt_min, opt_max),
            "label": lbl,
        }

    # 6. Score. Single-photo knee reliability depends on the (guessed) pedal
    # phase: mid-stroke = knee nearly ignored; near TDC/BDC = knee down-weighted
    # (the crank position can't be verified from one still).
    weights_override = None
    if sport == "bike":
        if pedal_phase == "mid_stroke":
            weights_override = PHOTO_CYCLING_WEIGHTS_MIDSTROKE
        elif pedal_phase in ("near_tdc", "near_bdc"):
            weights_override = PHOTO_CYCLING_WEIGHTS_PHASED
    score_data = _score_photo_angles(
        angles, optimal_ranges, sport, weights_override,
        cycling_position=cycling_position,
    )

    # 7. Thumbnail
    arc_triplets = _build_arc_triplets(sport, camera_side)
    thumbnail_bytes = _generate_photo_thumbnail(
        cv2, image, nl,
        angles, score_data, camera_side, sport, arc_triplets,
        cycling_position=cycling_position,
        hide_angle_values=hide_angle_values,
        optimal_ranges=optimal_ranges,
    )
    thumbnail_b64 = f"data:image/png;base64,{base64.b64encode(thumbnail_bytes).decode()}"

    processing_time = round(time.time() - start_time, 3)

    # 8. Build response
    response: dict[str, Any] = {
        "sport": sport,
        "camera_side": camera_side,
        "angles": angles,
        "angles_with_context": angles_with_context,
        "score": score_data,
        "thumbnail_base64": thumbnail_b64,
        "processing_time_seconds": processing_time,
        "warnings": warnings,
    }

    if sport == "bike":
        response["cycling_position"] = cycling_position
        response["cycling_position_label"] = get_position_label(cycling_position)
        response["pedal_phase"] = pedal_phase
        if pedal_phase == "mid_stroke":
            warnings.append(
                "Knee captured at mid-pedal-stroke -- knee angle scoring is reduced. "
                "For saddle height assessment, use a photo at the bottom of the pedal stroke."
            )

        # Position archetype classification
        from app.services.video_analysis.biomechanics.cycling_positions import detect_position_archetype
        archetype = detect_position_archetype(
            shoulder_angle=angles.get("shoulder", 0.0),
            elbow_angle=angles.get("elbow", 0.0),
            trunk_angle=angles.get("trunk", 0.0),
            hip_angle=angles.get("hip", 0.0),
            cycling_position=cycling_position,
        )
        if archetype:
            response["position_archetype"] = archetype

    logger.info(
        "PHOTO_ANALYSIS_COMPLETE",
        sport=sport,
        camera_side=camera_side,
        num_angles=len(angles),
        overall_score=score_data["overall_score"],
        grade=score_data["grade"],
        processing_time=processing_time,
    )

    return response
