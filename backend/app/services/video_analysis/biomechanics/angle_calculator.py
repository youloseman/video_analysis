"""Joint angle calculation formulas for biomechanical analysis.

All formulas adapted from MEDIAPIPE_PROJECT_ANALYSIS.md.
Uses world_landmarks (3D coordinates in meters) for angle computation.
"""

import math

import numpy as np

# Minimum per-landmark visibility to trust an angle measurement.
# Below this, any of the 3 landmarks is likely occluded and the angle unreliable.
# Bike is lower because the frame/handlebars frequently occlude landmarks.
MIN_LANDMARK_VISIBILITY = 0.7

SPORT_LANDMARK_VISIBILITY = {
    "run": 0.7,
    "bike": 0.45,
    "swim": 0.5,
}


def calculate_angle_3d(
    landmarks, idx_a: int, idx_b: int, idx_c: int,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> tuple[float, float]:
    """Calculate 3D angle at joint B between rays BA and BC.

    Formula: angle = arccos((BA . BC) / (|BA| * |BC|))

    Args:
        landmarks: list of landmarks with x, y, z attributes (world_landmarks)
        idx_a: index of point A
        idx_b: index of vertex B (where angle is measured)
        idx_c: index of point C

    Returns:
        (angle_degrees, avg_visibility) - angle in [0, 180] and average visibility
    """
    vis_a = getattr(landmarks[idx_a], "visibility", 1.0)
    vis_b = getattr(landmarks[idx_b], "visibility", 1.0)
    vis_c = getattr(landmarks[idx_c], "visibility", 1.0)
    avg_visibility = (vis_a + vis_b + vis_c) / 3.0

    # Reject if ANY landmark is below threshold
    if min(vis_a, vis_b, vis_c) < min_visibility:
        return float("nan"), avg_visibility

    a = np.array([landmarks[idx_a].x, landmarks[idx_a].y, landmarks[idx_a].z])
    b = np.array([landmarks[idx_b].x, landmarks[idx_b].y, landmarks[idx_b].z])
    c = np.array([landmarks[idx_c].x, landmarks[idx_c].y, landmarks[idx_c].z])

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if (
        np.isnan(norm_ba) or np.isnan(norm_bc)
        or norm_ba < 1e-6 or norm_bc < 1e-6
    ):
        return float("nan"), avg_visibility

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))

    return angle, avg_visibility


def calculate_angle_2d(
    landmarks, idx_a: int, idx_b: int, idx_c: int,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> tuple[float, float]:
    """Calculate 2D angle (ignoring Z depth) at joint B.

    Used when camera is perpendicular to the plane of movement.
    """
    vis_a = getattr(landmarks[idx_a], "visibility", 1.0)
    vis_b = getattr(landmarks[idx_b], "visibility", 1.0)
    vis_c = getattr(landmarks[idx_c], "visibility", 1.0)
    avg_visibility = (vis_a + vis_b + vis_c) / 3.0

    # Reject if ANY landmark is below threshold
    if min(vis_a, vis_b, vis_c) < min_visibility:
        return float("nan"), avg_visibility

    a = np.array([landmarks[idx_a].x, landmarks[idx_a].y])
    b = np.array([landmarks[idx_b].x, landmarks[idx_b].y])
    c = np.array([landmarks[idx_c].x, landmarks[idx_c].y])

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if (
        np.isnan(norm_ba) or np.isnan(norm_bc)
        or norm_ba < 1e-6 or norm_bc < 1e-6
    ):
        return float("nan"), avg_visibility

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = float(np.degrees(np.arccos(cosine)))

    return angle, avg_visibility


def calculate_segment_to_vertical(
    landmarks, idx_top: int, idx_bottom: int,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> float:
    """Calculate angle of body segment relative to vertical axis.

    Useful for trunk lean in running/cycling.
    Y-axis points DOWN in image coordinates.
    Returns np.nan if either landmark visibility is below threshold.
    """
    vis_top = getattr(landmarks[idx_top], "visibility", 1.0)
    vis_bottom = getattr(landmarks[idx_bottom], "visibility", 1.0)

    if min(vis_top, vis_bottom) < min_visibility:
        return float("nan")

    dx = landmarks[idx_bottom].x - landmarks[idx_top].x
    dy = landmarks[idx_bottom].y - landmarks[idx_top].y

    if math.isnan(dx) or math.isnan(dy):
        return float("nan")

    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return angle


def calculate_segment_to_vertical_from_points(
    top,
    bottom,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> float:
    """Like calculate_segment_to_vertical, but accepts arbitrary points.

    Mirror of the index-based variant for callers that pass synthetic
    or blended landmarks (e.g. C7 approximation built from shoulder +
    ear). Both points need x, y, and visibility attributes. Returns
    NaN on missing visibility, low confidence, or non-finite coords.
    """
    vis_top = getattr(top, "visibility", 1.0)
    vis_bottom = getattr(bottom, "visibility", 1.0)

    if min(vis_top, vis_bottom) < min_visibility:
        return float("nan")

    dx = bottom.x - top.x
    dy = bottom.y - top.y

    if math.isnan(dx) or math.isnan(dy):
        return float("nan")

    return math.degrees(math.atan2(abs(dx), abs(dy)))


def calculate_segment_to_horizontal(
    landmarks, idx_a: int, idx_b: int,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> float:
    """Angle of the segment a -> b relative to the horizontal (X) axis.

    Returns 0 when the segment is perfectly horizontal (body in a
    freestyle streamline), 90 when it is perfectly vertical. Mirror of
    calculate_segment_to_vertical with dx/dy swapped so the reference
    axis is horizontal instead of vertical. Returns NaN on low
    visibility or non-finite coords.
    """
    vis_a = getattr(landmarks[idx_a], "visibility", 1.0)
    vis_b = getattr(landmarks[idx_b], "visibility", 1.0)

    if min(vis_a, vis_b) < min_visibility:
        return float("nan")

    dx = landmarks[idx_b].x - landmarks[idx_a].x
    dy = landmarks[idx_b].y - landmarks[idx_a].y

    if math.isnan(dx) or math.isnan(dy):
        return float("nan")

    return math.degrees(math.atan2(abs(dy), abs(dx)))


def calculate_trunk_lean_midpoint(landmarks, logger=None) -> float:
    """Calculate trunk lean using midpoints of both shoulders and both hips.

    Uses X (lateral) and Y (vertical) only. Z (depth) is EXCLUDED because
    MediaPipe's monocular depth estimation has +/-10-15cm noise that inflates
    the lean angle significantly (e.g. 5 deg real -> 36 deg measured).

    MediaPipe world coords: X=lateral, Y=vertical(up), Z=towards camera(noisy).

    Returns angle in degrees from vertical (0 = upright, 4-8 = optimal running lean).
    """
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2

    if any(math.isnan(v) for v in (shoulder_x, shoulder_y, hip_x, hip_y)):
        return float("nan")

    dx = shoulder_x - hip_x
    dy = shoulder_y - hip_y

    horizontal_dist = abs(dx)
    vertical_dist = abs(dy)

    if vertical_dist < 1e-6:
        return float("nan")

    lean_deg = math.degrees(math.atan2(horizontal_dist, vertical_dist))

    if logger:
        logger.info(
            "TRUNK_LEAN_DEBUG",
            shoulder=f"x={shoulder_x:.4f} y={shoulder_y:.4f}",
            hip=f"x={hip_x:.4f} y={hip_y:.4f}",
            dx=f"{dx:.4f}",
            dy=f"{dy:.4f}",
            horizontal_dist=f"{horizontal_dist:.4f}",
            vertical_dist=f"{vertical_dist:.4f}",
            lean_deg=f"{lean_deg:.1f}",
        )

    return round(lean_deg, 1)


def calculate_body_rotation(
    landmarks, min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> float:
    """Calculate body rotation in horizontal plane using shoulder depth difference.

    For swimming - determines body roll angle.
    Uses Z-depth difference between left and right shoulder.
    Returns NaN when either shoulder is below the visibility threshold or
    carries NaN coordinates (post P0 visibility gating).
    """
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]

    vis_l = getattr(left_shoulder, "visibility", 1.0)
    vis_r = getattr(right_shoulder, "visibility", 1.0)
    if min(vis_l, vis_r) < min_visibility:
        return float("nan")

    dz = right_shoulder.z - left_shoulder.z
    dx = right_shoulder.x - left_shoulder.x

    if math.isnan(dz) or math.isnan(dx):
        return float("nan")

    return math.degrees(math.atan2(dz, dx))


def calculate_forearm_tilt_2d(
    landmarks, idx_elbow: int, idx_wrist: int,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> tuple[float, float]:
    """Angle of forearm relative to horizontal (2D, ignoring Z).

    Positive = wrist higher than elbow (modern aero bar tilt).
    Reference: Steinmetz - athletes adding 10-20 deg of bar tilt.

    Returns:
        (angle_degrees, avg_visibility) or (NaN, avg_vis) if below threshold.
    """
    vis_e = getattr(landmarks[idx_elbow], "visibility", 1.0)
    vis_w = getattr(landmarks[idx_wrist], "visibility", 1.0)
    avg_vis = (vis_e + vis_w) / 2.0

    if min(vis_e, vis_w) < min_visibility:
        return float("nan"), avg_vis

    ex, ey = landmarks[idx_elbow].x, landmarks[idx_elbow].y
    wx, wy = landmarks[idx_wrist].x, landmarks[idx_wrist].y

    dx = wx - ex
    dy = wy - ey  # image y increases downward

    # atan2(-dy, dx): negate dy so positive = wrist above elbow
    angle_deg = math.degrees(math.atan2(-dy, dx))
    return round(angle_deg, 1), avg_vis


def calculate_head_alignment_2d(
    landmarks, idx_shoulder: int, idx_hip: int,
    idx_ear: int, idx_nose: int = 0,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> tuple[float, float]:
    """How well the head hides behind the back line (score 0-100, 2D).

    Back line = hip -> shoulder, extended forward.
    Measures perpendicular deviation of ear (or nose) from this line.
    Higher score = head better tucked behind back line.

    Reference: Steinmetz - "helmet mates nicely to his back".

    Returns:
        (score_0_100, avg_visibility) or (NaN, avg_vis) if below threshold.
    """
    vis_s = getattr(landmarks[idx_shoulder], "visibility", 1.0)
    vis_h = getattr(landmarks[idx_hip], "visibility", 1.0)
    vis_ear = getattr(landmarks[idx_ear], "visibility", 1.0)
    vis_nose = getattr(landmarks[idx_nose], "visibility", 1.0)

    # Need shoulder + hip + at least ear or nose
    if min(vis_s, vis_h) < min_visibility:
        return float("nan"), (vis_s + vis_h) / 2.0

    use_ear = vis_ear >= min_visibility
    use_nose = vis_nose >= min_visibility
    if not use_ear and not use_nose:
        return float("nan"), (vis_s + vis_h + vis_ear) / 3.0

    shoulder = np.array([landmarks[idx_shoulder].x, landmarks[idx_shoulder].y])
    hip = np.array([landmarks[idx_hip].x, landmarks[idx_hip].y])

    # Back line direction (hip -> shoulder)
    back_vec = shoulder - hip
    back_length = float(np.linalg.norm(back_vec))
    if back_length < 1e-6:
        return float("nan"), (vis_s + vis_h) / 2.0

    back_unit = back_vec / back_length

    # Head point: prefer ear, fallback to nose
    if use_ear:
        head_point = np.array([landmarks[idx_ear].x, landmarks[idx_ear].y])
        avg_vis = (vis_s + vis_h + vis_ear) / 3.0
    else:
        head_point = np.array([landmarks[idx_nose].x, landmarks[idx_nose].y])
        avg_vis = (vis_s + vis_h + vis_nose) / 3.0

    # Vector from shoulder to head
    shoulder_to_head = head_point - shoulder

    # Perpendicular component (deviation from back line)
    parallel_len = float(np.dot(shoulder_to_head, back_unit))
    projection = parallel_len * back_unit
    perpendicular = shoulder_to_head - projection

    # In image coords y-down: negative perpendicular[1] = head ABOVE line
    deviation = -perpendicular[1]

    # Normalize by torso length
    ratio = deviation / back_length

    # Convert to score: 0 deviation = 100, ratio 0.5 = score 0
    score = max(0.0, min(100.0, 100.0 - ratio * 200.0))
    return round(score, 1), avg_vis


def calculate_line_tilt_2d(
    landmarks, idx_left: int, idx_right: int,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> tuple[float, float]:
    """Angle of line between two landmarks relative to horizontal (2D).

    Uses normalized landmarks (image coords, Y-down).
    Positive = left point HIGHER than right (left hip dropped = negative).

    Args:
        landmarks: list of landmarks with x, y attributes (normalized_landmarks)
        idx_left: index of left-side landmark
        idx_right: index of right-side landmark

    Returns:
        (angle_degrees, avg_visibility) or (NaN, avg_vis) if below threshold.
    """
    vis_l = getattr(landmarks[idx_left], "visibility", 1.0)
    vis_r = getattr(landmarks[idx_right], "visibility", 1.0)
    avg_vis = (vis_l + vis_r) / 2.0

    if min(vis_l, vis_r) < min_visibility:
        return float("nan"), avg_vis

    lx, ly = landmarks[idx_left].x, landmarks[idx_left].y
    rx, ry = landmarks[idx_right].x, landmarks[idx_right].y

    dx = rx - lx
    dy = ry - ly  # image y increases downward

    # abs(dx): handles rear-view where right landmark may be to left of left landmark
    # atan2(dy, abs(dx)): positive when left.y < right.y (left point HIGHER in image)
    angle_deg = math.degrees(math.atan2(dy, abs(dx)))
    return round(angle_deg, 1), avg_vis


def calculate_tilt_3d(
    left_landmark, right_landmark,
    min_visibility: float = MIN_LANDMARK_VISIBILITY,
) -> tuple[float, float]:
    """Angle of line between two 3D world landmarks in frontal plane.

    Projects onto XY plane (ignoring Z depth), giving tilt independent
    of camera perspective. Positive = left point HIGHER.

    Args:
        left_landmark: world landmark with x, y, z, visibility
        right_landmark: world landmark with x, y, z, visibility

    Returns:
        (angle_degrees, avg_visibility) or (NaN, avg_vis) if below threshold.
    """
    vis_l = getattr(left_landmark, "visibility", 1.0)
    vis_r = getattr(right_landmark, "visibility", 1.0)
    avg_vis = (vis_l + vis_r) / 2.0

    if min(vis_l, vis_r) < min_visibility:
        return float("nan"), avg_vis

    dx = right_landmark.x - left_landmark.x
    dy = right_landmark.y - left_landmark.y  # Y down in MediaPipe world

    if math.isnan(dx) or math.isnan(dy) or abs(dx) < 1e-6:
        return float("nan"), avg_vis

    # abs(dx): handles rear-view where right landmark may be to left of left
    # -dy: invert because Y points down in MediaPipe world coordinates
    angle_deg = math.degrees(math.atan2(-dy, abs(dx)))
    return round(angle_deg, 1), avg_vis


def calculate_distance_3d(landmarks, idx_a: int, idx_b: int) -> float:
    """Calculate 3D Euclidean distance between two landmarks."""
    a = landmarks[idx_a]
    b = landmarks[idx_b]
    return math.sqrt(
        (a.x - b.x) ** 2
        + (a.y - b.y) ** 2
        + (a.z - b.z) ** 2
    )
