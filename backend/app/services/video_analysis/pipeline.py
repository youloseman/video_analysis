"""Shared constants + overlay-drawing support extracted from Motus.

This is NOT the full Motus ``VideoAnalysisPipeline`` orchestrator. Milestone 1
did not port the DB / LLM / storage / async ``process_video`` path -- the
standalone driver in ``backend/scripts/analyze_local.py`` reproduces the
analysis path instead.

What lives here is the subset of ``pipeline.py`` that other copied modules
import (verbatim from Motus so behaviour is identical):

* ``SPORT_SAMPLE_RATES`` -- read by ``landmark_stabilizer`` (Butterworth fps).
* Overlay-drawing constants + helpers + a minimal ``VideoAnalysisPipeline``
  class exposing only ``_draw_angle_arc`` and ``_get_angle_display_config`` --
  imported by ``video_visualizer`` (Milestone 2). No DB/LLM/R2/thumbnails.
"""

import math
from typing import Any

# POSE_CONNECTIONS is the BlazePose-33 topology owned by the detector layer;
# re-exported here because Motus modules import it from ``pipeline``.
from app.services.video_analysis.detectors.base import POSE_CONNECTIONS  # noqa: F401

# --- frame sampling (read by landmark_stabilizer) --------------------------
FRAME_SAMPLE_RATE = 3
SPORT_SAMPLE_RATES = {
    "bike": 1,
    "run": 1,
    "swim": 1,
}

# --- overlay drawing (read by video_visualizer) ----------------------------

# Minimum landmark visibility to draw on the overlay (0.0 = invisible, 1.0 = fully visible).
# This is the *visualization* gate -- more permissive than the angle-trust gate in
# biomechanics.angle_calculator.MIN_LANDMARK_VISIBILITY, which decides whether an
# angle measurement is reliable enough to report.
MIN_OVERLAY_VISIBILITY = 0.5

# Angle arc triplets: (proximal_idx, vertex_idx, distal_idx) for each measured angle.
# The arc is drawn at the vertex landmark showing the angle between the two bones.
ARC_TRIPLETS: dict[str, dict[str, tuple[int, int, int]]] = {
    "bike": {
        "left_knee":      (23, 25, 27),
        "right_knee":     (24, 26, 28),
        "left_hip":       (11, 23, 25),
        "right_hip":      (12, 24, 26),
        "left_elbow":     (11, 13, 15),
        "right_elbow":    (12, 14, 16),
        "left_shoulder":  (13, 11, 23),   # Elbow -> Shoulder -> Hip
        "right_shoulder": (14, 12, 24),
        # forearm_tilt: no arc (angle to horizontal, not joint angle)
    },
    "run": {},  # Running uses unprefixed keys; arcs built dynamically by camera_side
    "swim": {
        "left_shoulder":  (23, 11, 13),
        "right_shoulder": (24, 12, 14),
        "left_elbow":     (11, 13, 15),
        "right_elbow":    (12, 14, 16),
        "left_knee":      (23, 25, 27),
    },
}


# Landmark indices per body side (for near/far skeleton coloring)
LEFT_SIDE_LANDMARKS = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
RIGHT_SIDE_LANDMARKS = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
# Midline landmarks (always drawn in near-side color)
MIDLINE_LANDMARKS = {0, 9, 10}


def _draw_dashed_line(
    cv2_mod, frame, pt1: tuple[int, int], pt2: tuple[int, int],
    color: tuple[int, int, int], thickness: int = 1,
    dash_len: int = 8, gap_len: int = 5,
) -> None:
    """Draw a dashed line between two points using OpenCV."""
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash_len, length)
        x1 = int(pt1[0] + ux * pos)
        y1 = int(pt1[1] + uy * pos)
        x2 = int(pt1[0] + ux * end_pos)
        y2 = int(pt1[1] + uy * end_pos)
        cv2_mod.line(frame, (x1, y1), (x2, y2), color, thickness, cv2_mod.LINE_AA)
        pos += dash_len + gap_len


def _is_near_side_landmark(idx: int, camera_side: str | None) -> bool:
    """Check if a landmark belongs to the near (camera-facing) side."""
    if idx in MIDLINE_LANDMARKS:
        return True
    if camera_side == "left":
        return idx in LEFT_SIDE_LANDMARKS
    elif camera_side == "right":
        return idx in RIGHT_SIDE_LANDMARKS
    return True  # No camera side info -> treat all as near


class VideoAnalysisPipeline:
    """Minimal carrier for the two overlay-drawing routines Motus kept on the
    pipeline class. Instantiable with no args; holds no state.

    The full Motus orchestrator (validation, detection, scoring, DB, LLM,
    thumbnails, R2) is intentionally NOT ported in this milestone.
    """

    @staticmethod
    def _draw_angle_arc(
        cv2_mod,
        frame,
        pixel_coords: list[tuple[int, int, float]],
        prox_idx: int,
        vert_idx: int,
        dist_idx: int,
        color: tuple[int, int, int],
        body_height_px: float,
    ) -> None:
        """Draw a small arc at the vertex joint showing the measured angle.

        Args:
            cv2_mod: cv2 module reference
            frame: image to draw on
            pixel_coords: list of (x, y, visibility) per landmark
            prox_idx, vert_idx, dist_idx: landmark indices (proximal, vertex, distal)
            color: BGR color tuple
            body_height_px: shoulder-to-hip pixel distance for scaling
        """
        max_idx = max(prox_idx, vert_idx, dist_idx)
        if max_idx >= len(pixel_coords):
            return

        px, py, pv = pixel_coords[prox_idx]
        vx, vy, vv = pixel_coords[vert_idx]
        dx, dy, dv = pixel_coords[dist_idx]

        if min(pv, vv, dv) < MIN_OVERLAY_VISIBILITY:
            return

        # Vectors from vertex to proximal and distal
        v1x, v1y = px - vx, py - vy
        v2x, v2y = dx - vx, dy - vy

        # Angles in degrees (OpenCV uses clockwise from 3-o'clock)
        ang1 = math.degrees(math.atan2(v1y, v1x))
        ang2 = math.degrees(math.atan2(v2y, v2x))

        # Ensure we draw the smaller arc between the two bones
        start = min(ang1, ang2)
        end = max(ang1, ang2)
        if end - start > 180:
            start, end = end, start + 360

        # Arc radius scales with body size
        radius = max(12, int(body_height_px * 0.08))

        cv2_mod.ellipse(
            frame,
            (vx, vy),
            (radius, radius),
            0,
            start,
            end,
            color,
            1,
            cv2_mod.LINE_AA,
        )

    def _get_angle_display_config(
        self, sport_type: str, summary: dict[str, Any] | None = None,
        frame_landmarks: Any = None,
        cycling_position: str | None = None,
    ) -> list[dict[str, Any]]:
        """Per-sport label configs with directional offsets to prevent overlap.

        Each entry:
          key:        angle_statistics key
          idx:        landmark index for joint position
          optimal:    (min, max) range for color coding
          name:       short display name
          offset_dir: direction to push label away from body
        """
        if sport_type == "run":
            # Running uses unprefixed keys (knee, hip, elbow, trunk)
            camera_side = (summary or {}).get("camera_side", "left")

            # Near-side landmark indices for label positioning
            if camera_side == "left":
                knee_idx, hip_idx, elbow_idx = 25, 23, 13
            else:
                knee_idx, hip_idx, elbow_idx = 26, 24, 14

            trunk_idx = 11 if camera_side == "left" else 12

            # Detect Stance/Swing from ankle Y (higher Y = lower in frame = stance)
            knee_label = "Knee"
            hip_label = "Hip"
            if frame_landmarks is not None:
                left_ankle_y = frame_landmarks[27].y
                right_ankle_y = frame_landmarks[28].y
                stance_side = "left" if left_ankle_y > right_ankle_y else "right"
                if camera_side == stance_side:
                    knee_label = "Stance"
                    hip_label = "Stance Hip"
                else:
                    knee_label = "Swing"
                    hip_label = "Swing Hip"

            return [
                {"key": "knee",   "idx": knee_idx,  "optimal": (80, 175),  "name": knee_label,  "offset_dir": "left"},
                {"key": "hip",    "idx": hip_idx,   "optimal": (150, 180), "name": hip_label,   "offset_dir": "up-left"},
                {"key": "trunk",  "idx": trunk_idx, "optimal": (0, 12),    "name": "Trunk",     "offset_dir": "up"},
                {"key": "elbow",  "idx": elbow_idx, "optimal": (85, 100),  "name": "Elbow",     "offset_dir": "up-left"},
            ]
        elif sport_type == "bike":
            from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference
            cycling_pos = cycling_position or (summary or {}).get("cycling_position")
            ref = get_cycling_reference(cycling_pos)
            camera_side = (summary or {}).get("camera_side", "left")

            if camera_side == "left":
                near_knee_key, near_knee_idx = "left_knee", 25
                near_hip_key, near_hip_idx = "left_hip", 23
                near_elbow_key, near_elbow_idx = "left_elbow", 13
                near_shoulder_key, near_shoulder_idx = "left_shoulder", 11
                near_forearm_key, near_wrist_idx = "left_forearm_tilt", 15
            else:
                near_knee_key, near_knee_idx = "right_knee", 26
                near_hip_key, near_hip_idx = "right_hip", 24
                near_elbow_key, near_elbow_idx = "right_elbow", 14
                near_shoulder_key, near_shoulder_idx = "right_shoulder", 12
                near_forearm_key, near_wrist_idx = "right_forearm_tilt", 16

            return [
                {"key": near_knee_key,     "idx": near_knee_idx,     "optimal": ref["knee_at_bdc"],   "name": "Knee",    "offset_dir": "left"},
                {"key": near_hip_key,      "idx": near_hip_idx,      "optimal": (30, 80),             "name": "Hip",     "offset_dir": "up-left"},
                {"key": "trunk_angle",     "idx": 11,                "optimal": ref["trunk_angle"],    "name": "Trunk",   "offset_dir": "up"},
                {"key": near_elbow_key,    "idx": near_elbow_idx,    "optimal": ref["elbow_angle"],    "name": "Elbow",   "offset_dir": "up-left"},
                {"key": near_shoulder_key, "idx": near_shoulder_idx, "optimal": ref["shoulder_angle"], "name": "Shldr",   "offset_dir": "right"},
                {"key": near_forearm_key,  "idx": near_wrist_idx,    "optimal": ref["forearm_tilt"],   "name": "Forearm", "offset_dir": "down-right"},
            ]
        elif sport_type == "swim":
            return [
                {"key": "left_shoulder",  "idx": 11, "optimal": (160, 180), "name": "L.Shldr", "offset_dir": "up"},
                {"key": "right_shoulder", "idx": 12, "optimal": (160, 180), "name": "R.Shldr", "offset_dir": "up-right"},
                {"key": "left_elbow",     "idx": 13, "optimal": (90, 120),  "name": "L.Elbow", "offset_dir": "left"},
                {"key": "right_elbow",    "idx": 14, "optimal": (90, 120),  "name": "R.Elbow", "offset_dir": "right"},
                {"key": "left_knee",      "idx": 25, "optimal": (150, 180), "name": "L.Knee",  "offset_dir": "down-left"},
            ]
        return []


__all__ = [
    "FRAME_SAMPLE_RATE",
    "SPORT_SAMPLE_RATES",
    "MIN_OVERLAY_VISIBILITY",
    "ARC_TRIPLETS",
    "LEFT_SIDE_LANDMARKS",
    "RIGHT_SIDE_LANDMARKS",
    "MIDLINE_LANDMARKS",
    "POSE_CONNECTIONS",
    "VideoAnalysisPipeline",
    "_draw_dashed_line",
    "_is_near_side_landmark",
]
