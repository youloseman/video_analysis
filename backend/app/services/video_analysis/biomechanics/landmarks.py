"""MediaPipe pose landmark indices and data structures."""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class PoseLandmark(IntEnum):
    """MediaPipe BlazePose 33 landmark indices."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Joint angle definitions: (point_a, vertex_b, point_c)
# Angle is measured at vertex B between rays BA and BC

COMMON_ANGLES = {
    "left_knee": (23, 25, 27),      # Hip -> Knee -> Ankle
    "right_knee": (24, 26, 28),
    "left_hip": (11, 23, 25),       # Shoulder -> Hip -> Knee
    "right_hip": (12, 24, 26),
}

RUNNING_ANGLES = {
    **COMMON_ANGLES,
    "left_ankle": (25, 27, 31),     # Knee -> Ankle -> FootIndex
    "right_ankle": (26, 28, 32),
    "left_elbow": (11, 13, 15),     # Shoulder -> Elbow -> Wrist
    "right_elbow": (12, 14, 16),
    # trunk_lean is computed directly via calculate_trunk_lean_midpoint(),
    # not as a 3-point angle (removed trunk_left/trunk_right triplets)
}

CYCLING_ANGLES = {
    **COMMON_ANGLES,
    "left_ankle": (25, 27, 31),
    "right_ankle": (26, 28, 32),
    "left_shoulder": (13, 11, 23),  # Elbow -> Shoulder -> Hip
    "right_shoulder": (14, 12, 24),
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
}

SWIMMING_ANGLES = {
    **COMMON_ANGLES,
    "left_shoulder": (23, 11, 13),  # Hip -> Shoulder -> Elbow
    "right_shoulder": (24, 12, 14),
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_ankle": (25, 27, 31),
    "right_ankle": (26, 28, 32),
}

SPORT_ANGLE_DEFINITIONS = {
    "run": RUNNING_ANGLES,
    "bike": CYCLING_ANGLES,
    "swim": SWIMMING_ANGLES,
}


@dataclass
class AngleResult:
    """Result of a single angle calculation."""

    name: str
    angle_degrees: float
    visibility: float
    landmark_indices: tuple[int, int, int]


@dataclass
class FrameAnalysis:
    """Analysis results for a single video frame."""

    timestamp_ms: float
    angles: dict[str, float] = field(default_factory=dict)
    visibility: dict[str, float] = field(default_factory=dict)
    extra_metrics: dict[str, Any] = field(default_factory=dict)
