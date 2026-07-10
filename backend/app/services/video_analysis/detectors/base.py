"""Abstract pose-detector interface.

All detectors produce frame-by-frame pose data in the BlazePose-33
landmark format: 33 landmarks per frame, each with normalised
``(x, y, z)`` plus visibility in ``[0, 1]``.

Coordinate conventions (all detectors MUST honour):

- Normalised ``x``, ``y``: ``[0, 1]``, origin top-left, x right, y down.
- Normalised ``z``: relative depth (negative toward camera, positive
  away). Scale is not standardised -- downstream code uses z only as a
  qualitative signal (for example flip-fix side detection).
- World landmarks: metres, origin at the pose centre (midpoint between
  hips), scale approximate.
- Visibility: ``[0, 1]``, interpretation as MediaPipe defines it.

Landmark indexing MUST match BlazePose:

    0  = nose
    11 = left_shoulder,  12 = right_shoulder
    13 = left_elbow,     14 = right_elbow
    15 = left_wrist,     16 = right_wrist
    23 = left_hip,       24 = right_hip
    25 = left_knee,      26 = right_knee
    27 = left_ankle,     28 = right_ankle
    29 = left_heel,      30 = right_heel
    31 = left_foot_idx,  32 = right_foot_idx

Detectors whose native format is different (COCO-17, OpenPose-25, ...)
MUST translate to BlazePose-33 before returning. Missing landmarks get
``visibility=0`` and ``(nan, nan, nan)`` coordinates -- never zero,
which would silently pollute averages downstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# BlazePose-33 skeleton topology. Shared by every detector (even those whose
# native model has a different topology -- they're responsible for mapping
# into this layout before returning DetectorFrames). Used by visualisation
# code to draw bones.
# ---------------------------------------------------------------------------
# fmt: off
POSE_CONNECTIONS: list[tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Upper body
    (11, 12),                           # shoulders
    (11, 13), (13, 15),                 # left arm
    (12, 14), (14, 16),                 # right arm
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Lower body
    (23, 25), (25, 27),                 # left leg
    (24, 26), (26, 28),                 # right leg
    # Hands
    (15, 17), (16, 18), (15, 19), (16, 20),
    (15, 21), (16, 22), (17, 19), (18, 20),
    (19, 21), (20, 22),
    # Feet
    (27, 29), (28, 30), (27, 31), (28, 32),
    (29, 31), (30, 32),
]
# fmt: on

BLAZEPOSE_LANDMARK_COUNT = 33


@dataclass
class DetectorLandmark:
    """One landmark in the BlazePose-33 format.

    Missing landmarks (e.g. when the native detector has no equivalent
    point) have ``visibility=0`` and NaN coordinates.
    """

    x: float
    y: float
    z: float
    visibility: float


@dataclass
class DetectorFrame:
    """One frame's detection result.

    ``normalized_landmarks`` and ``world_landmarks`` each carry exactly
    :data:`BLAZEPOSE_LANDMARK_COUNT` entries. The pipeline consumes
    these directly -- no further conversion happens before the
    stabilizer.
    """

    normalized_landmarks: list[DetectorLandmark]
    world_landmarks: list[DetectorLandmark]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    """Configuration passed to :meth:`PoseDetector.__init__`.

    The three confidence values are plumbed from
    :func:`detectors.mediapipe_detector.mediapipe_confidence` (or an
    equivalent for other detector families). All three default to the
    same value; detectors that distinguish detection/presence/tracking
    can tune them independently.
    """

    sport: str
    camera_angle: str | None
    min_detection_confidence: float
    min_presence_confidence: float
    min_tracking_confidence: float


class PoseDetector(ABC):
    """Abstract interface for a pose-detector plugin.

    Lifetime rules: instantiate one detector per analysis, call
    :meth:`detect` once per frame, then :meth:`close`. Detectors MUST
    be stateless across videos -- no process-wide singletons. A failing
    ``close()`` should never mask detection errors.
    """

    @abstractmethod
    def __init__(self, config: DetectorConfig) -> None:
        """Initialise the detector with sport/camera-specific config."""

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> DetectorFrame | None:
        """Run detection on a single RGB frame.

        Args:
            image_rgb: H x W x 3 uint8 array in RGB order.
            timestamp_ms: Optional monotonic timestamp in milliseconds.
                Detectors that require monotonic timestamps (e.g.
                MediaPipe Tasks API in VIDEO mode) will use it when
                provided and fall back to an internal counter otherwise.

        Returns:
            A :class:`DetectorFrame` on success, ``None`` when no pose
            was detected. When returning a DetectorFrame, every landmark
            slot must be filled -- missing landmarks use
            ``visibility=0`` and NaN coordinates, never zeros.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any native resources (GPU buffers, file handles).

        Idempotent. Exceptions MUST be caught inside the detector --
        failing to release resources should not break the pipeline.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in diagnostics, e.g. ``"mediapipe_tasks"``."""

    @property
    @abstractmethod
    def config(self) -> DetectorConfig:
        """The immutable :class:`DetectorConfig` the detector was built with.

        Exposed so the pipeline can surface confidence thresholds etc.
        in diagnostics without reaching into private attributes.
        """


__all__ = [
    "BLAZEPOSE_LANDMARK_COUNT",
    "DetectorConfig",
    "DetectorFrame",
    "DetectorLandmark",
    "POSE_CONNECTIONS",
    "PoseDetector",
]
