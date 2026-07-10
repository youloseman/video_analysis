"""Pose-detector subsystem.

Public API:
    PoseDetector              -- abstract base class
    DetectorConfig            -- sport + camera config passed to __init__
    DetectorLandmark          -- per-landmark (x, y, z, visibility) record
    DetectorFrame             -- one frame's detection result
    MediaPipePoseDetector     -- the default implementation
    build_detector            -- factory used by the pipeline
    POSE_CONNECTIONS          -- BlazePose-33 bone pairs (for visualisation)
"""

from app.services.video_analysis.detectors.base import (
    BLAZEPOSE_LANDMARK_COUNT,
    DetectorConfig,
    DetectorFrame,
    DetectorLandmark,
    POSE_CONNECTIONS,
    PoseDetector,
)
from app.services.video_analysis.detectors.mediapipe_detector import (
    MediaPipePoseDetector,
)


def build_detector(
    sport_type: str,
    camera_angle: str | None = None,
) -> PoseDetector:
    """Factory that returns the detector appropriate for the given sport.

    Currently always returns :class:`MediaPipePoseDetector`; future
    research work (YOLO-Pose under-water, MMPose elite swim, etc.)
    can branch here without touching the pipeline call-sites.
    """
    from app.services.video_analysis.detectors.mediapipe_detector import (
        DEFAULT_CONFIDENCE_TABLE,
        mediapipe_confidence,
    )

    confidence = mediapipe_confidence(
        DEFAULT_CONFIDENCE_TABLE, sport_type, camera_angle
    )
    config = DetectorConfig(
        sport=sport_type,
        camera_angle=camera_angle,
        min_detection_confidence=confidence,
        min_presence_confidence=confidence,
        min_tracking_confidence=confidence,
    )
    return MediaPipePoseDetector(config)


__all__ = [
    "BLAZEPOSE_LANDMARK_COUNT",
    "POSE_CONNECTIONS",
    "DetectorConfig",
    "DetectorFrame",
    "DetectorLandmark",
    "MediaPipePoseDetector",
    "PoseDetector",
    "build_detector",
]
