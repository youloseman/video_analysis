"""Base sport analyzer abstract class."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from app.services.video_analysis.biomechanics.landmarks import FrameAnalysis

# Midline angle names that are always valid regardless of camera side
MIDLINE_ANGLES = {"trunk_lean", "trunk_angle"}


class SportAnalyzer(ABC):
    """Abstract base class for sport-specific analyzers.

    Subclasses implement analyze_frame(), compute_summary(), detect_issues().
    """

    def __init__(self, sport_type: str, fps: float = 30.0):
        self.sport_type = sport_type
        self.fps = fps
        self.frame_results: list[FrameAnalysis] = []
        self.angle_history: dict[str, list[float]] = {}
        self.angle_timestamps: list[float] = []
        # Camera side detection (unilateral focus)
        self.camera_side: str | None = None
        self.camera_side_votes: list[str] = []

    def add_frame_result(self, result: FrameAnalysis) -> None:
        """Store a frame analysis result and update angle history."""
        self.frame_results.append(result)
        self.angle_timestamps.append(result.timestamp_ms / 1000.0)
        for angle_name, angle_value in result.angles.items():
            if angle_name not in self.angle_history:
                self.angle_history[angle_name] = []
            self.angle_history[angle_name].append(angle_value)

    def get_effective_fps(self) -> float:
        """Compute actual effective FPS from timestamp data."""
        if len(self.angle_timestamps) < 2:
            return self.fps
        total_duration = self.angle_timestamps[-1] - self.angle_timestamps[0]
        if total_duration <= 0:
            return self.fps
        return (len(self.angle_timestamps) - 1) / total_duration

    @abstractmethod
    def analyze_frame(
        self, world_landmarks: Any, normalized_landmarks: Any, timestamp_ms: float
    ) -> FrameAnalysis:
        """Analyze a single frame. Must be implemented by subclass."""
        ...

    @abstractmethod
    def compute_summary(self) -> dict[str, Any]:
        """Compute sport-specific summary metrics from all frames."""
        ...

    @abstractmethod
    def detect_issues(self) -> list[dict[str, Any]]:
        """Detect technique issues from aggregated frame data."""
        ...

    # --- Camera Side Detection (Unilateral Focus) ---

    def detect_camera_side(self, world_landmarks: Any) -> str:
        """Detect which side of the body faces the camera using Z-depth.

        MediaPipe world landmarks: Z increases AWAY from camera.
        The side with SMALLER Z values is closer to the camera.
        Uses shoulders + hips for robust detection (large landmarks, reliable Z).
        """
        left_z = (world_landmarks[11].z + world_landmarks[23].z) / 2
        right_z = (world_landmarks[12].z + world_landmarks[24].z) / 2
        return "left" if left_z < right_z else "right"

    def finalize_camera_side(self) -> None:
        """Determine camera side by majority vote across all frames."""
        if not self.camera_side_votes:
            self.camera_side = "left"
            return
        left_votes = self.camera_side_votes.count("left")
        right_votes = self.camera_side_votes.count("right")
        self.camera_side = "left" if left_votes >= right_votes else "right"

    def get_near_side_prefix(self) -> str:
        """Returns 'left' or 'right' -- the side facing camera."""
        return self.camera_side or "left"

    def get_far_side_prefix(self) -> str:
        """Returns the side AWAY from camera."""
        return "right" if self.get_near_side_prefix() == "left" else "left"

    def is_near_side_angle(self, angle_name: str) -> bool:
        """Check if an angle belongs to the near (camera-facing) side."""
        if angle_name in MIDLINE_ANGLES:
            return True
        return angle_name.startswith(self.get_near_side_prefix())

    # --- Statistics ---

    def compute_angle_statistics(self) -> dict[str, dict[str, float | None]]:
        """Compute min/max/mean/std/range statistics for all tracked angles.

        NaN-safe: filters out NaN values (from visibility threshold) and
        reports valid/nan frame counts. When a landmark is fully gated
        (zero valid frames), numeric stats are None -- the frontend then
        renders "--" instead of falsely reading 0.0 as a valid measurement
        and flagging a phantom left/right asymmetry.
        """
        stats: dict[str, dict[str, float | None]] = {}
        for name, values in self.angle_history.items():
            if not values:
                continue
            arr = np.array(values, dtype=float)
            valid = arr[~np.isnan(arr)]
            nan_count = int(np.isnan(arr).sum())

            if len(valid) == 0:
                stats[name] = {
                    "min": None, "max": None, "mean": None,
                    "std": None, "range": None,
                    "valid_frames": 0, "nan_frames": nan_count,
                    "nan_pct": 100.0,
                }
                continue

            stats[name] = {
                "min": round(float(np.min(valid)), 1),
                "max": round(float(np.max(valid)), 1),
                "mean": round(float(np.mean(valid)), 1),
                "std": round(float(np.std(valid)), 1),
                "range": round(float(np.max(valid) - np.min(valid)), 1),
                "valid_frames": int(len(valid)),
                "nan_frames": nan_count,
                "nan_pct": round(nan_count / len(arr) * 100, 1),
            }
        return stats
