"""MediaPipe BlazePose implementation of :class:`PoseDetector`.

Tries the Tasks API (``PoseLandmarker``) first and falls back to the
Legacy Solutions API (``mp.solutions.pose.Pose``) if Tasks-API
initialisation fails (model file missing, runtime incompatibility,
etc.). Both back-ends return :class:`DetectorFrame` values so the
pipeline cannot tell which one is active -- only diagnostics report
it via :attr:`name`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis.detectors.base import (
    BLAZEPOSE_LANDMARK_COUNT,
    DetectorConfig,
    DetectorFrame,
    DetectorLandmark,
    PoseDetector,
)

logger = structlog.get_logger()


# Per-sport MediaPipe confidence thresholds, keyed by
# ``(sport_type, camera_angle)``. ``camera_angle`` is ``None`` for non-
# swim sports. Swim above-water is strict -- glare and splash produce
# low-confidence hallucinations; swim under-water is lenient -- water
# distortion depresses confidence even when landmarks are reliable.
DEFAULT_CONFIDENCE_TABLE: dict[tuple[str, str | None], float] = {
    ("run",  None):          0.30,
    ("bike", None):          0.50,
    ("swim", "above_water"): 0.55,
    ("swim", "under_water"): 0.30,
    ("swim", None):          0.40,
}


def mediapipe_confidence(
    table: dict[tuple[str, str | None], float],
    sport_type: str,
    camera_angle: str | None,
) -> float:
    """Look up the right confidence for ``(sport, camera_angle)``.

    Falls back to the sport-only entry when the ``camera_angle``-specific
    one is absent, then to a global default of 0.4.
    """
    if (sport_type, camera_angle) in table:
        return table[(sport_type, camera_angle)]
    if (sport_type, None) in table:
        return table[(sport_type, None)]
    return 0.4


# ---------------------------------------------------------------------------
# Helpers: convert MediaPipe's native landmark types into DetectorFrame.
# Tasks API returns plain Python objects with .x/.y/.z/.visibility. Legacy
# Solutions returns protobuf messages with the same fields. Both paths hit
# the same converter.
# ---------------------------------------------------------------------------
def _landmark_to_detector(landmark: Any) -> DetectorLandmark:
    return DetectorLandmark(
        x=float(getattr(landmark, "x", math.nan)),
        y=float(getattr(landmark, "y", math.nan)),
        z=float(getattr(landmark, "z", math.nan)),
        visibility=float(getattr(landmark, "visibility", 0.0)),
    )


def _pad_landmarks(
    landmarks: list[Any] | None,
) -> list[DetectorLandmark]:
    """Return exactly :data:`BLAZEPOSE_LANDMARK_COUNT` entries.

    Missing native slots become NaN-coordinate, zero-visibility entries
    so downstream code can treat the list as always-full.
    """
    if landmarks is None:
        return [
            DetectorLandmark(math.nan, math.nan, math.nan, 0.0)
            for _ in range(BLAZEPOSE_LANDMARK_COUNT)
        ]
    out = [_landmark_to_detector(lm) for lm in landmarks[:BLAZEPOSE_LANDMARK_COUNT]]
    while len(out) < BLAZEPOSE_LANDMARK_COUNT:
        out.append(DetectorLandmark(math.nan, math.nan, math.nan, 0.0))
    return out


class MediaPipePoseDetector(PoseDetector):
    """BlazePose wrapped behind the :class:`PoseDetector` interface.

    Resource lifecycle:
        - ``__init__`` tries Tasks API; on any exception (missing model
          file, runtime incompatibility, ...) falls back to Legacy
          Solutions. The final state is recorded in :attr:`_backend`
          and surfaced via :attr:`name`.
        - :meth:`detect` dispatches to whichever back-end succeeded.
        - :meth:`close` releases the back-end-specific resource and is
          idempotent.
    """

    _MODEL_SEARCH_PATHS = (
        # backend/models/ -- primary location for this standalone project
        # (parents[4] == backend/ from detectors/mediapipe_detector.py).
        Path(__file__).resolve().parents[4] / "models" / "pose_landmarker_heavy.task",
        # Original Motus search paths kept as fallbacks.
        Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_heavy.task",
        Path(__file__).resolve().parent.parent / "biomechanics" / "pose_landmarker_heavy.task",
        Path("/app/models/pose_landmarker_heavy.task"),
        Path("pose_landmarker_heavy.task"),
    )

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._backend: str = "uninitialised"
        self._landmarker = None
        self._pose = None
        self._monotonic_ts: int = 0
        self._closed: bool = False

        try:
            import mediapipe as mp  # noqa: F401 -- import-for-side-effect check
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install with: pip install mediapipe"
            ) from exc

        # Try Tasks API first.
        if self._init_tasks_api():
            self._backend = "mediapipe_tasks"
            return

        # Fall back to Legacy Solutions.
        logger.info("Falling back to Legacy Solutions API")
        if self._init_legacy_api():
            self._backend = "mediapipe_legacy"
            return

        # Neither path worked -- fail with a diagnosable RuntimeError
        # rather than letting a raw AttributeError leak from inside.
        # Typical cause in CI: mediapipe importable but legacy API
        # removed (>= 0.11) AND no model file on disk for Tasks API.
        raise RuntimeError(
            "MediaPipe could not be initialised: Tasks API model file "
            "not found AND Legacy Solutions API unavailable in this "
            "mediapipe build. Provide a pose_landmarker_heavy.task "
            "file or install a mediapipe version that ships "
            "mp.solutions.pose."
        )

    # ------------------------------------------------------------------
    # Back-end initialisers
    # ------------------------------------------------------------------
    def _init_tasks_api(self) -> bool:
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker,
                PoseLandmarkerOptions,
                RunningMode,
            )

            model_path: str | None = None
            for candidate in self._MODEL_SEARCH_PATHS:
                if candidate.exists():
                    model_path = str(candidate)
                    break
            if not model_path:
                logger.warning(
                    "PoseLandmarker model file not found, falling back to Legacy API"
                )
                return False

            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self._config.min_detection_confidence,
                min_pose_presence_confidence=self._config.min_presence_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)
            return True
        except Exception as exc:
            logger.warning("Tasks API init failed", err=str(exc))
            return False

    def _init_legacy_api(self) -> bool:
        """Try the Legacy Solutions API.

        Returns True on success, False if the installed mediapipe build
        doesn't expose ``mp.solutions`` (dropped in recent releases)
        or the Pose constructor raises.
        """
        try:
            import mediapipe as mp

            mp_solutions = getattr(mp, "solutions", None)
            if mp_solutions is None:
                logger.warning(
                    "mediapipe has no `solutions` attribute -- "
                    "Legacy API was removed in this build"
                )
                return False
            mp_pose = mp_solutions.pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            return True
        except Exception as exc:
            logger.warning("Legacy API init failed", err=str(exc))
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        image_rgb: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> DetectorFrame | None:
        if self._closed:
            raise RuntimeError("detect() called after close()")

        if timestamp_ms is None:
            self._monotonic_ts += 1
            timestamp_ms = self._monotonic_ts
        else:
            self._monotonic_ts = max(self._monotonic_ts, int(timestamp_ms))

        if self._backend == "mediapipe_tasks":
            return self._detect_tasks(image_rgb, int(timestamp_ms))
        if self._backend == "mediapipe_legacy":
            return self._detect_legacy(image_rgb)
        raise RuntimeError(f"MediaPipePoseDetector not initialised ({self._backend})")

    def _detect_tasks(
        self, image_rgb: np.ndarray, timestamp_ms: int
    ) -> DetectorFrame | None:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)  # type: ignore[union-attr]

        world = (
            result.pose_world_landmarks[0]
            if result.pose_world_landmarks else None
        )
        normalised = (
            result.pose_landmarks[0]
            if result.pose_landmarks else None
        )
        if world is None or normalised is None:
            return None
        return DetectorFrame(
            normalized_landmarks=_pad_landmarks(list(normalised)),
            world_landmarks=_pad_landmarks(list(world)),
        )

    def _detect_legacy(self, image_rgb: np.ndarray) -> DetectorFrame | None:
        result = self._pose.process(image_rgb)  # type: ignore[union-attr]
        world = (
            result.pose_world_landmarks.landmark
            if result.pose_world_landmarks else None
        )
        normalised = (
            result.pose_landmarks.landmark
            if result.pose_landmarks else None
        )
        if world is None or normalised is None:
            return None
        return DetectorFrame(
            normalized_landmarks=_pad_landmarks(list(normalised)),
            world_landmarks=_pad_landmarks(list(world)),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception as exc:
            logger.warning("MediaPipe Tasks landmarker close failed", err=str(exc))
        try:
            if self._pose is not None:
                self._pose.close()
        except Exception as exc:
            logger.warning("MediaPipe Legacy pose close failed", err=str(exc))

    @property
    def name(self) -> str:
        return self._backend

    @property
    def config(self) -> DetectorConfig:
        return self._config


__all__ = [
    "DEFAULT_CONFIDENCE_TABLE",
    "MediaPipePoseDetector",
    "mediapipe_confidence",
]
