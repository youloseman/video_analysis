"""Minimal settings for the standalone video-analysis MVP (Milestone 1).

IMPORTANT: the copied biomechanics/detector core reads **no** ``settings.*``
fields at all -- a repo-wide search for ``settings.`` across
``app/services/video_analysis`` returns nothing. So this object is
deliberately tiny; it exists only to give the driver (and future
milestones) a single place to grow configuration.

No ``pydantic`` / ``pydantic-settings`` dependency on purpose: Milestone 1
requirements are limited to packages the core actually imports
(mediapipe, opencv, numpy, scipy, structlog). R2 / LLM keys are out of
scope for this milestone.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# backend/ directory (this file is backend/app/core/config.py -> parents[2]).
BACKEND_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Overridable via environment variables."""

    # Pose model. The detector searches several locations
    # (see detectors/mediapipe_detector.py::_MODEL_SEARCH_PATHS); this
    # is the primary one the driver checks for a friendly error message.
    models_dir: Path = BACKEND_DIR / "models"
    model_filename: str = "pose_landmarker_heavy.task"

    # Cap on frames analyzed for long clips (mirrors the Motus default
    # baked into the frame extractor). Kept here so it is tunable later.
    max_analysis_frames: int = 450

    # Where the API stores uploaded clips + generated overlays (M3).
    uploads_dir: Path = BACKEND_DIR / "uploads"

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename


def _load_settings() -> Settings:
    models_dir = os.environ.get("VA_MODELS_DIR")
    uploads_dir = os.environ.get("VA_UPLOADS_DIR")
    return Settings(
        models_dir=Path(models_dir).resolve() if models_dir else Settings.models_dir,
        uploads_dir=Path(uploads_dir).resolve() if uploads_dir else Settings.uploads_dir,
    )


settings = _load_settings()
