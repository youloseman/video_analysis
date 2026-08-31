"""Which pose model the video path actually loads.

Two places in this app answer questions about "the model": every guard
(``/health``'s ``model_present``, the 503 on each analyze endpoint) tests
``settings.model_path``, and the detector decides what to load from its own
search list. Those were wired separately and agreed only by coincidence.

The coincidence had a cost. ``model_filename`` was dead config for the video
path, so a model-comparison experiment on 2026-08-31 swapped it, measured the
SAME model three times, and reported "heavy, full and lite are identical" --
a result that would have been believed if 29 MB and 5.5 MB models producing
byte-identical analyses were not obviously impossible.
"""

from __future__ import annotations

from app.core.config import settings
from app.services.video_analysis.detectors.mediapipe_detector import (
    MediaPipePoseDetector,
)


def test_the_detector_looks_where_the_guards_check_first():
    """A guard that says "model present" about one file while the loader opens
    another is a health check that cannot fail for the right reason."""
    assert MediaPipePoseDetector._MODEL_SEARCH_PATHS[0] == settings.model_path


def test_the_configured_model_is_not_dead_config():
    """`settings.model_filename` has to be able to change what gets loaded, or
    it is documentation that lies."""
    paths = MediaPipePoseDetector._MODEL_SEARCH_PATHS
    assert settings.model_filename in str(paths[0])


def test_the_hardcoded_fallbacks_survive():
    """The original Motus locations stay as fallbacks: a deploy that ships the
    model somewhere unexpected should still start."""
    paths = MediaPipePoseDetector._MODEL_SEARCH_PATHS
    assert len(paths) >= 5
    assert any("pose_landmarker_heavy" in str(p) for p in paths[1:])


def test_the_default_resolves_to_the_heavy_model():
    """Measured 2026-08-31: full and lite are 1.3-1.6x faster on a whole
    analysis and change CONCLUSIONS -- on the aero bike clip the grade went
    B to A, `saddle_too_high` disappeared and the "lower the saddle"
    recommendation vanished, because the lighter models put the ankle up to
    35 degrees elsewhere. Speed is not worth a fit report that says a saddle
    is fine when it is not."""
    assert settings.model_filename == "pose_landmarker_heavy.task"
