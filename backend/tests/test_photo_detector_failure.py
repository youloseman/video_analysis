"""A photo the detector never looked at is our failure, not "no body".

Any exception in the pose detector (or a model file missing from the image)
fell through to "No body pose detected": a 422 telling the athlete to take a
better picture, logged as a rejection, so a broken detector looked exactly
like a run of bad photos. Now it is raised as a crash (500, report_crash),
and only a detector that looked and found nothing says "no body".
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from mediapipe.tasks.python import vision

from app.core.config import settings
from app.services.video_analysis import photo_analyzer as pa


@pytest.fixture
def model_file(tmp_path, monkeypatch):
    """The detector is stubbed; it only has to find a model file to open."""
    path = tmp_path / "pose_landmarker_heavy.task"
    path.write_bytes(b"stub")
    _use_model(monkeypatch, path)
    return path


def _use_model(monkeypatch, path):
    monkeypatch.setattr(pa, "settings", dataclasses.replace(
        settings, models_dir=path.parent, model_filename=path.name))


def _jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((800, 600, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_a_crashing_detector_is_a_crash(monkeypatch, model_file):
    def boom(_options):
        raise RuntimeError("landmarker exploded")

    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options", staticmethod(boom))
    with pytest.raises(RuntimeError) as err:
        pa.detect_pose_in_image(_jpeg())
    assert not isinstance(err.value, ValueError)
    assert "landmarker exploded" in str(err.value.__cause__)


def test_a_detector_that_looked_and_found_nothing_still_says_no_body(monkeypatch, model_file):
    class Empty:
        def detect(self, _img):
            return SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[],
                                   segmentation_masks=None)

        def close(self):
            pass

    monkeypatch.setattr(vision.PoseLandmarker, "create_from_options",
                        staticmethod(lambda _o: Empty()))
    with pytest.raises(ValueError, match="No body pose detected"):
        pa.detect_pose_in_image(_jpeg())


def test_a_missing_model_is_a_crash_not_a_bad_photo(monkeypatch, tmp_path):
    _use_model(monkeypatch, tmp_path / "absent.task")
    monkeypatch.chdir(tmp_path)
    if any(p.exists() for p in (
        Path(pa.__file__).parent / "models" / "pose_landmarker_heavy.task",
        Path("/app/models/pose_landmarker_heavy.task"),
    )):
        pytest.skip("a model file is installed on this machine")
    with pytest.raises(RuntimeError) as err:
        pa.detect_pose_in_image(_jpeg())
    assert isinstance(err.value.__cause__, FileNotFoundError)
