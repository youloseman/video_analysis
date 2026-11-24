"""Timing runs on the rate the frames decode at, not the header's (2026-11-24).

A clip trimmed in the iPhone's Photos app keeps an edit list, and its header
then misstates the rate: IMG_4091 says 707 frames at 27.93 fps and decodes as
655 at 29.98 (IMG_4088 57.5 for 60, IMG_4004 58.2 for 60). Cadence, contact
and flight were that much low. Untrimmed clips read 0.0% apart and are left
exactly as they were.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.services.video_analysis import runner


def test_a_true_header_is_left_alone():
    assert runner.decoded_fps(645, 0.0, 10735.0, 59.991) is None
    assert runner.decoded_fps(151, 0.0, 5000.0, 30.0) is None


def test_an_edit_list_header_is_replaced():
    real = runner.decoded_fps(655, 0.0, 21815.0, 27.934)
    assert real == pytest.approx(29.98, abs=0.01)
    assert runner.decoded_fps(570, 0.0, 9483.0, 57.475) == pytest.approx(60.0, abs=0.02)


def test_an_unreadable_clock_changes_nothing():
    assert runner.decoded_fps(655, None, 21815.0, 27.9) is None
    assert runner.decoded_fps(655, 0.0, 0.0, 27.9) is None
    assert runner.decoded_fps(5, 0.0, 200.0, 27.9) is None
    assert runner.decoded_fps(655, 0.0, 1800.0, 27.9) is None      # 363 fps: not a clock to trust


class _EditListCapture:
    """IMG_4091 as OpenCV sees it: the header says 707 at 27.93, 655 decode at 29.98."""

    HEADER_FPS, HEADER_FRAMES, FRAMES, REAL_FPS = 27.934, 707, 655, 29.97

    def __init__(self, *_a):
        self.i = -1

    def isOpened(self):  # noqa: N802 -- cv2's name
        return True

    def get(self, prop):
        import cv2

        return {cv2.CAP_PROP_FPS: self.HEADER_FPS, cv2.CAP_PROP_FRAME_COUNT: self.HEADER_FRAMES,
                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_POS_MSEC: self.i * 1000.0 / self.REAL_FPS}.get(prop, 0)

    def grab(self):
        if self.i + 1 >= self.FRAMES:
            return False
        self.i += 1
        return True

    def read(self):
        if not self.grab():
            return False, None
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        pass


class _Detector:
    name = "stub"

    def __init__(self):
        self.clock = []

    def detect(self, rgb, ts):
        self.clock.append(ts)
        return type("F", (), {"world_landmarks": [], "normalized_landmarks": []})()


def test_the_frames_carry_decoded_time_and_the_detector_keeps_its_clock(monkeypatch):
    monkeypatch.setattr(runner.cv2, "VideoCapture", _EditListCapture)
    det, meta = _Detector(), {}
    frames = runner.extract_frames("clip.mov", "bike", 27.934, det, meta=meta)
    assert len(frames) == 328                 # 707 by the header: thinned to every 2nd, as before
    assert meta["decoded_fps"] == pytest.approx(29.97, abs=0.01)
    assert meta["fps_from_header"] == pytest.approx(27.934)
    assert meta["decoded_frames"] == 655
    assert meta["duration_s"] == pytest.approx(21.9, abs=0.1)
    # frame 600 is 20.02 s into the clip, not the header's 21.48
    f = frames[300]
    assert f["frame_idx"] == 600
    assert f["timestamp_ms"] == pytest.approx(600 * 1000 / 29.97, rel=1e-3)
    # the tracker's own clock is the one it always had
    assert det.clock[300] == int(600 * 1000 / 27.934)


def test_the_rate_reaches_the_analysis_and_the_overlay():
    src = (runner.__file__ and open(runner.__file__, encoding="utf-8").read())
    assert 'video_info = {**video_info, "fps": fps, "frame_count": float(n), "duration": n / fps}' in src
    assert "VideoVisualizer(\n            video_path=video_path,\n            fps=fps," in src
