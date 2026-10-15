"""Stored clips shrink to 1080p and still line up frame for frame (clip_store.py)."""
from __future__ import annotations

import shutil

import cv2
import numpy as np
import pytest

from app.services import clip_store as CS


def _clip(path, w=640, h=360, n=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), (i * 8) % 255, np.uint8)
        cv2.putText(f, str(i), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)
        vw.write(f)
    vw.release()


def test_without_ffmpeg_the_clip_is_left_alone(tmp_path, monkeypatch):
    clip = tmp_path / "input.mov"
    clip.write_bytes(b"\0" * 100)
    monkeypatch.setattr(CS, "MIN_BYTES", 0)
    monkeypatch.setattr(CS.shutil, "which", lambda _: None)
    assert CS.compress_clip(clip) is None and clip.exists()


def test_a_small_clip_is_not_touched(tmp_path):
    clip = tmp_path / "input.mp4"
    _clip(clip)
    assert CS.compress_clip(clip) is None and clip.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_clip_is_replaced_with_every_frame_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "MIN_BYTES", 0)
    clip = tmp_path / "input.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    for i in range(30):
        f = np.full((360, 640, 3), (i * 8) % 255, np.uint8)
        cv2.putText(f, str(i), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 6)
        vw.write(f)
    vw.release()
    out = CS.compress_clip(clip)
    assert out == tmp_path / "input.mp4" and out.exists() and not clip.exists()
    assert CS._frame_count(out) == 30
    assert not list(tmp_path.glob("*.compressing.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_frame_count_mismatch_keeps_the_original(tmp_path, monkeypatch):
    monkeypatch.setattr(CS, "MIN_BYTES", 0)
    clip = tmp_path / "input.avi"
    vw = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
    for _ in range(10):
        vw.write(np.zeros((360, 640, 3), np.uint8))
    vw.release()
    counts = iter([10, 9])
    monkeypatch.setattr(CS, "_frame_count", lambda p: next(counts))
    assert CS.compress_clip(clip) is None
    assert clip.exists() and not list(tmp_path.glob("*.compressing.mp4"))
