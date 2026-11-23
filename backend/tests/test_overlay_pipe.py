"""The overlay is encoded once, through a pipe, while the next frames are drawn.

It was written twice: an XVID AVI through OpenCV (6 s of a 25 s render), then
decoded and re-encoded to H.264 by ffmpeg after the last frame. Frames now go
straight into ffmpeg's stdin. These need ffmpeg on PATH (prod has it; CI's
runner image does too); without it they skip, loudly.
"""

from __future__ import annotations

import inspect
import shutil
import subprocess

import numpy as np
import pytest

from app.services.video_analysis import runner
from app.services.video_analysis.video_visualizer import _FfmpegPipe

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")


def _probe(path) -> tuple[str, int, int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(",")
    return out[0], int(out[1]), int(out[2]), int(out[3])


@needs_ffmpeg
def test_piped_frames_come_out_as_h264_at_the_target_size(tmp_path):
    out = tmp_path / "o.mp4"
    pipe = _FfmpegPipe(str(out), 30.0, 120, 80, 120, 80)
    assert pipe.isOpened()
    for i in range(12):
        # A non-contiguous view, as ChipLayer used to return: must still go.
        frame = np.full((80, 120, 3), i * 20, dtype=np.uint8)[:, ::-1]
        pipe.write(frame)
    assert pipe.release() is True
    assert _probe(out) == ("h264", 120, 80, 12)


@needs_ffmpeg
def test_a_failed_encode_reports_false(tmp_path):
    # An output directory that does not exist: ffmpeg cannot open the file.
    pipe = _FfmpegPipe(str(tmp_path / "missing" / "o.mp4"), 30.0, 64, 64, 64, 64)
    for _ in range(3):
        pipe.write(np.zeros((64, 64, 3), dtype=np.uint8))
    assert pipe.release() is False


def test_no_ffmpeg_binary_is_a_closed_writer(monkeypatch, tmp_path):
    def nope(*_a, **_k):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(subprocess, "Popen", nope)
    pipe = _FfmpegPipe(str(tmp_path / "o.mp4"), 30.0, 64, 64, 64, 64)
    assert not pipe.isOpened()
    pipe.write(np.zeros((64, 64, 3), dtype=np.uint8))   # no raise
    assert pipe.release() is False


def test_the_overlay_render_is_joined_before_the_result_is_built():
    src = inspect.getsource(runner.analyze_from_frames)
    start = src.index("overlay_render.start()")
    join = src.index("overlay_render.join()")
    ret = src.rindex('"overlay_video_path": overlay_video_path')
    assert start < join < ret
