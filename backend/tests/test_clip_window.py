"""A long clip is cut to its clearest window, not thinned (2026-11-24)."""
from __future__ import annotations

import inspect
import shutil
from pathlib import Path

import pytest

from app.services.video_analysis import clip_window as cw

BACKEND = Path(__file__).resolve().parents[1]


def test_only_a_clip_the_extractor_would_thin_is_cut():
    assert cw.window_frames(600, 30.0, 450) is None          # 20 s at 30 fps: read whole
    assert cw.window_frames(675, 30.0, 450) == 450           # stride 2 from here: 15 s kept
    assert cw.window_frames(700, 60.0, 450) is None          # 60 fps thins to 30: the run's own rate
    assert cw.window_frames(1000, 25.0, 450) == 375          # 25 fps: 15 s
    assert cw.window_frames(2000, 120.0, 450) is None        # high speed: left to the extractor
    assert cw.window_frames(0, 30.0, 450) is None


def test_the_window_stays_at_full_rate():
    for fps in (24.0, 25.0, 29.97, 30.0):
        keep = cw.window_frames(5000, fps, 450)
        assert keep / 450 + 0.5 < 2, fps      # the extractor's nearest stride stays 1
        assert keep / fps <= cw.MAX_WINDOW_S + 1e-9


def test_the_window_goes_where_the_athlete_is():
    # 40 s clip, the runner in shot from 18 s to 34 s only
    times = [i * 0.5 for i in range(80)]
    feats = [{"trunk_deg": 80.0, "height_frac": 0.7} if 18 <= t <= 34 else None for t in times]
    start = cw.pick_start(times, cw.sample_scores(feats), 40.0, 15.0)
    assert 18 <= start and start + 15 <= 34.5


def test_a_steady_trainer_clip_is_cut_from_the_middle():
    times = [i * 0.5 for i in range(60)]
    feats = [{"trunk_deg": 20.0, "height_frac": 0.8} for _ in times]
    start = cw.pick_start(times, cw.sample_scores(feats), 30.0, 15.0)
    assert abs(start + 7.5 - 15.0) <= 0.5


def test_another_position_counts_half():
    s = cw.sample_scores([{"trunk_deg": 20.0, "height_frac": 0.85}] * 4
                         + [{"trunk_deg": 45.0, "height_frac": 0.85}, None])
    assert s[0] == pytest.approx(1.0) and s[4] == pytest.approx(0.5) and s[5] == 0.0


def test_the_note_says_which_part():
    assert cw.trimmed_note({"from_s": 8.2, "to_s": 23.2, "source_s": 34.0}) == "0:08-0:23 of a 34 s clip"
    assert cw.trimmed_note(None) is None


def test_it_is_opt_in_and_the_jobs_opt_in_by_flag():
    assert inspect.signature(__import__(
        "app.services.video_analysis.runner", fromlist=["run_analysis"]).run_analysis
    ).parameters["trim_long"].default is False
    main = (BACKEND / "app" / "main.py").read_text(encoding="utf-8")
    assert main.count("trim_long=sport in settings.clip_trim_sports") == 2
    spa = (BACKEND / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "(r.quality&&r.quality.clip_trimmed)||s.clip_trimmed" in spa


def _clip(path: Path, frames: int, fps: float = 30.0) -> None:
    import cv2
    import numpy as np

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for i in range(frames):
        img = np.full((240, 320, 3), i % 256, np.uint8)
        cv2.putText(img, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        vw.write(img)
    vw.release()


def _frames(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def test_nobody_in_shot_leaves_the_clip_alone(tmp_path, monkeypatch):
    clip = tmp_path / "in.mp4"
    _clip(clip, 900)
    before = clip.read_bytes()
    monkeypatch.setattr(cw, "_scan", lambda *a: ([i * 0.5 for i in range(20)], [None] * 20, 30.0))
    assert cw.trim_long_clip(clip, budget=450) is None
    assert clip.read_bytes() == before


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_the_cut_is_frame_exact_and_in_place(tmp_path, monkeypatch):
    clip = tmp_path / "in.mov"
    _clip(clip, 900)                                   # 30 s at 30 fps
    times = [i * 0.5 for i in range(60)]
    feats = [{"trunk_deg": 20.0, "height_frac": 0.8} if 10 <= t <= 26 else None for t in times]
    monkeypatch.setattr(cw, "_scan", lambda *a: (times, feats, 30.0))
    got = cw.trim_long_clip(clip, budget=450)
    assert got and got["frames"] == 450 and got["source_frames"] == 900
    assert 10 <= got["from_s"] and got["to_s"] <= 26.5
    assert clip.exists() and _frames(clip) == 450      # same path, the window only
    assert not list(tmp_path.glob("*.tmp"))


def test_bike_is_cut_by_default_and_the_variable_switches_it(monkeypatch):
    from app.core import config

    monkeypatch.delenv("VA_CLIP_TRIM_SPORTS", raising=False)
    assert config._load_settings().clip_trim_sports == ("bike",)
    for raw, want in (("none", ()), ("", ()), ("bike,run", ("bike", "run"))):
        monkeypatch.setenv("VA_CLIP_TRIM_SPORTS", raw)
        assert config._load_settings().clip_trim_sports == want, raw
