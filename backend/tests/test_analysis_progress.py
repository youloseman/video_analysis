"""The analysis reports where it is, and the job turns that into a bar.

A minute of "Detecting pose…" over a sweeping bar was indistinguishable
from a hang, and a hang is what people reload out of. Detection is the only
phase that can count (frames), so it owns most of the bar and moves through
it; the fixed-cost phases after it each take a fixed slice.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.video_analysis import runner
from app.services.video_analysis.progress import (
    PHASE_SPAN,
    job_progress_hook,
    phase_sentence,
    phase_to_progress,
)


# --------------------------------------------------------------------------
# phase -> bar
# --------------------------------------------------------------------------
def test_the_phases_tile_the_bar_in_order():
    spans = [PHASE_SPAN[p] for p in runner.PHASES]
    assert spans[0][0] == 0 and spans[-1][1] <= 100
    for (_, end), (start, _) in zip(spans, spans[1:], strict=False):
        assert end == start, "no gap and no overlap between phases"


def test_detection_moves_through_its_slice():
    lo, hi = PHASE_SPAN["detect"]
    assert phase_to_progress("detect", 0.0) == lo
    assert phase_to_progress("detect", 0.5) == round(lo + (hi - lo) / 2)
    assert phase_to_progress("detect", 1.0) == hi
    assert phase_to_progress("detect", 7.0) == hi, "a fraction past 1 is clamped"


def test_a_phase_without_a_count_sits_at_its_start():
    assert phase_to_progress("measure", None) == PHASE_SPAN["measure"][0]


def test_a_two_clip_job_narrows_the_bar_to_its_half():
    assert phase_to_progress("detect", 1.0, span=(0.0, 0.5)) == PHASE_SPAN["detect"][1] // 2
    second = phase_to_progress("detect", 0.0, span=(0.5, 1.0))
    assert second == 50


def test_progress_never_goes_backwards_across_the_run():
    seq = [("detect", 0.0), ("detect", 0.4), ("detect", 1.0), ("stabilize", None),
           ("measure", None), ("visuals", None), ("coach", None)]
    values = [phase_to_progress(p, f) for p, f in seq]
    assert values == sorted(values)


# --------------------------------------------------------------------------
# phase -> sentence
# --------------------------------------------------------------------------
def test_detection_says_how_far_and_the_sport():
    assert phase_sentence("detect", 0.43, "run") == "Finding the runner in every frame 43%…"
    assert phase_sentence("detect", 0.0, "bike") == "Finding the rider in every frame…"


def test_every_phase_has_a_sentence_for_both_sports():
    for phase in runner.PHASES:
        for sport in ("run", "bike"):
            assert phase_sentence(phase, None, sport).endswith("…")


def test_the_hook_writes_both_fields_onto_the_job():
    job: dict = {}
    hook = job_progress_hook(job, "run", prefix="Left side, 1 of 2 · ", span=(0.0, 0.5))
    hook("detect", 0.5)
    assert job["stage"].startswith("Left side, 1 of 2 · Finding the runner")
    assert job["progress"] == phase_to_progress("detect", 0.5, (0.0, 0.5))


# --------------------------------------------------------------------------
# the runner actually calls it
# --------------------------------------------------------------------------
class _BlindDetector:
    """Sees nothing: the loop still runs every frame, which is what is tested."""

    def detect(self, rgb, ts):
        return None

    def close(self):
        pass


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    assert w.isOpened(), "no MJPG encoder in this OpenCV build"
    for _ in range(40):
        w.write(np.zeros((48, 64, 3), dtype=np.uint8))
    w.release()
    return str(path)


def test_extract_frames_reports_detection_from_zero_to_one(clip):
    calls: list[tuple[str, float | None]] = []
    runner.extract_frames(clip, "run", 30.0, _BlindDetector(), meta={},
                          progress=lambda p, f: calls.append((p, f)))
    assert calls and all(p == "detect" for p, _ in calls)
    fractions = [f for _, f in calls]
    assert fractions[0] == 0.0
    assert fractions == sorted(fractions), "never backwards"
    assert 0.9 <= fractions[-1] <= 1.0


def test_a_failing_hook_never_fails_the_analysis(clip):
    def boom(p, f):
        raise RuntimeError("UI is broken")

    frames = runner.extract_frames(clip, "run", 30.0, _BlindDetector(), meta={}, progress=boom)
    assert frames == []   # blind detector: no frames, but no exception either


def test_no_hook_is_the_old_behaviour(clip):
    assert runner.extract_frames(clip, "run", 30.0, _BlindDetector(), meta={}) == []
