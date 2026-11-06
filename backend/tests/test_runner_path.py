"""The skeleton has to be on the runner, and a broken hip path says when it was not."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.services.video_analysis import runner
from app.services.video_analysis.biomechanics import runner_path
from app.services.video_analysis.biomechanics.quality_gate import evaluate_quality_gate
from app.services.video_analysis.biomechanics.runner_path import (
    FollowWindow,
    plan_follow,
    trace_runner,
)
from app.services.video_analysis.detectors.base import DetectorFrame, DetectorLandmark

W, H = 1000, 500


def lm(x: float, y: float) -> DetectorLandmark:
    return DetectorLandmark(x=x, y=y, z=0.1, visibility=0.9)


def marks(hip_x: float, hip_y: float, torso: float = 50.0) -> list[DetectorLandmark]:
    """33 landmarks with the hips at a pixel and the shoulders straight above."""
    out = [lm(hip_x / W, hip_y / H) for _ in range(33)]
    out[11] = out[12] = lm(hip_x / W, (hip_y - torso) / H)
    return out


def frames(path: list[tuple[float, float] | tuple[float, float, float]]) -> list[dict]:
    return [
        {"frame_idx": i, "frame_width": W, "frame_height": H, "normalized_landmarks": marks(*p)}
        for i, p in enumerate(path)
    ]


def still(n: int, x: float = 500.0, y: float = 400.0) -> list[tuple[float, float]]:
    return [(x, y) for _ in range(n)]


def test_a_whole_path_is_one_piece_and_nobody_was_lost():
    t = trace_runner(frames(still(120)), 60.0)
    assert t.pieces == 1
    assert t.off_runner == 0.0
    assert not t.lost


def test_a_sprinter_past_a_still_camera_is_not_a_jump():
    """10 m/s at 30 fps is two thirds of a torso a frame."""
    t = trace_runner(frames([(100 + i * 33.0, 400.0) for i in range(25)]), 30.0)
    assert t.pieces == 1


def test_an_excursion_onto_something_else_is_off_the_runner():
    path = still(60) + still(20, x=800.0, y=100.0) + still(60)
    t = trace_runner(frames(path), 60.0)
    assert t.pieces == 3
    assert not t.on_runner[60:80].any()
    assert t.on_runner[:60].all() and t.on_runner[80:].all()
    assert t.off_runner == pytest.approx(20 / 140)


def test_a_false_target_can_run_longer_than_any_one_stretch_of_the_runner():
    """Pieces that continue the longest stretch stand with it; the excavator does not."""
    path = still(50) + still(45, x=800.0, y=100.0) + still(40)
    t = trace_runner(frames(path), 60.0)
    assert t.on_runner[:50].all() and t.on_runner[95:].all()
    assert not t.on_runner[50:95].any()


def test_after_a_gap_he_is_expected_where_he_was_heading():
    """Past a still camera, he is two torsos on after a gap of half a second, and
    that is him. Being there with no way of having got there is not."""
    moving = [(100 + i * 5.0, 400.0) for i in range(120)]
    gone = moving[:60] + [(500.0, 100.0)] * 15 + moving[75:]
    t = trace_runner(frames(gone), 60.0)
    assert t.on_runner[:60].all() and t.on_runner[75:].all()
    assert not t.on_runner[60:75].any()


def test_a_body_of_another_size_is_someone_else():
    path = still(80) + still(10, x=510.0, y=400.0)
    fr = frames(path)
    for i in range(80, 90):
        fr[i]["normalized_landmarks"] = marks(510.0, 400.0, torso=90.0)
    t = trace_runner(fr, 60.0)
    assert not t.on_runner[80:].any()


@pytest.mark.parametrize("pieces,off,lost", [
    (1, 0.0, False),
    (3, 0.11, False),   # IMG_3314: a brief loss on a clip with nothing to gain from a crop
    (22, 0.01, False),  # jittery but on him
    (22, 0.14, True),   # Run00000851
])
def test_what_counts_as_lost(pieces, off, lost):
    on = np.ones(100, dtype=bool)
    on[: round(off * 100)] = False
    t = runner_path.RunnerTrace(
        np.arange(100.0), np.zeros(100), np.zeros(100), np.ones(100), on, pieces,
    )
    assert t.lost is lost
    assert t.summary()["lost"] is lost


def test_nothing_to_trace_is_not_a_failure():
    assert trace_runner([], 30.0) is None
    fr = frames(still(10))
    for f in fr:
        f["normalized_landmarks"] = [lm(math.nan, math.nan) for _ in range(33)]
    assert trace_runner(fr, 30.0) is None


# --- the window ------------------------------------------------------------


def test_the_window_is_sized_from_the_torso_and_centred_on_the_hips():
    t = trace_runner(frames(still(60, x=500.0, y=250.0)), 60.0)
    w = plan_follow(t, 60, 60.0, W, H)
    assert w.size == 366    # 50 / 0.30 * 2.2, even
    assert w.origin(30) == (500 - 183, 250 - 183)


def test_the_window_ignores_the_excursion_and_stays_in_the_picture():
    path = still(60, x=950.0, y=450.0) + still(20, x=100.0, y=60.0) + still(60, x=950.0, y=450.0)
    t = trace_runner(frames(path), 60.0)
    w = plan_follow(t, 140, 60.0, W, H)
    for i in (0, 70, 139):
        x0, y0 = w.origin(i)
        assert 0 <= x0 <= W - w.size and 0 <= y0 <= H - w.size
        assert x0 == W - w.size, "held on the runner through the excursion"


def test_before_the_path_starts_he_is_further_back_along_it():
    moving = [(300 + i * 4.0, 300.0) for i in range(60)]
    fr = frames(moving)
    for i, f in enumerate(fr):
        f["frame_idx"] = i + 30
    t = trace_runner(fr, 60.0)
    w = plan_follow(t, 120, 60.0, W, H)
    assert w.origin(0)[0] < w.origin(30)[0] < w.origin(89)[0] < w.origin(119)[0]


def test_a_frame_past_the_plan_uses_its_last_position():
    w = FollowWindow(np.array([1, 2]), np.array([3, 4]), 10, W, H)
    assert w.origin(99) == (2, 4)
    assert w.origin(-1) == (1, 3)


def test_landmarks_go_back_to_the_whole_frame():
    """Wherever the window was on that frame, the point lands where it really is."""
    w = FollowWindow(np.array([100, 100, 160]), np.array([50, 50, 20]), 200, W, H)
    seen = [DetectorLandmark(0.5, 0.25, 0.4, 0.8), DetectorLandmark(math.nan, math.nan, math.nan, 0.0)]
    fr = [{"normalized_landmarks": w.to_zoom(seen, 2), "frame_width": 200, "frame_height": 200}]
    w.restore(fr)
    out = fr[0]["normalized_landmarks"]
    assert out[0].x == pytest.approx((160 + 100) / W)
    assert out[0].y == pytest.approx((20 + 50) / H)
    assert out[0].z == pytest.approx(0.4 * 200 / W)
    assert out[0].visibility == 0.8
    assert math.isnan(out[1].x) and out[1].visibility == 0.0
    assert (fr[0]["frame_width"], fr[0]["frame_height"]) == (W, H)


def test_the_window_scale_keeps_the_real_motion():
    """The window follows the hips, so its travel is part of what they did. Leave
    it out and vertical oscillation loses part of the bounce: it scored 0 against
    76 on Run00000851."""
    w = FollowWindow(np.array([100, 100, 100]), np.array([50, 50, 90]), 200, W, H)
    seen = [DetectorLandmark(0.5, 0.5, 0.0, 0.9)]
    assert w.to_zoom(seen, 0)[0].y == pytest.approx(0.5)
    assert w.to_zoom(seen, 2)[0].y == pytest.approx(0.5 + 40 / 200)


# --- through the frame loop ------------------------------------------------


class _Seer:
    """Finds the same pose in the middle of whatever it is shown."""

    def __init__(self) -> None:
        self.shapes: list[tuple[int, int]] = []
        self.closed = False

    def detect(self, rgb, ts):
        self.shapes.append(rgb.shape[:2])
        marks_ = [lm(0.5, 0.5) for _ in range(33)]
        return DetectorFrame(normalized_landmarks=marks_, world_landmarks=marks_)

    def close(self):
        self.closed = True


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
    assert w.isOpened(), "no MJPG encoder in this OpenCV build"
    for _ in range(12):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()
    return str(path)


def test_the_detector_sees_the_window_and_the_result_lands_on_the_source(clip):
    seer = _Seer()
    window = FollowWindow(np.full(12, 40), np.full(12, 20), 100, 320, 240)
    out = runner.extract_frames(clip, "run", 30.0, seer, meta={}, follow=window)
    assert set(seer.shapes) == {(100, 100)}
    assert out[0]["normalized_landmarks"][0].x == 0.5, "the window's scale, for the stabiliser"
    assert out[0]["frame_width"] == 100 and out[0]["frame_height"] == 100
    window.restore(out)
    m = out[0]["normalized_landmarks"][0]
    assert m.x == pytest.approx((40 + 50) / 320)
    assert m.y == pytest.approx((20 + 50) / 240)
    assert out[0]["frame_width"] == 320 and out[0]["frame_height"] == 240


def test_no_window_is_the_old_behaviour(clip):
    seer = _Seer()
    out = runner.extract_frames(clip, "run", 30.0, seer, meta={})
    assert set(seer.shapes) == {(240, 320)}
    assert out[0]["normalized_landmarks"][0].x == 0.5


# --- the decision ----------------------------------------------------------


LOST = still(40) + still(5, x=800.0, y=100.0) + still(5) + still(5, x=100.0, y=100.0) + still(40)


def second_pass(monkeypatch, result):
    calls = []

    def fake(path, sport, fps, det, meta=None, progress=None, min_sample_rate=1, follow=None):
        calls.append(follow)
        if meta is not None:
            meta["sample_rate"] = 1
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner, "extract_frames", fake)
    monkeypatch.setattr(runner, "build_detector", lambda *a, **k: _Seer())
    return calls


def test_a_clip_that_stayed_on_the_runner_is_not_looked_at_twice(monkeypatch):
    """The window breaks treadmill clips: IMG_4004 lost 20 spm and its score."""
    calls = second_pass(monkeypatch, frames(still(95)))
    first = frames(still(95))
    out, meta, follow = runner._second_look_if_lost("x", first, {"sample_rate": 1}, 60.0, 95, None)
    assert calls == [] and follow is None
    assert out is first
    assert meta["runner_path"]["lost"] is False
    assert "second_look" not in meta


def test_a_lost_runner_gets_a_second_look_and_it_stands(monkeypatch):
    calls = second_pass(monkeypatch, frames(still(95)))
    first = frames(LOST)
    assert trace_runner(first, 60.0).lost
    out, meta, follow = runner._second_look_if_lost("x", first, {"sample_rate": 1}, 60.0, 95, None)
    assert len(calls) == 1 and calls[0] is follow
    assert out is not first
    assert meta["runner_path"]["lost"] is False
    assert meta["second_look"]["first_pass"]["lost"] is True
    assert meta["second_look"]["window_px"] == follow.size


def test_a_second_look_that_holds_him_for_less_does_not_stand(monkeypatch):
    second_pass(monkeypatch, frames(still(10)))
    first = frames(LOST)
    out, meta, follow = runner._second_look_if_lost("x", first, {"sample_rate": 1}, 60.0, 95, None)
    assert out is first and follow is None
    assert meta["runner_path"]["lost"] is True, "still lost: the gate withholds"


def test_a_second_look_that_fails_costs_nothing(monkeypatch):
    second_pass(monkeypatch, RuntimeError("no more memory"))
    first = frames(LOST)
    out, meta, follow = runner._second_look_if_lost("x", first, {"sample_rate": 1}, 60.0, 95, None)
    assert out is first and follow is None
    assert meta["runner_path"]["lost"] is True


def _gate(**kw):
    return evaluate_quality_gate(
        unknown_phase_pct=None, angle_statistics={}, valid_frames=100, frames_processed=100,
        sport="run", **kw,
    )


def test_a_runner_still_lost_gets_no_score():
    g = _gate(runner_lost=True)
    assert g["triggered"]
    assert "lost the runner" in g["reasons"][0]


@pytest.mark.parametrize("value", [False, None])
def test_a_runner_on_the_path_is_not_the_gates_business(value):
    assert not _gate(runner_lost=value)["triggered"]
