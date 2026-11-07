"""RTMPose-m refinement of MediaPipe's body joints (detectors/rtmpose_refiner.py).

The logic runs on a fake ONNX session, so these tests need neither onnxruntime
nor the 55 MB model. One test runs the real model when both are present.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.video_analysis.detectors import build_detector
from app.services.video_analysis.detectors import rtmpose_refiner as R
from app.services.video_analysis.detectors.base import (
    DetectorConfig,
    DetectorFrame,
    DetectorLandmark,
    PoseDetector,
)

W, H = 400, 300


def _mp_points() -> np.ndarray:
    """A side-view figure in pixels: left leg forward, right leg back."""
    p = np.full((33, 2), np.nan)
    p[0] = (200, 40)
    p[11], p[12] = (195, 80), (205, 80)
    p[13], p[14] = (215, 120), (185, 120)
    p[15], p[16] = (235, 150), (165, 150)
    p[23], p[24] = (198, 160), (202, 160)
    p[25], p[26] = (240, 210), (170, 215)
    p[27], p[28] = (235, 270), (150, 265)
    p[29], p[30] = (228, 278), (140, 272)
    p[31], p[32] = (255, 280), (165, 280)
    return p


class FakeInner(PoseDetector):
    def __init__(self, pts: np.ndarray | None) -> None:
        self.pts = pts
        self.closed = False

    def detect(self, image_rgb, timestamp_ms=None):
        if self.pts is None:
            return None
        lms = [DetectorLandmark(float(x / W), float(y / H), -0.1 * i, 0.9)
               for i, (x, y) in enumerate(np.nan_to_num(self.pts, nan=200.0))]
        world = [DetectorLandmark(0.0, 0.0, 0.0, 0.9) for _ in range(33)]
        return DetectorFrame(normalized_landmarks=lms, world_landmarks=world, extras={"k": 1})

    def close(self):
        self.closed = True

    @property
    def name(self):
        return "mediapipe_tasks"

    @property
    def config(self):
        return DetectorConfig("bike", None, 0.5, 0.5, 0.5)


def _session_answering(halpe_px: np.ndarray, scores: np.ndarray, box_holder: dict):
    """A fake ONNX session whose SimCC output decodes to ``halpe_px`` in image
    pixels, given the box the refiner will crop."""

    class S:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, _out, feed):
            x0, y0, x1, y1 = box_holder["box"]
            _img, center, scale = R._crop(np.zeros((H, W, 3), np.uint8), (x0, y0, x1, y1))
            loc = (halpe_px - center + scale / 2) / scale * np.array([R.MODEL_INPUT_W, R.MODEL_INPUT_H])
            bins = np.rint(loc * R.SIMCC_SPLIT_RATIO).astype(int)
            sx = np.zeros((1, 26, int(R.MODEL_INPUT_W * R.SIMCC_SPLIT_RATIO)), np.float32)
            sy = np.zeros((1, 26, int(R.MODEL_INPUT_H * R.SIMCC_SPLIT_RATIO)), np.float32)
            for k in range(26):
                sx[0, k, np.clip(bins[k, 0], 0, sx.shape[2] - 1)] = scores[k]
                sy[0, k, np.clip(bins[k, 1], 0, sy.shape[2] - 1)] = scores[k]
            return [sx, sy]

    return S()


def _halpe_from(blaze: np.ndarray) -> np.ndarray:
    h = np.zeros((26, 2))
    for b, k in R.BLAZE_FROM_HALPE.items():
        h[k] = blaze[b]
    return h


def _box(pts):
    """The box the refiner draws round what FakeInner reports."""
    p = np.nan_to_num(pts, nan=200.0)
    return (*p.min(0), *p.max(0))


def _refiner(mp, rtm_blaze, scores=None):
    holder = {"box": _box(mp)}
    sc = np.full(26, 0.9) if scores is None else scores
    return R.RTMPoseRefiner(FakeInner(mp), _session_answering(_halpe_from(rtm_blaze), sc, holder))


IMG = np.zeros((H, W, 3), np.uint8)


def test_body_joints_take_rtmposes_xy_and_keep_mediapipes_depth_and_visibility():
    mp = _mp_points()
    rtm = mp.copy()
    rtm[25] += (6, -4)    # left knee moved
    rtm[27] += (-5, 3)    # left ankle moved
    det = _refiner(mp, rtm)
    out = det.detect(IMG, 0)
    knee = out.normalized_landmarks[25]
    # SimCC resolves to half a model pixel; the crop here is ~1.5 px per model px.
    assert abs(knee.x * W - rtm[25][0]) < 1.5 and abs(knee.y * H - rtm[25][1]) < 1.5
    assert knee.z == pytest.approx(-2.5) and knee.visibility == pytest.approx(0.9)
    # The nose is not a body joint RTMPose replaces.
    assert out.normalized_landmarks[0].x * W == pytest.approx(200)
    assert out.world_landmarks[0].visibility == 0.9 and out.extras == {"k": 1}
    assert det.refined == 1


def test_an_unsure_joint_keeps_mediapipes_point():
    mp = _mp_points()
    rtm = mp.copy()
    rtm[25] += (20, 0)
    sc = np.full(26, 0.9)
    sc[R.BLAZE_FROM_HALPE[25]] = R.MIN_SCORE - 0.05
    out = _refiner(mp, rtm, sc).detect(IMG, 0)
    assert out.normalized_landmarks[25].x * W == pytest.approx(240)


def test_rtmposes_legs_follow_mediapipes_labels_as_whole_chains():
    mp = _mp_points()
    rtm = mp.copy()
    # RTMPose calls the forward leg "right": the whole chain comes back as left.
    for left, right in zip(R.LEG["left"], R.LEG["right"]):
        rtm[[left, right]] = mp[[right, left]] + (2, 1)
    out = _refiner(mp, rtm).detect(IMG, 0)
    assert abs(out.normalized_landmarks[25].x * W - (240 + 2)) < 1.5
    assert abs(out.normalized_landmarks[26].x * W - (170 + 2)) < 1.5


def test_a_frame_without_a_pose_stays_none_and_a_failure_keeps_mediapipe():
    assert R.RTMPoseRefiner(FakeInner(None), object()).detect(IMG, 0) is None

    class Broken:
        def get_inputs(self):
            raise RuntimeError("boom")

    det = R.RTMPoseRefiner(FakeInner(_mp_points()), Broken())
    out = det.detect(IMG, 0)
    assert out.normalized_landmarks[25].x * W == pytest.approx(240)
    assert det.failed == 1


def test_the_name_says_rtmpose_and_close_reaches_mediapipe():
    inner = FakeInner(_mp_points())
    det = R.RTMPoseRefiner(inner, object())
    assert det.name == "mediapipe_tasks+rtmpose_m" and det.config.sport == "bike"
    det.close()
    assert inner.closed


def _settings(monkeypatch, sports, model_dir):
    from app.core import config

    monkeypatch.setattr(config, "settings", dataclasses.replace(
        config.settings, rtmpose_sports=sports, models_dir=model_dir))


def test_off_unless_the_sport_is_listed_and_never_without_the_model(monkeypatch, tmp_path):
    inner = FakeInner(_mp_points())
    _settings(monkeypatch, (), tmp_path)
    assert R.maybe_refine(inner, "bike") is inner
    _settings(monkeypatch, ("bike",), tmp_path)          # listed, no model file
    assert R.maybe_refine(inner, "bike") is inner
    (tmp_path / "rtmpose-m_body7_halpe26.onnx").write_bytes(b"x")
    monkeypatch.setattr(R, "load_session", lambda p: object())
    assert isinstance(R.maybe_refine(inner, "bike"), R.RTMPoseRefiner)
    assert R.maybe_refine(inner, "run") is inner         # run not listed

    def broken(_p):
        raise RuntimeError("bad model")

    monkeypatch.setattr(R, "load_session", broken)
    assert R.maybe_refine(inner, "bike") is inner


def test_by_default_the_factory_still_builds_mediapipe_alone():
    from app.core.config import settings

    assert settings.rtmpose_sports == () or __import__("os").environ.get("VA_RTMPOSE_SPORTS")
    if settings.rtmpose_sports:
        pytest.skip("VA_RTMPOSE_SPORTS set in this environment")
    det = build_detector("bike", None)
    try:
        assert "rtmpose" not in det.name
    finally:
        det.close()


def test_the_env_switch_reads_bike_and_run_only(monkeypatch):
    from app.core import config

    monkeypatch.setenv("VA_RTMPOSE_SPORTS", " Bike, swim ,run,")
    assert config._load_settings().rtmpose_sports == ("bike", "run")
    monkeypatch.setenv("VA_RTMPOSE_SPORTS", "")
    assert config._load_settings().rtmpose_sports == ()


def test_the_real_model_finds_a_person_where_mediapipe_put_the_box():
    pytest.importorskip("onnxruntime")
    from app.core.config import settings

    if not settings.rtmpose_model_path.is_file():
        pytest.skip("RTMPose model not installed (baked into the image, see Dockerfile)")
    import cv2

    img = np.full((H, W, 3), 120, np.uint8)
    cv2.rectangle(img, (150, 40), (250, 290), (40, 40, 40), -1)
    kps, scores = R.predict(R.load_session(settings.rtmpose_model_path), img, (150, 40, 250, 290))
    assert kps.shape == (26, 2) and scores.shape == (26,)
    assert np.isfinite(kps).all()
