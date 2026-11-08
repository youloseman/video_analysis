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


def test_the_factory_wraps_only_the_sports_switched_on(monkeypatch, tmp_path):
    # No MediaPipe model in CI: the factory's MediaPipe is a stand-in here.
    import app.services.video_analysis.detectors as detectors

    monkeypatch.setattr(detectors, "MediaPipePoseDetector", lambda config: FakeInner(_mp_points()))
    _settings(monkeypatch, (), tmp_path)
    assert build_detector("bike", None).name == "mediapipe_tasks"
    (tmp_path / "rtmpose-m_body7_halpe26.onnx").write_bytes(b"x")
    monkeypatch.setattr(R, "load_session", lambda p: object())
    _settings(monkeypatch, ("bike",), tmp_path)
    assert build_detector("bike", None).name == "mediapipe_tasks+rtmpose_m"
    assert build_detector("run", None).name == "mediapipe_tasks"


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


def test_the_session_does_not_spawn_a_thread_per_visible_core(monkeypatch, tmp_path):
    # Railway shows 48 cores against an 8-CPU quota: the default ran 10x slower.
    seen = {}

    class FakeOrt:
        class SessionOptions:
            pass

        @staticmethod
        def InferenceSession(path, sess_options, providers):
            seen["opts"] = sess_options
            return object()

    monkeypatch.setitem(__import__("sys").modules, "onnxruntime", FakeOrt)
    monkeypatch.setattr(R, "_SESSIONS", {})
    R.load_session(tmp_path / "m.onnx")
    assert seen["opts"].intra_op_num_threads == R.INTRA_OP_THREADS == 4
    assert seen["opts"].inter_op_num_threads == 1


# --- a still (2026-11-08) -----------------------------------------------------
# A TT race photo came out of MediaPipe's low-threshold pass with the near knee at
# 0.66 and the ankle at 0.61: under 0.7, so "the leg could not be measured" on a
# leg in plain view. RTMPose had them at 0.79 and 0.90.

def _photo_landmarks(vis_knee=0.66, vis_ankle=0.61):
    pts = np.nan_to_num(_mp_points(), nan=200.0)
    out = []
    for i, (x, y) in enumerate(pts):
        v = 0.95
        if i == 26:
            v = vis_knee
        if i == 28:
            v = vis_ankle
        out.append(SimpleNamespace(x=x / W, y=y / H, z=0.1 * i, visibility=v, presence=0.9))
    return out


def _photo_scores():
    sc = np.full(26, 0.9)
    sc[R.BLAZE_FROM_HALPE[26]] = 0.79      # right knee
    sc[R.BLAZE_FROM_HALPE[28]] = 0.90      # right ankle
    sc[R.BLAZE_FROM_HALPE[15]] = 0.2       # left wrist: unsure, keeps MediaPipe
    return sc


def _photo_on(monkeypatch, tmp_path, sports=("bike",), shift=6.0, scores=None):
    _settings(monkeypatch, sports, tmp_path)
    (tmp_path / "rtmpose-m_body7_halpe26.onnx").write_bytes(b"x")
    monkeypatch.setattr(R, "load_session", lambda p: object())
    rtm = np.nan_to_num(_mp_points(), nan=200.0).copy()
    rtm[26] += (shift, 0.0)                # RTMPose puts the knee a little aside
    sc = _photo_scores() if scores is None else scores
    monkeypatch.setattr(R, "predict", lambda session, img, box: (_halpe_from(rtm), sc))
    return rtm


def test_a_photo_takes_rtmposes_xy_and_lifts_visibility_to_its_score(monkeypatch, tmp_path):
    rtm = _photo_on(monkeypatch, tmp_path)
    given = _photo_landmarks()
    out, stats = R.refine_photo(IMG, given, "bike")
    assert stats and stats["moved"] == 15 and stats["lifted"] == 2
    knee, ankle = out[26], out[28]
    assert knee.x == pytest.approx(rtm[26][0] / W) and knee.visibility == pytest.approx(0.79)
    assert ankle.visibility == pytest.approx(0.90)
    assert knee.z == given[26].z and knee.presence == 0.9            # depth and presence stay MediaPipe's
    assert out[11].visibility == pytest.approx(0.95)                 # a sure joint is not lowered to 0.9
    assert out[15] is given[15] and out[0] is given[0]               # unsure wrist, and the face, untouched
    from app.services.video_analysis.biomechanics.angle_calculator import MIN_LANDMARK_VISIBILITY
    assert knee.visibility >= MIN_LANDMARK_VISIBILITY > given[26].visibility


def test_on_a_photo_rtmposes_surer_leg_is_the_leg_facing_the_camera(monkeypatch, tmp_path):
    # MediaPipe's far (left) knee is nowhere near the leg, so by nearness RTMPose's
    # sure right leg would go to the LEFT label. The camera sees the right side:
    # the sure chain is the right leg, whatever MediaPipe's labels say.
    rtm = _photo_on(monkeypatch, tmp_path)          # MediaPipe's right knee 6 px off RTMPose's
    given = _photo_landmarks()
    given[25] = SimpleNamespace(x=rtm[26][0] / W, y=rtm[26][1] / H, z=0, visibility=0.01, presence=0.9)
    given[27] = SimpleNamespace(x=rtm[28][0] / W, y=rtm[28][1] / H, z=0, visibility=0.03, presence=0.9)
    sc = _photo_scores()
    sc[R.BLAZE_FROM_HALPE[25]], sc[R.BLAZE_FROM_HALPE[27]] = 0.38, 0.57   # RTMPose's left leg: unsure
    monkeypatch.setattr(R, "predict", lambda session, img, box: (_halpe_from(rtm), sc))
    out, _ = R.refine_photo(IMG, list(given), "bike", camera_side="right")
    assert out[26].visibility == pytest.approx(0.79) and out[28].visibility == pytest.approx(0.90)
    assert out[26].x == pytest.approx(rtm[26][0] / W)
    out, _ = R.refine_photo(IMG, list(given), "bike", camera_side="left")
    assert out[25].visibility == pytest.approx(0.79)                      # the sure chain follows the label in view


def test_a_photo_is_left_alone_when_off_without_a_model_or_on_a_failure(monkeypatch, tmp_path):
    given = _photo_landmarks()
    _settings(monkeypatch, (), tmp_path)
    out, stats = R.refine_photo(IMG, given, "bike")
    assert stats is None and out == given
    _photo_on(monkeypatch, tmp_path, sports=("run",))
    assert R.refine_photo(IMG, given, "bike")[1] is None
    _photo_on(monkeypatch, tmp_path)

    def broken(*_a):
        raise RuntimeError("no")

    monkeypatch.setattr(R, "predict", broken)
    out, stats = R.refine_photo(IMG, given, "bike")
    assert stats is None and out == given
