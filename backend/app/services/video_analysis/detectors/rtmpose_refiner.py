"""MediaPipe tracks, RTMPose-m places the body joints.

Why: on 480 labelled bike-fit frames RTMPose-m roughly halved MediaPipe's knee
scatter (sd 7.1 -> 3.9 deg) and its ankle error (16.4 -> 7.4 deg MAE), and on
drive/non-drive pairs it closed most of the 9 deg gap between the two sides of
the same rider (see docs: flapp-detector-benchmark, flapp-drive-side-mirror).

How: MediaPipe runs exactly as before -- it finds the athlete, follows them
from frame to frame, labels left and right, and supplies z, visibility and the
world landmarks. RTMPose-m (Halpe-26, SimCC) is then run once per frame on
MediaPipe's own box and replaces only the image x/y of the 16 body joints it
shares with BlazePose (shoulders, elbows, wrists, hips, knees, ankles, heels,
toes). RTMPose works frame by frame and has no idea of identity, so each of its
legs (and arms) is given to whichever of MediaPipe's labels it sits nearer to,
chain by chain. Leg identity, near-side detection and every gate therefore see
the same structure they always have.

A joint RTMPose is unsure of (score below ``MIN_SCORE``) keeps MediaPipe's
point. Per-joint "tracked" assignment was tried and drifted onto the wrong leg
(IMG_4090): don't.

The preprocessing and SimCC decoding follow rtmlib (Apache-2.0,
github.com/Tau-J/rtmlib), reimplemented here on onnxruntime + OpenCV so the
image does not pull rtmlib's own OpenCV build next to ours.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

from app.services.video_analysis.detectors.base import (
    DetectorConfig,
    DetectorFrame,
    DetectorLandmark,
    PoseDetector,
)

logger = structlog.get_logger(__name__)

MODEL_INPUT_W, MODEL_INPUT_H = 192, 256
BOX_PADDING = 1.25
SIMCC_SPLIT_RATIO = 2.0
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
# Below this SimCC score the joint keeps MediaPipe's point.
MIN_SCORE = 0.3
# onnxruntime's default spawns a thread per core it can SEE: 48 on Railway's
# host against a cgroup quota of 8 CPUs, which throttled it to 224 ms a frame.
# Measured in the container (2026-11-07): 1 thread 35 ms, 2 -> 33, 4 -> 23,
# 8 -> 21, 16 -> 34. Four is nearly the best and leaves cores to MediaPipe and
# to a second analysis running alongside.
INTRA_OP_THREADS = 4

# BlazePose index -> Halpe-26 index
BLAZE_FROM_HALPE = {11: 5, 12: 6, 13: 7, 14: 8, 15: 9, 16: 10, 23: 11, 24: 12,
                    25: 13, 26: 14, 27: 15, 28: 16, 29: 24, 30: 25, 31: 20, 32: 21}
LEG = {"left": [23, 25, 27, 29, 31], "right": [24, 26, 28, 30, 32]}
ARM = {"left": [11, 13, 15], "right": [12, 14, 16]}

NAME_SUFFIX = "+rtmpose_m"

# One ONNX session per model file for the process. The session holds weights,
# not state: ``run`` is thread-safe and nothing about one video reaches the
# next, so this does not break PoseDetector's "stateless across videos" rule.
# It saves loading 50 MB three times per analysis (main, half rate, second look).
_SESSIONS: dict[str, Any] = {}
_SESSIONS_LOCK = threading.Lock()


def load_session(model_path: Path) -> Any:
    key = str(model_path)
    with _SESSIONS_LOCK:
        if key not in _SESSIONS:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.intra_op_num_threads = INTRA_OP_THREADS
            opts.inter_op_num_threads = 1
            _SESSIONS[key] = ort.InferenceSession(
                key, sess_options=opts, providers=["CPUExecutionProvider"],
            )
        return _SESSIONS[key]


def _crop(image_rgb: np.ndarray, box: tuple[float, float, float, float]
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The box, padded and widened to the model's aspect, warped to its input."""
    x0, y0, x1, y1 = box
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])
    bw, bh = (x1 - x0) * BOX_PADDING, (y1 - y0) * BOX_PADDING
    aspect = MODEL_INPUT_W / MODEL_INPUT_H
    scale = np.array([bw, bw / aspect]) if bw > bh * aspect else np.array([bh * aspect, bh])
    s = MODEL_INPUT_W / scale[0]
    warp = np.array([[s, 0.0, MODEL_INPUT_W * 0.5 - s * center[0]],
                     [0.0, s, MODEL_INPUT_H * 0.5 - s * center[1]]], dtype=np.float64)
    img = cv2.warpAffine(image_rgb, warp, (MODEL_INPUT_W, MODEL_INPUT_H), flags=cv2.INTER_LINEAR)
    return img, center, scale


def predict(session: Any, image_rgb: np.ndarray, box: tuple[float, float, float, float]
            ) -> tuple[np.ndarray, np.ndarray]:
    """Halpe-26 keypoints in image pixels (26, 2) and their SimCC scores (26,)."""
    img, center, scale = _crop(image_rgb, box)
    x = ((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    inp = session.get_inputs()[0].name
    simcc_x, simcc_y = session.run(None, {inp: np.ascontiguousarray(x, dtype=np.float32)})[:2]
    sx, sy = simcc_x[0], simcc_y[0]
    locs = np.stack([np.argmax(sx, axis=1), np.argmax(sy, axis=1)], axis=-1).astype(np.float64)
    scores = 0.5 * (np.amax(sx, axis=1) + np.amax(sy, axis=1))
    kps = locs / SIMCC_SPLIT_RATIO / np.array([MODEL_INPUT_W, MODEL_INPUT_H]) * scale + center - scale / 2
    kps[scores <= 0] = np.nan
    return kps, scores


def assign_to_mediapipe(rtm: np.ndarray, rsc: np.ndarray, mp_px: np.ndarray) -> None:
    """Swap RTMPose's left and right chains, in place, where MediaPipe's labels
    sit nearer the other way round. Chain by chain: legs as a whole, arms as a
    whole, so a knee never travels without its ankle."""
    for groups in (LEG, ARM):
        L, R = groups["left"], groups["right"]
        same = np.nansum(np.linalg.norm(rtm[L] - mp_px[L], axis=1)) + \
            np.nansum(np.linalg.norm(rtm[R] - mp_px[R], axis=1))
        swap = np.nansum(np.linalg.norm(rtm[R] - mp_px[L], axis=1)) + \
            np.nansum(np.linalg.norm(rtm[L] - mp_px[R], axis=1))
        if swap < same:
            rtm[L + R], rsc[L + R] = rtm[R + L].copy(), rsc[R + L].copy()


class RTMPoseRefiner(PoseDetector):
    """A MediaPipe detector whose body-joint x/y come from RTMPose-m."""

    def __init__(self, inner: PoseDetector, session: Any) -> None:  # noqa: D107
        self._inner = inner
        self._session = session
        self.refined = 0
        self.failed = 0

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int | None = None) -> DetectorFrame | None:
        frame = self._inner.detect(image_rgb, timestamp_ms)
        if frame is None:
            return None
        try:
            return self._refine(image_rgb, frame)
        except Exception as e:  # noqa: BLE001 -- MediaPipe's frame is still a frame
            self.failed += 1
            if self.failed == 1:
                logger.warning("RTMPOSE_REFINE_FAILED", err=str(e))
            return frame

    def _refine(self, image_rgb: np.ndarray, frame: DetectorFrame) -> DetectorFrame:
        h, w = image_rgb.shape[:2]
        mp_px = np.array([[lm.x * w, lm.y * h] for lm in frame.normalized_landmarks], dtype=np.float64)
        ok = np.isfinite(mp_px).all(1)
        if ok.sum() < 5:
            return frame
        x0, y0 = mp_px[ok].min(0)
        x1, y1 = mp_px[ok].max(0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        kps, sc = predict(self._session, image_rgb, (x0, y0, x1, y1))
        rtm = np.full((33, 2), np.nan)
        rsc = np.zeros(33)
        for b, hh in BLAZE_FROM_HALPE.items():
            rtm[b], rsc[b] = kps[hh], sc[hh]
        assign_to_mediapipe(rtm, rsc, mp_px)
        out = list(frame.normalized_landmarks)
        for b in BLAZE_FROM_HALPE:
            if rsc[b] >= MIN_SCORE and np.isfinite(rtm[b]).all():
                lm = out[b]
                out[b] = DetectorLandmark(float(rtm[b][0] / w), float(rtm[b][1] / h), lm.z, lm.visibility)
        self.refined += 1
        return DetectorFrame(normalized_landmarks=out, world_landmarks=frame.world_landmarks,
                             extras=frame.extras)

    def close(self) -> None:
        self._inner.close()

    @property
    def name(self) -> str:
        return f"{self._inner.name}{NAME_SUFFIX}"

    @property
    def config(self) -> DetectorConfig:
        return self._inner.config


def maybe_refine(inner: PoseDetector, sport_type: str) -> PoseDetector:
    """``inner`` wrapped in RTMPose when the sport is switched on and the model
    loads; ``inner`` itself otherwise. Never raises."""
    from app.core.config import settings

    if sport_type not in settings.rtmpose_sports:
        return inner
    path = settings.rtmpose_model_path
    if not path.is_file():
        logger.warning("RTMPOSE_MODEL_MISSING", path=str(path))
        return inner
    try:
        session = load_session(path)
    except Exception as e:  # noqa: BLE001 -- MediaPipe alone still analyses
        logger.warning("RTMPOSE_LOAD_FAILED", path=str(path), err=str(e))
        return inner
    return RTMPoseRefiner(inner, session)
