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

# On the bike the box RTMPose is cropped on spans the whole last second, not
# the one frame. MediaPipe's box is the min/max of its points, and its bottom
# edge rides the pedalling foot (~10 px a frame on IMG_5572), so the crop's
# scale changed every frame and the whole body rocked at cadence -- the elbow,
# which barely moves, five times as far per frame as MediaPipe's (2026-11-09).
# The box's extent around the torso is the widest of the last second (about a
# pedal revolution); its position follows the torso on this frame, so a rider
# passing the camera is not chased by a box that lags behind.
#
# OFF (2026-11-09). On four clips it steadied the arms, but it also moves the
# legs: the TT reference clip's knee at BDC went 150.0 -> 151.3, the saddle
# verdict flipped from acceptable to too high and the score 86 -> 81. Without
# labelled clips there is no telling which knee is right, and filtering the
# arms at 3 Hz (landmark_stabilizer.BIKE_ARM_CUTOFF_RATIO) already took most
# of the shake out without touching a leg. Turn on with "bike" once a labelled
# set says the knee moves the right way.
STEADY_BOX_SPORTS: frozenset[str] = frozenset()

# Which of RTMPose's legs (and arms) is "left" is decided once for the whole
# clip on the bike, not frame by frame. Frame by frame each RTMPose chain went
# to whichever MediaPipe label it sat nearer, so every time MediaPipe swapped
# the legs -- often on the chainring side, where the far leg shows through the
# frame -- the refined skeleton swapped with it. RTMPose's own labels did not
# swap once on five bike clips; through MediaPipe's they swapped 16 times on
# IMG_4090 and 24 on IMG_5577, whose knee at the bottom then read anywhere from
# 120 to 141 deg and told the athlete to raise the saddle 43 mm (2026-11-09).
# Now RTMPose keeps its own labels, flipped as a whole only if MediaPipe's
# labels disagree with them on most frames -- MediaPipe still names the sides,
# by majority. Bike only: the run has its own leg-identity resolver.
CLIP_LABEL_SPORTS = frozenset({"bike"})
STEADY_BOX_MS = 1000
_TORSO = (11, 12, 23, 24)

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
        sport = getattr(getattr(inner, "config", None), "sport", None)
        self._steady = sport in STEADY_BOX_SPORTS
        self._boxes: list[tuple[float, np.ndarray]] = []  # (ms, box around the torso)
        self._frame = 0
        self._clip_labels = sport in CLIP_LABEL_SPORTS
        # Per refined frame: (output landmarks, MediaPipe's landmarks, RTMPose's
        # own-label points, their scores, MediaPipe's points, width, height).
        self._kept: list[tuple[list, list, np.ndarray, np.ndarray, np.ndarray, int, int]] = []

    def _box(self, mp_px: np.ndarray, ok: np.ndarray, w: int, h: int,
             timestamp_ms: int | None) -> tuple[float, float, float, float]:
        """This frame's box, or on the bike the last second's (STEADY_BOX_MS)."""
        x0, y0 = mp_px[ok].min(0)
        x1, y1 = mp_px[ok].max(0)
        if not self._steady:
            return x0, y0, x1, y1
        torso = mp_px[list(_TORSO)]
        torso = torso[np.isfinite(torso).all(1)]
        anchor = torso.mean(0) if len(torso) else np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        now = float(timestamp_ms) if timestamp_ms is not None else self._frame * 1000.0 / 30.0
        self._frame += 1
        self._boxes.append((now, np.array([x0, y0, x1, y1]) - np.tile(anchor, 2)))
        self._boxes = [(t, b) for t, b in self._boxes if now - t <= STEADY_BOX_MS]
        rel = np.array([b for _, b in self._boxes])
        bx0, by0 = rel[:, 0].min() + anchor[0], rel[:, 1].min() + anchor[1]
        bx1, by1 = rel[:, 2].max() + anchor[0], rel[:, 3].max() + anchor[1]
        return max(bx0, 0.0), max(by0, 0.0), min(bx1, w - 1.0), min(by1, h - 1.0)

    def detect(self, image_rgb: np.ndarray, timestamp_ms: int | None = None) -> DetectorFrame | None:
        frame = self._inner.detect(image_rgb, timestamp_ms)
        if frame is None:
            return None
        try:
            return self._refine(image_rgb, frame, timestamp_ms)
        except Exception as e:  # noqa: BLE001 -- MediaPipe's frame is still a frame
            self.failed += 1
            if self.failed == 1:
                logger.warning("RTMPOSE_REFINE_FAILED", err=str(e))
            return frame

    def _refine(self, image_rgb: np.ndarray, frame: DetectorFrame,
                timestamp_ms: int | None = None) -> DetectorFrame:
        h, w = image_rgb.shape[:2]
        mp_px = np.array([[lm.x * w, lm.y * h] for lm in frame.normalized_landmarks], dtype=np.float64)
        ok = np.isfinite(mp_px).all(1)
        if ok.sum() < 5:
            return frame
        x0, y0, x1, y1 = self._box(mp_px, ok, w, h, timestamp_ms)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return frame
        kps, sc = predict(self._session, image_rgb, (x0, y0, x1, y1))
        rtm = np.full((33, 2), np.nan)
        rsc = np.zeros(33)
        for b, hh in BLAZE_FROM_HALPE.items():
            rtm[b], rsc[b] = kps[hh], sc[hh]
        own, own_sc = rtm.copy(), rsc.copy()
        assign_to_mediapipe(rtm, rsc, mp_px)
        out = list(frame.normalized_landmarks)
        if self._clip_labels:
            self._kept.append((out, list(frame.normalized_landmarks), own, own_sc, mp_px, w, h))
        for b in BLAZE_FROM_HALPE:
            if rsc[b] >= MIN_SCORE and np.isfinite(rtm[b]).all():
                lm = out[b]
                out[b] = DetectorLandmark(float(rtm[b][0] / w), float(rtm[b][1] / h), lm.z, lm.visibility)
        self.refined += 1
        return DetectorFrame(normalized_landmarks=out, world_landmarks=frame.world_landmarks,
                             extras=frame.extras)

    def finish(self) -> dict[str, Any] | None:
        """After the last frame, on the bike: give every refined frame RTMPose's
        own leg and arm labels, flipped as a whole where MediaPipe's labels
        disagree with them on most frames (CLIP_LABEL_SPORTS). The landmark
        lists the frames were returned with are rewritten in place. Returns
        what it did, or None when there was nothing to do. Never raises."""
        kept, self._kept = self._kept, []
        if not kept:
            return None
        try:
            stats: dict[str, Any] = {"frames": len(kept)}
            for name, groups in (("legs", LEG), ("arms", ARM)):
                L, R = groups["left"], groups["right"]
                both = L + R
                swapped_vs_mp = []
                for _out, _mp, own, _sc, mp_px, _w, _h in kept:
                    same = np.nansum(np.linalg.norm(own[L] - mp_px[L], axis=1)) +                         np.nansum(np.linalg.norm(own[R] - mp_px[R], axis=1))
                    cross = np.nansum(np.linalg.norm(own[R] - mp_px[L], axis=1)) +                         np.nansum(np.linalg.norm(own[L] - mp_px[R], axis=1))
                    swapped_vs_mp.append(cross < same)
                flip = sum(swapped_vs_mp) > len(kept) / 2
                for (out, mp_lms, own, own_sc, _px, w, h), sw in zip(kept, swapped_vs_mp):
                    pts, sc = own.copy(), own_sc.copy()
                    if flip:
                        pts[both], sc[both] = own[R + L], own_sc[R + L]
                    # This frame's MediaPipe names the legs the other way round
                    # from the clip's labels: its z and visibility (and its
                    # point, where RTMPose is unsure) come from its other side.
                    other = sw != flip
                    for b, o in zip(both, R + L):
                        lm = mp_lms[o] if other else mp_lms[b]
                        if sc[b] >= MIN_SCORE and np.isfinite(pts[b]).all():
                            out[b] = DetectorLandmark(float(pts[b][0] / w), float(pts[b][1] / h),
                                                      lm.z, lm.visibility)
                        else:
                            out[b] = lm
                stats[name] = {"flipped": bool(flip),
                               "mediapipe_disagreed": int(sum(s != flip for s in swapped_vs_mp))}
            logger.info("RTMPOSE_CLIP_LABELS", **stats)
            return stats
        except Exception as e:  # noqa: BLE001 -- the frame-by-frame labels stand
            logger.warning("RTMPOSE_CLIP_LABELS_FAILED", err=str(e))
            return None

    def close(self) -> None:
        self._inner.close()

    @property
    def name(self) -> str:
        return f"{self._inner.name}{NAME_SUFFIX}"

    @property
    def config(self) -> DetectorConfig:
        return self._inner.config


def assign_near_side(rtm: np.ndarray, rsc: np.ndarray, camera_side: str) -> None:
    """Swap RTMPose's chains, in place, so its surer leg (and arm) carries the
    label of the side facing the camera.

    On a still, MediaPipe's far-side points are often nowhere (the TT photo:
    far knee at visibility 0.01, a quarter of the frame from the leg), so the
    clip's nearest-label rule can hand the near leg's points to the far label
    and the far leg's to the near one. The leg in view is the one RTMPose is
    sure of; the label of the side in view is ``camera_side``.
    """
    near, far = ("left", "right") if camera_side == "left" else ("right", "left")
    for groups in (LEG, ARM):
        N, F = groups[near], groups[far]
        if np.nanmean(rsc[F]) > np.nanmean(rsc[N]):
            rtm[N + F], rsc[N + F] = rtm[F + N].copy(), rsc[F + N].copy()


def refine_photo(image_rgb: np.ndarray, landmarks: Any, sport_type: str,
                 camera_side: str | None = None,
                 ) -> tuple[list[Any], dict[str, int] | None]:
    """One still through RTMPose: ``(landmarks, stats)``.

    The rules of a clip (``RTMPoseRefiner._refine``), with two changes a still
    needs. Chains go to the side in view by RTMPose's own confidence
    (``assign_near_side``) when ``camera_side`` is given, by nearness to
    MediaPipe's labels otherwise. And a joint's visibility rises to RTMPose's
    score where that is the higher.

    On the 30 bike photos in upload/Check (2026-11-08) this measured the leg
    on 7 that MediaPipe alone had withheld and withheld none it had measured;
    on gym photos, where MediaPipe's left/right was wrong about half the time
    (treadmill study), the knee moves by tens of degrees.
    A photo has no neighbouring frames to borrow a reading from, and the angle
    functions read nothing under MIN_LANDMARK_VISIBILITY (0.7). The TT race
    photo that started this (2026-11-08) came out of MediaPipe's low-threshold
    pass with a leg in plain view at knee 0.66 and ankle 0.61, so the report
    said the leg could not be measured; RTMPose put them at 0.79 and 0.90. On
    a photo a joint is now read when either model is 0.7 sure of it. Clips
    keep MediaPipe's visibility: the golden baselines pin them.

    ``stats`` is None when nothing ran (sport not switched on, no model, a
    failure) and the landmarks come back as given. Never raises.
    """
    from types import SimpleNamespace

    from app.core.config import settings

    given = list(landmarks)
    if sport_type not in settings.rtmpose_sports:
        return given, None
    path = settings.rtmpose_model_path
    if not path.is_file():
        return given, None
    try:
        session = load_session(path)
        h, w = image_rgb.shape[:2]
        mp_px = np.array([[float(lm.x) * w, float(lm.y) * h] for lm in given], dtype=np.float64)
        ok = np.isfinite(mp_px).all(1)
        if ok.sum() < 5:
            return given, None
        x0, y0 = mp_px[ok].min(0)
        x1, y1 = mp_px[ok].max(0)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return given, None
        kps, sc = predict(session, image_rgb, (x0, y0, x1, y1))
    except Exception as e:  # noqa: BLE001 -- MediaPipe's pose still analyses
        logger.warning("RTMPOSE_PHOTO_FAILED", err=str(e))
        return given, None
    rtm = np.full((33, 2), np.nan)
    rsc = np.zeros(33)
    for b, hh in BLAZE_FROM_HALPE.items():
        rtm[b], rsc[b] = kps[hh], sc[hh]
    if camera_side in ("left", "right"):
        assign_near_side(rtm, rsc, camera_side)
    else:
        assign_to_mediapipe(rtm, rsc, mp_px)
    out = given
    moved = lifted = 0
    for b in BLAZE_FROM_HALPE:
        if rsc[b] >= MIN_SCORE and np.isfinite(rtm[b]).all():
            lm = out[b]
            vis = getattr(lm, "visibility", 1.0)
            vis = 1.0 if vis is None else float(vis)
            new_vis = max(vis, float(rsc[b]))
            lifted += int(new_vis > vis)
            moved += 1
            out[b] = SimpleNamespace(
                x=float(rtm[b][0] / w), y=float(rtm[b][1] / h),
                z=getattr(lm, "z", 0.0), visibility=new_vis,
                presence=getattr(lm, "presence", None),
            )
    return out, {"moved": moved, "lifted": lifted}


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
