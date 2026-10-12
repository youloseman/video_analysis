"""Helmet above the top of the back, for a clip.

The same reading the photo report gives (biomechanics/head_silhouette.py),
so one rider stops getting two verdicts about one head: the clip used to say
"head position 100/100" -- the ear-under-the-joint-line score every aero
rider passed -- while a photo of the same position said "+27 %, amber".

A clip carries no person mask: the pose model runs in VIDEO mode for the
landmarks, and asking it for a mask on every frame would cost memory on every
analysis for one number. So a handful of frames, spread across the clip, are
re-read from the video and run through the IMAGE model with the mask on. The
head barely moves through a pedal stroke, so their median is the reading, and
a frame whose silhouette cannot carry it simply drops out.

Each frame is measured against the clip's own landmarks (filtered, and with
the athlete's corrections if any), not the second model pass's: the mask only
supplies the outline.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any

import structlog

from app.services.video_analysis.biomechanics.head_silhouette import (
    helmet_above_back,
    top_contour,
)

logger = structlog.get_logger()

SAMPLES = 5
# The same cap the clip's own detection uses (runner.DETECT_MAX_LONG_EDGE):
# the model is not more accurate on a 4K frame, only slower.
MAX_LONG_EDGE = 1280
# Near-side shoulder / hip; the nose is landmark 0 on either side.
_NEAR = {"left": (11, 23), "right": (12, 24)}


def _xy(lm: Any) -> tuple[float, float] | None:
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if x is None or y is None or (isinstance(x, float) and math.isnan(x)) or (
            isinstance(y, float) and math.isnan(y)):
        return None
    return float(x), float(y)


def measure_clip_helmet(
    video_path: str, frame_data: list[dict[str, Any]], camera_side: str,
    samples: int = SAMPLES,
) -> dict[str, Any] | None:
    """Median helmet-above-back over ``samples`` frames of the clip, or None.

    Returns ``value`` (% of torso length), ``check`` (the reading flagged
    itself on most of the frames: the helmet tail lies flush on the back and
    the top-of-back point may sit on it), ``values`` and ``frame_idx`` (one
    per measured frame).
    """
    import cv2
    import mediapipe as mp

    from app.services.video_analysis.photo_analyzer import _segmentation_mask_supported

    if camera_side not in _NEAR or not _segmentation_mask_supported():
        return None
    usable = [fd for fd in frame_data
              if fd.get("normalized_landmarks") and fd.get("frame_idx") is not None]
    if not usable:
        return None
    step = max(1, len(usable) // samples)
    picks = usable[step // 2::step][:samples]

    from app.core.config import settings
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        RunningMode,
    )

    landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(settings.model_path)),
        running_mode=RunningMode.IMAGE, num_poses=1, output_segmentation_masks=True,
    ))
    cap = cv2.VideoCapture(video_path)
    sh_i, hp_i = _NEAR[camera_side]
    readings: list[tuple[float, bool, int]] = []
    try:
        for fd in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fd["frame_idx"])
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            k = min(1.0, MAX_LONG_EDGE / max(h, w))
            if k < 1.0:
                frame = cv2.resize(frame, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not res.segmentation_masks:
                continue
            nl = fd["normalized_landmarks"]
            pts = [_xy(nl[i]) for i in (sh_i, hp_i, 0)]
            if any(p is None for p in pts):
                continue
            got = helmet_above_back(top_contour(res.segmentation_masks[0].numpy_view()),
                                    pts[0], pts[1], pts[2], w / h)
            if got is not None:
                readings.append((got["value"], bool(got.get("check")), int(fd["frame_idx"])))
    finally:
        cap.release()
        landmarker.close()
    if not readings:
        logger.info("CLIP_HELMET_NONE", frames=len(picks))
        return None
    values = [r[0] for r in readings]
    flagged = sum(r[1] for r in readings)
    out = {
        "value": round(median(values), 1),
        "check": flagged * 2 > len(readings),
        "values": values,
        "frame_idx": [r[2] for r in readings],
    }
    logger.info("CLIP_HELMET", **{k: v for k, v in out.items() if k != "frame_idx"})
    return out
