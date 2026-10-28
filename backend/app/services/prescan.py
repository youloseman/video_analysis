"""Pre-scan: what a clip shows, read from a few frames before it is uploaded.

On a phone the footage comes first now, and the choices that used to come
before it -- run or bike, and on a bike the position -- are proposed from the
footage itself. The browser cuts a handful of small frames out of the clip it
already holds, this module finds a pose in each and says what they look like.

Only the two things the picture can show are proposed:

* **sport** -- a runner stands nearly upright with the hands by the body; a
  rider leans over the bike with the hands out on the bars.
* **position family** on a bike -- aero (forearms resting on pads, elbows
  bent near a right angle), road (arms long to the hoods or drops) or upright.

What the picture cannot show is left to the athlete: time trial against
triathlon, hoods against drops. The answer carries a confidence, and the app
only fills a choice in when the frames agree; otherwise it asks.
"""
from __future__ import annotations

import math
from statistics import median
from typing import Any

# MediaPipe pose indices.
NOSE = 0
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28

MIN_VIS = 0.5


def _pt(lms: Any, i: int, w: int, h: int) -> tuple[float, float, float]:
    p = lms[i]
    return p.x * w, p.y * h, float(getattr(p, "visibility", 1.0) or 0.0)


def _angle(a: tuple, b: tuple, c: tuple) -> float:
    """Included angle at b, degrees."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*v1) * math.hypot(*v2)
    if n == 0:
        return float("nan")
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / n))
    return math.degrees(math.acos(cos))


def frame_features(lms: Any, w: int, h: int) -> dict[str, float] | None:
    """The handful of numbers one frame contributes, or None without a body.

    Distances are in trunk lengths so the size of the rider in the frame does
    not matter; ``facing`` is +1 when the athlete faces right in the image.
    """
    sh = [_pt(lms, i, w, h) for i in (L_SH, R_SH)]
    hp = [_pt(lms, i, w, h) for i in (L_HIP, R_HIP)]
    if min(p[2] for p in sh + hp) < MIN_VIS * 0.6:
        return None
    msh = ((sh[0][0] + sh[1][0]) / 2, (sh[0][1] + sh[1][1]) / 2)
    mhp = ((hp[0][0] + hp[1][0]) / 2, (hp[0][1] + hp[1][1]) / 2)
    trunk = math.hypot(msh[0] - mhp[0], msh[1] - mhp[1])
    if trunk < 1:
        return None
    nose = _pt(lms, NOSE, w, h)
    facing = 1.0 if nose[0] >= mhp[0] else -1.0
    # 0 = flat, 90 = upright
    trunk_deg = math.degrees(math.atan2(mhp[1] - msh[1], abs(msh[0] - mhp[0])))

    # The arm and leg the camera sees best.
    arm = max(((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)),
              key=lambda c: sum(_pt(lms, i, w, h)[2] for i in c))
    s, e, wr = (_pt(lms, i, w, h) for i in arm)
    out: dict[str, float] = {"trunk_deg": trunk_deg}
    if min(s[2], e[2], wr[2]) >= MIN_VIS * 0.6:
        out["wrist_fwd"] = facing * (wr[0] - s[0]) / trunk
        out["wrist_down"] = (wr[1] - s[1]) / trunk
        out["elbow_deg"] = _angle(s, e, wr)
        # forearm above the horizontal, degrees: + when the hand is higher
        out["forearm_deg"] = math.degrees(math.atan2(e[1] - wr[1], abs(wr[0] - e[0])))
    backs, drops = [], []
    for hi, ai in ((L_HIP, L_AN), (R_HIP, R_AN)):
        hx, hy, hv = _pt(lms, hi, w, h)
        ax, ay, av = _pt(lms, ai, w, h)
        if min(hv, av) >= MIN_VIS * 0.6:
            backs.append(-facing * (ax - hx) / trunk)
            drops.append((ay - hy) / trunk)
    if backs:
        out["ankle_back"] = max(backs)
        out["ankle_drop"] = max(drops)
    out["hip_x"] = mhp[0] / trunk
    out["hip_y"] = mhp[1] / trunk
    return out


def summarize(frames: list[dict[str, float]]) -> dict[str, float]:
    """Medians across frames, plus how much the hips moved and how far back a
    foot ever went (a runner's back-swing shows in one frame, not the median)."""
    keys = {k for f in frames for k in f}
    s = {k: median(f[k] for f in frames if k in f) for k in keys
         if k not in ("hip_x", "hip_y", "ankle_back")}
    backs = [f["ankle_back"] for f in frames if "ankle_back" in f]
    if backs:
        s["ankle_back_max"] = max(backs)
    ys = [f["hip_y"] for f in frames]
    if len(ys) >= 2:
        s["hip_y_range"] = max(ys) - min(ys)
    s["n"] = len(frames)
    return s


# One photo of a runner (see classify): upright, a foot swung back behind.
PHOTO_RUN_TRUNK = 80.0
PHOTO_RUN_BACK = 0.3


def _valid(f: dict[str, float]) -> bool:
    # A trunk "below the horizon" is a pose turned upside down by a bad
    # detection, not a rider; such a frame says nothing (5 of 156 race photos).
    return -5.0 <= f["trunk_deg"] <= 95.0


def classify(frames: list[dict[str, float]], *, photo: bool = False) -> dict[str, Any]:
    """What the frames show: ``sport`` (run / bike / None) and on a bike the
    position ``family`` (aero / road / upright / None). None means "could
    not tell", and the app then asks instead of guessing.

    Thresholds from 50 local clips and 156 race photos with a visible arm
    (2026-10-21): runners 72-89 deg trunk with a foot 0.8-1.1 trunk lengths
    behind the hip; riders 12-45 deg with it at most 0.3. Aero: elbow 65-110
    with the hand above the elbow; road on the hoods: elbow 130-157 with the
    hand 25-52 deg below it.

    A photo is a bike unless it is plainly a runner: trunk at 80 deg or more
    with a foot at least 0.3 trunk lengths behind the hip. On 27 running photos
    (runphoto2026, 2026-10-28: treadmill, track, street, every stride instant)
    the trunk sat at 83-90 and the back foot at 0.32-1.20; of ~150 bike photos
    the few as upright as that had the foot ahead of the hip (-0.17, -0.26).
    The app proposes "Run · photo" only where the account has it.
    """
    good = [f for f in frames if _valid(f)]
    out: dict[str, Any] = {"sport": None, "family": None, "frames": len(good)}
    if not good or (not photo and len(good) < 3):
        return out
    s = summarize(good)
    out["measured"] = {k: round(v, 2) for k, v in s.items() if k != "n"}
    trunk = s["trunk_deg"]
    back = s.get("ankle_back_max", 0.0)
    fwd = s.get("wrist_fwd", 0.0)
    upright_share = sum(f["trunk_deg"] >= 60 for f in good) / len(good)

    if photo:
        out["sport"] = "run" if (trunk >= PHOTO_RUN_TRUNK and back >= PHOTO_RUN_BACK) else "bike"
    elif trunk >= 65 and back >= 0.6 and upright_share >= 2 / 3:
        out["sport"] = "run"
    elif trunk <= 45 and upright_share <= 1 / 3:
        out["sport"] = "bike"
    elif trunk <= 70 and fwd >= 0.35 and back < 0.55:
        # Sitting up (a city bike), hands still out on the bars, and no frame
        # with a foot swung back behind the hip the way a runner's goes.
        out["sport"] = "bike"
    if out["sport"] != "bike" or "elbow_deg" not in s:
        return out

    elbow, forearm = s["elbow_deg"], s["forearm_deg"]
    if elbow <= 115 and forearm >= -10:
        out["family"] = "aero"
    elif trunk >= 50:
        out["family"] = "upright"
    elif elbow >= 125 and forearm <= -15:
        out["family"] = "road"
    return out


# ---- pose on the frames ---------------------------------------------------

def _model_path() -> str | None:
    from pathlib import Path

    from app.core.config import settings

    for p in (settings.model_path,
              Path(__file__).parent / "video_analysis" / "models" / "pose_landmarker_heavy.task",
              Path("/app/models/pose_landmarker_heavy.task")):
        if Path(p).exists():
            return str(p)
    return None


def scan_images(images: list[bytes], *, photo: bool = False) -> dict[str, Any]:
    """Find a pose in each small JPEG and classify them together. One
    landmarker for all of them: building the heavy model is most of the cost
    of a single detection."""
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    path = _model_path()
    if not path:
        return {"sport": None, "family": None, "frames": 0}
    opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=path),
                                 running_mode=RunningMode.IMAGE, num_poses=1)
    feats: list[dict[str, float]] = []
    with PoseLandmarker.create_from_options(opts) as lm:
        for raw in images:
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if res.pose_landmarks:
                f = frame_features(res.pose_landmarks[0], img.shape[1], img.shape[0])
                if f:
                    feats.append(f)
    return classify(feats, photo=photo)
