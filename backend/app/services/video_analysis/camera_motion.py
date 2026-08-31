"""Did the camera move, and enough to matter?

Vertical oscillation is measured as the rise and fall of the hips in the
picture. If the person holding the phone is also rising and falling, that
lands in the same number -- and it is not a slow drift a detrend removes,
because a camera held by someone standing still bobs at roughly the frequency
the runner's steps do. The two signals are on top of each other.

Whether that actually happens to Flapp's clips is not known. Neither clip in
the repo has a moving camera, so building a compensator would be fixing a
problem nobody has demonstrated. This module measures instead, and reports;
the same shape as the residual-analysis diagnostic in ``butterworth_filter``,
and for the same reason. When enough clips carry a reading, the question of
whether to compensate answers itself.

How
---
The method is Kinovea's, down to the constants (``CameraTracker.cs``): ORB
features, brute-force matched with a symmetry cross-check, and a homography
between consecutive frames estimated with USAC_MAGSAC. Two departures, both
forced by what this pipeline is for:

* **The athlete is masked out.** Kinovea masks static overlays out of a scene
  that is otherwise all background; here the largest moving thing in frame is
  the subject, and matching features on a running person would measure the
  runner rather than the camera. Pose landmarks give the mask for free.
* **Only the translation is kept.** A full homography can express pan, tilt,
  roll, zoom and perspective; what contaminates vertical oscillation is the
  up-and-down, so that is what comes out. The rest is thrown away rather than
  reported as precision nobody asked for.

The headline number is deliberately not "the camera moved N pixels" -- that
means nothing without knowing how big the athlete was. It is the camera's
vertical movement **as a fraction of the hip movement being measured**, which
is exactly the proportion by which the reading could be wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Kinovea's matcher settings (Kinovea.ScreenManager/Measurement/CameraMotion).
ORB_FEATURES = 500
RANSAC_REPROJ_PX = 1.25
RANSAC_CONFIDENCE = 0.995
RANSAC_MAX_ITERS = 2000

# Below this many inlier matches the homography is a guess, and a guess about
# camera motion is worse than admitting the frame pair could not be read.
MIN_INLIERS = 12

# How much of the frame around the athlete to exclude. The pose bbox clips the
# silhouette; hair, hands and a trailing foot sit outside it, and a feature on
# a moving limb is the one thing that must not enter the estimate. The mask
# also runs down to the bottom edge -- see ``_subject_mask``.
SUBJECT_MARGIN = 0.12

# Frames actually compared. Camera motion is low-frequency compared with the
# frame rate, and ORB on every frame of a 500-frame clip is seconds of CPU for
# a diagnostic. Consecutive SAMPLED frames are still consecutive in time, just
# further apart.
MAX_PAIRS = 60

# Long edge the frames are shrunk to before ORB runs on them. Measured on a
# 1072x1920 clip: detectAndCompute costs 62.3 ms per frame at full size and
# 10.7 ms at half -- and this module was 9 of the 38 seconds a running
# analysis took, for a diagnostic.
#
# Shrinking is safe here in a way it would not be for pose estimation, because
# what comes out is a rigid translation of the whole frame: halving the picture
# halves the measured shift exactly, and it is scaled back below. What it does
# cost is features -- ORB has fewer pixels to find corners in -- so 960 is
# chosen to stay well clear of MIN_INLIERS on ordinary backgrounds rather than
# to be as small as possible. Frames already at or under this are left alone;
# nothing is ever upscaled.
MOTION_MAX_LONG_EDGE = 960

# Vertical camera movement as a share of the hip movement being measured.
# Under the first, the reading is unaffected in any way that matters; over the
# second, the number is substantially the camera's rather than the athlete's.
SHARE_WARN = 0.15
SHARE_BAD = 0.35

# Where the frame is split for the rigidity check. Above the line is the part
# of a scene that holds still -- ceilings, walls, distant background; below it
# is the ground the athlete is on and any equipment standing on it.
SPLIT_LINE = 0.45

# How far the two halves may disagree before the scene is declared non-rigid.
# Measured on a treadmill clip: the halves reported 157 px and 3 px of bounce,
# because the machine vibrates and the building does not. A camera genuinely
# shaking moves both halves together, so honest disagreement stays small.
RIGIDITY_TOLERANCE_PX = 8.0
RIGIDITY_TOLERANCE_RATIO = 2.5


def _subject_mask(
    cv2_mod: Any, shape: tuple[int, int], landmarks: Any,
) -> "np.ndarray | None":
    """255 where features may be taken, 0 over the athlete."""
    height, width = shape[:2]
    xs, ys = [], []
    for lm in landmarks or []:
        x, y = getattr(lm, "x", None), getattr(lm, "y", None)
        vis = getattr(lm, "visibility", 0.0) or 0.0
        if x is None or y is None or vis < 0.3:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < 4:
        return None
    mask = np.full((height, width), 255, dtype=np.uint8)
    mx = (max(xs) - min(xs)) * SUBJECT_MARGIN + SUBJECT_MARGIN * 0.5
    my = (max(ys) - min(ys)) * SUBJECT_MARGIN + SUBJECT_MARGIN * 0.5
    x0 = int(max(0.0, min(xs) - mx) * width)
    x1 = int(min(1.0, max(xs) + mx) * width)
    y0 = int(max(0.0, min(ys) - my) * height)
    # Down to the bottom edge rather than to the feet. What is under a runner
    # is the surface they are running on, and on a treadmill that surface
    # moves -- belt texture sliding past would be matched as the camera
    # sliding the other way, and a machine that reports a shaking tripod is
    # worse than one that reports nothing. Outdoors this costs a strip of
    # ground there is no shortage of.
    mask[y0:height, x0:x1] = 0
    # A mask that leaves almost no background is not a mask, it is a refusal.
    if float(np.count_nonzero(mask)) / mask.size < 0.25:
        return None
    return mask


def _fit_translation(
    cv2_mod: Any, src: Any, dst: Any, width: int, height: int,
) -> tuple[float, float] | None:
    """Camera translation from one set of matched points, or None."""
    if len(src) < MIN_INLIERS:
        return None
    homography, inliers = cv2_mod.findHomography(
        src, dst, cv2_mod.USAC_MAGSAC, RANSAC_REPROJ_PX,
        maxIters=RANSAC_MAX_ITERS, confidence=RANSAC_CONFIDENCE,
    )
    if homography is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return None
    # Where the centre of the frame went. Reading the translation off the
    # matrix elements directly would ignore the perspective terms; mapping a
    # point through it does not.
    centre = np.float32([[[width / 2.0, height / 2.0]]])
    moved = cv2_mod.perspectiveTransform(centre, homography)[0][0]
    return (float(moved[0] - width / 2.0), float(moved[1] - height / 2.0))


def _translation_between(
    cv2_mod: Any, prev_gray: Any, gray: Any,
    prev_mask: Any, mask: Any, orb: Any, matcher: Any,
) -> tuple[float, float, float | None, float | None] | None:
    """Frame-to-frame camera translation in pixels, or None if unreadable.

    Returns ``(dx, dy, dy_upper, dy_lower)``. The last two are the same
    measurement taken from the top and bottom of the frame separately, and
    exist to catch a scene that is not rigid: a treadmill vibrating under a
    runner moves its own half of the picture while the building holds still,
    and a single homography over both averages the two into a camera shake
    that never happened. Either may be None when that half held too few
    points to fit.
    """
    kp1, des1 = orb.detectAndCompute(prev_gray, prev_mask)
    kp2, des2 = orb.detectAndCompute(gray, mask)
    if des1 is None or des2 is None or len(kp1) < MIN_INLIERS or len(kp2) < MIN_INLIERS:
        return None
    matches = matcher.match(des1, des2)
    if len(matches) < MIN_INLIERS:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    height, width = gray.shape[:2]

    whole = _fit_translation(cv2_mod, src, dst, width, height)
    if whole is None:
        return None

    # The same fit over the top and the bottom of the frame on their own. One
    # ORB pass already produced the points; splitting them costs two more
    # RANSAC fits and is what separates "the camera moved" from "something in
    # the scene did".
    ys = dst[:, 0, 1]
    upper = ys < height * SPLIT_LINE
    lower = ~upper
    dy_upper = dy_lower = None
    if int(upper.sum()) >= MIN_INLIERS:
        fit = _fit_translation(cv2_mod, src[upper], dst[upper], width, height)
        dy_upper = fit[1] if fit else None
    if int(lower.sum()) >= MIN_INLIERS:
        fit = _fit_translation(cv2_mod, src[lower], dst[lower], width, height)
        dy_lower = fit[1] if fit else None

    return (whole[0], whole[1], dy_upper, dy_lower)


def estimate_camera_motion(
    video_path: str,
    frame_data_list: list[dict[str, Any]],
    *,
    hip_amplitude_norm: float | None = None,
) -> dict[str, Any] | None:
    """Measure how much the CAMERA moved during the clip.

    ``hip_amplitude_norm`` is the athlete's hip rise-and-fall in normalized
    units, which is what turns a pixel count into the only figure worth
    reporting: what share of the measured oscillation could be the camera.

    Returns None when the clip cannot be read at all. Never raises.
    """
    try:
        import cv2
    except Exception as e:  # noqa: BLE001
        logger.warning("CAMERA_MOTION_NO_CV2", err=str(e))
        return None

    if len(frame_data_list) < 4:
        return None

    step = max(1, len(frame_data_list) // MAX_PAIRS)
    wanted = [frame_data_list[i] for i in range(0, len(frame_data_list), step)]
    targets = {fd["frame_idx"]: fd for fd in wanted}

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    dxs: list[float] = []
    dys: list[float] = []
    dy_upper: list[float] = []
    dy_lower: list[float] = []
    unreadable = 0
    prev_gray = prev_mask = None
    # The frame height in ORIGINAL pixels, taken from the first frame actually
    # decoded. It used to be read by opening the video a second time purely for
    # CAP_PROP_FRAME_HEIGHT, after the decode loop had already had every frame
    # in hand.
    height = 1
    scale = 1.0

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        idx = 0
        last = max(targets)
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            fd = targets.get(idx)
            idx += 1
            if fd is None:
                continue
            if height == 1:
                height = frame.shape[0] or 1
                long_edge = max(frame.shape[0], frame.shape[1])
                if long_edge > MOTION_MAX_LONG_EDGE:
                    scale = MOTION_MAX_LONG_EDGE / long_edge
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale < 1.0:
                # INTER_AREA: the correct filter for shrinking, and the one that
                # matters here -- a nearest-neighbour shrink aliases exactly the
                # high-frequency detail ORB looks for corners in.
                gray = cv2.resize(
                    gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA,
                )
            mask = _subject_mask(
                cv2, gray.shape, fd.get("normalized_landmarks"),
            )
            if prev_gray is not None:
                shift = _translation_between(
                    cv2, prev_gray, gray, prev_mask, mask, orb, matcher,
                )
                if shift is None:
                    unreadable += 1
                else:
                    # Back to original pixels immediately, so everything below
                    # keeps the units it was calibrated in -- RIGIDITY_TOLERANCE_PX
                    # was measured at full size on a real treadmill clip, and a
                    # tolerance whose meaning depends on the input resolution
                    # would be a different check on every phone.
                    back = 1.0 / scale
                    dxs.append(shift[0] * back)
                    dys.append(shift[1] * back)
                    if shift[2] is not None and shift[3] is not None:
                        dy_upper.append(shift[2] * back)
                        dy_lower.append(shift[3] * back)
            prev_gray, prev_mask = gray, mask
    finally:
        cap.release()

    if len(dys) < 3:
        logger.info("CAMERA_MOTION_UNREADABLE", pairs=len(dys), skipped=unreadable)
        return None

    dy = np.array(dys, dtype=np.float64)
    dx = np.array(dxs, dtype=np.float64)
    # Cumulative path = a deliberate pan; the per-step spread = the shake.
    pan_px = float(np.abs(np.cumsum(dx))[-1])
    tilt_px = float(np.abs(np.cumsum(dy))[-1])
    # Peak-to-peak of the accumulated vertical position, with the steady drift
    # removed: what is left is the bounce that lands on top of the athlete's.
    bounce_px = _bounce(dy)

    # Is one rigid motion even the right description of this scene? A
    # treadmill vibrating under a runner moves its half of the picture while
    # the building holds still, and a homography over both splits the
    # difference into a camera shake that never happened. Measured on exactly
    # that clip: 157 px of "bounce" from the mixed fit, 3 px from the ceiling.
    rigid = True
    if len(dy_upper) >= 3 and len(dy_lower) >= 3:
        up = _bounce(np.array(dy_upper, dtype=np.float64))
        low = _bounce(np.array(dy_lower, dtype=np.float64))
        gap = abs(up - low)
        rigid = gap <= RIGIDITY_TOLERANCE_PX or (
            gap <= RIGIDITY_TOLERANCE_RATIO * min(up, low)
        )

    share = None
    if rigid and hip_amplitude_norm and hip_amplitude_norm > 1e-6:
        hip_px = hip_amplitude_norm * height
        if hip_px > 1e-6:
            share = round(bounce_px / hip_px, 3)

    result = {
        "pairs": len(dys),
        "unreadable_pairs": unreadable,
        "pan_px": round(pan_px, 1),
        "tilt_px": round(tilt_px, 1),
        "vertical_bounce_px": round(bounce_px, 1),
        "frame_height_px": height,
        "vertical_share_of_hip_motion": share,
        "scene_rigid": rigid,
        # Without a rigid scene there is no single camera motion to report, and
        # blaming the athlete's tripod for a machine's vibration is worse than
        # saying nothing. "unknown" drops the row from the capture report.
        "verdict": _verdict(share) if rigid else "unknown",
    }
    logger.info("CAMERA_MOTION", **result)
    return result


def _bounce(steps: np.ndarray) -> float:
    """Peak-to-peak of the accumulated position with the steady drift removed.

    What is left after detrending is the shake that lands on top of the
    athlete's own rise and fall; the drift itself is a pan or a tilt, which is
    reported separately and is harmless.
    """
    if steps.size < 2:
        return 0.0
    track = np.cumsum(steps)
    detrended = track - np.linspace(track[0], track[-1], track.size)
    return float(np.percentile(detrended, 95) - np.percentile(detrended, 5))


def _verdict(share: float | None) -> str:
    if share is None:
        return "unknown"
    if share >= SHARE_BAD:
        return "bad"
    if share >= SHARE_WARN:
        return "warn"
    return "good"


__all__ = [
    "SHARE_BAD",
    "SHARE_WARN",
    "estimate_camera_motion",
]
