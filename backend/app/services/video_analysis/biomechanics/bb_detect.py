"""Find the crank axle in the picture, so the athlete only has to check it.

The pose model knows nothing about the bicycle, and a guess from the leg
alone (the hip-over-BB angle the field says is typical) only repeats the
field back: on the hip/knee study's 72 hand-checked photos it put the axle
a median 14 % of the hip-to-crank distance away (~12 cm). The picture does
better, with two plain circle searches (Hough) scaled to the rider's leg:

* **The wheels.** Two circles of the same size, level, ~2.9 radii apart,
  either side of the hip and below it. On a road/tri bike the crank axle
  sits ~1.17 wheel radii ahead of the rear axle along the axle line and
  ~0.24 radii below it -- ~40 cm of chainstay and ~8 cm of bottom-bracket
  drop -- measured on the study's photos (leave-one-out). Alone: median
  error 1.0 deg of the hip-over-BB angle, but a wrong pair is a wrong axle.
* **The chainring / crank boss.** A circle 0.07-0.17 of the leg across,
  near the wheel prediction (or the leg guess when there are no wheels --
  a direct-drive trainer has no rear wheel). Confirmed this way: median
  0.6 deg, worst 2.9 deg on 50 photos; and on clips, where the median of the
  frames wipes the legs out and leaves the bike, 8 of 8 trainer clips.

Whatever it finds is a STARTING point the athlete confirms or drags; the
result records which (``c_source``). The wheel axles are offered as the
level only when the chainring confirmed the pair.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Crank axle in the rear-to-front axle frame, in wheel radii (study medians).
WHEEL_U = 1.17
WHEEL_V = 0.235  # below the axle line
_LEG_PX = 360.0  # every search runs on a copy scaled so thigh+shank is this long


def _circles(gray: Any, cv2: Any, rmin: float, rmax: float, *, dp: float, p2: int,
             min_dist: float, maxn: int) -> list[tuple[float, float, float]]:
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=110,
                         param2=p2, minRadius=int(rmin), maxRadius=int(rmax))
    if c is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in c[0][:maxn]]


def _wheel_pair(wheels, hip, facing):
    best, best_score = None, -1e9
    for i, a in enumerate(wheels):
        for b in wheels[i + 1:]:
            rm = (a[2] + b[2]) / 2
            if abs(a[2] - b[2]) > 0.18 * rm:
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            sep = math.hypot(dx, dy)
            if not (2.3 * rm < sep < 3.6 * rm) or abs(dy) > 0.25 * rm:
                continue
            if not (min(a[0], b[0]) < hip[0] < max(a[0], b[0])) or min(a[1], b[1]) < hip[1]:
                continue
            score = -abs(a[2] - b[2]) / rm - abs(dy) / rm - abs(sep / rm - 2.9)
            if score > best_score:
                best_score, best = score, (a, b)
    if best is None:
        return None
    a, b = best
    return (a, b) if (a[0] - b[0]) * facing < 0 else (b, a)  # rear, front


def _from_wheels(rear, front):
    rm = (rear[2] + front[2]) / 2
    ax, ay = front[0] - rear[0], front[1] - rear[1]
    n = math.hypot(ax, ay)
    ux, uy = ax / n, ay / n
    vx, vy = -uy, ux
    if vy < 0:  # the perpendicular that points DOWN the image, whichever way the bike faces
        vx, vy = -vx, -vy
    return (rear[0] + (WHEEL_U * ux + WHEEL_V * vx) * rm,
            rear[1] + (WHEEL_U * uy + WHEEL_V * vy) * rm)


def _leg_guess(hip, leg, facing):
    d = 0.86 * leg
    return (hip[0] + facing * d * math.cos(math.radians(86)), hip[1] + d * math.sin(math.radians(86)))


def detect_crank(
    image_bgr: Any, hip_px: tuple[float, float], leg_px: float, facing: int,
) -> dict[str, Any]:
    """The crank axle (and, when trustworthy, both wheel axles) in this picture.

    ``hip_px`` and ``leg_px`` (thigh + shank) come from the pose, in the
    image's own pixels; ``facing`` is +1 when the rider faces image-right.
    Returns normalized points and ``method`` = chainring / wheels / guess
    with ``confidence`` high / medium / low. Never raises on a picture.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    out: dict[str, Any] = {"method": "guess", "confidence": "low"}
    leg = max(float(leg_px or 0.0), 1.0)
    guess = _leg_guess(hip_px, leg, facing)
    try:
        k = _LEG_PX / leg
        small = cv2.resize(image_bgr, (max(1, int(w * k)), max(1, int(h * k))),
                           interpolation=cv2.INTER_AREA)
        gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
        wheels = _circles(gray, cv2, 0.33 * _LEG_PX, 0.52 * _LEG_PX, dp=1.5, p2=45,
                          min_dist=0.6 * _LEG_PX, maxn=12)
        rings = _circles(gray, cv2, 0.07 * _LEG_PX, 0.17 * _LEG_PX, dp=1.2, p2=28,
                         min_dist=0.05 * _LEG_PX, maxn=25)
    except Exception:  # noqa: BLE001 -- a picture we cannot search is a guess, not an error
        wheels, rings = [], []
        k = 1.0
    wheels = [(x / k, y / k, r / k) for x, y, r in wheels]
    rings = [(x / k, y / k, r / k) for x, y, r in rings]

    pair = _wheel_pair(wheels, hip_px, facing)
    prior = _from_wheels(*pair) if pair else guess
    reach = 0.30 * ((pair[0][2] + pair[1][2]) / 2) if pair else 0.30 * leg
    near = [c for c in rings if math.dist(c[:2], prior) < reach]
    ring = min(near, key=lambda c: math.dist(c[:2], prior)) if near else None

    if ring is not None:
        c, out["method"], out["confidence"] = ring[:2], "chainring", "high"
        out["ring_radius"] = round(ring[2] / max(w, h), 5)
    elif pair is not None:
        # Wheels without a chainring to confirm them were the weak case on the
        # study (median 4.7 deg): better than the leg guess, still "check it".
        c, out["method"], out["confidence"] = prior, "wheels", "low"
    else:
        c = guess

    # A body would not do this: hand it back as a guess.
    vx, vy = (hip_px[0] - c[0]) * facing, c[1] - hip_px[1]
    esa = 180.0 - math.degrees(math.atan2(vy, vx))
    dist = math.dist(hip_px, c) / leg
    if out["method"] != "guess" and not (0.6 <= dist <= 1.15 and 70.0 <= esa <= 106.0):
        c, out["method"], out["confidence"] = guess, "guess", "low"
        out.pop("ring_radius", None)

    clamp = lambda v: round(min(0.995, max(0.005, v)), 5)  # noqa: E731
    out["C"] = [clamp(c[0] / w), clamp(c[1] / h)]
    if pair is not None and out["method"] == "chainring":
        rear, front = pair
        out["RA"] = [clamp(rear[0] / w), clamp(rear[1] / h)]
        out["FA"] = [clamp(front[0] / w), clamp(front[1] / h)]
    return out


def leg_and_hip(landmarks: Any, w: int, h: int, side: str) -> tuple[tuple[float, float], float, int] | None:
    """Hip (px), thigh+shank (px) and facing from one set of normalized landmarks."""
    ids = (11, 23, 25, 27) if side == "left" else (12, 24, 26, 28)
    pts = []
    for i in ids:
        lm = landmarks[i]
        x, y = float(getattr(lm, "x", math.nan)), float(getattr(lm, "y", math.nan))
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        pts.append((x * w, y * h))
    S, H, K, A = pts
    leg = math.dist(H, K) + math.dist(K, A)
    if leg < 10:
        return None
    return H, leg, (1 if S[0] >= H[0] else -1)


def median_frame(video_path: str, n: int = 41) -> Any | None:
    """The median of ``n`` frames spread over the clip: the legs blur out,
    the bike (which does not move on a trainer) stays sharp."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return None
        frames = []
        for idx in np.linspace(0, total - 1, min(n, total)).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        if len(frames) < 5:
            return None
        return np.median(np.stack(frames), axis=0).astype(np.uint8)
    finally:
        cap.release()


__all__ = ["WHEEL_U", "WHEEL_V", "detect_crank", "leg_and_hip", "median_frame"]
