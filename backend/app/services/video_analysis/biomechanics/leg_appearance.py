"""What each leg LOOKS like, so a crossing can be broken by pixels.

WHY. Run leg identity is decided per link: between two frames the labels
either stayed on their legs or swapped, and the resolver picks the cheaper
matching. Its evidence is geometry -- how far each joint moved -- and geometry
goes blind at exactly the moment identity is decided. When the legs cross,
both matchings cost nearly the same, and the decision falls to hysteresis.
Measured on 2026-08-29, that is where the swaps come from: on one of Artur's
clips the labels ended up on the wrong leg for a stride at a time.

Pixels do not go blind there. Two legs that occupy the same place in the image
still look different -- one is in front, lit and sharp; the other is behind,
shadowed and partly hidden -- and on a real athlete they differ outright
(a tattooed calf, a different sock). So each leg gets a small appearance
descriptor, and the link cost gains a term the crossing cannot flatten.

WHAT IS STORED, AND WHY IT IS TINY. Two patches down the shin, greyscale,
shrunk to ``PATCH_N`` square and z-normalised: about a kilobyte per frame for
both legs. The prototype matched full 60x60 patches, which would have carried
tens of megabytes through the pipeline for the same answer.

The shin, not the thigh: it is the least occluded part of the leg through the
stride, and it carries whatever actually distinguishes one leg from the other
(sock, tattoo, shoe collar). The radius is a fraction of the shin's own
length, so the descriptor sees the same anatomy whether the athlete fills the
frame or sits in a corner of it.

Z-normalising each patch is what makes the comparison about texture rather
than exposure: the near leg is often brighter than the far one, and a raw
brightness difference would let the descriptor "identify" legs by lighting
that changes with the stride.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Landmarks the descriptor is anchored on, per side: (knee, ankle).
_SHIN = {"left": (25, 27), "right": (26, 28)}

# Where down the shin to sample. Below the knee's soft tissue, above the shoe.
_STATIONS = (0.40, 0.75)

# Half-width of a patch, as a fraction of the shin's length in this frame.
_PATCH_FRAC = 0.28
_PATCH_MIN_PX = 8

# Descriptor resolution. Measured against the swap metric: the full-resolution
# prototype and this size resolve the same crossings, and this one is small
# enough to ride along on every frame.
PATCH_N = 12


def _pt(landmarks: Any, idx: int, w: int, h: int) -> tuple[float, float] | None:
    if landmarks is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return None
    return (x * w, y * h)


def describe_legs(
    bgr_frame: Any, normalized_landmarks: Any,
) -> dict[str, list] | None:
    """Appearance descriptors for both legs, or None when they can't be taken.

    Called with the frame the detector just saw, so it costs a few crops and
    no second decode. Returns ``{"left": [...], "right": [...]}`` with each
    entry a list of ``PATCH_N x PATCH_N`` float32 arrays, or None for a leg
    whose shin is missing or too small to sample.
    """
    if bgr_frame is None or normalized_landmarks is None:
        return None
    try:
        import cv2
    except ImportError:  # pragma: no cover - cv2 is a hard dep of this path
        return None
    h, w = bgr_frame.shape[:2]
    if h < 16 or w < 16:
        return None

    out: dict[str, list] = {}
    grey = None
    for side, (knee_i, ankle_i) in _SHIN.items():
        knee = _pt(normalized_landmarks, knee_i, w, h)
        ankle = _pt(normalized_landmarks, ankle_i, w, h)
        if knee is None or ankle is None:
            out[side] = None
            continue
        shin = math.dist(knee, ankle)
        r = int(max(_PATCH_MIN_PX, round(_PATCH_FRAC * shin)))
        if grey is None:
            grey = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        patches = []
        for t in _STATIONS:
            cx = int(round(knee[0] + t * (ankle[0] - knee[0])))
            cy = int(round(knee[1] + t * (ankle[1] - knee[1])))
            x0, x1 = max(0, cx - r), min(w, cx + r)
            y0, y1 = max(0, cy - r), min(h, cy + r)
            if x1 - x0 < 6 or y1 - y0 < 6:
                patches = None
                break
            patch = cv2.resize(
                grey[y0:y1, x0:x1], (PATCH_N, PATCH_N),
                interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
            std = float(patch.std())
            if std < 1e-3:
                # A flat patch carries no texture to match on -- usually the
                # leg left the frame and this is a crop of blown-out wall.
                patches = None
                break
            patches.append((patch - patch.mean()) / std)
        out[side] = patches
    if out.get("left") is None and out.get("right") is None:
        return None
    return out


def patch_distance(a: list | None, b: list | None) -> float | None:
    """Mean absolute difference between two legs' descriptors, or None."""
    if not a or not b or len(a) != len(b):
        return None
    try:
        return float(np.mean([np.mean(np.abs(x - y)) for x, y in zip(a, b)]))
    except (TypeError, ValueError):
        return None


def link_costs(
    prev: dict[str, list] | None, cur: dict[str, list] | None,
) -> tuple[float, float] | None:
    """``(stay, cross)`` appearance cost between two frames, or None.

    ``stay`` matches each raw label to itself across the link, ``cross`` swaps
    them -- the same two hypotheses the geometry scores, so the two pieces of
    evidence can simply be added.

    Deliberately independent of the parity the resolver is choosing: a parity
    flip relabels BOTH frames, so the raw-label pairing is unchanged and this
    cost stays valid whichever branch the path is on.
    """
    if not prev or not cur:
        return None
    stay_l = patch_distance(prev.get("left"), cur.get("left"))
    stay_r = patch_distance(prev.get("right"), cur.get("right"))
    cross_l = patch_distance(prev.get("left"), cur.get("right"))
    cross_r = patch_distance(prev.get("right"), cur.get("left"))
    if None in (stay_l, stay_r, cross_l, cross_r):
        return None
    return stay_l + stay_r, cross_l + cross_r
