"""How far the helmet sits above the line of the back, read off the silhouette.

Replaces the photo "Head position" score (``calculate_head_alignment_2d``),
which asked whether the ear landmark sat below a line through the hip and
shoulder JOINT CENTRES. That line runs inside the body -- the shoulder joint is
well below the top of the back -- and the ear sits at cheekbone height, so the
answer was "yes" for every aero rider: all fifteen Kona 2026 pros scored
100/100, and a reader rightly called it meaningless (2026-10-11). It also never
saw the helmet, which is what actually sticks up into the wind.

This reads the person mask the pose model already produces (MediaPipe >= 1.0;
the float-mask accessor crashes the process in 0.10.35, on Linux too):

1. The top edge of the silhouette, one value per image column.
2. A straight line fitted to that edge along the mid-back, 20-75 % of the way
   from hip to shoulder. The upper end stops short of the shoulder because an
   aero helmet's tail lies on the upper back and would drag the line up.
3. The line extended forward over the head. The reading is how far the
   highest point of the head/helmet rises above it, as a percentage of torso
   length (hip to shoulder). Negative = the helmet stays under the back line.

Measured, not graded: no fit source gives a target for it. What the rider gets
alongside it is the field it was calibrated on -- the Kona 2026 professional
positions, -3 % to +16 %, median +4 %.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Columns kept of the silhouette's top edge. 512 resolves a helmet on any
# photo the app accepts (a 1179 px wide frame -> 2.3 px per column) and keeps
# the signed pose blob a few kilobytes.
CONTOUR_COLUMNS = 512

# Mid-back span the line is fitted on, as fractions of hip -> shoulder.
BACK_SPAN = (0.20, 0.75)
# Head zone: from the shoulder forward to the nose plus this much torso.
HEAD_REACH = 0.15
# A back edge that is not close to straight is not a back (a jersey flapping,
# an arm or a bottle merged into the silhouette): no reading. The fifteen
# Kona positions fitted within 0.4-1.6 % of torso length.
MAX_FIT_RMS = 0.03
MIN_BACK_COLUMNS = 8

# The Kona 2026 pro field this was calibrated on (n=15, hand-checked points).
KONA_PRO_RANGE = (-3.0, 16.0)
KONA_PRO_MEDIAN = 4.0


def top_contour(mask: np.ndarray, columns: int = CONTOUR_COLUMNS) -> list[float | None]:
    """The silhouette's top edge: per column, the highest person pixel as a
    fraction of image height (0 = top), or None where the column is empty."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[..., 0]
    m = m > 0.5 if m.dtype != bool else m
    h, w = m.shape
    has = m.any(axis=0)
    first = m.argmax(axis=0)
    out: list[float | None] = []
    for j in range(columns):
        x0 = int(j * w / columns)
        x1 = max(x0 + 1, int((j + 1) * w / columns))
        sel = has[x0:x1]
        if not sel.any():
            out.append(None)
            continue
        out.append(round(float(first[x0:x1][sel].min()) / h, 4))
    return out


def helmet_above_back(
    contour: list[float | None] | None,
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    nose: tuple[float, float],
    aspect: float,
) -> dict[str, Any] | None:
    """Helmet height above the back line, % of torso length -- or None.

    ``shoulder``/``hip``/``nose`` are normalized image coordinates (x of the
    width, y of the height) of the near side; ``aspect`` is width / height, so
    x * aspect and y share one unit. None when the silhouette cannot carry the
    reading: no contour, a back edge too short or too crooked to be a back, no
    head in front of the shoulder, or a helmet cut off by the top of the frame.
    """
    if not contour or aspect <= 0:
        return None
    n = len(contour)
    sx, sy = shoulder
    hx, hy = hip
    torso = math.hypot((sx - hx) * aspect, sy - hy)
    if torso <= 1e-6 or abs(sx - hx) * aspect < 0.25 * torso:
        return None  # torso near vertical in the image: no back to run a line along
    facing = 1.0 if sx > hx else -1.0

    def col(x: float) -> int:
        return int(min(n - 1, max(0, math.floor(x * n))))

    def x_of(j: int) -> float:
        return (j + 0.5) / n

    a, b = (col(hx + f * (sx - hx)) for f in BACK_SPAN)
    back = [(x_of(j) * aspect, contour[j]) for j in range(min(a, b), max(a, b) + 1)
            if contour[j] is not None]
    span = abs(b - a) + 1
    if len(back) < max(MIN_BACK_COLUMNS, 0.6 * span):
        return None
    xs = np.array([p[0] for p in back])
    ys = np.array([p[1] for p in back])
    k, c = np.polyfit(xs, ys, 1)
    rms = float(np.sqrt(np.mean((ys - (k * xs + c)) ** 2)))
    if rms / torso > MAX_FIT_RMS:
        return None

    reach = nose[0] + facing * HEAD_REACH * torso / aspect
    h0, h1 = sorted((col(sx), col(reach)))
    best = None
    for j in range(h0, h1 + 1):
        y = contour[j]
        if y is None:
            continue
        X = x_of(j) * aspect
        above = (k * X + c) - y
        if best is None or above > best[0]:
            best = (above, j, y)
    if best is None:
        return None
    above, j, y = best
    if y <= 0.002:
        return None  # the silhouette runs off the top of the frame: helmet cut off

    def line_y(x: float) -> float:
        return float(k * x * aspect + c)

    x_back = x_of(min(a, b) if facing > 0 else max(a, b))
    return {
        "value": round(float(above) / torso * 100.0, 1),
        "line": [[round(x_back, 4), round(line_y(x_back), 4)],
                 [round(x_of(j), 4), round(line_y(x_of(j)), 4)]],
        "head_point": [round(x_of(j), 4), round(float(y), 4)],
        "fit_rms_pct": round(rms / torso * 100.0, 2),
    }
