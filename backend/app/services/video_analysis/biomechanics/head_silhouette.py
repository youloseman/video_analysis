"""How far the helmet rises above the top of the back, read off the silhouette.

Replaces the photo "Head position" score (``calculate_head_alignment_2d``),
which asked whether the ear landmark sat below a line through the hip and
shoulder JOINT CENTRES. That line runs inside the body -- the shoulder joint is
well below the top of the back -- and the ear sits at cheekbone height, so the
answer was "yes" for every aero rider: all fifteen Kona 2026 pros scored
100/100, and a reader rightly called it meaningless (2026-10-11). It also never
saw the helmet, which is what actually sticks up into the wind.

The reference is HORIZONTAL, because the wind is (Artur, 2026-10-11). A first
version carried a line fitted along the back forward over the head; a back
that climbs toward the shoulders then lifted the line into the helmet and a
flat one left it low, so two riders with the same helmet height read -3 % and
+16 %.

From the person mask the pose model already produces (MediaPipe >= 1.0; the
float-mask accessor crashes the process in 0.10.35, on Linux too):

1. The top edge of the silhouette, one value per image column.
2. Where the back ends and the helmet begins. An aero helmet's tail often lies
   on the upper back, right where the back is highest, so the boundary is
   found, not assumed: walking forward from 75 % of hip -> shoulder, the first
   column where the edge climbs steeper than 45 deg (a back or a shoulder never
   does; a helmet's rear edge does) or sits STEP above a line fitted to the
   mid-back.
3. The top of the back = the highest edge from 20 % of hip -> shoulder up to
   that boundary. The top of the helmet = the highest edge from there to just
   past the nose. The reading is the vertical gap, as a percentage of torso
   length (hip to shoulder).

Where the tail lies flush on the back there is no step to find and the
boundary is soft: about +/-3 % on the Kona photos.

Measured, not graded: no fit source gives a target for it. What the rider gets
alongside it is the field it was calibrated on -- the Kona 2026 professional
positions (see KONA_PRO_RANGE).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# Columns kept of the silhouette's top edge. 512 resolves a helmet on any
# photo the app accepts (a 1179 px wide frame -> 2.3 px per column) and keeps
# the signed pose blob a few kilobytes.
CONTOUR_COLUMNS = 512

# Mid-back span: the back-surface fit, and where the back's top is searched from.
BACK_SPAN = (0.20, 0.75)
# Head zone reaches this much torso past the nose.
HEAD_REACH = 0.15
# The helmet starts where the edge sits this far above the mid-back fit...
STEP = 0.025
# ...or climbs steeper than 45 deg over this window (both in torso lengths).
CLIMB_WINDOW = 0.02
# A back edge that is not close to straight is not a back (a jersey flapping,
# an arm or a bottle merged into the silhouette): no reading. The fifteen
# Kona positions fitted within 0.4-1.6 % of torso length.
MAX_FIT_RMS = 0.03
MIN_BACK_COLUMNS = 8

# The Kona pro field this was calibrated on (n=15, 2026-10-11): 6.1-23.2 %,
# median 14.5, middle half 12-20. Moving the shoulder and hip points by hand
# changed no rider by more than 0.7 %.
KONA_PRO_RANGE = (6.0, 23.0)
KONA_PRO_MEDIAN = 14.5


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
    """Helmet top above the top of the back, % of torso length -- or None.

    ``shoulder``/``hip``/``nose`` are normalized image coordinates (x of the
    width, y of the height) of the near side; ``aspect`` is width / height, so
    x * aspect and y share one unit. None when the silhouette cannot carry the
    reading: no contour, a back edge too short or too crooked to be a back, no
    head in front of the back, or a helmet cut off by the top of the frame.
    """
    if not contour or aspect <= 0:
        return None
    n = len(contour)
    sx, sy = shoulder
    hx, hy = hip
    torso = math.hypot((sx - hx) * aspect, sy - hy)
    if torso <= 1e-6 or abs(sx - hx) * aspect < 0.25 * torso:
        return None  # torso near vertical in the image: no back to measure along
    step = 1 if sx > hx else -1

    def col(x: float) -> int:
        return int(min(n - 1, max(0, math.floor(x * n))))

    def x_of(j: int) -> float:
        return (j + 0.5) / n

    a, b = (col(hx + f * (sx - hx)) for f in BACK_SPAN)
    mid = [(x_of(j) * aspect, contour[j]) for j in range(min(a, b), max(a, b) + 1)
           if contour[j] is not None]
    if len(mid) < max(MIN_BACK_COLUMNS, 0.6 * (abs(b - a) + 1)):
        return None
    xs = np.array([p[0] for p in mid])
    ys = np.array([p[1] for p in mid])
    k, c = np.polyfit(xs, ys, 1)
    rms = float(np.sqrt(np.mean((ys - (k * xs + c)) ** 2)))
    if rms / torso > MAX_FIT_RMS:
        return None

    # Walk forward from 75 % to where the helmet begins.
    end = col(nose[0])
    win = max(1, round(CLIMB_WINDOW * torso / aspect * n))
    boundary = None
    j = b
    while (end - j) * step > 0:
        y = contour[j]
        ahead = contour[min(n - 1, max(0, j + step * win))]
        if y is not None:
            climbs = ahead is not None and (y - ahead) > CLIMB_WINDOW * torso
            if climbs or (k * x_of(j) * aspect + c) - y > STEP * torso:
                boundary = j
                break
        j += step
    if boundary is None:
        boundary = col(sx)

    def highest(j0: int, j1: int) -> tuple[float, int] | None:
        best = None
        for jj in range(min(j0, j1), max(j0, j1) + 1):
            y = contour[jj]
            if y is not None and (best is None or y < best[0]):
                best = (y, jj)
        return best

    back = highest(a, boundary)
    head = highest(boundary, col(nose[0] + step * HEAD_REACH * torso / aspect))
    if back is None or head is None:
        return None
    if head[0] <= 0.002:
        return None  # the silhouette runs off the top of the frame: helmet cut off

    back_y, back_j = back
    head_y, head_j = head
    return {
        "value": round(float(back_y - head_y) / torso * 100.0, 1),
        # A level line through the top of the back, out to the helmet's top.
        "line": [[round(x_of(back_j), 4), round(back_y, 4)],
                 [round(x_of(head_j), 4), round(back_y, 4)]],
        "head_point": [round(x_of(head_j), 4), round(head_y, 4)],
        "boundary_x": round(x_of(boundary), 4),
        "fit_rms_pct": round(rms / torso * 100.0, 2),
    }
