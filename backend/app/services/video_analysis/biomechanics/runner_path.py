"""Is the skeleton on the runner for the whole clip, and if not, where was he?

MediaPipe looks for a person on the whole frame, shrunk to a small square. A
runner far from the camera is a few dozen pixels in that square, and it will
settle on something else instead. Run00000851 (4K, a stadium) spent frames 30-54
on an excavator and 94-102 on a grandstand, and it scored 95/A: nothing
downstream can tell. Leg identity asks which leg a label is on, never whether
the leg belongs to the runner, so instability read 0.0 throughout.

A body cannot jump, though. The middle of the hips is the steadiest point on the
skeleton and leg confusion does not move it, so a break in its path is the
detector changing its mind about who it is looking at. This module cuts the path
at those breaks, works out which pieces are the runner, and plans a window that
follows him: seen through it, from the source pixels, he fills the search square
and the excavator is not in the picture.

It is not for every clip. The same window breaks treadmill clips, where the
athlete already fills the frame and the crop hands the neighbouring machine to
the detector at higher resolution: IMG_4004 lost 20 spm and its score. So it
only runs when the path says the runner was lost, and if the second look does
not bring him back the score is withheld rather than computed off an excavator.

What this does not see: a torso that stays on the athlete while a leg lands on
the next machine. The hips are fine there and the path is whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24

# A jump the runner cannot make, in torso lengths. A torso is about half a metre,
# so 30 a second is 15 m/s of image motion: a sprinter past a still camera plus
# room for a pan. The floor keeps two neighbouring frames of a high-rate clip from
# being cut over jitter.
MAX_TORSOS_PER_SECOND = 30.0
MIN_JUMP_TORSOS = 0.5
MAX_TORSO_CHANGE = 0.4          # share of the median torso, between two frames

# A piece joins the runner if it lands where he was heading.
_PIECE_MIN_FRAMES = 3
_HEADING_SECONDS = 0.5
_PIECE_MISS_TORSOS = 1.0
_PIECE_MISS_TORSOS_PER_SECOND = 3.0
_PIECE_TORSO_CHANGE = 0.3

# Measured on every running clip we hold, 2026-11-06. See tests/test_runner_path.py.
LOST_MIN_PIECES = 5
LOST_MIN_OFF_RUNNER = 0.05

# The window, in body heights. 1.6 cut the feet off and read a false 206 spm on a
# clip counted at 186 by hand. Torso is about 0.3 of standing height.
WINDOW_BODIES = 2.2
_TORSO_OF_BODY = 0.30


def _xy(mark: Any) -> tuple[float, float]:
    if isinstance(mark, dict):
        return float(mark.get("x", np.nan)), float(mark.get("y", np.nan))
    return float(mark.x), float(mark.y)


@dataclass
class RunnerTrace:
    """Where the middle of the hips went, in source pixels."""

    frame_idx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    torso: np.ndarray
    on_runner: np.ndarray
    pieces: int

    @property
    def off_runner(self) -> float:
        return 1.0 - float(self.on_runner.mean()) if len(self.on_runner) else 0.0

    @property
    def lost(self) -> bool:
        return self.pieces >= LOST_MIN_PIECES and self.off_runner >= LOST_MIN_OFF_RUNNER

    def summary(self) -> dict[str, Any]:
        return {
            "pieces": self.pieces,
            "detected": len(self.on_runner),
            "off_runner_pct": round(self.off_runner * 100, 1),
            "lost": self.lost,
        }


def trace_runner(frames: list[dict[str, Any]], fps: float) -> RunnerTrace | None:
    """Follow the hips through a pass. Reads the raw landmarks, before the
    stabiliser has had a chance to smooth a jump into a glide."""
    rows = []
    for f in frames:
        marks = f.get("normalized_landmarks")
        if not marks or len(marks) <= _R_HIP:
            continue
        w, h = f.get("frame_width") or 0, f.get("frame_height") or 0
        lh, rh, ls, rs = (_xy(marks[i]) for i in (_L_HIP, _R_HIP, _L_SHOULDER, _R_SHOULDER))
        hx, hy = (lh[0] + rh[0]) / 2 * w, (lh[1] + rh[1]) / 2 * h
        sx, sy = (ls[0] + rs[0]) / 2 * w, (ls[1] + rs[1]) / 2 * h
        if all(np.isfinite(v) for v in (hx, hy, sx, sy)):
            rows.append((f.get("frame_idx", len(rows)), hx, hy, float(np.hypot(hx - sx, hy - sy))))
    if len(rows) < _PIECE_MIN_FRAMES:
        return None

    a = np.asarray(rows, dtype=float)
    idx, x, y, torso = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    median = float(np.median(torso))
    if median <= 0:
        return None
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    cuts = [0]
    for j in range(1, len(a)):
        limit = median * max(MIN_JUMP_TORSOS, MAX_TORSOS_PER_SECOND * dt * (idx[j] - idx[j - 1]))
        if (np.hypot(x[j] - x[j - 1], y[j] - y[j - 1]) > limit
                or abs(torso[j] - torso[j - 1]) > MAX_TORSO_CHANGE * median):
            cuts.append(j)
    cuts.append(len(a))
    pieces = [(cuts[j], cuts[j + 1]) for j in range(len(cuts) - 1)]

    # The longest stretch without a break is the runner: a false target holds for
    # a moment, because nothing on a track looks like a person for long.
    best = max(pieces, key=lambda p: p[1] - p[0])
    on = np.zeros(len(a), dtype=bool)
    on[best[0]:best[1]] = True
    best_torso = float(np.median(torso[best[0]:best[1]]))

    for p in sorted(pieces, key=lambda p: p[0] - p[1]):
        if p == best or p[1] - p[0] < _PIECE_MIN_FRAMES:
            continue
        if abs(float(np.median(torso[p[0]:p[1]])) - best_torso) >= _PIECE_TORSO_CHANGE * best_torso:
            continue
        kept = np.where(on)[0]
        before, after = kept[kept < p[0]], kept[kept >= p[1]]
        # From the nearest side that has anything to say.
        if len(before) and (not len(after) or p[0] - before[-1] <= after[0] - (p[1] - 1)):
            end, frm = p[0], int(before[-1])
            near = before[idx[before] >= idx[frm] - _HEADING_SECONDS / dt]
        else:
            end, frm = p[1] - 1, int(after[0])
            near = after[idx[after] <= idx[frm] + _HEADING_SECONDS / dt]
        vx = vy = 0.0
        span = idx[near[-1]] - idx[near[0]] if len(near) >= 2 else 0.0
        if span > 0:
            vx = float(x[near[-1]] - x[near[0]]) / span
            vy = float(y[near[-1]] - y[near[0]]) / span
        gap = idx[end] - idx[frm]
        miss = np.hypot(x[end] - (x[frm] + vx * gap), y[end] - (y[frm] + vy * gap))
        if miss < best_torso * (_PIECE_MISS_TORSOS + _PIECE_MISS_TORSOS_PER_SECOND * abs(gap) * dt):
            on[p[0]:p[1]] = True

    return RunnerTrace(idx, x, y, torso, on, len(pieces))


@dataclass
class FollowWindow:
    """A square that follows the runner through the source frames."""

    x0: np.ndarray
    y0: np.ndarray
    size: int
    frame_width: int
    frame_height: int

    @property
    def _centre(self) -> tuple[float, float]:
        return float(np.median(self.x0)), float(np.median(self.y0))

    def origin(self, frame_idx: int) -> tuple[int, int]:
        i = min(max(0, frame_idx), len(self.x0) - 1)
        return int(self.x0[i]), int(self.y0[i])

    def to_zoom(self, marks: list[Any], frame_idx: int) -> list[Any]:
        """Landmarks read through the window, at the window's scale, with the
        window's own travel put back.

        The scale is what the stabiliser needs: its limits are fractions of the
        frame, and the same detections at source scale read 193 spm against 187,
        with 186 counted by hand. The travel has to come back because the window
        follows the hips, so leaving it out subtracts part of the bounce from
        vertical oscillation, which then scored 0 against 76.
        """
        ox, oy = self.origin(frame_idx)
        cx, cy = self._centre
        return [
            type(m)(
                x=m.x + (ox - cx) / self.size,
                y=m.y + (oy - cy) / self.size,
                z=m.z,
                visibility=m.visibility,
            )
            for m in marks
        ]

    def restore(self, frames: list[dict[str, Any]]) -> None:
        """Put stabilised frames back on the whole picture, in place.

        Everything after this draws on the source video: the overlay, the
        kinogram, the player. None of it may know a window ever existed. z shares
        x's scale in MediaPipe's normalised space, so it shrinks with it.
        """
        cx, cy = self._centre
        sx, sy = self.size / self.frame_width, self.size / self.frame_height
        for f in frames:
            f["normalized_landmarks"] = [
                type(m)(
                    x=cx / self.frame_width + m.x * sx,
                    y=cy / self.frame_height + m.y * sy,
                    z=m.z * sx,
                    visibility=m.visibility,
                )
                for m in f["normalized_landmarks"]
            ]
            f["frame_width"], f["frame_height"] = self.frame_width, self.frame_height


def plan_follow(
    trace: RunnerTrace, frame_count: int, fps: float,
    frame_width: int, frame_height: int,
) -> FollowWindow | None:
    """The window for a second look, or None when there is nothing to follow."""
    if not trace.on_runner.any() or frame_width <= 0 or frame_height <= 0:
        return None
    idx, x, y = trace.frame_idx[trace.on_runner], trace.x[trace.on_runner], trace.y[trace.on_runner]
    body = float(np.median(trace.torso[trace.on_runner])) / _TORSO_OF_BODY
    size = int(min(frame_width, frame_height, body * WINDOW_BODIES)) & ~1
    if size < 16:
        return None
    n = max(int(frame_count), int(idx[-1]) + 1)
    every = np.arange(n, dtype=float)
    lead = min(max(2, int(fps * _HEADING_SECONDS)), len(idx))

    def along(values: np.ndarray) -> np.ndarray:
        out = np.interp(every, idx, values)
        # Beyond the ends he keeps going the way he was going: a runner past a
        # still camera is somewhere else by the time the path starts.
        if lead >= 2:
            first, last = every < idx[0], every > idx[-1]
            out[first] = values[0] + np.polyfit(idx[:lead], values[:lead], 1)[0] * (every[first] - idx[0])
            out[last] = values[-1] + np.polyfit(idx[-lead:], values[-lead:], 1)[0] * (every[last] - idx[-1])
        # The hips bob; the window should not.
        kernel = max(3, int(fps / 4)) | 1
        pad = kernel // 2
        return np.convolve(np.pad(out, pad, mode="edge"), np.ones(kernel) / kernel, "valid")

    x0 = np.clip(np.round(along(x) - size / 2), 0, frame_width - size).astype(int)
    y0 = np.clip(np.round(along(y) - size / 2), 0, frame_height - size).astype(int)
    return FollowWindow(x0, y0, size, frame_width, frame_height)
