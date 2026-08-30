"""The stabilized landmark frames of an analysis, kept on disk.

Everything the report measures is a function of these frames (see the module
notes in ``runner``), and until now they existed for the length of one call and
were gone. That made every analysis a one-shot: an athlete who could see the
hip point sitting on the saddle had no way to move it, because there was
nothing left to move it on. Storing the frames is what makes a re-analysis
possible without a second MediaPipe pass.

What is stored is the frames AFTER ``stabilize_landmarks`` -- leg identities
resolved, low-visibility points gated to NaN, the smoothing pass applied. A
re-analysis therefore enters the pipeline after stabilization and must not run
it again; see ``runner.analyze_from_frames``.

Format: one ``.npz`` per analysis. Coordinates are float64 so a load followed
by a re-analysis reproduces the original numbers exactly rather than to within
float32 rounding -- the difference is a few hundred kilobytes, and "the same
frames give the same report" is a property worth not having to argue about.
The per-frame dict keys the measuring half reads are all here; ``leg_patches``
(run) is not, because it is consumed inside stabilization and nothing after
reads it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

FORMAT_VERSION = 1
N_LANDMARKS = 33
_SIDES = ("left", "right")
_LANDMARK_KEYS = ("world_landmarks", "normalized_landmarks")


def _num(v: Any) -> float:
    if v is None:
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _landmarks_to_array(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    """``[n_frames, 33, 4]`` of (x, y, z, visibility); missing -> NaN."""
    out = np.full((len(frames), N_LANDMARKS, 4), np.nan, dtype=np.float64)
    for i, frame in enumerate(frames):
        lms = frame.get(key) or []
        for j in range(min(N_LANDMARKS, len(lms))):
            lm = lms[j]
            out[i, j, 0] = _num(getattr(lm, "x", None))
            out[i, j, 1] = _num(getattr(lm, "y", None))
            out[i, j, 2] = _num(getattr(lm, "z", None))
            out[i, j, 3] = _num(getattr(lm, "visibility", 1.0))
    return out


def _array_to_landmarks(arr: np.ndarray) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            x=float(row[0]), y=float(row[1]), z=float(row[2]),
            visibility=float(row[3]),
        )
        for row in arr
    ]


def save_frames(
    path: str | Path, frames: list[dict[str, Any]], *, meta: dict[str, Any],
) -> Path:
    """Write ``frames`` (+ a JSON-safe ``meta`` dict) to ``path``.

    ``meta`` is whatever the caller needs to re-run the measuring half later:
    the sport, the position, the video info, the sampling and stabilization
    diagnostics. It is stored verbatim and handed back by :func:`load_frames`.
    """
    path = Path(path)
    n = len(frames)
    gate = np.zeros((n, len(_SIDES)), dtype=np.uint8)
    for i, frame in enumerate(frames):
        filled = frame.get("leg_gate_filled") or ()
        for s, side in enumerate(_SIDES):
            if side in filled:
                gate[i, s] = 1

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.array(FORMAT_VERSION),
        meta=np.array(json.dumps(meta or {})),
        frame_idx=np.array([int(f.get("frame_idx", i)) for i, f in enumerate(frames)], dtype=np.int64),
        timestamp_ms=np.array([_num(f.get("timestamp_ms")) for f in frames], dtype=np.float64),
        frame_width=np.array([int(f.get("frame_width") or 0) for f in frames], dtype=np.int64),
        frame_height=np.array([int(f.get("frame_height") or 0) for f in frames], dtype=np.int64),
        world=_landmarks_to_array(frames, "world_landmarks"),
        normalized=_landmarks_to_array(frames, "normalized_landmarks"),
        leg_gate_filled=gate,
    )
    return path


def load_frames(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a store back into the frame dicts the pipeline works on.

    Raises ``FileNotFoundError`` when there is nothing at ``path`` (an expired
    clip, or an analysis that predates the store) and ``ValueError`` for a
    store this code cannot read.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with np.load(path, allow_pickle=False) as z:
        version = int(z["version"])
        if version > FORMAT_VERSION:
            raise ValueError(
                f"landmark store version {version} is newer than this code ({FORMAT_VERSION})"
            )
        meta = json.loads(str(z["meta"]))
        frame_idx = z["frame_idx"]
        timestamp_ms = z["timestamp_ms"]
        frame_width = z["frame_width"]
        frame_height = z["frame_height"]
        world = z["world"]
        normalized = z["normalized"]
        gate = z["leg_gate_filled"]

    frames: list[dict[str, Any]] = []
    for i in range(len(frame_idx)):
        frame: dict[str, Any] = {
            "world_landmarks": _array_to_landmarks(world[i]),
            "normalized_landmarks": _array_to_landmarks(normalized[i]),
            "timestamp_ms": float(timestamp_ms[i]),
            "frame_idx": int(frame_idx[i]),
            "frame_width": int(frame_width[i]),
            "frame_height": int(frame_height[i]),
        }
        filled = {side for s, side in enumerate(_SIDES) if gate[i, s]}
        if filled:
            # Present only when set, matching the stabilizer's setdefault: a
            # reader testing ``frame.get("leg_gate_filled")`` must see the same
            # falsy/absent shape it saw on the live frames.
            frame["leg_gate_filled"] = filled
        frames.append(frame)
    return frames, meta


__all__ = ["FORMAT_VERSION", "save_frames", "load_frames"]
