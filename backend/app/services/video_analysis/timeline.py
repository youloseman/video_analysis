"""The per-frame record of an analysis, for the player.

Everything the report prints is a summary: a mean, a p95, an angle at one
frame the analyzer chose. The overlay video shows the frames, but as pixels
-- the numbers are burned in, and nothing on the page can ask "where is the
knee at 2.3 s" or "jump to the next foot strike". This is the record that
lets the page ask.

One compact dict per analysis, built once beside the overlay and carried on
the result:

* ``frames`` -- the ANALYZED video frame indices (adaptive sampling on long
  clips skips frames, so these are not 0..n). Time is ``i / fps``, the same
  mapping the overlay video and the joint editor use.
* ``points`` -- the near-side skeleton per analyzed frame, normalized
  against the decoded frame like every landmark in the pipeline. Only the
  joints the editor's skeleton draws (``LMED_BONES`` in the client); the
  face and the hands are the overlay video's business.
* ``series`` -- one angle per frame per callout, from ``angle_history`` --
  the filtered series the overlay prints and the summary reads, so the curve
  under the video is the same signal as the number on it.
* ``metrics`` -- what each series is: its name, its reference band, the
  joint it hangs on, and whether the overlay gave it a callout.
* ``phases`` / ``events`` -- run: the gait phase per frame and the debounced
  foot strikes and toe-offs, counted exactly as the overlay's stride counter
  counts them. Bike: the BDC/TDC frames the knee extremes were read from.

Paid only, by construction: it is not in the free allowlist
(``result_gating._SAFE_KEYS``), and a free reader never gets an overlay to
play it over anyway.

Sizes: ~350 analyzed frames x 9 joints x 3 numbers, plus 4-6 series, comes
to ~60 KB of JSON. The result already carries two base64 stills larger than
that.
"""

from __future__ import annotations

import math
from typing import Any

import structlog

logger = structlog.get_logger()

# The near-side skeleton the client draws: ear, shoulder, elbow, wrist, hip,
# knee, ankle, heel, toe. Mirrors the joint editor's LMED_BONES so the
# "skeleton" layer of the player and the editor are one drawing.
SKELETON_BONES: dict[str, list[tuple[int, int]]] = {
    "left": [(7, 11), (11, 13), (13, 15), (11, 23), (23, 25), (25, 27),
             (27, 29), (27, 31), (29, 31)],
    "right": [(8, 12), (12, 14), (14, 16), (12, 24), (24, 26), (26, 28),
              (28, 30), (28, 32), (30, 32)],
}

# Run: a state has to hold this many analyzed frames before it counts as a
# stance or a swing. Same debounce as the overlay's stride counter
# (video_visualizer), so the player's "next contact" is the overlay's next
# stride.
MIN_RUN_STATE_FRAMES = 3
RUN_STANCE_PHASES = frozenset({
    "initial_contact", "loading_response", "midstance",
    "terminal_stance", "pre_swing",
})
# The order the legend lists the run phases in -- stance first, then swing.
RUN_PHASE_ORDER = [
    "initial_contact", "loading_response", "midstance", "terminal_stance",
    "pre_swing", "initial_swing", "mid_swing", "terminal_swing", "unknown",
]


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not (
        isinstance(v, float) and math.isnan(v)
    )


def _round(v: Any, nd: int) -> float | None:
    return round(float(v), nd) if _finite(v) else None


def _skeleton_points(
    frame_data: list[dict[str, Any]], side: str,
) -> tuple[list[int], dict[str, list[list[float] | None]]]:
    """``(joint indices, {index: [[x, y, vis] | None per frame]})``."""
    bones = SKELETON_BONES.get(side) or SKELETON_BONES["left"]
    joints = sorted({i for bone in bones for i in bone})
    points: dict[str, list[list[float] | None]] = {str(j): [] for j in joints}
    for fd in frame_data:
        lms = fd.get("normalized_landmarks") or []
        for j in joints:
            lm = lms[j] if j < len(lms) else None
            x = getattr(lm, "x", None) if lm is not None else None
            y = getattr(lm, "y", None) if lm is not None else None
            if not (_finite(x) and _finite(y)):
                points[str(j)].append(None)
                continue
            vis = getattr(lm, "visibility", 1.0)
            vis = 1.0 if vis is None or not _finite(vis) else float(vis)
            points[str(j)].append([round(float(x), 4), round(float(y), 4), round(vis, 2)])
    return joints, points


def _run_events(phases: list[str]) -> list[dict[str, Any]]:
    """Debounced foot strikes and toe-offs, as analyzed-frame positions.

    A contact is the first frame of a stance run that goes on to hold for
    ``MIN_RUN_STATE_FRAMES``; a toe-off the first frame of such a swing run.
    The event sits on the FIRST frame of the streak, not the frame the streak
    was confirmed on -- the foot landed when it landed, the debounce only
    decides whether to believe it.
    """
    events: list[dict[str, Any]] = []
    in_stance = False
    stance_streak = swing_streak = 0
    for k, phase in enumerate(phases):
        if phase in RUN_STANCE_PHASES:
            stance_streak += 1
            swing_streak = 0
        else:
            swing_streak += 1
            stance_streak = 0
        if not in_stance and stance_streak >= MIN_RUN_STATE_FRAMES:
            in_stance = True
            events.append({"k": k - MIN_RUN_STATE_FRAMES + 1, "kind": "contact"})
        elif in_stance and swing_streak >= MIN_RUN_STATE_FRAMES:
            in_stance = False
            events.append({"k": k - MIN_RUN_STATE_FRAMES + 1, "kind": "toe_off"})
    return events


def _bike_events(analyzer: Any, side: str) -> list[dict[str, Any]]:
    """The BDC/TDC frames of the near knee, from the analyzer's own peak pass.

    Empty when the extremes came from the percentile fallback (no per-stroke
    frames exist then) -- the player then has no stroke markers, which is
    the truth of that clip rather than a guess.
    """
    diag = (getattr(analyzer, "_bdc_tdc_diag", None) or {}).get(side) or {}
    events = [{"k": int(k), "kind": "bdc"} for k in diag.get("bdc_indices") or []]
    events += [{"k": int(k), "kind": "tdc"} for k in diag.get("tdc_indices") or []]
    events.sort(key=lambda e: e["k"])
    return events


def build_timeline(
    *,
    analyzer: Any,
    frame_data: list[dict[str, Any]],
    sport_type: str,
    video_info: dict[str, Any],
    label_configs: list[dict[str, Any]],
    material_keys: set[str] | None = None,
    phase_sequence: list[str] | None = None,
    cycle_numbers: list[int] | None = None,
) -> dict[str, Any] | None:
    """The record described in the module notes, or None on an empty clip.

    ``label_configs`` / ``material_keys`` are the overlay's own
    (``VideoVisualizer.label_configs`` / ``_material_keys``) so the player's
    metric list is the overlay's callout list, bands included.
    """
    if not frame_data:
        return None
    n = len(frame_data)
    side = analyzer.camera_side if analyzer.camera_side in ("left", "right") else "left"
    fps = float(video_info.get("fps") or analyzer.fps or 30.0)

    joints, points = _skeleton_points(frame_data, side)

    def _drawn_joint(idx: int) -> int | None:
        # The joint the callout hangs on, as a joint the skeleton layer draws.
        # A midline callout (the bike trunk hangs on landmark 11 whichever
        # side is near) is moved to its mirror when that is the drawn one;
        # BlazePose pairs its sides as (odd, even) from 7/8 up.
        if idx in joints:
            return idx
        mirror = idx + 1 if idx % 2 else idx - 1
        return mirror if idx >= 7 and mirror in joints else None

    series: dict[str, list[float | None]] = {}
    metrics: list[dict[str, Any]] = []
    history = getattr(analyzer, "angle_history", {}) or {}
    for cfg in label_configs:
        key = str(cfg["key"])
        values = history.get(key)
        if not values:
            continue
        vals = [_round(v, 1) for v in values[:n]]
        vals += [None] * (n - len(vals))
        if all(v is None for v in vals):
            continue
        series[key] = vals
        lo, hi = cfg["optimal"]
        # Where the band applies. The bike knee band is a statement about
        # the BOTTOM of the stroke and the hip band about the TOP -- a knee
        # at 90 deg mid-stroke is not out of range, it is mid-stroke. A curve
        # coloured against such a band frame by frame reads as a fault for
        # most of every revolution; the client grades those at their event
        # instead. Every other band is a whole-cycle band.
        # And the other kind of band: one about the whole-clip MEAN (the run
        # elbow's ~90 deg carry, the bike trunk). Coloured frame by frame, a
        # curve that swings 40 deg every stride reads as a fault twice a
        # stride; the client grades the mean line instead. Same vocabulary
        # as sport_configs.BAND_APPLIES.
        at = None
        if sport_type == "bike":
            if key.endswith("_knee"):
                at = "bdc"
            elif key.endswith("_hip"):
                at = "tdc"
            elif key in ("trunk_angle",) or key.endswith(("_elbow", "_shoulder", "_forearm_tilt")):
                at = "mean"
        elif sport_type == "run" and key in ("elbow", "trunk"):
            at = "mean"
        metrics.append({
            "key": key,
            "name": str(cfg["name"]),
            "joint": _drawn_joint(int(cfg["idx"])),
            "band": [float(lo), float(hi)],
            "open_ok": bool(cfg.get("open_ok")),
            "at": at,
            "callout": (material_keys is None) or (key in material_keys),
        })

    out: dict[str, Any] = {
        "version": 1,
        "sport": sport_type,
        "camera_side": side,
        "fps": round(fps, 4),
        "frame_count": int(video_info.get("frame_count") or 0) or None,
        "width": frame_data[0].get("frame_width"),
        "height": frame_data[0].get("frame_height"),
        "frames": [int(fd["frame_idx"]) for fd in frame_data],
        "bones": [list(b) for b in (SKELETON_BONES.get(side) or [])],
        "points": points,
        "series": series,
        "metrics": metrics,
    }

    if sport_type == "run" and phase_sequence:
        phases = list(phase_sequence[:n]) + ["unknown"] * max(0, n - len(phase_sequence))
        legend = [p for p in RUN_PHASE_ORDER if p in set(phases)]
        code = {p: i for i, p in enumerate(legend)}
        out["phase_legend"] = legend
        out["phases"] = [code.get(p, code.get("unknown", len(legend) - 1)) for p in phases]
        out["stance_phases"] = [p for p in legend if p in RUN_STANCE_PHASES]
        if cycle_numbers:
            out["cycles"] = [int(c) for c in cycle_numbers[:n]]
        out["events"] = _run_events(phases)
    elif sport_type == "bike":
        out["events"] = _bike_events(analyzer, side)
    else:
        out["events"] = []

    # Positions are analyzed-frame ordinals (k); the client maps k -> video
    # frame through ``frames``. Stated here so a reader of the JSON does not
    # take them for video frames.
    out["events_index"] = "analyzed"
    return out
