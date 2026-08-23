"""Two things a side-view clip gets silently wrong, measured rather than assumed.

**Framing.** MediaPipe is handed a frame capped at ``DETECT_MAX_LONG_EDGE``, so
what matters is not the camera's megapixels but how many pixels of that frame
the athlete actually occupies. Filmed from the far side of a track the runner
can be 140 px tall, and at that size the two legs are a few pixels apart for
much of the stride -- left/right assignment becomes a coin flip no downstream
smoothing can repair. Worth saying out loud, because "stand closer" is a fix
the athlete can actually apply.

**Which leg faces the camera.** Today that is decided from MediaPipe's depth
(``z``) alone, and on a perpendicular side view the two hips project to nearly
the same pixel -- the model is guessing. These cues ask the image instead, and
each is independent of that guess:

* *visibility* -- the far leg is periodically hidden behind the near leg and
  the torso, and the tracker reports it less confidently.
* *contact depth* -- perspective: for a camera above the ground plane, the
  nearer foot lands LOWER in the image. Measured at each leg's lowest point
  (its ground contact) relative to the hips, so a panning camera cancels out.
* *segment length* -- the nearer limb subtends more pixels.
* *z bias* -- MediaPipe's own depth, kept as one voter rather than the judge.

Every cue is signed the same way: **positive means evidence for LEFT being the
near side**. Nothing here decides anything yet -- the runner records the numbers
so they can be checked against real clips before they are trusted.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# hip, knee, ankle, foot-index per side.
_LEFT_LEG = (23, 25, 27, 31)
_RIGHT_LEG = (24, 26, 28, 32)
_LEFT_LEG_VIS = (25, 27, 29, 31)   # knee, ankle, heel, toe
_RIGHT_LEG_VIS = (26, 28, 30, 32)

# Each cue's noise floor: how large a reading has to be before it means
# anything at all. These are not vote thresholds -- they are the denominator
# that turns four measurements in four different units into one comparable
# number. A cue reading twice its floor carries twice the evidence.
_DEADBAND = {
    "visibility_pp": 3.0,        # percentage points of visibility
    "contact_depth_pct": 1.5,    # % of leg length
    "segment_length_pct": 1.5,   # % of leg length
    "z_bias_pct": 5.0,           # % of leg length
}

# Past this multiple of its own noise floor, a cue is as certain as it is going
# to get, and letting it grow further only lets one mistracked stretch decide
# the answer on its own.
_CUE_CAP = 3.0

# Net evidence needed before a side is named. One cue's worth, clear of the
# noise floor and of whatever points the other way.
_MIN_NEAR_SIDE_SCORE = 1.0

# Athlete height in the frame MediaPipe actually sees. Below "tiny" the legs
# are too few pixels apart to tell reliably; between the two it is workable but
# degraded.
#
# Raised 2026-08-21 from 150/250 on measured evidence. One clip, cropped three
# ways, with everything else held constant:
#
#     116 px -> legs swapped on 42.7% of frames
#     255 px -> legs swapped on 45.6% of frames   <- the old bar called this OK
#     452 px -> legs swapped on  1.8% of frames
#
# The old "ok" bar handed a clean bill of health to a clip whose left and right
# legs were interchangeable half the time -- a false all-clear on the one
# measurement everything else rests on.
#
# Three crops of one clip is calibration, not proof, so the bars are placed
# where that evidence AND a geometric argument agree. What the pose model has
# to resolve is not the athlete's height but the width of a limb: a thigh is
# roughly a tenth of body height, so a 400 px athlete gives it ~40 px of thigh
# to separate from the other one, and a 255 px athlete gives it ~25. The "tiny"
# bar sits above the largest height we have measured failing, and the "small"
# bar at the limb width where it demonstrably works.
FRAMING_TINY_PX = 300
FRAMING_SMALL_PX = 400


def _stack(frame_results: list[dict[str, Any]], key: str, attr: str) -> np.ndarray:
    """(n_frames, 33) array of one landmark attribute; missing -> NaN."""
    n = len(frame_results)
    out = np.full((n, 33), np.nan, dtype=float)
    for i, frame in enumerate(frame_results):
        lms = frame.get(key) or []
        for j in range(min(33, len(lms))):
            v = getattr(lms[j], attr, None)
            if v is None:
                continue
            try:
                out[i, j] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def _nanmedian(values: np.ndarray) -> float | None:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.median(values))


def _seg_len(x: np.ndarray, y: np.ndarray, a: int, b: int) -> np.ndarray:
    return np.hypot(x[:, a] - x[:, b], y[:, a] - y[:, b])


def compute_framing(
    frame_results: list[dict[str, Any]],
    detect_height_px: float | None = None,
) -> dict[str, Any]:
    """How much of the frame the athlete fills, and how many pixels that is.

    ``detect_height_px`` is the height of the frame as handed to the detector
    (after the long-edge cap), so the pixel figure reflects what MediaPipe
    actually saw rather than the camera's raw resolution.
    """
    if not frame_results:
        return {"subject_height_frac": None, "subject_height_px": None, "verdict": None}

    ny = _stack(frame_results, "normalized_landmarks", "y")
    # Nose to the lower ankle: robust to one foot being mid-swing. Rows where
    # both ankles were gated out are all-NaN -- nanmax warns and yields NaN,
    # which the median then ignores, so silence the warning rather than let it
    # reach the logs on every heavily-gated clip.
    top = ny[:, 0]
    if np.isfinite(top).sum() < max(3, 0.2 * len(frame_results)):
        top = np.nanmean(ny[:, [11, 12]], axis=1)   # fall back to shoulders
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            bottom = np.nanmax(ny[:, [27, 28]], axis=1)
    frac = _nanmedian(bottom - top)
    # A head below the feet is not a small athlete, it is broken tracking.
    # Reporting "-10 px tall" would be worse than admitting we cannot tell.
    if frac is not None and frac <= 0.01:
        frac = None

    px = None
    if frac is not None and detect_height_px:
        px = round(frac * float(detect_height_px))

    # Key is "verdict", not "level": structlog reserves "level" for the log
    # level and silently drops a colliding field.
    verdict = None
    if px is not None:
        verdict = "tiny" if px < FRAMING_TINY_PX else (
            "small" if px < FRAMING_SMALL_PX else "ok"
        )
    return {
        "subject_height_frac": round(frac, 4) if frac is not None else None,
        "subject_height_px": px,
        "verdict": verdict,
    }


def compute_near_side_cues(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Image-space evidence for which leg faces the camera.

    Positive values favour LEFT. Returns the individual cues, each cue's vote,
    and the majority suggestion with how many cues backed it. Cues that cannot
    be measured (or that sit inside their deadband) abstain rather than guess.
    """
    if len(frame_results) < 5:
        return {"suggested_near_side": None, "agreement": 0, "cues_voting": 0}

    nx = _stack(frame_results, "normalized_landmarks", "x")
    ny = _stack(frame_results, "normalized_landmarks", "y")
    vis = _stack(frame_results, "normalized_landmarks", "visibility")
    wx = _stack(frame_results, "world_landmarks", "x")
    wy = _stack(frame_results, "world_landmarks", "y")
    wz = _stack(frame_results, "world_landmarks", "z")

    # Leg length in normalized units -- the scale every length cue is read in.
    left_len = _seg_len(nx, ny, 23, 25) + _seg_len(nx, ny, 25, 27)
    right_len = _seg_len(nx, ny, 24, 26) + _seg_len(nx, ny, 26, 28)
    med_left, med_right = _nanmedian(left_len), _nanmedian(right_len)
    leg_len = None
    if med_left is not None and med_right is not None:
        leg_len = (med_left + med_right) / 2.0
    elif med_left is not None or med_right is not None:
        leg_len = med_left if med_left is not None else med_right

    cues: dict[str, float | None] = {}

    # 1. Visibility asymmetry (occlusion): the far leg is tracked less surely.
    vl = _nanmedian(np.nanmean(vis[:, list(_LEFT_LEG_VIS)], axis=1))
    vr = _nanmedian(np.nanmean(vis[:, list(_RIGHT_LEG_VIS)], axis=1))
    cues["visibility_pp"] = (
        round((vl - vr) * 100, 2) if vl is not None and vr is not None else None
    )

    # 2. Contact depth (perspective): each leg's lowest point relative to the
    #    hips. Hip-relative so a panning or drifting camera cancels out; the
    #    90th percentile stands in for "this leg's ground contact".
    hip_y = np.nanmean(ny[:, [23, 24]], axis=1)
    cues["contact_depth_pct"] = None
    if leg_len and leg_len > 1e-6:
        dl, dr = ny[:, 27] - hip_y, ny[:, 28] - hip_y
        dl, dr = dl[np.isfinite(dl)], dr[np.isfinite(dr)]
        if dl.size >= 5 and dr.size >= 5:
            drop = float(np.percentile(dl, 90) - np.percentile(dr, 90))
            cues["contact_depth_pct"] = round(drop / leg_len * 100, 2)

    # 3. Apparent segment length: the nearer limb subtends more pixels.
    cues["segment_length_pct"] = None
    if med_left is not None and med_right is not None:
        mean_len = (med_left + med_right) / 2.0
        if mean_len > 1e-6:
            cues["segment_length_pct"] = round(
                (med_left - med_right) / mean_len * 100, 2
            )

    # 4. MediaPipe depth, kept as one voter. Smaller z = nearer the camera, so
    #    (right - left) positive means the left leg is the nearer one.
    cues["z_bias_pct"] = None
    zl = _nanmedian(np.nanmean(wz[:, [25, 27]], axis=1))
    zr = _nanmedian(np.nanmean(wz[:, [26, 28]], axis=1))
    if zl is not None and zr is not None:
        world_len = _nanmedian(
            _seg_len(wx, wy, 23, 25) + _seg_len(wx, wy, 25, 27)
        )
        if world_len and world_len > 1e-6:
            cues["z_bias_pct"] = round((zr - zl) / world_len * 100, 2)

    # Weigh each cue by how far it clears its OWN noise floor, and add them up.
    #
    # This used to be a vote: clear your deadband, cast one ballot, ties are a
    # conflict. That threw away the one thing separating a cue that barely
    # cleared its floor from one that cleared it sevenfold. Measured on
    # IMG_3979, whose near side is confirmed RIGHT: contact depth cleared by
    # 1.2x and said left, segment length cleared by 7.4x and said right, and
    # the two ballots tied. The clip was declared conflicted and the metrics
    # that hang on the near side were withheld -- on a clip where three of the
    # four cues, and the answer, were right.
    #
    # A deadband is a noise floor, so value/floor is a signal-to-noise ratio,
    # which is the natural weight and makes four readings in four different
    # units comparable without inventing anything.
    return {**cues, **score_near_side_cues(cues)}


def score_near_side_cues(cues: dict[str, float | None]) -> dict[str, Any]:
    """Combine the four cues into one verdict, weighed by evidence.

    This used to be a vote: clear your deadband, cast one ballot, ties are a
    conflict. That threw away the one thing separating a cue that barely
    cleared its floor from one that cleared it sevenfold. Measured on
    IMG_3979, whose near side the athlete confirmed is RIGHT: contact depth
    cleared by 1.2x and said left, segment length cleared by 7.4x and said
    right, and the two ballots tied. The clip was declared conflicted and the
    metrics that hang on the near side were withheld -- on a clip where three
    of the four cues, and the answer, were right.

    A deadband is a noise floor, so value/floor is a signal-to-noise ratio,
    which is the natural weight and makes four readings in four different units
    comparable without inventing anything.
    """
    votes: dict[str, str] = {}
    strengths: dict[str, float] = {}
    score = 0.0
    total = 0.0
    for name, value in cues.items():
        floor = _DEADBAND.get(name)
        if value is None or not floor:
            continue
        ratio = max(-_CUE_CAP, min(_CUE_CAP, value / floor))
        strengths[name] = round(ratio, 2)
        score += ratio
        total += abs(ratio)
        if abs(value) >= floor:
            votes[name] = "left" if value > 0 else "right"

    decided = abs(score) >= _MIN_NEAR_SIDE_SCORE
    return {
        "cue_votes": votes,
        "cues_voting": len(votes),
        # How one-sided the evidence is, 0 to 1: everything pointing one way is
        # 1.0, an even split is 0.0. Replaces a count of ballots, which said
        # nothing about how strongly any of them were cast.
        "agreement": round(abs(score) / total, 2) if total > 1e-9 else 0.0,
        "cue_strengths": strengths,
        "near_side_score": round(score, 2),
        # Genuinely balanced evidence, not merely two ballots disagreeing.
        "conflict": not decided,
        "suggested_near_side": (
            ("left" if score > 0 else "right") if decided else None
        ),
    }


__all__ = [
    "FRAMING_SMALL_PX",
    "FRAMING_TINY_PX",
    "compute_framing",
    "compute_near_side_cues",
]
