"""Did the legs keep their identity through this clip? Measure, don't hope.

WHY THIS EXISTS. Run leg identity is decided per link, and on a hard clip it
still comes out wrong for a stride at a time -- the overlay lives on the far
leg and the near-leg metrics are spliced from both. Two rounds of work cut
that a long way (2026-08-29) but did not end it, and no further improvement
could be VALIDATED: every structural fix would be scored by a structural
metric, and appearance turned out far too weak to arbitrate (twenty injected
swaps moved it by 2%, less than the noise between two runs).

So the honest move is the one this codebase already applies elsewhere: when a
number cannot be made reliable, say when it is not, rather than shipping a
confident-looking skeleton that jumps.

WHAT IS MEASURED. Running guarantees a structure the labels cannot argue
with: the two ankles swap vertical order EXACTLY TWICE per stride cycle -- one
foot rises while the other falls, they cross, and later cross back. Count the
order changes, compare with two per cycle, and the excess is the number of
times the labels traded legs. No ground truth, no reference clip, no
assumption about the athlete.

The count is taken through a deadband so that noise while the ankles are level
is not read as a swap, and the cycle length comes from the ankle's own
autocorrelation rather than from cadence (which is computed downstream and
would make this circular).

WHAT IT IS NOT. It cannot say WHICH stride went wrong, and it is blind to a
clip whose labels were wrong from the first frame to the last -- consistently
wrong is still consistent. It is a reliability signal, not a correction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

L_ANKLE, R_ANKLE = 27, 28

# The ankles have to separate by this share of their typical separation before
# a crossing counts, so that level-ankle jitter is not mistaken for a swap.
_DEADBAND_FRAC = 0.18

# Stride cycles are looked for between these bounds. 0.30 s is faster than any
# human stride; 1.6 s is slower than any running one.
_MIN_CYCLE_S = 0.30
_MAX_CYCLE_S = 1.6

_MIN_FRAMES = 60

# Above this share of excess crossings the clip's leg identity is not something
# to build a report on. Measured over the run fixtures: the clips that read
# clean sit at or below 0.05, the one the athlete called unusable sits at 0.21.
INSTABILITY_WARN = 0.15


def _ankle_series(frame_results: list[dict[str, Any]]) -> tuple[Any, Any]:
    n = len(frame_results)
    left = np.full(n, np.nan)
    right = np.full(n, np.nan)
    for k, fr in enumerate(frame_results):
        lms = fr.get("normalized_landmarks")
        if not lms or len(lms) <= R_ANKLE:
            continue
        for idx, arr in ((L_ANKLE, left), (R_ANKLE, right)):
            y = getattr(lms[idx], "y", None)
            if y is None:
                continue
            y = float(y)
            if not math.isnan(y):
                arr[k] = y
    return left, right


def _cycles(sig: Any, fps: float) -> float | None:
    """Stride cycles in the clip, from the ankle's own autocorrelation."""
    s = sig[np.isfinite(sig)]
    if len(s) < _MIN_FRAMES or fps <= 0:
        return None
    s = s - s.mean()
    if not np.any(np.abs(s) > 1e-9):
        return None
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    lo = max(4, int(_MIN_CYCLE_S * fps))
    hi = min(len(ac) - 1, int(_MAX_CYCLE_S * fps))
    if hi <= lo:
        return None
    lag = lo + int(np.argmax(ac[lo:hi]))
    return (len(s) / lag) if lag else None


def _order_changes(diff: Any) -> int:
    """Order changes of the two ankles, with a deadband (Schmitt trigger)."""
    d = diff[np.isfinite(diff)]
    if len(d) < 20:
        return 0
    thr = _DEADBAND_FRAC * float(np.percentile(np.abs(d), 90))
    if thr <= 0:
        return 0
    # The first excursion only establishes which ankle starts lower; it is not
    # itself a crossing. Counting it on one side and not the other -- which an
    # earlier version did -- biases every clip by one and hides a real swap
    # behind the offset.
    state = 0
    count = 0
    for v in d:
        if v < -thr:
            if state == 1:
                count += 1
            state = -1
        elif v > thr:
            if state == -1:
                count += 1
            state = 1
    return count


def measure_leg_identity_stability(
    frame_results: list[dict[str, Any]], fps: float,
) -> dict[str, Any] | None:
    """How often the legs traded labels, as a share of what gait allows.

    Returns None when the clip cannot support the measurement (too short, no
    resolvable stride, ankles never tracked) -- absence, not a reassuring
    zero.
    """
    if not frame_results or len(frame_results) < _MIN_FRAMES:
        return None
    left, right = _ankle_series(frame_results)
    cycles = _cycles(left, fps)
    if cycles is None or cycles < 2:
        return None
    expected = 2.0 * cycles
    observed = _order_changes(left - right)
    if observed == 0:
        return None
    excess = max(0.0, observed - expected)
    return {
        "cycles": round(cycles, 1),
        "expected_crossings": round(expected, 1),
        "observed_crossings": int(observed),
        "excess_crossings": round(excess, 1),
        # Excess as a share of what the gait itself requires, so a long clip
        # and a short one are comparable.
        "instability": round(excess / expected, 3),
        "unstable": bool(excess / expected > INSTABILITY_WARN),
    }
