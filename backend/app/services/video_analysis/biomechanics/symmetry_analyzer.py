"""Module 3: Bilateral symmetry analysis via Continuous Relative Phase (CRP).

Computes phase angles for left/right joint pairs, then CRP (difference in phase).
CRP near 0 = in-phase (symmetric), CRP near 180 = anti-phase.

Also computes Bilateral Symmetry Index (BSI) as a simple amplitude comparison.
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Left/Right joint pairs per sport
SPORT_PAIRS = {
    "run": [("left_knee", "right_knee"), ("left_hip", "right_hip"), ("left_elbow", "right_elbow")],
    "bike": [("left_knee", "right_knee")],
    "swim": [("left_shoulder", "right_shoulder"), ("left_elbow", "right_elbow")],
}

MIN_POINTS = 15

# CRP threshold for declaring "no lag" (degrees)
LAG_THRESHOLD = 5.0


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal to [-1, 1] range. Returns zeros when the signal
    contains no finite samples (all NaN from P0 visibility gating)."""
    if signal.size == 0 or not np.any(np.isfinite(signal)):
        return np.zeros_like(signal)
    mn, mx = np.nanmin(signal), np.nanmax(signal)
    rng = mx - mn
    if not np.isfinite(rng) or rng < 1e-6:
        return np.zeros_like(signal)
    return 2.0 * (signal - mn) / rng - 1.0


def _compute_phase_angle(
    angles: np.ndarray, timestamps: np.ndarray
) -> np.ndarray:
    """Compute instantaneous phase angle using Hilbert-like approach.

    Phase = arctan2(velocity_normalized, angle_normalized).
    Returns phase in degrees [-180, 180].
    """
    n = len(angles)
    if n < 3:
        return np.zeros(n)

    # Normalize angle to [-1, 1]
    angle_norm = _normalize_signal(angles)

    # Compute angular velocity via central differences
    velocity = np.zeros(n)
    for i in range(1, n - 1):
        dt = timestamps[i + 1] - timestamps[i - 1]
        if dt > 0:
            velocity[i] = (angles[i + 1] - angles[i - 1]) / dt
    dt0 = timestamps[1] - timestamps[0]
    if dt0 > 0:
        velocity[0] = (angles[1] - angles[0]) / dt0
    dtn = timestamps[-1] - timestamps[-2]
    if dtn > 0:
        velocity[-1] = (angles[-1] - angles[-2]) / dtn

    velocity_norm = _normalize_signal(velocity)

    # Phase angle
    phase = np.degrees(np.arctan2(velocity_norm, angle_norm))
    return phase


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap angle to [-180, 180] range."""
    return ((angle + 180.0) % 360.0) - 180.0


def _compute_bsi(left_values: np.ndarray, right_values: np.ndarray) -> float:
    """Compute Bilateral Symmetry Index (0-100).

    BSI = 100 * (1 - |L_mean - R_mean| / avg_mean).
    100 = perfect symmetry, 0 = maximally asymmetric.
    """
    # Guard against all-NaN inputs from fully gated landmarks.
    if not np.any(np.isfinite(left_values)) or not np.any(np.isfinite(right_values)):
        return 100.0  # No data either side -> not informative, emit neutral

    l_mean = float(np.nanmean(left_values))
    r_mean = float(np.nanmean(right_values))
    avg = (abs(l_mean) + abs(r_mean)) / 2.0

    if avg < 1.0:
        return 100.0  # Both near zero = symmetric

    asymmetry = abs(l_mean - r_mean) / avg
    bsi = max(0.0, 100.0 * (1.0 - asymmetry))
    return round(bsi, 1)


def compute_symmetry(
    angle_history: dict[str, list[float]],
    timestamps: list[float],
    sport_type: str,
    camera_side: str | None = None,
) -> dict[str, Any]:
    """Compute bilateral symmetry metrics for L/R joint pairs.

    For side-view sports (run, bike), far-side landmarks are unreliable,
    so symmetry analysis is skipped entirely. Only swimming (both sides
    visible) gets full symmetry computation.

    Args:
        angle_history: Filtered angle time-series.
        timestamps: Time in seconds for each frame.
        sport_type: 'run', 'bike', or 'swim'.
        camera_side: 'left' or 'right' (unused, for API consistency).

    Returns:
        Dict with BSI, CRP timeline, lagging side per pair, and overall score.
    """
    # Side-view sports: far side is unreliable, skip symmetry
    if sport_type in ("run", "bike"):
        logger.info("SYMMETRY_SKIP", sport=sport_type, reason="side_view_unreliable")
        return {"pairs": [], "bilateral_symmetry_index": None}

    pairs = SPORT_PAIRS.get(sport_type, SPORT_PAIRS["run"])
    ts = np.array(timestamps, dtype=np.float64)

    pair_results: list[dict[str, Any]] = []
    bsi_scores: list[float] = []

    for left_name, right_name in pairs:
        if left_name not in angle_history or right_name not in angle_history:
            continue

        left_vals = np.array(angle_history[left_name], dtype=np.float64)
        right_vals = np.array(angle_history[right_name], dtype=np.float64)

        min_len = min(len(left_vals), len(right_vals), len(ts))
        if min_len < MIN_POINTS:
            continue

        left_vals = left_vals[:min_len]
        right_vals = right_vals[:min_len]
        t = ts[:min_len]

        # BSI (amplitude-based)
        bsi = _compute_bsi(left_vals, right_vals)
        bsi_scores.append(bsi)

        # Phase angles
        phase_left = _compute_phase_angle(left_vals, t)
        phase_right = _compute_phase_angle(right_vals, t)

        # CRP = phase_left - phase_right, wrapped
        crp = _wrap_angle(phase_left - phase_right)
        if not np.any(np.isfinite(crp)):
            # All-NaN CRP -> skip pair rather than emitting NaN mean/std
            continue
        mean_crp = float(np.nanmean(crp))
        std_crp = float(np.nanstd(crp))

        # Determine lagging side
        if abs(mean_crp) < LAG_THRESHOLD:
            lagging_side = "none"
        elif mean_crp > 0:
            lagging_side = "right"
        else:
            lagging_side = "left"

        # Downsample CRP timeline for charts (max 200 points)
        step = max(1, len(crp) // 200)
        crp_timeline = [
            {"time": round(float(t[i]), 3), "crp": round(float(crp[i]), 1)}
            for i in range(0, len(crp), step)
        ]

        pair_label = f"{left_name.replace('left_', '')}".capitalize()
        pair_results.append({
            "pair": pair_label,
            "left": left_name,
            "right": right_name,
            "bsi": bsi,
            "mean_crp": round(mean_crp, 1),
            "std_crp": round(std_crp, 1),
            "lagging_side": lagging_side,
            "crp_timeline": crp_timeline,
        })

    overall_bsi = round(float(np.mean(bsi_scores)), 1) if bsi_scores else 50.0

    logger.info(
        "SYMMETRY_DEBUG",
        sport=sport_type,
        pairs_analyzed=len(pair_results),
        overall_bsi=overall_bsi,
    )

    return {
        "pairs": pair_results,
        "bilateral_symmetry_index": overall_bsi,
    }
