"""Module 4: Joint coordination analysis (angle-angle diagrams).

Analyzes proximal-distal joint coupling by:
1. Splitting angle data into gait cycles (from phase portrait cycle detection)
2. Time-normalizing each cycle to 101 points via cubic spline
3. Computing coupling angles (vector coding method)
4. Computing cross-correlation between proximal and distal angles
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Proximal-distal joint pairs per sport (default: left side)
# For run/bike, dynamically replaced with near-side based on camera_side
SPORT_COORD_PAIRS = {
    "run": [("hip", "knee"), ("knee", "ankle")],  # Unprefixed: near-side only
    "bike": [("left_hip", "left_knee"), ("left_knee", "left_ankle")],
    "swim": [("left_shoulder", "left_elbow")],
}


def _get_coord_pairs(sport_type: str, camera_side: str | None) -> list[tuple[str, str]]:
    """Get coordination pairs, using near-side for side-view sports."""
    base_pairs = SPORT_COORD_PAIRS.get(sport_type, SPORT_COORD_PAIRS["run"])
    # Running: already unprefixed, no replacement needed
    if sport_type in ("swim", "run") or not camera_side:
        return base_pairs

    # Replace "left_" with near-side prefix for bike
    near = camera_side
    result = []
    for prox, dist in base_pairs:
        prox_new = prox.replace("left_", f"{near}_")
        dist_new = dist.replace("left_", f"{near}_")
        result.append((prox_new, dist_new))
    return result

# Number of points for time-normalized cycles
NORM_POINTS = 101

MIN_CYCLE_SAMPLES = 6


def _time_normalize_cycle(
    values: np.ndarray, n_points: int = NORM_POINTS
) -> np.ndarray:
    """Time-normalize a cycle to fixed number of points using cubic spline.

    Returns array of n_points values representing 0-100% of the cycle.
    """
    from scipy.interpolate import CubicSpline

    n = len(values)
    if n < 8:
        # Not enough for cubic spline, use linear interpolation
        x_old = np.linspace(0, 100, n)
        x_new = np.linspace(0, 100, n_points)
        return np.interp(x_new, x_old, values)

    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)
    cs = CubicSpline(x_old, values)
    return cs(x_new)


def _compute_coupling_angle(
    proximal: np.ndarray, distal: np.ndarray
) -> np.ndarray:
    """Compute coupling angle using vector coding method.

    Coupling angle = arctan2(delta_distal, delta_proximal) at each time step.
    Returns angles in degrees [0, 360].

    Interpretation:
    - 0/360 = proximal only (in-phase, proximal leads)
    - 90 = distal only
    - 45 = equal contribution (1:1 coupling)
    - 180 = proximal only (anti-phase)
    """
    d_proximal = np.diff(proximal)
    d_distal = np.diff(distal)

    # Avoid division issues
    coupling = np.degrees(np.arctan2(d_distal, d_proximal))
    # Wrap to [0, 360]
    coupling = coupling % 360.0

    return coupling


def _cross_correlation(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Compute normalized cross-correlation between two signals.

    Returns Pearson r and optimal lag.
    """
    if len(a) != len(b) or len(a) < 3:
        return {"r": 0.0, "lag": 0}

    # Zero-lag correlation first
    r_zero = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0

    return {
        "r": round(r_zero, 3),
        "lag": 0,
    }


def compute_coordination(
    angle_history: dict[str, list[float]],
    timestamps: list[float],
    sport_type: str,
    phase_data: dict[str, Any] | None = None,
    camera_side: str | None = None,
) -> dict[str, Any]:
    """Compute coordination metrics for proximal-distal joint pairs.

    Args:
        angle_history: Filtered angle time-series.
        timestamps: Time in seconds per frame.
        sport_type: 'run', 'bike', or 'swim'.
        phase_data: Output from phase_portrait module (for cycle boundaries).
        camera_side: 'left' or 'right' for near-side pair selection.

    Returns:
        Dict with per-pair coordination data.
    """
    pairs = _get_coord_pairs(sport_type, camera_side)

    pair_results: list[dict[str, Any]] = []

    for proximal_name, distal_name in pairs:
        if proximal_name not in angle_history or distal_name not in angle_history:
            continue

        proximal = np.array(angle_history[proximal_name], dtype=np.float64)
        distal = np.array(angle_history[distal_name], dtype=np.float64)

        min_len = min(len(proximal), len(distal))
        if min_len < MIN_CYCLE_SAMPLES:
            continue

        proximal = proximal[:min_len]
        distal = distal[:min_len]

        # Get cycles from phase portrait data (proximal joint)
        cycles = []
        if phase_data and "joints" in phase_data:
            joint_data = phase_data["joints"].get(proximal_name, {})
            cycles = joint_data.get("cycles", [])

        # Normalize each cycle
        normalized_cycles_prox: list[np.ndarray] = []
        normalized_cycles_dist: list[np.ndarray] = []

        for cycle in cycles:
            start = cycle["start_idx"]
            end = cycle["end_idx"]
            if end - start < MIN_CYCLE_SAMPLES:
                continue
            if end > min_len:
                continue

            norm_prox = _time_normalize_cycle(proximal[start:end])
            norm_dist = _time_normalize_cycle(distal[start:end])
            normalized_cycles_prox.append(norm_prox)
            normalized_cycles_dist.append(norm_dist)

        # Skip pairs with 0 normalized cycles (no chart data to show)
        if len(normalized_cycles_prox) == 0:
            logger.info(
                "COORDINATION_SKIP",
                pair=f"{proximal_name}-{distal_name}",
                reason="0_cycles",
            )
            continue

        # Compute mean cycle (ensemble average)
        mean_cycle_prox = None
        mean_cycle_dist = None
        coupling_angle_mean = None
        coupling_angle_std = None
        variability_score = None

        if normalized_cycles_prox:
            prox_stack = np.array(normalized_cycles_prox)
            dist_stack = np.array(normalized_cycles_dist)
            mean_cycle_prox = np.mean(prox_stack, axis=0)
            mean_cycle_dist = np.mean(dist_stack, axis=0)

            # Coupling angles per cycle
            all_coupling = []
            for i in range(len(normalized_cycles_prox)):
                ca = _compute_coupling_angle(normalized_cycles_prox[i], normalized_cycles_dist[i])
                all_coupling.append(ca)

            if all_coupling:
                coupling_stack = np.array(all_coupling)
                coupling_angle_mean = np.mean(coupling_stack, axis=0)
                coupling_angle_std = np.std(coupling_stack, axis=0)
                # Variability: mean of std across time points (lower = more consistent)
                mean_var = float(np.mean(coupling_angle_std))
                # Score: 100 at var=0, 0 at var=90
                variability_score = round(max(0.0, 100.0 - mean_var * (100.0 / 90.0)), 1)

        # Cross-correlation: prefer ensemble-averaged mean cycles (less noisy)
        if mean_cycle_prox is not None and mean_cycle_dist is not None:
            xcorr = _cross_correlation(mean_cycle_prox, mean_cycle_dist)
        else:
            # Fallback to raw signal if no cycles detected
            xcorr = _cross_correlation(proximal, distal)

        # Build chart data for angle-angle diagram
        # Downsample individual cycles to max 5 cycles for display
        cycle_data_for_chart: list[list[dict[str, float]]] = []
        for i, (cp, cd) in enumerate(zip(normalized_cycles_prox[:5], normalized_cycles_dist[:5])):
            cycle_points = [
                {"proximal": round(float(cp[j]), 1), "distal": round(float(cd[j]), 1), "pct": j}
                for j in range(0, NORM_POINTS, 2)  # Every other point
            ]
            cycle_data_for_chart.append(cycle_points)

        # Mean cycle for chart
        mean_cycle_chart = None
        if mean_cycle_prox is not None:
            mean_cycle_chart = [
                {
                    "proximal": round(float(mean_cycle_prox[j]), 1),
                    "distal": round(float(mean_cycle_dist[j]), 1),
                    "pct": j,
                }
                for j in range(0, NORM_POINTS, 2)
            ]

        # Strip side prefix for display label
        prox_label = proximal_name.replace("left_", "").replace("right_", "").capitalize()
        dist_label = distal_name.replace("left_", "").replace("right_", "").capitalize()
        pair_label = f"{prox_label}-{dist_label}"
        pair_results.append({
            "pair": pair_label,
            "proximal": proximal_name,
            "distal": distal_name,
            "num_cycles": len(normalized_cycles_prox),
            "cross_correlation": xcorr,
            "variability_score": variability_score,
            "coupling_angle_mean": round(float(np.mean(coupling_angle_mean)), 1) if coupling_angle_mean is not None else None,
            "cycle_data": cycle_data_for_chart,
            "mean_cycle": mean_cycle_chart,
        })

    logger.info(
        "COORDINATION_DEBUG",
        sport=sport_type,
        pairs_analyzed=len(pair_results),
    )

    return {
        "pairs": pair_results,
    }
