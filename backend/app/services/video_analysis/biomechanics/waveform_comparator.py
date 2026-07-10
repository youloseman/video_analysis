"""Module 5: Waveform comparison against reference gait curves.

Compares the athlete's ensemble-averaged gait cycle against published
reference data (e.g., Novacheck 1998) using:
- RMSD (root mean square deviation)
- Cross-correlation (Pearson r)
- Deviation zones (consecutive frames where z-score > threshold)
- Similarity score combining r and RMSD
"""

import json
import os
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Joints to compare per sport (must have reference data available)
SPORT_COMPARE_JOINTS = {
    "run": {
        "knee": "knee",  # Unprefixed: near-side only (set by running_analyzer)
        "hip": "hip",
    },
    "bike": {
        "left_knee": "knee",
        "right_knee": "knee",
        "left_hip": "hip",
        "right_hip": "hip",
        "left_ankle": "ankle",
        "right_ankle": "ankle",
    },
    "swim": {
        "left_shoulder": "shoulder",
        "right_shoulder": "shoulder",
        "left_elbow": "elbow",
        "right_elbow": "elbow",
    },
}

NORM_POINTS = 101
MIN_CYCLES = 1

# Z-score threshold for deviation zones
Z_THRESHOLD = 2.0
# Minimum consecutive points for a deviation zone
MIN_CONSECUTIVE = 3

# Reference data cache
_reference_cache: dict[str, Any] = {}


def _load_reference(sport_type: str) -> dict[str, Any] | None:
    """Load reference gait data for the given sport."""
    if sport_type in _reference_cache:
        return _reference_cache[sport_type]

    ref_dir = os.path.join(os.path.dirname(__file__), "reference_data")
    ref_file = os.path.join(ref_dir, f"{sport_type}ning_reference.json" if sport_type == "run" else f"{sport_type}_reference.json")

    if not os.path.exists(ref_file):
        return None

    with open(ref_file, "r") as f:
        data = json.load(f)

    _reference_cache[sport_type] = data
    return data


def _compute_ensemble_average(
    angle_history: dict[str, list[float]],
    joint_name: str,
    cycles: list[dict[str, Any]],
) -> np.ndarray | None:
    """Compute ensemble average (mean cycle) for a joint.

    Returns 101-point array or None if insufficient data. NaN-safe:
    CubicSpline rejects non-finite y, so gated samples are linearly
    interpolated before spline fitting (and the cycle is dropped if it
    has fewer than 4 finite points).
    """
    from scipy.interpolate import CubicSpline

    def _prep(segment: np.ndarray) -> np.ndarray | None:
        """Return a NaN-free copy of segment via linear interpolation,
        or None if fewer than 4 finite samples remain."""
        finite = np.isfinite(segment)
        if finite.sum() < 4:
            return None
        if finite.all():
            return segment
        idx = np.arange(len(segment))
        return np.interp(idx, idx[finite], segment[finite])

    values = np.array(angle_history.get(joint_name, []), dtype=np.float64)
    if len(values) == 0:
        return None

    normalized: list[np.ndarray] = []
    for cycle in cycles:
        start = cycle["start_idx"]
        end = cycle["end_idx"]
        if end - start < 6 or end > len(values):
            continue

        segment = _prep(values[start:end])
        if segment is None:
            continue
        n = len(segment)
        x_old = np.linspace(0, 100, n)
        x_new = np.linspace(0, 100, NORM_POINTS)
        if n < 4:
            norm = np.interp(x_new, x_old, segment)
        else:
            cs = CubicSpline(x_old, segment)
            norm = cs(x_new)
        normalized.append(norm)

    if not normalized:
        # Fallback: use entire signal as one "cycle"
        if len(values) < 6:
            return None
        prepped = _prep(values)
        if prepped is None:
            return None
        n = len(prepped)
        x_old = np.linspace(0, 100, n)
        x_new = np.linspace(0, 100, NORM_POINTS)
        cs = CubicSpline(x_old, prepped)
        return cs(x_new)

    return np.mean(np.array(normalized), axis=0)


def _compute_rmsd(athlete: np.ndarray, reference: np.ndarray) -> float:
    """Compute RMSD between athlete and reference curves."""
    return float(np.sqrt(np.mean((athlete - reference) ** 2)))


def _find_deviation_zones(
    athlete: np.ndarray, ref_mean: np.ndarray, ref_std: np.ndarray
) -> list[dict[str, Any]]:
    """Find zones where athlete deviates significantly from reference.

    A deviation zone is where |z-score| > threshold for MIN_CONSECUTIVE+ points.
    """
    # Avoid division by zero
    std_safe = np.where(ref_std > 0.5, ref_std, 0.5)
    z_scores = (athlete - ref_mean) / std_safe

    zones = []
    in_zone = False
    zone_start = 0

    for i in range(len(z_scores)):
        if abs(z_scores[i]) > Z_THRESHOLD:
            if not in_zone:
                in_zone = True
                zone_start = i
        else:
            if in_zone:
                length = i - zone_start
                if length >= MIN_CONSECUTIVE:
                    mean_z = float(np.mean(z_scores[zone_start:i]))
                    zones.append({
                        "start_pct": zone_start,
                        "end_pct": i,
                        "mean_z_score": round(mean_z, 2),
                        "direction": "above" if mean_z > 0 else "below",
                    })
                in_zone = False

    # Handle zone at end
    if in_zone:
        length = len(z_scores) - zone_start
        if length >= MIN_CONSECUTIVE:
            mean_z = float(np.mean(z_scores[zone_start:]))
            zones.append({
                "start_pct": zone_start,
                "end_pct": len(z_scores) - 1,
                "mean_z_score": round(mean_z, 2),
                "direction": "above" if mean_z > 0 else "below",
            })

    return zones


def _auto_correct_alignment(
    athlete_curve: np.ndarray, ref_mean: np.ndarray, initial_r: float,
) -> tuple[np.ndarray, float, str | None]:
    """Try phase shifts and convention flips to fix negative/low correlations.

    Returns (corrected_curve, best_r, correction_name_or_None).
    """
    best_curve = athlete_curve
    best_r = initial_r
    correction = None

    # Try phase shifts: 25%, 50%, 75% of cycle
    if best_r < 0.3:
        for shift_pct in [0.5, 0.25, 0.75]:
            shift_n = int(len(athlete_curve) * shift_pct)
            shifted = np.roll(athlete_curve, shift_n)
            if np.std(shifted) > 0 and np.std(ref_mean) > 0:
                r = float(np.corrcoef(shifted, ref_mean)[0, 1])
                if r > best_r:
                    best_curve = shifted
                    best_r = r
                    correction = f"phase_shift_{int(shift_pct * 100)}pct"

    # Try convention flip: 180 - athlete
    if best_r < 0.3:
        flipped = 180.0 - athlete_curve
        if np.std(flipped) > 0:
            r = float(np.corrcoef(flipped, ref_mean)[0, 1])
            if r > best_r:
                best_curve = flipped
                best_r = r
                correction = "convention_flip_180"

    # Try flip + shift combinations
    if best_r < 0.3:
        flipped = 180.0 - athlete_curve
        for shift_pct in [0.25, 0.5, 0.75]:
            shift_n = int(len(flipped) * shift_pct)
            shifted_flipped = np.roll(flipped, shift_n)
            if np.std(shifted_flipped) > 0:
                r = float(np.corrcoef(shifted_flipped, ref_mean)[0, 1])
                if r > best_r:
                    best_curve = shifted_flipped
                    best_r = r
                    correction = f"flip_180_shift_{int(shift_pct * 100)}pct"

    if correction:
        logger.info(
            "WAVEFORM_AUTO_CORRECT",
            correction=correction,
            original_r=round(initial_r, 3),
            corrected_r=round(best_r, 3),
        )

    return best_curve, best_r, correction


def _get_compare_joints(sport_type: str, camera_side: str | None) -> dict[str, str]:
    """Get joints to compare, using near-side only for side-view sports."""
    base_map = SPORT_COMPARE_JOINTS.get(sport_type, {})
    if sport_type == "swim" or not camera_side:
        return base_map

    # For run/bike: only compare near-side joints
    near = camera_side
    return {k: v for k, v in base_map.items() if k.startswith(near)}


def compute_waveform_comparison(
    angle_history: dict[str, list[float]],
    timestamps: list[float],
    sport_type: str,
    phase_data: dict[str, Any] | None = None,
    camera_side: str | None = None,
) -> dict[str, Any]:
    """Compare athlete waveforms against reference data.

    Args:
        angle_history: Filtered angle time-series.
        timestamps: Time in seconds per frame.
        sport_type: 'run', 'bike', or 'swim'.
        phase_data: Phase portrait output (for cycle boundaries).
        camera_side: 'left' or 'right' for near-side joint selection.

    Returns:
        Dict with per-joint comparisons and overall similarity score.
    """
    joint_map = _get_compare_joints(sport_type, camera_side)
    if not joint_map:
        return {"comparisons": [], "overall_similarity_score": None}

    reference = _load_reference(sport_type)
    if not reference:
        logger.info("WAVEFORM_SKIP", reason="no_reference_data", sport=sport_type)
        return {"comparisons": [], "overall_similarity_score": None}

    ref_joints = reference.get("joints", {})
    comparisons: list[dict[str, Any]] = []
    similarity_scores: list[float] = []

    for joint_name, ref_key in joint_map.items():
        if joint_name not in angle_history:
            continue
        if ref_key not in ref_joints:
            continue

        ref_data = ref_joints[ref_key]
        ref_mean = np.array(ref_data["mean"], dtype=np.float64)
        ref_std = np.array(ref_data["std"], dtype=np.float64)

        # Get cycles for this joint
        cycles = []
        if phase_data and "joints" in phase_data:
            joint_phase = phase_data["joints"].get(joint_name, {})
            cycles = joint_phase.get("cycles", [])

        # Ensemble average
        athlete_curve = _compute_ensemble_average(angle_history, joint_name, cycles)
        if athlete_curve is None:
            continue

        # Ensure same length
        if len(athlete_curve) != len(ref_mean):
            continue

        # Cross-correlation (Pearson r)
        original_r = 0.0
        if np.std(athlete_curve) > 0 and np.std(ref_mean) > 0:
            original_r = float(np.corrcoef(athlete_curve, ref_mean)[0, 1])

        # Auto-correct alignment if correlation is poor
        r = original_r
        correction_applied = None
        display_curve = athlete_curve
        if original_r < 0.3:
            display_curve, r, correction_applied = _auto_correct_alignment(
                athlete_curve, ref_mean, original_r,
            )

        # RMSD (using potentially corrected curve)
        rmsd = _compute_rmsd(display_curve, ref_mean)

        # Deviation zones
        deviation_zones = _find_deviation_zones(display_curve, ref_mean, ref_std)

        # Similarity: 70% shape (r), 30% offset (RMSD with gentler penalty)
        r_component = max(0.0, r)
        rmsd_component = max(0.0, 1.0 - rmsd / 30.0)
        similarity = 100.0 * (0.7 * r_component + 0.3 * rmsd_component)
        similarity = round(min(100.0, max(0.0, similarity)), 1)
        similarity_scores.append(similarity)

        # Chart data (downsample for JSON)
        step = max(1, NORM_POINTS // 50)  # ~50 points
        chart_data = [
            {
                "pct": i,
                "athlete": round(float(display_curve[i]), 1),
                "ref_mean": round(float(ref_mean[i]), 1),
                "ref_upper": round(float(ref_mean[i] + ref_std[i]), 1),
                "ref_lower": round(float(ref_mean[i] - ref_std[i]), 1),
            }
            for i in range(0, NORM_POINTS, step)
        ]

        comp_entry: dict[str, Any] = {
            "joint": joint_name,
            "ref_key": ref_key,
            "rmsd": round(rmsd, 2),
            "correlation_r": round(r, 3),
            "similarity_score": similarity,
            "deviation_zones": deviation_zones,
            "chart_data": chart_data,
        }
        if correction_applied:
            comp_entry["correction_applied"] = correction_applied
            comp_entry["original_r"] = round(original_r, 3)

        comparisons.append(comp_entry)

    overall = round(float(np.mean(similarity_scores)), 1) if similarity_scores else None

    logger.info(
        "WAVEFORM_DEBUG",
        sport=sport_type,
        joints_compared=len(comparisons),
        overall_similarity=overall,
    )

    return {
        "comparisons": comparisons,
        "overall_similarity_score": overall,
    }
