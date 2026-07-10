"""Module 2: Phase portraits (angle vs angular velocity).

Computes angular velocity via central differences, detects gait cycles
via peak detection, and measures movement stability via ConvexHull area
on the phase portrait (angle, angular_velocity) scatter.

Smaller hull area = more consistent/stable movement pattern.
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Joints to analyze per sport
SPORT_JOINTS = {
    "run": ["knee", "hip"],  # Unprefixed: near-side only (set by running_analyzer)
    "bike": ["left_knee", "right_knee", "left_hip", "right_hip", "left_ankle", "right_ankle"],
    "swim": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"],
}

# Minimum data points for meaningful phase portrait
MIN_POINTS = 15

# Minimum cycles for stability scoring
MIN_CYCLES = 2

# Expected ROM per joint per sport (degrees) for ROM penalty
# Sources: Bini et al. 2011, Novacheck 1998, Fonda & Sarabon 2012
EXPECTED_ROM = {
    "run": {
        "knee": 80, "hip": 45,  # Unprefixed: near-side only
    },
    "bike": {
        "left_knee": 70, "right_knee": 70,
        "left_hip": 40, "right_hip": 40,
        "left_ankle": 30, "right_ankle": 30,
    },
    "swim": {
        "left_shoulder": 160, "right_shoulder": 160,
        "left_elbow": 90, "right_elbow": 90,
    },
}


def _compute_angular_velocity(angles: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Compute angular velocity using central differences.

    Returns array of same length (endpoints use forward/backward difference).
    Units: degrees/second. NaN values in angles propagate to velocity.
    """
    n = len(angles)
    velocity = np.full(n, np.nan)

    if n < 3:
        return velocity

    # Central differences for interior points
    for i in range(1, n - 1):
        dt = timestamps[i + 1] - timestamps[i - 1]
        if dt > 0 and not (np.isnan(angles[i + 1]) or np.isnan(angles[i - 1])):
            velocity[i] = (angles[i + 1] - angles[i - 1]) / dt

    # Forward difference for first point
    dt0 = timestamps[1] - timestamps[0]
    if dt0 > 0 and not (np.isnan(angles[0]) or np.isnan(angles[1])):
        velocity[0] = (angles[1] - angles[0]) / dt0

    # Backward difference for last point
    dtn = timestamps[-1] - timestamps[-2]
    if dtn > 0 and not (np.isnan(angles[-1]) or np.isnan(angles[-2])):
        velocity[-1] = (angles[-1] - angles[-2]) / dtn

    return velocity


def _detect_cycles(angles: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
    """Detect movement cycles using peak detection on the angle signal.

    NaN-safe: filters out NaN before peak detection, maps indices back.
    Returns list of cycle dicts with start/end indices and timestamps.
    """
    from scipy.signal import find_peaks

    # Filter out NaN for peak detection
    valid_mask = ~np.isnan(angles)
    valid_angles = angles[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    if len(valid_angles) < MIN_POINTS:
        return []

    # Find peaks (maxima in angle signal = e.g., knee extension peaks)
    # Distance: at least 5 samples between peaks to avoid false positives
    prominence = max(5.0, np.nanstd(valid_angles) * 0.5)
    peaks, properties = find_peaks(valid_angles, distance=5, prominence=prominence)

    if len(peaks) < MIN_CYCLES:
        # Try with lower prominence
        peaks, properties = find_peaks(valid_angles, distance=4, prominence=prominence * 0.5)

    # Map peak indices back to original array coordinates
    original_peaks = valid_indices[peaks]

    cycles = []
    for i in range(len(original_peaks) - 1):
        start_idx = int(original_peaks[i])
        end_idx = int(original_peaks[i + 1])
        cycles.append({
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_time": float(timestamps[start_idx]),
            "end_time": float(timestamps[end_idx]),
            "duration": float(timestamps[end_idx] - timestamps[start_idx]),
        })

    return cycles


def _compute_hull_area(angles: np.ndarray, velocities: np.ndarray) -> float:
    """Compute ConvexHull area of the phase portrait scatter.

    Returns area in (degrees * degrees/second) units.
    Returns 0 if hull computation fails (e.g., collinear points).
    """
    from scipy.spatial import ConvexHull

    points = np.column_stack([angles, velocities])

    # Remove duplicates and NaN
    valid = ~(np.isnan(points).any(axis=1))
    points = points[valid]

    if len(points) < 4:
        return 0.0

    # Check for degeneracy (all points collinear)
    if np.std(points[:, 0]) < 1e-6 or np.std(points[:, 1]) < 1e-6:
        return 0.0

    try:
        hull = ConvexHull(points)
        return float(hull.volume)  # In 2D, volume = area
    except Exception:
        return 0.0


def _hull_area_to_score(
    area: float, angle_range: float,
    joint_name: str = "", sport_type: str = "run",
) -> float:
    """Convert hull area to stability score (0-100).

    Normalized by angle range to make scores comparable across joints.
    Smaller area = higher score (more stable pattern).
    Applies ROM penalty if observed ROM is below expected for the joint/sport.
    """
    if angle_range < 1.0:
        return 50.0  # Can't judge stability with no movement

    # Normalize area by angle_range^2 to get a dimensionless measure
    normalized = area / (angle_range ** 2)

    # Empirical mapping: normalized area of ~50 = decent, ~200+ = poor
    # Score: 100 at normalized=0, decreasing to 0
    score = max(0.0, 100.0 - normalized * 0.5)

    # ROM penalty: if joint shows less ROM than expected, discount the score
    expected_rom = EXPECTED_ROM.get(sport_type, {}).get(joint_name, 0)
    if expected_rom > 0:
        rom_factor = min(1.0, angle_range / expected_rom)
        score = score * rom_factor

    return round(score, 1)


def _get_joints_for_sport(sport_type: str, camera_side: str | None) -> list[str]:
    """Get joints to analyze, ordered near-side first for side-view sports."""
    base_joints = SPORT_JOINTS.get(sport_type, SPORT_JOINTS["run"])
    # Running: unprefixed keys, no reordering needed
    if sport_type in ("swim", "run") or not camera_side:
        return base_joints

    # Bike: reorder near-side first, then far-side
    near = camera_side
    far = "right" if near == "left" else "left"
    near_joints = [j for j in base_joints if j.startswith(near)]
    far_joints = [j for j in base_joints if j.startswith(far)]
    midline = [j for j in base_joints if not j.startswith("left") and not j.startswith("right")]
    return near_joints + midline + far_joints


def compute_phase_portraits(
    angle_history: dict[str, list[float]],
    timestamps: list[float],
    sport_type: str,
    camera_side: str | None = None,
) -> dict[str, Any]:
    """Compute phase portraits for sport-relevant joints.

    Args:
        angle_history: Filtered angle time-series (from Butterworth).
        timestamps: Time in seconds for each frame.
        sport_type: 'run', 'bike', or 'swim'.
        camera_side: 'left' or 'right' -- side facing camera (for tagging).

    Returns:
        Dict with per-joint phase portrait data and overall stability score.
    """
    joints = _get_joints_for_sport(sport_type, camera_side)
    ts = np.array(timestamps, dtype=np.float64)

    joint_results: dict[str, Any] = {}
    stability_scores: list[float] = []

    for joint in joints:
        if joint not in angle_history or len(angle_history[joint]) < MIN_POINTS:
            continue

        angles = np.array(angle_history[joint], dtype=np.float64)

        # Ensure same length as timestamps
        min_len = min(len(angles), len(ts))
        angles = angles[:min_len]
        t = ts[:min_len]

        # Compute angular velocity
        velocity = _compute_angular_velocity(angles, t)

        # Detect cycles
        cycles = _detect_cycles(angles, t)

        # Skip joints with 0 cycles (far-side noisy data, no meaningful pattern)
        if len(cycles) == 0:
            logger.info("PHASE_PORTRAIT_SKIP", joint=joint, reason="0_cycles")
            continue

        # Compute hull area for stability
        hull_area = _compute_hull_area(angles, velocity)
        valid_angles = angles[~np.isnan(angles)]
        angle_range = float(np.max(valid_angles) - np.min(valid_angles)) if len(valid_angles) > 0 else 0.0
        stability = _hull_area_to_score(hull_area, angle_range, joint_name=joint, sport_type=sport_type)
        stability_scores.append(stability)

        # Downsample data points for JSON (keep max 200 points, skip NaN)
        step = max(1, len(angles) // 200)
        data_points = [
            {"angle": round(float(angles[i]), 1), "velocity": round(float(velocity[i]), 1)}
            for i in range(0, len(angles), step)
            if not (np.isnan(angles[i]) or np.isnan(velocity[i]))
        ]

        valid_velocity = velocity[~np.isnan(velocity)]
        vel_range = float(np.max(valid_velocity) - np.min(valid_velocity)) if len(valid_velocity) > 0 else 0.0

        # Tag near/far side for side-view sports
        if sport_type == "run":
            # Running uses unprefixed keys -- always near-side, always reliable
            side = "near"
            reliable = True
        elif camera_side and sport_type == "bike":
            side = "near" if joint.startswith(camera_side) else "far"
            reliable = side == "near"
        else:
            side = "both"
            reliable = True

        joint_results[joint] = {
            "data_points": data_points,
            "hull_area": round(hull_area, 1),
            "stability_score": stability,
            "cycles": cycles,
            "angle_range": round(angle_range, 1),
            "velocity_range": round(vel_range, 1),
            "side": side,
            "reliable": reliable,
        }

    overall_stability = round(float(np.mean(stability_scores)), 1) if stability_scores else 50.0

    logger.info(
        "PHASE_PORTRAIT_DEBUG",
        sport=sport_type,
        camera_side=camera_side,
        joints_analyzed=len(joint_results),
        overall_stability=overall_stability,
    )

    return {
        "joints": joint_results,
        "overall_stability_score": overall_stability,
    }
