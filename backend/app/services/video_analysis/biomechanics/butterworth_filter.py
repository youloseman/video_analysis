"""Module 1: Butterworth low-pass filter for angle time-series.

Applies a zero-phase 2nd-order Butterworth low-pass filter (effective 4th order
due to forward-backward pass) to each angle in angle_history.

Key design decisions:
- Filter ANGLES not raw landmarks (10 channels vs 99 coordinate channels)
- Batch post-hoc filtering (sosfiltfilt needs full signal, not real-time)
- Cutoff capped to 0.9*Nyquist (FRAME_SAMPLE_RATE=3 gives ~10fps, Nyquist=5Hz)
- NaN handling: linear interpolation before filter, restore NaN positions after
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Sport-specific cutoff frequencies (Hz)
# These capture the fundamental movement frequencies while removing noise
SPORT_CUTOFFS = {
    "run": 4.0,   # Gait cycle ~3Hz, need headroom
    "bike": 3.0,  # Pedaling ~1.5Hz, more static
    "swim": 3.5,  # Stroke rate ~1Hz, moderate movement
}

# Minimum samples required for filtering (need enough for filter startup)
MIN_SAMPLES = 13

# Butterworth filter order (2nd order SOS, effective 4th order with filtfilt)
FILTER_ORDER = 2


def _interpolate_nans(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace NaN values with linear interpolation.

    Returns (interpolated_data, nan_mask) so NaNs can be restored after filtering.
    """
    nan_mask = np.isnan(data)
    if not np.any(nan_mask):
        return data.copy(), nan_mask

    valid = ~nan_mask
    if np.sum(valid) < 2:
        return data.copy(), nan_mask

    interpolated = data.copy()
    indices = np.arange(len(data))
    interpolated[nan_mask] = np.interp(
        indices[nan_mask], indices[valid], data[valid]
    )
    return interpolated, nan_mask


def apply_butterworth_filter(
    angle_history: dict[str, list[float]],
    effective_fps: float,
    sport_type: str,
) -> dict[str, Any]:
    """Apply Butterworth low-pass filter to all angles in-place.

    Args:
        angle_history: Dict of angle_name -> list of values. MUTATED in-place.
        effective_fps: Actual frames per second from timestamps.
        sport_type: 'run', 'bike', or 'swim'.

    Returns:
        Info dict with filtered angle names and parameters used.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = effective_fps / 2.0
    desired_cutoff = SPORT_CUTOFFS.get(sport_type, 4.0)

    # Clamp cutoff to 90% of Nyquist to avoid filter instability
    cutoff = min(desired_cutoff, 0.9 * nyquist)

    if cutoff <= 0 or nyquist <= 0:
        logger.warning(
            "BUTTERWORTH_SKIP",
            reason="invalid_frequencies",
            effective_fps=effective_fps,
            nyquist=nyquist,
        )
        return {"filtered": [], "skipped": list(angle_history.keys()), "reason": "invalid_fps"}

    # Normalized cutoff for scipy (fraction of Nyquist)
    wn = cutoff / nyquist

    # Design filter once (reuse for all angles)
    sos = butter(FILTER_ORDER, wn, btype="low", output="sos")

    filtered_names: list[str] = []
    skipped_names: list[str] = []

    for angle_name, values in angle_history.items():
        if len(values) < MIN_SAMPLES:
            skipped_names.append(angle_name)
            continue

        data = np.array(values, dtype=np.float64)

        # Handle NaN values
        interpolated, nan_mask = _interpolate_nans(data)

        if np.sum(~nan_mask) < MIN_SAMPLES:
            skipped_names.append(angle_name)
            continue

        try:
            filtered = sosfiltfilt(sos, interpolated)

            # Restore NaN for large gaps (>5 consecutive NaN in original).
            # Interpolation over large gaps is unreliable -- mark center as NaN.
            MAX_INTERP_GAP = 5
            MARGIN = 2  # keep interpolated values at gap edges
            gap_start = None
            for i in range(len(nan_mask)):
                if nan_mask[i]:
                    if gap_start is None:
                        gap_start = i
                else:
                    if gap_start is not None:
                        gap_len = i - gap_start
                        if gap_len > MAX_INTERP_GAP:
                            restore_from = min(gap_start + MARGIN, i)
                            restore_to = max(i - MARGIN, gap_start)
                            if restore_from < restore_to:
                                filtered[restore_from:restore_to] = np.nan
                        gap_start = None
            # Handle gap at end of signal
            if gap_start is not None:
                gap_len = len(nan_mask) - gap_start
                if gap_len > MAX_INTERP_GAP:
                    restore_from = min(gap_start + MARGIN, len(nan_mask))
                    if restore_from < len(nan_mask):
                        filtered[restore_from:] = np.nan

            # Mutate in-place
            for i in range(len(values)):
                values[i] = float(filtered[i])

            nan_pct = np.sum(nan_mask) / len(nan_mask) * 100
            if nan_pct > 0:
                logger.info(
                    "BUTTERWORTH_NAN",
                    angle=angle_name,
                    nan_frames=int(np.sum(nan_mask)),
                    nan_pct=round(nan_pct, 1),
                )

            filtered_names.append(angle_name)
        except Exception as e:
            logger.warning("BUTTERWORTH_ANGLE_FAIL", angle=angle_name, err=str(e))
            skipped_names.append(angle_name)

    logger.info(
        "BUTTERWORTH_DEBUG",
        sport=sport_type,
        effective_fps=round(effective_fps, 1),
        nyquist=round(nyquist, 1),
        cutoff_hz=round(cutoff, 2),
        filter_order=FILTER_ORDER,
        filtered_count=len(filtered_names),
        skipped_count=len(skipped_names),
    )

    return {
        "filtered": filtered_names,
        "skipped": skipped_names,
        "cutoff_hz": round(cutoff, 2),
        "effective_fps": round(effective_fps, 1),
        "filter_order": FILTER_ORDER,
    }
