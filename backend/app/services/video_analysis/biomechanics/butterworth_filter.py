"""Module 1: Butterworth low-pass filter for angle time-series.

Applies a zero-phase 2nd-order Butterworth low-pass filter (effective 4th order
due to forward-backward pass) to each angle in angle_history.

Key design decisions:
- Filter ANGLES not raw landmarks (10 channels vs 99 coordinate channels)
- Batch post-hoc filtering (sosfiltfilt needs full signal, not real-time)
- The cutoff is CHOSEN PER ANGLE from the data, not fixed per sport -- see below
- NaN handling: linear interpolation before filter, restore NaN positions after

Choosing the cutoff: what was tried, and why it does not decide yet
-------------------------------------------------------------------
The obvious upgrade to one constant per sport is residual analysis (Winter,
*Biomechanics and Motor Control of Human Movement*, 4th ed., 2009) -- the method
Kinovea uses for the same job:

    for each candidate cutoff, filter, and look at the residual (raw minus
    filtered). If the residual is still autocorrelated, the filter took signal
    as well as noise. If it looks like white noise, exactly the noise came out.
    Pick the cutoff whose residual is least autocorrelated.

Autocorrelation is measured with the Durbin-Watson statistic, which sits at 2.0
for white noise, so ``|2 - DW| / 2`` is a score to minimise. That is implemented
here, in :func:`select_cutoff`, and it demonstrably works: given the same
movement with more noise on it, it picks a lower cutoff.

**It does not drive this filter, though, and the reason matters.** The method
assumes raw digitised coordinates with a broadband noise floor to find. By the
time a signal reaches this module that assumption is already false:
``landmark_stabilizer`` has low-passed the landmark COORDINATES at 6-8 Hz before
any angle was computed from them. There is no noise floor left above ~8 Hz, so
the sweep finds white residuals everywhere in its band and saturates at whatever
ceiling it is given -- on real clips, 10.0 Hz for run and 8.0 Hz for bike, i.e.
"this needs no further smoothing", which is the correct answer to the question
the method actually asked.

Acting on that answer would not be a precision improvement, it would be turning
the second smoothing stage off. Measured on one clip, that widened the knee's
p05-p95 band by about 11 degrees -- and those percentiles are what the scoring
grades against. So the operating cutoff stays the sport constant, and residual
analysis runs alongside it as a REPORTED DIAGNOSTIC (``suggested_cutoffs`` in
the returned meta) so the question can be settled on many real clips instead of
on one.

The open question it exposes is worth stating plainly: this pipeline filters
twice -- 8 Hz on landmarks, then 4 Hz on the angles derived from them -- and the
second pass is the binding constraint on every angle metric. It was chosen to
preserve movement frequencies ("gait cycle ~3 Hz, need headroom"), not to remove
noise, and the cascade has never been validated end to end.

Two guards on the diagnostic, because the method is not self-limiting:

* **A physiological clamp per sport.** Unbounded, the sweep will happily pick
  0.5 Hz for a near-static trunk and erase the real motion along with the noise.
* **Left/right pairs share one suggestion** (the lower of the two). Filtering
  the left knee at 8 Hz and the right at 3 Hz would make the symmetry and
  coupling analyses downstream partly a measurement of the filter rather than
  the athlete. Zero-phase filtering leaves relative PHASE intact whatever the
  cutoff, so this is about amplitude content, not lag.
"""

from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# The OPERATING cutoffs (Hz) -- what actually filters the angles. Chosen to
# preserve movement frequencies rather than to remove noise; see the module
# docstring for why residual analysis does not currently override them.
SPORT_CUTOFFS = {
    "run": 4.0,   # Gait cycle ~3Hz, need headroom
    "bike": 3.0,  # Pedaling ~1.5Hz, more static
    "swim": 3.5,  # Stroke rate ~1Hz, moderate movement
}

# How far the residual-analysis DIAGNOSTIC is allowed to roam, per sport (Hz).
#
# The ceilings are the standard gait-lab low-pass range rather than anything
# derived here: running kinematics are conventionally filtered at 6-10 Hz
# (Winter 2009), and pedalling, whose fundamental is ~1.5 Hz at 90 rpm with
# harmonics to ~4 Hz at the dead centres, needs less. The floors exist to stop
# the sweep erasing a real, slow signal -- a trunk angle that genuinely only
# moves at 1 Hz has a residual that looks white at almost any cutoff, and
# without a floor the score would happily choose the lowest one offered.
SPORT_CUTOFF_BOUNDS: dict[str, tuple[float, float]] = {
    "run": (1.5, 10.0),
    "bike": (1.0, 8.0),
    "swim": (1.0, 8.0),
}
DEFAULT_CUTOFF_BOUNDS = (1.0, 8.0)

# Minimum samples required for filtering (need enough for filter startup)
MIN_SAMPLES = 13

# Minimum samples before residual analysis is worth running at all.
# Durbin-Watson on a short series is dominated by its own sampling noise; at a
# typical 50 fps this is about 0.8 s of signal.
AUTO_CUTOFF_MIN_SAMPLES = 40

# Candidate cutoffs tried per channel. The score curve is shallow near its
# minimum, so a finer sweep buys precision nobody can use.
CUTOFF_SWEEP_STEPS = 24

# Fraction of Nyquist the cutoff may never exceed, whatever the sport bound
# says -- filtering close to Nyquist rings.
NYQUIST_SAFETY = 0.9

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


def durbin_watson(residuals: np.ndarray) -> float:
    """Autocorrelation of a residual series: 2.0 means white noise.

    Below 2 is positive autocorrelation (the filter is still removing signal);
    above 2 is negative autocorrelation (over-fitting the noise). Returns NaN
    for a degenerate series, which callers treat as "no answer" rather than as
    a good or bad score.
    """
    if residuals.size < 3:
        return float("nan")
    denominator = float(np.sum(residuals ** 2))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.sum(np.diff(residuals) ** 2) / denominator)


def select_cutoff(
    data: np.ndarray, effective_fps: float, bounds: tuple[float, float],
) -> tuple[float | None, float]:
    """Residual-analysis cutoff for one channel: ``(cutoff_hz, score)``.

    Sweeps the allowed band and returns the cutoff whose residual is least
    autocorrelated, with its ``|2 - DW| / 2`` score (0 is ideal). Returns
    ``(None, inf)`` when no candidate produced a usable statistic -- a flat
    channel, a series too short to filter, or an empty band.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = effective_fps / 2.0
    if nyquist <= 0 or data.size < MIN_SAMPLES:
        return None, float("inf")

    lo, hi = bounds
    hi = min(hi, NYQUIST_SAFETY * nyquist)
    lo = min(lo, hi)
    if hi <= 0 or not np.isfinite(hi) or hi <= lo:
        return None, float("inf")

    best_fc: float | None = None
    best_score = float("inf")
    for fc in np.linspace(lo, hi, CUTOFF_SWEEP_STEPS):
        try:
            sos = butter(FILTER_ORDER, fc / nyquist, btype="low", output="sos")
            residual = data - sosfiltfilt(sos, data)
        except ValueError:
            continue
        dw = durbin_watson(residual)
        if not np.isfinite(dw):
            continue
        score = abs(2.0 - dw) / 2.0
        if score < best_score:
            best_fc, best_score = float(fc), score
    return best_fc, best_score


def _base_name(angle_name: str) -> str:
    """``left_knee`` / ``right_knee`` -> ``knee``; anything else unchanged.

    Left and right of the same joint are compared with each other downstream
    (symmetry, continuous relative phase), so they have to be filtered alike.
    """
    for prefix in ("left_", "right_"):
        if angle_name.startswith(prefix):
            return angle_name[len(prefix):]
    return angle_name


def suggest_cutoffs(
    prepared: dict[str, np.ndarray], effective_fps: float, sport_type: str,
) -> dict[str, float]:
    """What residual analysis would choose for each channel, if it decided.

    Reported rather than applied -- see the module docstring. Channels too short
    for a trustworthy Durbin-Watson, and channels the sweep cannot answer for,
    are simply absent from the result rather than filled in with a guess.

    Left/right of the same joint are collapsed onto the lower of the two, so
    the numbers reported are the ones that would actually be used.
    """
    bounds = SPORT_CUTOFF_BOUNDS.get(sport_type, DEFAULT_CUTOFF_BOUNDS)
    suggested: dict[str, float] = {}
    for name, data in prepared.items():
        if data.size < AUTO_CUTOFF_MIN_SAMPLES:
            continue
        cutoff, _score = select_cutoff(data, effective_fps, bounds)
        if cutoff is not None:
            suggested[name] = cutoff

    group_min: dict[str, float] = {}
    for name, cutoff in suggested.items():
        base = _base_name(name)
        group_min[base] = min(group_min.get(base, cutoff), cutoff)
    return {
        name: round(group_min[_base_name(name)], 2) for name in suggested
    }


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
        Info dict with the filtered angle names, the cutoff used, and -- as a
        diagnostic that changes nothing -- what residual analysis would have
        chosen per channel instead.
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = effective_fps / 2.0
    ceiling = NYQUIST_SAFETY * nyquist
    # The operating cutoff: one per sport, clamped off Nyquist. Unchanged.
    cutoff = min(SPORT_CUTOFFS.get(sport_type, 4.0), ceiling)

    if nyquist <= 0 or ceiling <= 0:
        logger.warning(
            "BUTTERWORTH_SKIP",
            reason="invalid_frequencies",
            effective_fps=effective_fps,
            nyquist=nyquist,
        )
        return {
            "filtered": [], "skipped": list(angle_history.keys()),
            "reason": "invalid_fps",
        }

    # -- pass 1: prepare each channel ---------------------------------------
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    skipped_names: list[str] = []

    for angle_name, values in angle_history.items():
        if len(values) < MIN_SAMPLES:
            skipped_names.append(angle_name)
            continue
        data = np.array(values, dtype=np.float64)
        interpolated, nan_mask = _interpolate_nans(data)
        if np.sum(~nan_mask) < MIN_SAMPLES:
            skipped_names.append(angle_name)
            continue
        prepared[angle_name] = (interpolated, nan_mask)

    # -- residual-analysis diagnostic (reported, not applied) ---------------
    suggested: dict[str, float] = {}
    try:
        suggested = suggest_cutoffs(
            {k: v[0] for k, v in prepared.items()}, effective_fps, sport_type,
        )
    except Exception as e:  # noqa: BLE001 -- a diagnostic must not filter nothing
        logger.warning("BUTTERWORTH_SUGGEST_FAILED", err=str(e))

    # -- pass 2: filter -----------------------------------------------------
    filtered_names: list[str] = []
    for angle_name, (interpolated, nan_mask) in prepared.items():
        values = angle_history[angle_name]
        try:
            sos = butter(
                FILTER_ORDER, cutoff / nyquist, btype="low", output="sos",
            )
            filtered = sosfiltfilt(sos, interpolated)

            # Interpolation exists ONLY as scaffolding for the filter: every
            # sample that was NaN going in comes back NaN. The old policy
            # kept gaps of up to 5 frames -- ~170 ms at 30 fps -- as smooth
            # interpolated values in the series, plus 2 frames at the edges
            # of longer gaps, and everything downstream of this in-place
            # mutation (angle statistics, the quality gate, confidence, the
            # overlay's per-frame chips) then counted the invented samples
            # as measured. The identity gates upstream go to great lengths
            # to refuse a frame they cannot vouch for; a filter quietly
            # un-refusing them defeated that chain.
            filtered[nan_mask] = np.nan

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

    # Only report suggestions for channels that were actually filtered, so the
    # diagnostic describes the same set of signals the result rests on.
    suggested = {k: v for k, v in suggested.items() if k in filtered_names}
    gaps = [round(v - cutoff, 2) for v in suggested.values()]

    logger.info(
        "BUTTERWORTH_DEBUG",
        sport=sport_type,
        effective_fps=round(effective_fps, 1),
        nyquist=round(nyquist, 1),
        cutoff_hz=round(cutoff, 2),
        filter_order=FILTER_ORDER,
        filtered_count=len(filtered_names),
        skipped_count=len(skipped_names),
        # Diagnostic only -- see the module docstring. A large positive gap on
        # every channel means the angles arrived already denoised by the
        # landmark stage and this filter is smoothing signal, not noise.
        suggested_cutoffs=suggested,
        suggested_gap_hz=(
            [min(gaps), max(gaps)] if gaps else None
        ),
    )

    return {
        "filtered": filtered_names,
        "skipped": skipped_names,
        "cutoff_hz": round(cutoff, 2),
        "effective_fps": round(effective_fps, 1),
        "filter_order": FILTER_ORDER,
        # What residual analysis would have chosen, had it decided. Reported so
        # the cascade question can be settled across many real clips.
        "suggested_cutoffs": suggested,
        "suggested_method": "winter_residual_analysis",
        "suggested_applied": False,
    }
