"""Post-processing stabilizer for MediaPipe landmarks.

Three passes over raw frame results:
1. Visibility gating -- low-confidence landmarks are replaced with NaN so
   downstream consumers (angle calculator, Butterworth) skip them instead
   of smoothing noise into the signal. Threshold is keyed by
   (sport_type, camera_angle) so swim above/under water get different floors.
2. Anti-flip correction -- detects when MediaPipe swaps left/right sides
   and swaps them back using Z-depth consistency of hips.
3. One Euro Filter smoothing on (x, y, z) per landmark to reduce jitter
   with velocity-adaptive cutoff (replaces the old fixed-alpha EMA).

Applied BETWEEN MediaPipe detection and sport-specific analysis.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis.biomechanics.one_euro import OneEuro

logger = structlog.get_logger(__name__)

# All left/right landmark pairs to swap when a flip is detected
_SWAP_PAIRS = [
    (11, 12),  # shoulders
    (13, 14),  # elbows
    (15, 16),  # wrists
    (17, 18),  # pinkies
    (19, 20),  # index fingers
    (21, 22),  # thumbs
    (23, 24),  # hips
    (25, 26),  # knees
    (27, 28),  # ankles
    (29, 30),  # heels
    (31, 32),  # foot index
    (1, 4),    # eye inner
    (2, 5),    # eye
    (3, 6),    # eye outer
    (7, 8),    # ears
    (9, 10),   # mouth
]


# Visibility gate -- below this, landmark coordinates become NaN.
# Swim above-water is strict: glare/splash makes low-conf = hallucination.
# Swim under-water is lenient: distortion depresses confidence but points
# are usually valid.
MIN_VISIBILITY: dict[tuple[str, str | None], float] = {
    ("run",  None):          0.3,
    ("bike", None):          0.4,
    ("swim", "above_water"): 0.6,
    ("swim", "under_water"): 0.3,
    ("swim", None):          0.5,  # fallback
}


# One Euro params per (sport, camera_angle).
# Above-water: low min_cutoff + low beta -> aggressive smoothing, tolerates
# lag to kill splash jitter.
# Under-water: high min_cutoff + higher beta -> responsive, captures fast
# catch kinematics.
ONE_EURO_PARAMS: dict[tuple[str, str | None], dict[str, float]] = {
    ("run",  None):          dict(min_cutoff=1.7, beta=0.10, d_cutoff=1.0),
    ("bike", None):          dict(min_cutoff=1.5, beta=0.05, d_cutoff=1.0),
    ("swim", "above_water"): dict(min_cutoff=0.6, beta=0.01, d_cutoff=1.0),
    ("swim", "under_water"): dict(min_cutoff=3.0, beta=0.70, d_cutoff=1.0),
    ("swim", None):          dict(min_cutoff=1.0, beta=0.05, d_cutoff=1.0),
}


def _lookup(
    table: dict[tuple[str, str | None], Any],
    sport: str,
    camera_angle: str | None,
    default: Any,
) -> Any:
    if (sport, camera_angle) in table:
        return table[(sport, camera_angle)]
    if (sport, None) in table:
        return table[(sport, None)]
    return default


def _to_mutable(landmark: Any) -> SimpleNamespace:
    """Convert a MediaPipe landmark (possibly protobuf) to a mutable object."""
    return SimpleNamespace(
        x=landmark.x,
        y=landmark.y,
        z=landmark.z,
        visibility=getattr(landmark, "visibility", 1.0),
    )


def _ensure_mutable(frame_results: list[dict[str, Any]]) -> None:
    """Convert all landmarks in frame_results to mutable SimpleNamespace objects.

    MediaPipe Tasks API returns protobuf objects that don't support setattr.
    This converts them once so all downstream code can freely mutate coordinates.
    """
    for frame in frame_results:
        for key in ("world_landmarks", "normalized_landmarks"):
            landmarks = frame[key]
            if landmarks and isinstance(landmarks[0], SimpleNamespace):
                continue
            frame[key] = [_to_mutable(lm) for lm in landmarks]


def _use_butterworth_landmarks(
    sport_type: str, camera_angle: str | None, camera_view: str | None,
) -> bool:
    """Decide which landmark smoother applies for this clip.

    Butterworth (zero-phase, post-hoc) is chosen when the whole signal
    is available up front and skeleton stability beats real-time
    responsiveness. One Euro (causal, adaptive) is the default elsewhere.

    Cases routed to Butterworth:
      - swim under-water: One Euro lags the fast catch transient.
      - bike side-view: visualizer landmark jitter — need landmark-level
        smoothing, not just the angle Butterworth that runs later.
        Excludes bike rear-view, which has its own 1.2 Hz Butterworth
        inside ``PelvicStabilityAnalyzer`` and would over-smooth if
        filtered twice.
    """
    if sport_type == "swim" and camera_angle == "under_water":
        return True
    if sport_type == "bike" and camera_view != "rear":
        return True
    return False


def stabilize_landmarks(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None = None,
    fps: float = 30.0,
    context: dict[str, Any] | None = None,
    camera_view: str | None = None,
) -> list[dict[str, Any]]:
    """Stabilize landmark sequence: gate low-visibility, fix flips, smooth.

    Args:
        frame_results: List of frame dicts with 'world_landmarks' and
            'normalized_landmarks' keys (from pipeline _iterate_video_frames).
        sport_type: "run" | "bike" | "swim".
        camera_angle: For swim, "above_water" or "under_water". None for other sports.
        fps: Video frame rate, used by the One Euro / Butterworth filter.
        context: Optional mutable dict — when provided, the function
            populates ``context["butterworth_meta"]`` with cutoff
            diagnostics so the caller (pipeline.py) can surface warnings
            without changing this function's return type.
        camera_view: For bike/run, "side" or "rear". Used to keep bike
            rear-view on One Euro (PelvicStabilityAnalyzer applies its
            own 1.2 Hz Butterworth). None for swim and for legacy callers.

    Returns:
        The same list, mutated in place.
    """
    if len(frame_results) < 3:
        return frame_results

    _ensure_mutable(frame_results)
    dropped = _gate_by_visibility(frame_results, sport_type, camera_angle)
    flips = _fix_flips(frame_results, sport_type)
    if _use_butterworth_landmarks(sport_type, camera_angle, camera_view):
        smoothed, butter_meta = _apply_butterworth_landmarks(
            frame_results, sport_type, camera_angle, fps,
        )
        if context is not None:
            context["butterworth_meta"] = butter_meta
    else:
        smoothed = _apply_one_euro(
            frame_results, sport_type, camera_angle, fps,
        )

    logger.info(
        "LANDMARK_STABILIZER",
        sport=sport_type,
        camera_angle=camera_angle,
        camera_view=camera_view,
        frames=len(frame_results),
        gated_out=dropped,
        flips_corrected=flips,
        smoothed_landmarks=smoothed,
    )

    return frame_results


def _gate_by_visibility(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None,
) -> int:
    """Mark low-confidence landmarks as NaN.

    Downstream angle calculation and Butterworth filtering already handle
    NaN (see butterworth_filter.restore_nans) -- gating here propagates
    gaps through cleanly instead of letting the smoother blend noise into
    the signal.
    """
    thr = _lookup(MIN_VISIBILITY, sport_type, camera_angle, 0.5)
    dropped = 0
    for frame in frame_results:
        for key in ("world_landmarks", "normalized_landmarks"):
            for lm in frame[key]:
                if getattr(lm, "visibility", 1.0) < thr:
                    lm.x = math.nan
                    lm.y = math.nan
                    lm.z = math.nan
                    dropped += 1
    return dropped


def _fix_flips(frame_results: list[dict[str, Any]], sport_type: str) -> int:
    """Detect and correct left/right skeleton flips.

    MediaPipe can swap left and right sides between frames, especially in
    profile views (cycling, running). We detect this by checking Z-depth
    consistency of hips: in a stable side-view, the near-side hip should
    always have a smaller Z than the far-side hip.

    DISABLED FOR BIKE: anti-flip relies on world_landmarks Z, which is
    unreliable when the far-side hip is occluded by the torso (always the
    case for bike side-view). Strict Z comparison without hysteresis flips
    landmarks frame-to-frame on noise, corrupting knee_angle histories and
    tripping the pedal-stroke quality gate. Riders don't physically flip
    during a recording, so anti-flip provides no value here. See
    diagnostic notes 2026-04-29.
    """
    if sport_type == "bike":
        logger.info("LANDMARK_STABILIZER anti-flip skipped (bike)")
        return 0

    if len(frame_results) < 5:
        return 0

    calibration_frames = min(10, len(frame_results))
    left_closer_votes = 0

    for i in range(calibration_frames):
        wl = frame_results[i]["world_landmarks"]
        try:
            z_left, z_right = wl[23].z, wl[24].z
            if math.isnan(z_left) or math.isnan(z_right):
                continue
            if z_left < z_right:
                left_closer_votes += 1
        except (IndexError, AttributeError):
            continue

    expect_left_closer = left_closer_votes > calibration_frames / 2
    flip_count = 0

    for i in range(len(frame_results)):
        wl = frame_results[i]["world_landmarks"]

        try:
            z_left, z_right = wl[23].z, wl[24].z
            if math.isnan(z_left) or math.isnan(z_right):
                continue
            hip_left_closer = z_left < z_right
        except (IndexError, AttributeError):
            continue

        if hip_left_closer == expect_left_closer:
            continue

        # Confirm with shoulders to reduce false positives
        try:
            zs_left, zs_right = wl[11].z, wl[12].z
            if not (math.isnan(zs_left) or math.isnan(zs_right)):
                sh_left_closer = zs_left < zs_right
                if sh_left_closer == expect_left_closer:
                    continue
        except (IndexError, AttributeError):
            pass

        for key in ("world_landmarks", "normalized_landmarks"):
            lm = frame_results[i][key]
            for left_idx, right_idx in _SWAP_PAIRS:
                try:
                    lm[left_idx], lm[right_idx] = lm[right_idx], lm[left_idx]
                except IndexError:
                    continue
        flip_count += 1

    return flip_count


def _apply_one_euro(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None,
    fps: float,
) -> int:
    """Apply One Euro Filter to landmark (x, y, z) across frames.

    One filter instance per (landmark_set, landmark_idx, coord). NaN inputs
    pass through as the previous estimate so visibility-gated gaps are
    preserved without breaking the recursion.

    Bike is strictly 2D sagittal-plane: z is unused downstream and the
    far-side hip occlusion makes z noise meaningless. Skip z smoothing for
    bike to avoid spending filter state on a coordinate we then ignore.
    Run and swim continue smoothing all three coordinates as before.
    """
    if len(frame_results) < 2:
        return 0

    params = _lookup(
        ONE_EURO_PARAMS,
        sport_type,
        camera_angle,
        dict(min_cutoff=1.0, beta=0.05, d_cutoff=1.0),
    )

    first_wl = frame_results[0]["world_landmarks"]
    first_nl = frame_results[0]["normalized_landmarks"]
    n_landmarks = min(len(first_wl), len(first_nl), 33)

    smooth_z = sport_type != "bike"

    filters = {
        "world_landmarks": [
            [OneEuro(freq=fps, **params) for _ in range(3)]
            for _ in range(n_landmarks)
        ],
        "normalized_landmarks": [
            [OneEuro(freq=fps, **params) for _ in range(3)]
            for _ in range(n_landmarks)
        ],
    }

    for frame in frame_results:
        for key in ("world_landmarks", "normalized_landmarks"):
            lms = frame[key]
            for i in range(n_landmarks):
                lm = lms[i]
                lm.x = filters[key][i][0](lm.x)
                lm.y = filters[key][i][1](lm.y)
                if smooth_z:
                    lm.z = filters[key][i][2](lm.z)

    return n_landmarks


# Zero-phase Butterworth params for landmark coordinate smoothing.
# Keyed by (sport, camera_angle). swim_under and bike side-view use this
# path; other modes keep the causal One Euro filter.
BUTTER_LANDMARK_CUTOFF_HZ: dict[tuple[str, str | None], float] = {
    # 8 Hz lets the sharp catch-phase elbow flexion and wrist-y peaks
    # through. Relies on SPORT_SAMPLE_RATES["swim"] == 1 so effective
    # fps is 30 Hz -> Nyquist 15 Hz, leaving headroom above the clamp.
    ("swim", "under_water"): 8.0,
    # Bike side-view: pedal cadence is ~1.5 Hz at 90 RPM; harmonic
    # content extends to ~3-4 Hz at TDC/BDC direction reversals. 6 Hz
    # cutoff preserves real motion (4x fundamental) while removing
    # MediaPipe re-detection jitter (typical 8-15 Hz). The fps-adaptive
    # cap (0.2 * effective_fps) usually binds first at 30 fps -> 6 Hz,
    # at 60 fps the target wins (cap 12, target 6).
    ("bike", None): 6.0,
}
BUTTER_LANDMARK_ORDER = 4
# filtfilt needs enough samples beyond the padlen (default 3*order*2 for
# SOS) to be numerically stable. 24 matches what the existing angle
# Butterworth helper uses as its minimum.
MIN_BUTTER_SAMPLES = 24

# Filter-stability ceiling: cutoff must stay well below Nyquist to avoid
# ringing at the edge. 0.85 is empirically safe for swim-catch energy.
NYQUIST_SAFETY_FACTOR = 0.85

# Adaptive cutoff cap: cutoff should not exceed FPS_ADAPTIVE_FRACTION *
# fps. This ties smoothing to the actual sampling frequency so that a
# 30 fps video gets proportionally more smoothing than a 60 fps one.
# Rationale: at 30 fps the fixed 8 Hz target is 53% of Nyquist -- that
# passes almost all high-frequency content through, so the skeleton
# appears to jitter frame-to-frame. Capping at 0.2*fps gives a
# consistent smoothness profile across source framerates.
#
# With the default constants, the adaptive cap is always tighter than
# the Nyquist safety ceiling (0.2*fps < 0.85*fps/2 = 0.425*fps), so
# "fps_adaptive" is the reason that fires in practice. "nyquist_safety"
# remains in the reason-selection logic for non-default call sites that
# pass a larger fraction.
FPS_ADAPTIVE_FRACTION = 0.2


def _compute_safe_butterworth_cutoff(
    target_cutoff_hz: float,
    effective_fps: float,
    nyquist_safety_factor: float = NYQUIST_SAFETY_FACTOR,
    fps_adaptive_fraction: float = FPS_ADAPTIVE_FRACTION,
) -> dict[str, Any]:
    """Compute a clamped Butterworth cutoff and quantify any degradation.

    The actual cutoff is the minimum of three bounds:
      - ``target_cutoff_hz`` (the ideal cutoff for the sport/mode)
      - ``fps_adaptive_fraction * effective_fps`` (fps-adaptive cap that
        keeps the smoothness profile consistent across source
        framerates -- stops a 30 fps clip from looking jittery while
        leaving 60 fps untouched)
      - ``nyquist_safety_factor * nyquist`` (hard stability ceiling
        against ringing at the Nyquist edge)

    Returns a metadata dict with the actual cutoff, reduction
    percentage, which bound was binding (``reduction_reason``), and a
    human-readable warning when the reduction exceeds 20%.
    """
    nyquist = effective_fps / 2.0
    safety_ceiling = nyquist_safety_factor * nyquist
    adaptive_cap = fps_adaptive_fraction * effective_fps

    actual = min(target_cutoff_hz, adaptive_cap, safety_ceiling)
    reduction_pct = (
        round((target_cutoff_hz - actual) / target_cutoff_hz * 100, 1)
        if target_cutoff_hz > 0 else 0.0
    )

    reduction_reason: str | None = None
    if actual < target_cutoff_hz:
        # If both caps are below target, the tighter one wins; report
        # that as the reason so triage knows which bound to tune.
        reduction_reason = (
            "fps_adaptive" if adaptive_cap <= safety_ceiling
            else "nyquist_safety"
        )

    warning: str | None = None
    if reduction_pct > 20:
        warning = (
            f"Effective fps {effective_fps:.1f} is low. Smoothing cutoff "
            f"reduced from {target_cutoff_hz:.1f} to {actual:.1f} Hz "
            f"({reduction_pct:.0f}%) to maintain skeleton stability."
        )
    return {
        "target_cutoff_hz": target_cutoff_hz,
        "actual_cutoff_hz": round(actual, 2),
        "effective_fps": round(effective_fps, 2),
        "nyquist_hz": round(nyquist, 2),
        "reduction_pct": reduction_pct,
        "reduction_reason": reduction_reason,
        "warning": warning,
    }


def _apply_butterworth_landmarks(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None,
    fps: float,
) -> tuple[int, dict[str, Any]]:
    """Zero-phase 4th-order Butterworth on every (landmark, axis) series.

    Non-causal (sosfiltfilt = forward + backward) so there is no phase
    lag through fast transients like the underwater swim catch. NaN gaps
    from the P0 visibility gate are linearly interpolated before
    filtering and re-masked afterwards so gated frames stay hidden
    downstream.

    Returns (count_of_filtered_series, butterworth_metadata_dict).
    Falls back to _apply_one_euro when the frame count is below the
    stable filtfilt minimum (returns 0 metadata in that case).
    """
    n = len(frame_results)
    if n < MIN_BUTTER_SAMPLES:
        # NOTE: This fallback path is defensive as of 2026-04-17.
        # After the 2-second minimum duration check in validator.py, no
        # normal swim upload can reach this branch (MIN_BUTTER_SAMPLES=24
        # ≈ 0.8 s at 30 fps). If this fires in production, something
        # upstream failed to validate.
        if sport_type == "swim" and camera_angle == "under_water":
            logger.warning(
                "BUTTERWORTH_FALLBACK",
                sport=sport_type,
                camera_angle=camera_angle,
                frame_count=n,
                min_required=MIN_BUTTER_SAMPLES,
                msg=(
                    "Butterworth fallback triggered for swim_under. "
                    "This should not happen after the 2s minimum duration "
                    "validation -- investigate upstream."
                ),
            )
        count = _apply_one_euro(frame_results, sport_type, camera_angle, fps)
        fallback_meta: dict[str, Any] = {
            "fallback_triggered": True,
            "fallback_reason": "insufficient_samples",
            "frame_count": n,
            "min_required": MIN_BUTTER_SAMPLES,
            "warning": (
                "Video was too short for optimal filtering. Analysis used "
                "a fallback smoother -- results may be less precise. "
                "Consider a longer clip (3+ seconds)."
            ) if sport_type == "swim" and camera_angle == "under_water" else None,
        }
        return count, fallback_meta

    from scipy.signal import butter, sosfiltfilt

    try:
        from app.services.video_analysis.pipeline import SPORT_SAMPLE_RATES
        sample_rate = SPORT_SAMPLE_RATES.get(sport_type, 1)
    except Exception:
        sample_rate = 1
    effective_fps = max(fps / max(sample_rate, 1), 1.0)
    target_cutoff = BUTTER_LANDMARK_CUTOFF_HZ.get((sport_type, camera_angle), 4.0)
    cutoff_info = _compute_safe_butterworth_cutoff(target_cutoff, effective_fps)
    cutoff_hz = cutoff_info["actual_cutoff_hz"]
    nyquist = cutoff_info["nyquist_hz"]
    wn = cutoff_hz / nyquist
    sos = butter(BUTTER_LANDMARK_ORDER, wn, btype="low", output="sos")

    first_wl = frame_results[0]["world_landmarks"]
    first_nl = frame_results[0]["normalized_landmarks"]
    n_lm = min(len(first_wl), len(first_nl), 33)

    # Bike never reads z downstream (cycling_analyzer projects z=0 via
    # _strip_z, and One Euro also skips z for bike). Skip Butterworth on
    # z too to keep behavior consistent and avoid wasted filter work.
    axes = ("x", "y") if sport_type == "bike" else ("x", "y", "z")

    filtered_series = 0
    for key in ("world_landmarks", "normalized_landmarks"):
        for i in range(n_lm):
            for attr in axes:
                series = np.array(
                    [getattr(frame_results[f][key][i], attr) for f in range(n)],
                    dtype=float,
                )
                nan_mask = np.isnan(series)
                valid = ~nan_mask
                if valid.sum() < MIN_BUTTER_SAMPLES:
                    continue
                # Linear-interp NaN gaps so filtfilt sees a finite input
                idx = np.arange(n)
                series[nan_mask] = np.interp(
                    idx[nan_mask], idx[valid], series[valid]
                )
                try:
                    out = sosfiltfilt(sos, series)
                except ValueError:
                    continue  # padlen mismatch -- leave original
                # Restore NaN so visibility-gated positions stay hidden
                out[nan_mask] = np.nan
                for f in range(n):
                    setattr(frame_results[f][key][i], attr, float(out[f]))
                filtered_series += 1

    cutoff_info["series_filtered"] = filtered_series
    cutoff_info["frames"] = n
    cutoff_info["order"] = BUTTER_LANDMARK_ORDER
    cutoff_info["fallback_triggered"] = False
    cutoff_info["fallback_reason"] = None

    logger.info(
        "LANDMARK_BUTTERWORTH",
        sport=sport_type,
        camera_angle=camera_angle,
        **cutoff_info,
    )
    return n_lm, cutoff_info
