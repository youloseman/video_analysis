"""Analysis Confidence scorer.

Combines multiple quality signals into a single confidence tier
("high" / "medium" / "low") with a transparent factors dict so
the user (and support engineers) can understand why confidence
dropped. All thresholds are module-level constants.
"""

from __future__ import annotations

import math
from typing import Any

THRESHOLDS = {
    "nan_pct_angle_strict": 40.0,
    "nan_pct_angle_high": 60.0,
    "valid_frames_ratio_medium": 0.70,
    "valid_frames_ratio_low": 0.40,
    "max_warnings_for_high": 0,
    "max_warnings_for_medium": 2,
    "cutoff_reduction_pct_medium": 40.0,
    "unknown_phase_pct_medium": 40.0,   # > this -> at least medium
    "unknown_phase_pct_low": 60.0,      # > this -> low
    # Tracking stability (run side-view). Signals measured by the
    # landmark stabilizer + camera-side voting; all ~0 on a clean clip.
    # A healthy 60 fps run clip probes at ~5% on each (scissor crossings),
    # so the medium bar sits above that and the low bar means the model
    # demonstrably could not hold left/right identity.
    "leg_swap_pct_medium": 6.0,
    "leg_swap_pct_low": 15.0,
    "leg_collapse_pct_medium": 12.0,
    "leg_collapse_pct_low": 25.0,
    "side_disagreement_pct_medium": 15.0,
    "side_disagreement_pct_low": 30.0,
    "flip_pct_medium": 8.0,
    "flip_pct_low": 20.0,
}

# tracking_stability dict key -> THRESHOLDS prefix
_TRACKING_CHECKS = [
    ("leg_swap_pct", "leg_swap_pct"),
    ("leg_collapse_pct", "leg_collapse_pct"),
    ("side_vote_disagreement_pct", "side_disagreement_pct"),
    ("flip_pct", "flip_pct"),
]


def assess_tracking_stability(
    tracking_stability: dict[str, Any] | None,
) -> str:
    """Grade left/right tracking stability: ``"ok" | "mild" | "severe"``.

    Shared by the confidence factor below and the runner's user-facing
    quality warning, so both fire from one set of thresholds. ``None``
    values (signal not measured -- e.g. bike, where the side is pre-locked
    and the leg pass is off) are skipped, never treated as evidence of
    stability.
    """
    if not tracking_stability:
        return "ok"

    # Whole-clip identity resolution changes what leg_swap_pct MEANS here.
    # It counts how noisy the labels ARRIVED; when the resolver reports the
    # repair itself is solid (nearly no links decided by hysteresis, every
    # segment's naming backed by cues), that arrival noise is history, and
    # grading it "severe" fires the "could not tell the legs apart" warning
    # on a clip whose output is demonstrably on the right legs. So a resolved
    # clip's swap figure may still say "mild" -- the athlete should know the
    # input was rough -- but not "severe". Unresolved clips grade exactly as
    # before.
    identity = tracking_stability.get("leg_identity") or {}
    identity_resolved = (
        identity.get("method") == "dp"
        and (identity.get("ambiguous_pair_pct") or 100.0) <= 5.0
        and (identity.get("naming_unconfident_pct")
             if identity.get("naming_unconfident_pct") is not None else 100.0) <= 10.0
    )

    severity = "ok"
    for key, thr in _TRACKING_CHECKS:
        v = tracking_stability.get(key)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or math.isnan(v):
            continue
        if v >= THRESHOLDS[f"{thr}_low"]:
            if key == "leg_swap_pct" and identity_resolved:
                severity = "mild"
                continue
            return "severe"
        if v >= THRESHOLDS[f"{thr}_medium"]:
            severity = "mild"
    return severity

# Humanized fallback-reason strings for the user-facing explanation
# text. Keys match phase_calibrator fallback_reason values; the
# frontend carries its own mapping for the tooltip in
# confidence-badge.tsx -- keep the two in sync when adding keys.
_FALLBACK_REASON_LABELS = {
    "insufficient_samples": "not enough data",
    "low_variance": "low motion variance",
    "narrow_range": "narrow angle range",
    "sanity_violation": "unstable thresholds",
    "out_of_bounds": "out-of-bounds thresholds",
}


def _humanize_fallback_reason(reason: str) -> str:
    """Map a phase_calibrator fallback_reason to a user-facing phrase.

    Unknown reasons pass through verbatim so new backend values surface
    visibly instead of being silently hidden.
    """
    return _FALLBACK_REASON_LABELS.get(reason, reason)


def compute_analysis_confidence(
    angle_statistics: dict[str, dict[str, Any]],
    frames_processed: int,
    butterworth_meta: dict[str, Any] | None,
    analysis_warnings: list[str],
    landmark_quality: dict[str, Any] | None = None,
    phase_diagnostics: dict[str, Any] | None = None,
    tracking_stability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute a single confidence tier from multiple quality signals.

    ``phase_diagnostics`` is the ``diagnostics.phase_thresholds`` dict
    written by the swim pipeline (``None`` for other sports). When it
    carries ``unknown_phase_pct`` above the medium/low thresholds, the
    level is downgraded and the explanation gains a swim-specific hint.

    Returns::

        {
            "level": "high" | "medium" | "low",
            "factors": {
                "landmark_quality_pct": float | None,
                "angles_with_high_nan": [str, ...],
                "majority_angles_gated": bool,
                "fallback_triggered": bool,
                "cutoff_reduced": bool,
                "warning_count": int,
                "tracking_severity": "ok" | "mild" | "severe",
                # flat copies when tracking_stability is supplied:
                "leg_swap_pct" / "leg_collapse_pct" /
                "side_vote_disagreement_pct" / "flip_pct": float | None,
                # present only when phase_diagnostics is supplied:
                "high_unknown_phases": bool,
                "unknown_phase_pct": float,
                "phase_calibration_source": "fixed" | "calibrated",
            },
            "explanation": str,
        }
    """
    level = "high"
    reasons: list[str] = []

    # --- factor: landmark quality (overall detection percentage) ---
    lq_pct: float | None = None
    if landmark_quality:
        lq_pct = landmark_quality.get("overall_pct")
        lq_conf = landmark_quality.get("confidence", "high")
        if lq_conf == "low":
            level = _downgrade(level, "low")
            reasons.append(
                "Body tracking was poor -- many landmarks were undetected."
            )
        elif lq_conf == "medium":
            level = _downgrade(level, "medium")
            reasons.append(
                "Some body landmarks had intermittent detection."
            )

    # --- factor: per-angle NaN percentage ---
    angles_with_high_nan: list[str] = []
    total_angles = 0
    angles_above_strict = 0
    for name, stats in angle_statistics.items():
        nan_pct = stats.get("nan_pct")
        if nan_pct is None:
            continue
        total_angles += 1
        if nan_pct > THRESHOLDS["nan_pct_angle_high"]:
            angles_with_high_nan.append(name)
            level = _downgrade(level, "medium")
        if nan_pct > THRESHOLDS["nan_pct_angle_strict"]:
            angles_above_strict += 1

    majority_gated = (
        total_angles > 0
        and angles_above_strict > total_angles / 2
    )
    if majority_gated:
        level = _downgrade(level, "medium")
        reasons.append(
            "More than half of measured angles had significant gaps."
        )
    if angles_with_high_nan:
        reasons.append(
            f"{len(angles_with_high_nan)} angle(s) had very high gap rates "
            f"({', '.join(angles_with_high_nan)})."
        )

    # --- factor: Butterworth fallback / cutoff reduction ---
    fallback_triggered = False
    cutoff_reduced = False
    if butterworth_meta:
        if butterworth_meta.get("fallback_triggered"):
            fallback_triggered = True
            level = _downgrade(level, "medium")
            reasons.append(
                "A fallback smoother was used because the clip was short."
            )
        reduction = butterworth_meta.get("reduction_pct", 0)
        if reduction > THRESHOLDS["cutoff_reduction_pct_medium"]:
            cutoff_reduced = True
            level = _downgrade(level, "medium")
            reasons.append(
                "Video length forced the filter to reduce precision."
            )

    # --- factor: left/right tracking stability (run side-view) ---
    # Leg-identity swaps keep MediaPipe visibility high and produce no NaN
    # angles, so no other factor can see them -- this one reads the
    # stabilizer's correction counts and the camera-side vote spread.
    tracking_severity = assess_tracking_stability(tracking_stability)
    if tracking_severity == "severe":
        level = _downgrade(level, "low")
        reasons.append(
            "Pose tracking kept mixing up the left and right legs -- "
            "typical for backlit or silhouette footage where the body "
            "reads as one dark shape. Leg angles, cadence and stride "
            "metrics are unreliable on this clip."
        )
    elif tracking_severity == "mild":
        level = _downgrade(level, "medium")
        reasons.append(
            "Pose tracking occasionally confused the left and right "
            "legs; leg-based metrics may be less precise."
        )

    # --- factor: analysis warnings ---
    warning_count = len(analysis_warnings)
    if warning_count > THRESHOLDS["max_warnings_for_medium"]:
        level = _downgrade(level, "low")
        reasons.append(
            "Multiple quality issues were detected."
        )
    elif warning_count > THRESHOLDS["max_warnings_for_high"]:
        level = _downgrade(level, "medium")
        reasons.append(
            "A quality warning was raised during processing."
        )

    # --- factor: phase-classification quality (swim only) ---
    phase_factors: dict[str, Any] = {}
    if phase_diagnostics is not None:
        unknown_pct = phase_diagnostics.get("unknown_phase_pct")
        source = phase_diagnostics.get("source")
        fallback_reason = phase_diagnostics.get("fallback_reason")
        high_unknown = False
        if isinstance(unknown_pct, (int, float)):
            if unknown_pct > THRESHOLDS["unknown_phase_pct_low"]:
                high_unknown = True
                level = _downgrade(level, "low")
            elif unknown_pct > THRESHOLDS["unknown_phase_pct_medium"]:
                high_unknown = True
                level = _downgrade(level, "medium")
            if high_unknown:
                # Three mutually exclusive branches keyed on calibration
                # state. The guidance each offers is different -- never
                # suggest enabling adaptive when it's already on.
                base = (
                    "A significant portion of swimming phases couldn't "
                    "be classified."
                )
                if source == "calibrated":
                    detail = (
                        " Adaptive phase calibration was applied, but a "
                        "large share of phases still couldn't be "
                        "classified. This usually means the video "
                        "quality limits what pose tracking can resolve "
                        "-- try a clearer underwater angle, better "
                        "lighting, or a clip with less splash/glare."
                    )
                elif source == "fixed" and fallback_reason:
                    humanized = _humanize_fallback_reason(fallback_reason)
                    detail = (
                        f" Adaptive phase calibration was attempted but "
                        f"fell back to fixed thresholds ({humanized}). "
                        f"Analysis proceeded with reduced precision for "
                        f"phase-dependent metrics."
                    )
                else:
                    detail = (
                        " This often happens when the swimmer's "
                        "technique differs from typical recreational "
                        "form. Consider enabling adaptive phase "
                        "calibration in advanced options for more "
                        "accurate results."
                    )
                reasons.append(base + detail)
            phase_factors["high_unknown_phases"] = high_unknown
            phase_factors["unknown_phase_pct"] = unknown_pct
        if source is not None:
            phase_factors["phase_calibration_source"] = source

    # --- build explanation ---
    if level == "high":
        explanation = "Analysis ran on clean data with good landmark detection."
    else:
        action = (
            "Try a clearer video with the whole body in frame, or "
            "film from a different angle."
        )
        explanation = " ".join(reasons) + " " + action

    factors: dict[str, Any] = {
        "landmark_quality_pct": lq_pct,
        "angles_with_high_nan": angles_with_high_nan,
        "majority_angles_gated": majority_gated,
        "fallback_triggered": fallback_triggered,
        "cutoff_reduced": cutoff_reduced,
        "warning_count": warning_count,
        "tracking_severity": tracking_severity,
    }
    if tracking_stability:
        # Flat keys so the UI factors line can read them directly.
        for _k in (
            "leg_swap_pct", "leg_collapse_pct",
            "side_vote_disagreement_pct", "flip_pct",
        ):
            factors[_k] = tracking_stability.get(_k)
    factors.update(phase_factors)

    return {
        "level": level,
        "factors": factors,
        "explanation": explanation,
    }


def _downgrade(current: str, target: str) -> str:
    """Return the worse of current and target confidence levels."""
    order = {"high": 0, "medium": 1, "low": 2}
    return current if order.get(current, 0) >= order.get(target, 0) else target
