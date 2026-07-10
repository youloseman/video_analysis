"""Technique scoring system (0-100 with letter grades).

Adapted from MEDIAPIPE_PROJECT_ANALYSIS.md section 17.
Scores each metric based on deviation from optimal range,
then computes weighted overall score.
"""

import math
from typing import Any

import numpy as np


# A joint pair is excluded from the Asymmetry Check when either side
# had more than this percentage of raw NaN frames (pre-interpolation).
# Interpolation fills gaps for the filter pass but the downstream mean
# is then compared against a mostly-synthetic series, which massively
# over-reports asymmetry for occluded limbs.
ASYMMETRY_NAN_PCT_LIMIT = 40.0


# Scoring weights per sport (higher = more important)
# Unilateral focus: symmetry/bilateral_crp removed (unreliable from side view).
# Redistributed to near-side metrics and advanced biomechanics.
RUNNING_WEIGHTS = {
    "trunk_lean": 0.15,
    "knee_angles": 0.22,
    "cadence": 0.18,
    "elbow_swing": 0.10,
    "vertical_osc": 0.10,
    "phase_stability": 0.15,
    "waveform_similarity": 0.10,
}

CYCLING_WEIGHTS = {
    "knee_bdc": 0.22,
    "trunk_angle": 0.18,
    "knee_tdc": 0.15,
    "elbow_angle": 0.10,
    "shoulder_angle": 0.10,
    "head_alignment": 0.10,
    "pelvic_ratio": 0.05,
    "forearm_tilt": 0.05,
    "saddle_fit": 0.05,
}

SWIMMING_WEIGHTS = {
    "elbow_catch": 0.18,
    "body_rotation": 0.18,
    "streamline": 0.15,
    "stroke_rate": 0.10,
    "symmetry": 0.09,
    "entry_angle": 0.10,
    "head_position": 0.10,
    "kick_amplitude": 0.10,
}


def score_in_range(value: float, optimal_min: float, optimal_max: float) -> float:
    """Score a single value 0-100 based on distance from optimal range.

    Returns 100 if in range, decreasing score as distance increases.
    """
    if optimal_min <= value <= optimal_max:
        return 100.0

    range_size = optimal_max - optimal_min
    if range_size <= 0:
        range_size = 1.0

    if value < optimal_min:
        distance = optimal_min - value
    else:
        distance = value - optimal_max

    # Penalty: lose ~10 points per unit of range-width deviation
    penalty_factor = 100.0 / (range_size * 2)
    score = max(0.0, 100.0 - distance * penalty_factor)
    return score


def compute_weighted_score(
    component_scores: dict[str, float], weights: dict[str, float],
) -> int:
    """Compute weighted average score from component scores."""
    total_weight = 0.0
    weighted_sum = 0.0

    for component, weight in weights.items():
        if component in component_scores:
            weighted_sum += component_scores[component] * weight
            total_weight += weight

    if total_weight <= 0:
        return 50  # Default if no data

    return int(np.clip(weighted_sum / total_weight, 0, 100))


def assign_grade(score: int) -> str:
    """Convert 0-100 score to letter grade."""
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def score_running(
    summary: dict[str, Any], angle_stats: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Score running technique using near-side angles only."""
    from app.services.video_analysis.biomechanics.sport_configs import RUNNING_REFERENCE

    components: dict[str, float] = {}

    # Trunk lean
    trunk = summary.get("trunk_lean_avg", 0)
    if trunk > 0:
        components["trunk_lean"] = score_in_range(trunk, *RUNNING_REFERENCE["trunk_lean"])

    # Cadence
    cadence = summary.get("cadence_spm", 0)
    if cadence > 0:
        components["cadence"] = score_in_range(cadence, *RUNNING_REFERENCE["cadence_spm"])

    # Knee angles (unprefixed key -- near-side only from running_analyzer).
    # mean may be None when the joint was fully gated.
    if "knee" in angle_stats:
        mean_val = angle_stats["knee"].get("mean")
        if mean_val is not None:
            components["knee_angles"] = score_in_range(mean_val, *RUNNING_REFERENCE["knee_at_midstance"])

    if "elbow" in angle_stats:
        elbow_mean = angle_stats["elbow"].get("mean")
        if elbow_mean is not None:
            components["elbow_swing"] = score_in_range(
                elbow_mean, *RUNNING_REFERENCE["elbow_angle"]
            )

    # Vertical oscillation
    vert_osc = summary.get("vertical_oscillation_m", 0)
    if vert_osc > 0:
        vert_osc_cm = vert_osc * 100
        components["vertical_osc"] = score_in_range(
            vert_osc_cm, *RUNNING_REFERENCE["vertical_oscillation_cm"]
        )

    # Advanced biomechanics components
    bio = summary.get("biomechanics")
    if bio:
        # Phase stability (from phase portraits module)
        pp = bio.get("phase_portraits")
        if pp and "overall_stability_score" in pp:
            components["phase_stability"] = float(pp["overall_stability_score"])

        # Waveform similarity (from waveform comparator module)
        wf = bio.get("waveform")
        if wf and "overall_similarity_score" in wf and wf["overall_similarity_score"] is not None:
            components["waveform_similarity"] = float(wf["overall_similarity_score"])

    overall = compute_weighted_score(components, RUNNING_WEIGHTS)
    return {
        "overall_score": overall,
        "letter_grade": assign_grade(overall),
        "component_scores": components,
    }


def score_cycling(
    summary: dict[str, Any],
    angle_stats: dict[str, dict[str, float]],
    cycling_position: str | None = None,
) -> dict[str, Any]:
    """Score cycling technique (bike fit) using near-side angles."""
    from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference

    ref = get_cycling_reference(cycling_position)
    components: dict[str, float] = {}
    near = summary.get("near_side", "left")

    # Knee at BDC (near-side primary, fall back to any available)
    near_bdc = summary.get(f"{near}_knee_at_bdc", 0)
    if near_bdc > 0:
        components["knee_bdc"] = score_in_range(near_bdc, *ref["knee_at_bdc"])
    else:
        # Fallback: check backward-compat key
        bdc = summary.get("knee_at_bdc", 0)
        if bdc > 0:
            components["knee_bdc"] = score_in_range(bdc, *ref["knee_at_bdc"])

    # Knee at TDC (near-side primary)
    near_tdc = summary.get(f"{near}_knee_at_tdc", 0)
    if near_tdc > 0:
        components["knee_tdc"] = score_in_range(near_tdc, *ref["knee_at_tdc"])
    else:
        tdc = summary.get("knee_at_tdc", 0)
        if tdc > 0:
            components["knee_tdc"] = score_in_range(tdc, *ref["knee_at_tdc"])

    # Trunk angle
    trunk = summary.get("trunk_angle_avg", 0)
    if trunk > 0:
        components["trunk_angle"] = score_in_range(trunk, *ref["trunk_angle"])

    # Saddle fit (derived from near-side BDC)
    saddle = summary.get("saddle_height_assessment", "")
    saddle_map = {"optimal": 100, "acceptable": 75, "too_low": 40, "too_high": 40}
    if saddle in saddle_map:
        components["saddle_fit"] = float(saddle_map[saddle])

    # Elbow angle (near-side from elbow_angle_avg which is now near-side)
    elbow = summary.get("elbow_angle_avg", 0)
    if elbow > 0:
        components["elbow_angle"] = score_in_range(elbow, *ref["elbow_angle"])

    # Shoulder angle (near-side)
    shoulder = summary.get("shoulder_angle_avg", 0)
    if shoulder > 0:
        components["shoulder_angle"] = score_in_range(shoulder, *ref["shoulder_angle"])

    # Forearm tilt (can be negative for some positions)
    forearm = summary.get("forearm_tilt_avg")
    if forearm is not None and not (forearm == 0 and summary.get("frames_analyzed", 0) == 0):
        components["forearm_tilt"] = score_in_range(forearm, *ref["forearm_tilt"])

    # Head alignment (already a score 0-100, use directly)
    head = summary.get("head_alignment_avg", 0)
    if head > 0:
        components["head_alignment"] = head

    # Pelvic ratio (scored against ratio thresholds)
    pelvic = summary.get("pelvic_ratio", 0)
    if pelvic > 0:
        components["pelvic_ratio"] = score_in_range(pelvic, *ref["pelvic_ratio"])

    overall = compute_weighted_score(components, CYCLING_WEIGHTS)
    return {
        "overall_score": overall,
        "letter_grade": assign_grade(overall),
        "component_scores": components,
    }


def score_swimming(
    summary: dict[str, Any],
    angle_stats: dict[str, dict[str, float]],
    landmark_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score swimming technique."""
    from app.services.video_analysis.biomechanics.sport_configs import SWIMMING_REFERENCE

    components: dict[str, float] = {}

    # Swim summary fields may be None when a landmark was fully gated by
    # P0 visibility. Treat None as "no measurement" -- skip that component
    # rather than crashing on NoneType > int comparisons.
    def _num(key: str) -> float | None:
        v = summary.get(key)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if math.isnan(f) or math.isinf(f):
            return None
        return f

    catch = _num("elbow_at_catch_avg")
    if catch is not None and catch > 0:
        components["elbow_catch"] = score_in_range(catch, *SWIMMING_REFERENCE["elbow_at_catch"])

    rotation = _num("body_rotation_avg")
    if rotation is not None and rotation > 0:
        components["body_rotation"] = score_in_range(rotation, *SWIMMING_REFERENCE["body_rotation"])

    streamline = _num("streamline_avg")
    if streamline is not None and streamline > 0:
        components["streamline"] = score_in_range(streamline, *SWIMMING_REFERENCE["streamline"])

    stroke_rate = _num("stroke_rate_spm")
    if stroke_rate is not None and stroke_rate > 0:
        components["stroke_rate"] = score_in_range(stroke_rate, *SWIMMING_REFERENCE["stroke_rate_spm"])

    # Symmetry (left vs right elbow). angle_statistics now carries None
    # instead of 0.0 when a landmark was fully gated, so guard both sides.
    # Also exclude pairs where either side had >40% NaN in the raw signal
    # -- interpolation fills those gaps so the mean looks plausible, but
    # asymmetry comparisons against a mostly-synthetic series produce
    # huge false positives.
    if "left_elbow" in angle_stats and "right_elbow" in angle_stats:
        le = angle_stats["left_elbow"]
        re = angle_stats["right_elbow"]
        l_mean = le.get("mean")
        r_mean = re.get("mean")
        l_nan_pct = float(le.get("nan_pct") or 0.0)
        r_nan_pct = float(re.get("nan_pct") or 0.0)
        if (
            l_mean is not None and r_mean is not None
            and l_nan_pct <= ASYMMETRY_NAN_PCT_LIMIT
            and r_nan_pct <= ASYMMETRY_NAN_PCT_LIMIT
        ):
            avg = (l_mean + r_mean) / 2
            if avg > 0:
                asym_pct = abs(l_mean - r_mean) / avg * 100
                components["symmetry"] = max(0.0, 100.0 - asym_pct * 5)

    # Entry angle
    entry = summary.get("entry_angle_avg")
    if entry is not None:
        components["entry_angle"] = score_in_range(entry, *SWIMMING_REFERENCE["entry_angle"])

    # Head position (0 is perfect -- in optimal range 0-15)
    head = summary.get("head_position_avg")
    if head is not None:
        components["head_position"] = score_in_range(head, *SWIMMING_REFERENCE["head_position"])

    # Kick amplitude
    kick = summary.get("kick_amplitude_avg")
    if kick is not None and kick > 0:
        components["kick_amplitude"] = score_in_range(kick, *SWIMMING_REFERENCE["kick_amplitude"])

    overall = compute_weighted_score(components, SWIMMING_WEIGHTS)
    result: dict[str, Any] = {
        "overall_score": overall,
        "letter_grade": assign_grade(overall),
        "component_scores": components,
    }

    # Add confidence metadata when landmark quality is available
    if landmark_quality:
        result["analysis_confidence"] = landmark_quality.get("confidence", "unknown")
        if landmark_quality.get("confidence") == "low":
            result["confidence_note"] = (
                "Low landmark detection quality. "
                "Score may not reflect actual technique."
            )

    return result


def score_analysis(
    sport_type: str,
    summary: dict[str, Any],
    angle_stats: dict[str, dict[str, float]],
    cycling_position: str | None = None,
    landmark_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score technique for any sport type."""
    if sport_type == "bike":
        return score_cycling(summary, angle_stats, cycling_position=cycling_position)

    if sport_type == "swim":
        return score_swimming(summary, angle_stats, landmark_quality=landmark_quality)

    if sport_type == "run":
        return score_running(summary, angle_stats)

    return {"overall_score": 50, "letter_grade": "C", "component_scores": {}}


# ---------------------------------------------------------------------------
# Evidence-based severity classification (uses META warning thresholds)
# ---------------------------------------------------------------------------

def classify_metric_severity(
    position: str | None, metric_key: str, value: float,
) -> str:
    """Classify a cycling metric as OPTIMAL, ACCEPTABLE, WARNING, CRITICAL, or MEDICAL_RISK.

    Uses CYCLING_POSITIONS_META warning thresholds for richer classification
    than the simple in-range/out-of-range check.
    """
    from app.services.video_analysis.biomechanics.cycling_positions import (
        CYCLING_POSITIONS_META,
        get_cycling_reference,
    )

    ref = get_cycling_reference(position)
    opt = ref.get(metric_key)
    if not opt:
        return "UNKNOWN"

    opt_min, opt_max = opt
    if opt_min <= value <= opt_max:
        return "OPTIMAL"

    # Check META for warning thresholds
    pos = position or "road_hoods"
    meta = CYCLING_POSITIONS_META.get(pos, {}).get(metric_key, {})
    warning_low = meta.get("warning_low")
    warning_high = meta.get("warning_high")
    has_medical = "medical_warning" in meta

    # ASYMMETRIC scoring for hip angle: above optimal = comfort (OK), below = risk
    if metric_key == "hip_angle_max":
        if value > opt_max:
            # Above optimal = more open hip = comfort trade-off, always OK
            return "ACCEPTABLE"
        # Below optimal = closing hip = bad
        if value < 45 and has_medical:
            return "MEDICAL_RISK"
        if warning_low is not None and value >= warning_low:
            return "ACCEPTABLE"
        distance = opt_min - value
        if distance > 15:
            return "CRITICAL"
        return "WARNING"

    # Symmetric scoring for all other metrics
    # Check medical risk first (hip angle closure)
    if has_medical and value < 45:
        return "MEDICAL_RISK"

    # Within warning thresholds -> ACCEPTABLE
    effective_low = warning_low if warning_low is not None else (opt_min - 5)
    effective_high = warning_high if warning_high is not None else (opt_max + 5)

    if effective_low <= value <= effective_high:
        return "ACCEPTABLE"

    # Beyond warning thresholds
    if value < opt_min:
        distance = opt_min - value
    else:
        distance = value - opt_max

    if distance > 15:
        return "CRITICAL"
    return "WARNING"


def build_severity_map(
    position: str | None, summary: dict[str, Any],
) -> dict[str, str]:
    """Build severity classification for all cycling metrics in a summary.

    Returns dict like {"trunk_angle": "WARNING", "knee_at_bdc": "OPTIMAL", ...}
    """
    metric_keys_map = {
        "trunk_angle_avg": "trunk_angle",
        "elbow_angle_avg": "elbow_angle",
        "shoulder_angle_avg": "shoulder_angle",
        "forearm_tilt_avg": "forearm_tilt",
        "head_alignment_avg": "head_alignment",
        "pelvic_ratio": "pelvic_ratio",
    }

    severity_map: dict[str, str] = {}

    # Near-side knee angles
    near = summary.get("near_side", "left")
    bdc = summary.get(f"{near}_knee_at_bdc") or summary.get("knee_at_bdc")
    if bdc and bdc > 0:
        severity_map["knee_at_bdc"] = classify_metric_severity(position, "knee_at_bdc", bdc)

    tdc = summary.get(f"{near}_knee_at_tdc") or summary.get("knee_at_tdc")
    if tdc and tdc > 0:
        severity_map["knee_at_tdc"] = classify_metric_severity(position, "knee_at_tdc", tdc)

    # Hip angle
    hip = summary.get("hip_angle_max") or summary.get("hip_angle_avg")
    if hip and hip > 0:
        severity_map["hip_angle_max"] = classify_metric_severity(position, "hip_angle_max", hip)

    # Other metrics
    for summary_key, ref_key in metric_keys_map.items():
        val = summary.get(summary_key)
        if val is not None and (not isinstance(val, (int, float)) or val != 0):
            severity_map[ref_key] = classify_metric_severity(position, ref_key, float(val))

    return severity_map
