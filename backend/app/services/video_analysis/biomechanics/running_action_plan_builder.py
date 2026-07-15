"""Running Action Plan Builder -- deterministic diagnostics for running gait analysis.

Mirrors the cycling action_plan_builder.py architecture:
  1. Python makes ALL diagnostic decisions (thresholds, drills, priorities)
  2. LLM only translates the plan JSON into prose (copywriter role)
  3. Fallback generates readable markdown without any LLM

Diagnosis order enforces a running-specific "kinematic chain":
  cadence -> vertical_oscillation -> trunk_lean -> knee_contact -> elbow_angle -> knee_swing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.video_analysis.biomechanics.sport_configs import RUNNING_REFERENCE

# ---------------------------------------------------------------------------
# Diagnosis order (strict, like cycling fitting_sequence)
# ---------------------------------------------------------------------------
RUNNING_DIAGNOSIS_ORDER: list[str] = [
    "cadence",        # 1. Cadence -- foundation of running efficiency
    "overstride",     # 2. Overstride -- braking + impact at foot-strike
    "vertical_osc",   # 3. Vertical oscillation -- energy waste
    "trunk_lean",     # 4. Trunk lean -- propulsive force direction
    "knee_contact",   # 5. Knee angle at contact -- impact loading
    "elbow_angle",    # 6. Elbow angle -- arm swing efficiency
    "knee_swing",     # 7. Knee angle in swing -- stride length
]

# Academy article slugs by metric (optional). A drill links to the article
# only when the slug is present AND the article exists -- the builder never
# hard-depends on an article being written, so new drills degrade to "no link"
# rather than a dead link. Add slugs here as Academy articles are published.
RUNNING_ACADEMY_SLUGS: dict[str, str] = {
    "cadence": "running-cadence",
    # overstride / foot-strike / vertical_osc articles: add slugs when written.
}

# ---------------------------------------------------------------------------
# Optimal ranges  (from RUNNING_REFERENCE in sport_configs.py)
# ---------------------------------------------------------------------------
_RANGES: dict[str, tuple[float, float]] = {
    "cadence":      (170, 190),   # steps per minute
    "overstride":   (0.0, 0.15),  # foot-ahead / leg-length at contact (lower better)
    "vertical_osc": (6, 10),      # centimeters
    "trunk_lean":   (4, 8),       # degrees from vertical
    "knee_contact": (160, 175),   # degrees  (knee_max from summary)
    "elbow_angle":  (85, 100),    # degrees  (elbow_mean from summary)
    "knee_swing":   (80, 100),    # degrees  (knee_min from summary)
}

# Map summary keys -> our metric names
_SUMMARY_KEY_MAP: dict[str, str] = {
    "cadence":      "cadence_spm",
    "overstride":   "overstride_ratio",
    "vertical_osc": "vertical_oscillation_m",   # needs *100 -> cm
    "trunk_lean":   "trunk_lean_avg",
    "knee_contact": "knee_max",
    "elbow_angle":  "elbow_mean",
    "knee_swing":   "knee_min",
}

# Biomechanically implausible values are treated as missing data
# rather than measurements. Defense-in-depth against regressions
# that re-introduce a "phantom 0" when cadence detection fails on
# a short clip (the failure mode that motivated this guard --
# 0.0 spm reaching the builder gets flagged "critical low" and
# becomes the AI Coach's #1 priority). Mirrors the
# _IMPLAUSIBLE_BOUNDS pattern in swimming_action_plan_builder.py.
#
# Cadence < 80 = walking, not running. > 220 = unsustainable
# sprint or false-positive spike from peak-detection noise.
# Vertical oscillation in cm: 0 = "no data" (analyzer default
# when norm_hip_y_history is too short); 25 cm = an unphysical
# bounce. Anything outside [1, 25] is noise.
_IMPLAUSIBLE_BOUNDS: dict[str, dict[str, float]] = {
    "cadence": {
        "min": 80.0,
        "max": 220.0,
    },
    "vertical_osc": {
        "min": 1.0,
        "max": 25.0,
    },
    "trunk_lean": {
        # Defensive floor against the analyzer's pre-hotfix 0.0
        # fallback when no valid trunk samples were captured. A
        # genuine reading is at least a fraction of a degree;
        # exactly 0.0 in legacy stored data is the failure mode.
        "min": 0.1,
        "max": 60.0,
    },
}
# Backward-compat alias -- only the ``min`` side. New code should
# use _IMPLAUSIBLE_BOUNDS and consult both ends.
_IMPLAUSIBLE_FLOORS: dict[str, float] = {
    metric: bounds["min"]
    for metric, bounds in _IMPLAUSIBLE_BOUNDS.items()
    if "min" in bounds
}


# Display precision per metric. Used to compute a boundary
# tolerance so a value identical to the displayed bound (after
# rounding) is classified as "optimal" -- prevents float-64
# representation from pushing 0.1m -> 10.000000000000002 cm into
# "minor" severity when displayed as 10.0 cm. The fact "Vertical
# Oscillation is 10.0 (target 6-10)" is self-contradictory after
# rounding, and the LLM resolves the contradiction by emitting
# the heading but no body, producing the empty issue card the
# user reported.
#
# All run metrics today display 1 decimal place; the dict is
# kept per-metric so future integer-displayed metrics can opt
# into the right tolerance without re-plumbing.
_DISPLAY_PRECISION: dict[str, int] = {
    "cadence": 1,
    "overstride": 2,   # small ratio -- shown to 2 decimals
    "vertical_osc": 1,
    "trunk_lean": 1,
    "knee_swing": 1,
    "knee_contact": 1,
    "elbow_angle": 1,
}
_DEFAULT_DISPLAY_PRECISION = 1


# Severity leniency by athlete level. The optimal TARGET ranges are
# identical for everyone (efficient running form is the same goal at
# any level) -- what changes is how harshly a deviation is graded.
# A developing runner a fixed distance outside the optimal band should
# not be told their form is "critical"; the same deviation in an elite
# runner is a more meaningful flag. The multiplier widens the
# deviation thresholds that escalate minor -> significant -> critical.
# 1.0 = no change; >1.0 = more forgiving.
_LEVEL_SEVERITY_LENIENCY: dict[str, float] = {
    "developing": 1.5,
    "good": 1.0,
    "elite": 1.0,
}


def _boundary_tolerance(metric: str | None = None) -> float:
    """Half a display-rounding ULP -- the largest float-64 error
    that displays identically to the bound. For 1-decimal metrics
    this is 0.05; for integer-displayed metrics it would be 0.5.
    """
    precision = _DISPLAY_PRECISION.get(
        metric or "", _DEFAULT_DISPLAY_PRECISION,
    )
    return 0.5 * (10 ** -precision)

# ---------------------------------------------------------------------------
# Drill library (English only -- no unicode symbols)
# ---------------------------------------------------------------------------
RUNNING_DRILLS: dict[str, dict[str, Any]] = {
    "cadence_low": {
        "problem": "Low cadence ({value} spm, optimal 170-190)",
        "drill": "High Knees Drill",
        "instruction": (
            "Run in place with high knee lift 3x30 seconds. "
            "Use a metronome app set to 180 bpm."
        ),
        "cue": "Imagine running on hot coals -- quick, light steps",
        "weeks_to_improvement": 3,
    },
    "overstride_high": {
        "problem": "Overstriding -- foot lands {value}x leg length ahead of the hip (optimal under 0.15)",
        "drill": "Quick-Feet Cadence Drill",
        "instruction": (
            "Run 4x20 seconds at a deliberately higher step rate on a "
            "metronome set to 180 bpm, focusing on landing with the foot "
            "under your hips rather than reaching out in front."
        ),
        "cue": "Land under your body, not out in front -- shorter, quicker steps",
        "weeks_to_improvement": 4,
    },
    "vertical_osc_high": {
        "problem": "High vertical oscillation ({value} cm, optimal 6-10)",
        "drill": "Bounding Drill",
        "instruction": (
            "Alternate 30s low-oscillation running + 30s normal running. "
            "Focus: push backward, not upward."
        ),
        "cue": "Glide forward like a cross-country skier, do not bounce",
        "weeks_to_improvement": 4,
    },
    "trunk_lean_insufficient": {
        "problem": "Insufficient trunk lean ({value} deg, optimal 4-8 deg)",
        "drill": "Wall Lean Drill",
        "instruction": (
            "Lean hands against a wall at 45 deg, body straight. "
            "Run in place maintaining that lean 3x20 seconds."
        ),
        "cue": "Fall forward with your whole body, not just your head",
        "weeks_to_improvement": 2,
    },
    "trunk_lean_excessive": {
        "problem": "Excessive trunk lean ({value} deg, optimal 4-8 deg)",
        "drill": "Tall Running Drill",
        "instruction": (
            "Run with a book on your head 3x50 m. "
            "Focus: pull the crown of your head upward, keep torso upright."
        ),
        "cue": "Grow tall from the ground -- imagine a string pulling you up by the crown",
        "weeks_to_improvement": 2,
    },
    "knee_contact_insufficient_flexion": {
        "problem": "Insufficient knee flexion at contact ({value} deg, optimal 160-175 deg)",
        "drill": "Pose Running Drill",
        "instruction": (
            "Practice landing with foot under center of gravity "
            "by standing on one leg 3x30 seconds each leg."
        ),
        "cue": "Land under your hips, not in front of you",
        "weeks_to_improvement": 4,
    },
    "elbow_angle_too_wide": {
        "problem": "Elbow angle too wide ({value} deg, optimal 85-100 deg)",
        "drill": "Arm Swing Drill",
        "instruction": (
            "Standing, practice pendulum arm swing with elbows at 90 deg. "
            "3x30 seconds, then transfer to running."
        ),
        "cue": "Hold a crisp in your fist without breaking it -- soft hands, sharp angle",
        "weeks_to_improvement": 2,
    },
    "elbow_angle_too_narrow": {
        "problem": "Elbow angle too narrow ({value} deg, optimal 85-100 deg)",
        "drill": "Relaxed Arm Swing",
        "instruction": (
            "Run with deliberately relaxed arms 2x100 m. "
            "Shake out your hands every 200 m."
        ),
        "cue": "Arms like pendulums -- no tension, no flailing",
        "weeks_to_improvement": 1,
    },
    "foot_strike_heel": {
        "problem": "Heel-first foot-strike, which usually pairs with overstriding and a braking force at contact",
        "drill": "Midfoot Landing Drill",
        "instruction": (
            "Run 4x20 seconds barefoot on grass or in minimal shoes, letting "
            "the midfoot/forefoot make first contact. Keep steps short and "
            "quick; do not force a hard toe-strike."
        ),
        "cue": "Land softly on the middle of your foot, under your hips",
        "weeks_to_improvement": 6,
    },
    "knee_swing_insufficient": {
        "problem": "Insufficient knee flexion in swing ({value} deg, optimal 80-100 deg)",
        "drill": "High Knee March",
        "instruction": (
            "March with maximum knee lift 3x20 m. "
            "Knee should rise to hip level."
        ),
        "cue": "Pull your heel toward your glute in the swing phase",
        "weeks_to_improvement": 3,
    },
}

# Banned words for running reports (cycling terms + absolutes + medical)
RUNNING_BANNED_WORDS: list[str] = [
    "saddle", "handlebars", "stem", "crank", "pedal",
    "always", "never",
]


# ---------------------------------------------------------------------------
# Metric aliases (for LLM label-binding validator)
# ---------------------------------------------------------------------------
# Each metric maps to a set of phrases the LLM might use to refer
# to it. The disjointness invariant (enforced at module load by
# _validate_running_aliases_disjoint) is: no phrase in set A is a
# case-insensitive substring of any phrase in set B for A != B.
# Without disjointness, the label-binding validator cannot
# reliably attribute a number to its metric -- the same failure
# mode that bit swim before Hotfix-2 ("Elbow at Catch (EVF)").
#
# Conservative principle: prefer specific multi-word phrases over
# bare nouns. "knee" alone is too ambiguous (matches both
# knee_swing and knee_contact); use "knee swing", "knee min",
# "knee at contact" instead.
RUNNING_METRIC_ALIASES: dict[str, set[str]] = {
    "cadence": {
        "cadence", "step rate", "stride rate", "spm",
        "steps per minute",
    },
    "vertical_osc": {
        "vertical oscillation", "vertical bounce",
    },
    "trunk_lean": {
        "trunk lean", "torso lean", "forward lean", "body lean",
    },
    "knee_swing": {
        "knee flexion in swing", "knee swing angle",
        "minimum knee angle", "knee min",
    },
    "knee_contact": {
        "knee at contact", "knee at strike",
        "knee max", "knee extension at contact",
    },
    "elbow_angle": {
        "elbow angle", "arm swing angle",
    },
}


def _validate_running_aliases_disjoint() -> None:
    """Ensure every alias phrase is disjoint across metrics.

    Disjointness means: for any two DIFFERENT metrics A and B, no
    alias in A is a case-insensitive substring of any alias in B
    (and vice versa). Raises ``ValueError`` at import time if
    violated, so a bad alias addition fails before reaching
    production.
    """
    items = list(RUNNING_METRIC_ALIASES.items())
    for i, (metric_a, aliases_a) in enumerate(items):
        for j in range(i + 1, len(items)):
            metric_b, aliases_b = items[j]
            for a in aliases_a:
                a_low = a.lower()
                for b in aliases_b:
                    b_low = b.lower()
                    if a_low in b_low or b_low in a_low:
                        raise ValueError(
                            f"RUNNING_METRIC_ALIASES disjointness violated: "
                            f"'{a}' ({metric_a}) overlaps with '{b}' "
                            f"({metric_b}). Aliases for different metrics "
                            f"must not be substrings of each other."
                        )


_validate_running_aliases_disjoint()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class RunningDiagnosis:
    """One metric diagnosis with drill recommendation."""

    metric_name: str
    current_value: float
    optimal_min: float
    optimal_max: float
    severity: str            # "optimal" | "minor" | "significant" | "critical"
    drill_key: str | None
    problem_description: str
    drill_name: str | None
    drill_instruction: str | None
    coaching_cue: str | None
    weeks_to_improvement: int | None
    linked_to: str | None = None
    academy_slug: str | None = None


@dataclass
class RunningActionPlan:
    """Complete running diagnostics plan."""

    sport: str = "running"
    athlete_level: str = "developing"   # "developing" | "good" | "elite"
    overall_score: int = 0

    diagnostics: list[RunningDiagnosis] = field(default_factory=list)
    good_metrics: list[dict[str, Any]] = field(default_factory=list)
    top_priorities: list[RunningDiagnosis] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
def build_running_action_plan(
    analysis_result: dict[str, Any],
    athlete_level: str | None = None,
) -> RunningActionPlan:
    """Build a deterministic running action plan from analyzer output.

    ``analysis_result`` should contain:
      - ``score`` (int, 0-100)
      - ``summary`` dict with keys from RunningAnalyzer.compute_summary()
    """
    plan = RunningActionPlan()

    score = analysis_result.get("score", 0) or 0
    plan.overall_score = score
    if athlete_level:
        plan.athlete_level = athlete_level
    else:
        plan.athlete_level = (
            "elite" if score >= 85
            else "good" if score >= 65
            else "developing"
        )

    summary = analysis_result.get("summary", {})

    for metric in RUNNING_DIAGNOSIS_ORDER:
        diagnosis = _diagnose_metric(metric, summary, plan.athlete_level)
        if diagnosis is None:
            continue
        if diagnosis.severity == "optimal":
            plan.good_metrics.append({
                "metric": metric,
                "value": diagnosis.current_value,
                "range": (diagnosis.optimal_min, diagnosis.optimal_max),
                "description": (
                    f"{_METRIC_LABELS.get(metric, metric)}: "
                    f"{diagnosis.current_value:.1f} -- in optimal range"
                ),
            })
        else:
            plan.diagnostics.append(diagnosis)

    # Foot-strike is categorical (heel/midfoot/forefoot), not a numeric range,
    # so it's diagnosed separately. Only a HEEL strike is flagged as
    # actionable -- midfoot/forefoot are treated as fine (no universal
    # "correct" strike, but heel-first is the one that pairs with overstride
    # and braking, and the one a cue can shift). Missing = not diagnosed.
    foot_strike = summary.get("foot_strike")
    if foot_strike == "heel":
        drill = RUNNING_DRILLS["foot_strike_heel"]
        plan.diagnostics.append(RunningDiagnosis(
            metric_name="foot_strike",
            current_value=0.0,          # categorical -- value unused
            optimal_min=0.0, optimal_max=0.0,
            severity="minor",
            drill_key="foot_strike_heel",
            problem_description=drill["problem"],
            drill_name=drill["drill"],
            drill_instruction=drill["instruction"],
            coaching_cue=drill["cue"],
            weeks_to_improvement=drill["weeks_to_improvement"],
            academy_slug=RUNNING_ACADEMY_SLUGS.get("foot_strike"),
        ))

    # Apply kinematic chain links
    _apply_kinematic_links(plan)

    # Top 3 priorities -- most severe first, then by diagnosis order
    severity_order = {"critical": 0, "significant": 1, "minor": 2}
    sorted_issues = sorted(
        plan.diagnostics,
        key=lambda d: severity_order.get(d.severity, 3),
    )
    plan.top_priorities = sorted_issues[:3]

    # Medical warnings
    if summary.get("cadence_spm") and summary["cadence_spm"] < 160:
        plan.warnings.append(
            "Very low cadence (<160 spm) increases ground reaction forces "
            "and may raise injury risk."
        )
    vert_cm = _get_vertical_osc_cm(summary)
    if vert_cm is not None and vert_cm > 12:
        plan.warnings.append(
            "High vertical oscillation (>12 cm) significantly increases "
            "impact loading per step."
        )

    return plan


# ---------------------------------------------------------------------------
# Metric labels
# ---------------------------------------------------------------------------
_METRIC_LABELS: dict[str, str] = {
    "cadence": "Cadence",
    "overstride": "Overstride",
    "foot_strike": "Foot Strike",
    "vertical_osc": "Vertical Oscillation",
    "trunk_lean": "Trunk Lean",
    "knee_contact": "Knee Angle at Contact",
    "elbow_angle": "Elbow Angle",
    "knee_swing": "Knee Flexion in Swing",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_vertical_osc_cm(summary: dict[str, Any]) -> float | None:
    """Get vertical oscillation in cm (summary stores meters)."""
    val = summary.get("vertical_oscillation_m")
    if val is None:
        return None
    # If value > 1, it might already be in cm (legacy data)
    if val > 1:
        return val
    return val * 100


def _get_value(metric: str, summary: dict[str, Any]) -> float | None:
    """Extract metric value from summary, converting units as needed.

    Returns None when the key is absent, explicitly None, or
    outside the per-metric implausibility bounds (see
    :data:`_IMPLAUSIBLE_BOUNDS`). The bounds prevent a phantom 0.0
    cadence from a failed peak-detection fallback being marked
    "critical low" in the diagnostic plan.
    """
    key = _SUMMARY_KEY_MAP.get(metric)
    if key is None:
        return None
    val = summary.get(key)
    if val is None:
        return None
    # vertical_oscillation is stored in meters but the diagnostic
    # ranges and bounds are expressed in cm. Convert before the
    # bounds check.
    if metric == "vertical_osc":
        val = _get_vertical_osc_cm(summary)
        if val is None:
            return None
    try:
        val_f = float(val)
    except (TypeError, ValueError):
        return None
    bounds = _IMPLAUSIBLE_BOUNDS.get(metric)
    if bounds is not None:
        lo = bounds.get("min")
        hi = bounds.get("max")
        if lo is not None and val_f < lo:
            return None
        if hi is not None and val_f > hi:
            return None
    return val_f


def _diagnose_metric(
    metric: str, summary: dict[str, Any], athlete_level: str = "good"
) -> RunningDiagnosis | None:
    """Diagnose a single metric. Returns None if data is missing.

    ``athlete_level`` only affects severity escalation (via
    :data:`_LEVEL_SEVERITY_LENIENCY`), never the optimal target range
    or the in-range "optimal" classification -- targets are the same
    for everyone; only the harshness of out-of-range grading scales.
    """
    value = _get_value(metric, summary)
    if value is None:
        return None

    opt_min, opt_max = _RANGES[metric]

    # Apply display-rounding tolerance to the boundary check. A
    # value within tol of the range (i.e., one that rounds to the
    # boundary at display precision) is treated as in-range so we
    # don't emit a "minor" diagnostic that the LLM can't sensibly
    # discuss when the rounded value matches the displayed bound.
    # Severity escalation below uses the RAW bounds because the
    # deviation is physically meaningful once we're outside the
    # tolerance window.
    tol = _boundary_tolerance(metric)
    if (opt_min - tol) <= value <= (opt_max + tol):
        severity = "optimal"
        drill_key = None
    else:
        deviation = min(abs(value - opt_min), abs(value - opt_max))
        range_size = opt_max - opt_min
        if range_size == 0:
            range_size = 1

        # Level-graduated escalation: developing runners get wider
        # thresholds before a deviation is called significant/critical.
        leniency = _LEVEL_SEVERITY_LENIENCY.get(athlete_level, 1.0)
        critical_thresh = range_size * leniency
        significant_thresh = range_size * 0.5 * leniency

        if deviation > critical_thresh:
            severity = "critical"
        elif deviation > significant_thresh:
            severity = "significant"
        else:
            severity = "minor"

        drill_key = _get_drill_key(metric, value, opt_min, opt_max)

    drill_data = RUNNING_DRILLS.get(drill_key, {}) if drill_key else {}

    return RunningDiagnosis(
        metric_name=metric,
        current_value=round(value, 1),
        optimal_min=opt_min,
        optimal_max=opt_max,
        severity=severity,
        drill_key=drill_key,
        problem_description=(
            drill_data.get("problem", "").format(value=f"{value:.1f}")
            if drill_key
            else ""
        ),
        drill_name=drill_data.get("drill"),
        drill_instruction=drill_data.get("instruction"),
        coaching_cue=drill_data.get("cue"),
        weeks_to_improvement=drill_data.get("weeks_to_improvement"),
        academy_slug=RUNNING_ACADEMY_SLUGS.get(metric) if drill_key else None,
    )


def _get_drill_key(
    metric: str, value: float, opt_min: float, opt_max: float
) -> str | None:
    """Map metric deviation direction to a drill key."""
    below = value < opt_min

    drill_map: dict[str, tuple[str | None, str | None]] = {
        "cadence":      ("cadence_low", None),
        "overstride":   (None, "overstride_high"),   # only "too high" matters
        "vertical_osc": (None, "vertical_osc_high"),
        "trunk_lean":   ("trunk_lean_insufficient", "trunk_lean_excessive"),
        "knee_contact": ("knee_contact_insufficient_flexion", None),
        "elbow_angle":  ("elbow_angle_too_narrow", "elbow_angle_too_wide"),
        "knee_swing":   ("knee_swing_insufficient", None),
    }

    if metric not in drill_map:
        return None

    below_key, above_key = drill_map[metric]
    return below_key if below else above_key


def _apply_kinematic_links(plan: RunningActionPlan) -> None:
    """Mark linked diagnostics in the kinematic chain.

    - Excessive trunk lean + low knee contact angle are linked
    - Wide elbows + upright trunk are linked
    """
    diag_map = {d.metric_name: d for d in plan.diagnostics}

    trunk = diag_map.get("trunk_lean")
    knee = diag_map.get("knee_contact")
    elbow = diag_map.get("elbow_angle")

    if trunk and knee:
        knee.linked_to = "trunk_lean"

    if elbow and trunk and trunk.current_value < trunk.optimal_min:
        elbow.linked_to = "trunk_lean"


# ---------------------------------------------------------------------------
# Serialization for LLM
# ---------------------------------------------------------------------------
def running_action_plan_to_json(plan: RunningActionPlan) -> dict[str, Any]:
    """Serialize plan to JSON dict for the LLM copywriter."""
    return {
        "sport": plan.sport,
        "athlete_level": plan.athlete_level,
        "overall_score": plan.overall_score,
        "strong_points": plan.good_metrics,
        "issues": [
            {
                "metric": d.metric_name,
                "label": _METRIC_LABELS.get(d.metric_name, d.metric_name),
                "value": d.current_value,
                "optimal_range": f"{d.optimal_min}-{d.optimal_max}",
                "severity": d.severity,
                "problem": d.problem_description,
                "drill": d.drill_name,
                "instruction": d.drill_instruction,
                "cue": d.coaching_cue,
                "weeks": d.weeks_to_improvement,
                "linked_to": d.linked_to,
                "academy_slug": d.academy_slug,
            }
            for d in plan.diagnostics
        ],
        "top_3_priorities": [
            {
                "priority": i + 1,
                "metric": d.metric_name,
                "label": _METRIC_LABELS.get(d.metric_name, d.metric_name),
                "drill": d.drill_name,
                "why": d.problem_description,
                "instruction": d.drill_instruction,
                "cue": d.coaching_cue,
                "weeks": d.weeks_to_improvement,
                "academy_slug": d.academy_slug,
            }
            for i, d in enumerate(plan.top_priorities)
        ],
        "warnings": plan.warnings,
    }


# ---------------------------------------------------------------------------
# Fallback report (no LLM)
# ---------------------------------------------------------------------------
def running_action_plan_fallback_report(plan: RunningActionPlan) -> str:
    """Generate a markdown coaching report without LLM.

    Used when Gemini is unavailable or fails banned-word validation.
    """
    lines: list[str] = []

    # Strong Points
    lines.append("## Strong Points")
    if plan.good_metrics:
        for m in plan.good_metrics[:4]:
            lines.append(f"- {m['description']}")
    else:
        lines.append(
            f"- Overall technique score: {plan.overall_score}/100."
        )
    lines.append("")

    # Areas to Improve
    lines.append("## Areas to Improve")
    issues = [d for d in plan.diagnostics if d.severity != "optimal"]
    if issues:
        for d in issues[:4]:
            label = _METRIC_LABELS.get(d.metric_name, d.metric_name)
            lines.append(f"**{label}** ({d.severity})")
            if d.problem_description:
                lines.append(d.problem_description)
            if d.drill_instruction:
                lines.append(f"*Drill:* {d.drill_instruction}")
            if d.linked_to:
                linked_label = _METRIC_LABELS.get(d.linked_to, d.linked_to)
                lines.append(
                    f"(Connected to {linked_label} -- "
                    "fixing that first may help resolve this.)"
                )
            lines.append("")
    else:
        lines.append(
            "Your running form looks solid -- no major changes needed."
        )
        lines.append("")

    # Warnings
    if plan.warnings:
        for w in plan.warnings:
            lines.append(f"**Note:** {w}")
        lines.append("")

    # Top 3 Priority Actions
    lines.append("## Top 3 Priority Actions")
    for i, d in enumerate(plan.top_priorities, 1):
        label = d.drill_name or _METRIC_LABELS.get(d.metric_name, d.metric_name)
        lines.append(f"{i}. **{label}**")
        if d.coaching_cue:
            lines.append(f"   *Key cue:* {d.coaching_cue}")
        if d.weeks_to_improvement:
            lines.append(
                f"   *Expected improvement in:* {d.weeks_to_improvement} weeks"
            )

    for i in range(len(plan.top_priorities) + 1, 4):
        lines.append(
            f"{i}. Maintain current form. "
            "Re-analyze after 4-6 weeks of training."
        )

    return "\n".join(lines)
