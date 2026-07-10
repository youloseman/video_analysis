"""Deterministic Action Plan Builder for cycling video analysis.

Builds a structured action plan from measured angles and position-specific
optimal ranges. The plan is consumed by the LLM copywriter (see
llm_recommendations._generate_cycling_recommendations_v2) which only
translates it into readable prose -- the LLM does NOT decide what to
recommend.

All decision-making lives here: fitting order, kinematic chains,
medical warnings, and position-specific terminology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.services.video_analysis.biomechanics.cycling_positions import (
    CYCLING_POSITIONS,
    get_cycling_reference,
    get_medical_warnings,
    get_position_label,
)


# ---------------------------------------------------------------------------
# Implausibility bounds + summary-key map (mirrors swim hotfix-5 pattern)
# ---------------------------------------------------------------------------
# Biomechanically implausible cycling values are treated as missing data,
# not measurements. These are *outer envelopes* -- wider than any single
# position's optimal range -- so a road_hoods rider at knee_at_bdc=130 deg
# (suboptimal but legitimate) still reaches the diagnostic, while
# knee_at_bdc=60 (measurement failure) is dropped.
#
# Why each metric is bounded:
#   knee_at_bdc < 90  : knee fully bent at extension = bike massively
#                       undersized OR pedal-stroke detector misfired.
#   knee_at_bdc > 175 : hyperextension; setup fault or measurement.
#   knee_at_tdc < 30  : extreme flexion; physically impossible fitting.
#   knee_at_tdc > 110 : barely bending at top; saddle absurdly high or fail.
#   trunk_angle < 5   : near-horizontal posture (impossible riding).
#   trunk_angle > 80  : near-vertical (also unphysical for cycling).
#   elbow_angle < 60  : extreme flexion; outside any cycling window.
#   elbow_angle > 180 : numerically impossible (joint angle).
#   shoulder_angle    : 50-150 covers all cycling positions including TT/casual.
#   hip_angle         : 25-100 covers TT aero (low) through casual (open).
#   pelvic_ratio      : <1.0 means trunk > hip (formula breaks); >8.0 noise.
#   forearm_tilt      : -30..45 covers UCI-legal aero pad tilts and road.
# head_alignment_avg is already a 0-100 score -- every value is structurally
# valid -- so it has no entry here.
_IMPLAUSIBLE_BOUNDS: dict[str, dict[str, float]] = {
    "knee_at_bdc":     {"min": 90.0,   "max": 175.0},
    "knee_at_tdc":     {"min": 30.0,   "max": 110.0},
    "trunk_angle":     {"min": 5.0,    "max": 80.0},
    "elbow_angle":     {"min": 60.0,   "max": 180.0},
    "shoulder_angle":  {"min": 50.0,   "max": 150.0},
    "hip_angle":       {"min": 25.0,   "max": 100.0},
    "pelvic_ratio":    {"min": 1.0,    "max": 8.0},
    "forearm_tilt":    {"min": -30.0,  "max": 45.0},
}

# Map metric name -> summary key the analyzer writes.
_SUMMARY_KEY_MAP: dict[str, str] = {
    "knee_at_bdc":      "knee_at_bdc",
    "knee_at_tdc":      "knee_at_tdc",
    "trunk_angle":      "trunk_angle_avg",
    "elbow_angle":      "elbow_angle_avg",
    "shoulder_angle":   "shoulder_angle_avg",
    "hip_angle":        "hip_angle_avg",
    "pelvic_ratio":     "pelvic_ratio",
    "forearm_tilt":     "forearm_tilt_avg",
    "head_alignment":   "head_alignment_avg",
}


def _get_value(metric: str, summary: dict[str, Any]) -> float | None:
    """Extract a metric from the summary, applying implausibility bounds.

    Returns None when the key is absent, explicitly None, or outside the
    per-metric implausibility bounds. The bounds protect the plan builder
    against phantom measurements -- e.g. a 0.0 knee_at_bdc from the
    analyzer's failure path being read as "knee fully extended (critical)".
    """
    key = _SUMMARY_KEY_MAP.get(metric)
    if key is None:
        return None
    val = summary.get(key)
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


# ---------------------------------------------------------------------------
# Boundary tolerance for severity classification
# ---------------------------------------------------------------------------
# A value identical to the displayed bound (after rounding) must be
# classified "in range" rather than "needs_adjustment". Without this,
# float-64 representation pushes e.g. 145.0 -> 145.00000000000003,
# which fails ``<= 145`` and triggers a "lower saddle" diagnostic
# even though current_value rounds back to 145.0 in the report.
#
# All cycling metrics today display 1 decimal place, so the
# tolerance is uniformly 0.05. The dict structure mirrors the
# run / swim builders so future integer-displayed metrics can opt
# into 0.5 without re-plumbing.
_DISPLAY_PRECISION: dict[str, int] = {
    "knee_at_bdc": 1,
    "knee_at_tdc": 1,
    "trunk_angle": 1,
    "elbow_angle": 1,
    "shoulder_angle": 1,
    "hip_angle": 1,
    "pelvic_ratio": 1,
    "forearm_tilt": 1,
    "head_alignment": 1,
}
_DEFAULT_DISPLAY_PRECISION = 1


def _boundary_tolerance(metric: str | None = None) -> float:
    """Half a display-rounding ULP. For 1-decimal metrics this is
    0.05; for integer-displayed metrics it would be 0.5."""
    precision = _DISPLAY_PRECISION.get(
        metric or "", _DEFAULT_DISPLAY_PRECISION,
    )
    return 0.5 * (10 ** -precision)


def _below_min(value: float, opt_min: float, metric: str | None = None) -> bool:
    """True iff value is below opt_min by more than display tolerance."""
    return value < (opt_min - _boundary_tolerance(metric))


def _above_max(value: float, opt_max: float, metric: str | None = None) -> bool:
    """True iff value is above opt_max by more than display tolerance."""
    return value > (opt_max + _boundary_tolerance(metric))


# ---------------------------------------------------------------------------
# Clinical deadband -- the "don't fire a 5mm fix for a sub-noise miss" band
# ---------------------------------------------------------------------------
# The display tolerance above (0.05 deg) only absorbs float-rounding ULPs.
# It is NOT the measurement uncertainty of a 2D pose pipeline: the bike
# reference curves carry per-joint SDs of 6-9 deg, so a knee_at_bdc that
# is 2 deg outside the optimal band is well inside the method's own noise.
# Issuing "lower saddle 5mm" off such a miss is false precision -- the
# exact failure surfaced by a real fit (knee 147 vs 145 ceiling, trunk
# 19.1 vs 20 floor both triggered actions that experienced fitters
# disagreed with).
#
# So the *recommendation* branches (saddle / fore-aft / bar) use this
# wider, clinically-meaningful deadband. Values strictly outside the
# optimal band but within the deadband are reported as on-target
# ("borderline"), not as an action item. Values beyond it still flag.
# Magnitudes are conservative fractions of each joint's reference SD.
_CLINICAL_DEADBAND: dict[str, float] = {
    "knee_at_bdc": 3.0,
    "knee_at_tdc": 3.0,
    "trunk_angle": 2.0,
    "hip_angle": 3.0,
    "elbow_angle": 5.0,
    "shoulder_angle": 5.0,
    "forearm_tilt": 2.0,
    "head_alignment": 3.0,
    "pelvic_ratio": 0.2,
}
_DEFAULT_CLINICAL_DEADBAND = 2.0


def _clinical_deadband(metric: str | None = None) -> float:
    """Measurement-uncertainty deadband for recommendation decisions.

    Always at least the display tolerance so the borderline band can
    never invert (deadband >= display ULP by construction).
    """
    db = _CLINICAL_DEADBAND.get(metric or "", _DEFAULT_CLINICAL_DEADBAND)
    return max(db, _boundary_tolerance(metric))


def _classify(
    value: float, opt_min: float, opt_max: float, metric: str | None = None,
) -> str:
    """Classify a value against an optimal band with a clinical deadband.

    Returns one of: ``"in_range"``, ``"borderline_low"``,
    ``"borderline_high"``, ``"out_low"``, ``"out_high"``.

    - in_range      : within the displayed band (+- display ULP)
    - borderline_*  : outside the band but within the clinical deadband
                      -- report as on-target, no action
    - out_*         : beyond the deadband -- a real, actionable miss
    """
    disp = _boundary_tolerance(metric)
    db = _clinical_deadband(metric)
    if value > opt_max + db:
        return "out_high"
    if value > opt_max + disp:
        return "borderline_high"
    if value < opt_min - db:
        return "out_low"
    if value < opt_min - disp:
        return "borderline_low"
    return "in_range"


def _borderline_note(side: str, opt_min: float, opt_max: float) -> str:
    """Short copy for a metric that is marginally out of band but within
    measurement tolerance -- consumed by the LLM copywriter so it frames
    the value as on-target instead of inventing a fix."""
    where = "above" if side == "high" else "below"
    return (
        f"Marginally {where} the {opt_min:g}-{opt_max:g} optimal band but "
        f"within measurement tolerance -- treat as on-target, no change needed."
    )


# ---------------------------------------------------------------------------
# Metric aliases (for LLM label-binding validator) -- Sprint B2
# ---------------------------------------------------------------------------
# Each metric maps to a set of phrases the LLM might use to refer
# to it. Disjointness is enforced at module load by
# _validate_cycling_aliases_disjoint -- without that, the
# label-binding validator cannot reliably attribute a number to
# its metric (the production "Elbow at Catch (EVF)" failure mode
# from swim Hotfix-2 in a different costume).
#
# Position-specific terminology ("pad stack", "stem spacers",
# "hoods") is intentionally NOT in this dict. Those words appear
# in drill text, not as metric references the LLM might invent
# numbers next to. Keeping the alias dict position-agnostic also
# keeps the disjointness check simple.
CYCLING_METRIC_ALIASES: dict[str, set[str]] = {
    "knee_at_bdc": {
        "knee at bdc", "knee at bottom dead center",
        "knee extension at bdc", "saddle height",
    },
    "knee_at_tdc": {
        "knee at tdc", "knee at top dead center",
        "knee flexion at tdc", "crank length",
    },
    "trunk_angle": {
        "trunk angle", "torso angle", "back angle", "bar position",
    },
    "elbow_angle": {
        "elbow angle",
    },
    "shoulder_angle": {
        "shoulder angle", "shoulder reach",
    },
    "hip_angle": {
        "hip angle", "hip flexion", "hip openness",
    },
    "pelvic_ratio": {
        "pelvic ratio", "pelvic rotation",
    },
    "forearm_tilt": {
        "forearm tilt", "forearm angle",
    },
    "head_alignment": {
        "head alignment", "head tuck", "aero tuck",
    },
}


def _validate_cycling_aliases_disjoint() -> None:
    """Ensure every alias phrase is disjoint across metrics.

    Disjointness: for any two DIFFERENT metrics A and B, no alias
    in A is a case-insensitive substring of any alias in B (and
    vice versa). Raises ``ValueError`` at import time if violated.
    """
    items = list(CYCLING_METRIC_ALIASES.items())
    for i, (metric_a, aliases_a) in enumerate(items):
        for j in range(i + 1, len(items)):
            metric_b, aliases_b = items[j]
            for a in aliases_a:
                a_low = a.lower()
                for b in aliases_b:
                    b_low = b.lower()
                    if a_low in b_low or b_low in a_low:
                        raise ValueError(
                            f"CYCLING_METRIC_ALIASES disjointness violated: "
                            f"'{a}' ({metric_a}) overlaps with '{b}' "
                            f"({metric_b}). Aliases for different metrics "
                            f"must not be substrings of each other."
                        )


_validate_cycling_aliases_disjoint()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    """A single fitting diagnostic with action recommendation."""

    component: str  # "saddle_height" | "saddle_fore_aft" | "bar_position" | "crank_length"
    status: str  # "needs_adjustment" | "optimal"
    metric_name: str  # "knee_at_bdc" | "hip_angle_avg" | "trunk_angle" | "knee_at_tdc"
    current_value: float
    target_range: tuple[float, float]
    action: str  # "raise_saddle" | "lower_saddle" | "move_saddle_back" | etc.
    amount: str  # "5mm" | ""
    reason: str  # Human-readable explanation
    priority: int  # 1=saddle height, 2=fore/aft, 3=bar, 4=cranks
    linked_to: str | None = None


@dataclass
class ActionPlan:
    """Complete deterministic action plan for a cycling video analysis."""

    position: str
    position_label: str
    terminology: dict[str, Any]
    technique_score: int
    letter_grade: str
    diagnostics: list[Diagnostic] = field(default_factory=list)
    good_metrics: list[dict[str, Any]] = field(default_factory=list)
    medical_warnings: list[dict[str, str]] = field(default_factory=list)
    fitting_sequence_note: str = ""


# ---------------------------------------------------------------------------
# Position-specific terminology for LLM copywriter
# ---------------------------------------------------------------------------

_POSITION_TERMINOLOGY: dict[str, dict[str, Any]] = {
    "tt_aero": {
        "bike_type": "TT bike",
        "hand_position": "aerobars (pads and extensions)",
        "bar_height_term": "pad stack",
        "bar_reach_term": "pad reach",
        "bar_angle_term": "extension angle",
        "banned_words": [
            "hoods", "drops", "tops", "stem", "spacers", "headset",
            "bar tape", "road bike",
        ],
    },
    "triathlon": {
        "bike_type": "triathlon bike",
        "hand_position": "aerobars (pads and extensions)",
        "bar_height_term": "pad stack",
        "bar_reach_term": "pad reach",
        "bar_angle_term": "extension angle",
        "banned_words": [
            "hoods", "drops", "tops", "stem", "spacers", "headset",
            "bar tape", "road bike",
        ],
    },
    "road_hoods": {
        "bike_type": "road bike",
        "hand_position": "hoods",
        "bar_height_term": "stem spacers and stem angle",
        "bar_reach_term": "stem length",
        "bar_angle_term": "bar angle",
        "banned_words": [],
    },
    "road_drops": {
        "bike_type": "road bike",
        "hand_position": "drops",
        "bar_height_term": "stem spacers and stem angle",
        "bar_reach_term": "stem length",
        "bar_angle_term": "bar angle",
        "banned_words": [],
    },
    "casual": {
        "bike_type": "casual / commuter bike",
        "hand_position": "handlebars",
        "bar_height_term": "handlebar height",
        "bar_reach_term": "handlebar reach",
        "bar_angle_term": "handlebar angle",
        "banned_words": [],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_terminology(position: str) -> dict[str, Any]:
    """Get position-specific terminology, falling back to road_hoods."""
    return _POSITION_TERMINOLOGY.get(position, _POSITION_TERMINOLOGY["road_hoods"])


def _is_tt_category(position: str) -> bool:
    """Check if position is in the TT/triathlon category."""
    pos_data = CYCLING_POSITIONS.get(position, {})
    return pos_data.get("category") == "tt"


def _add_good_metric(
    good_metrics: list[dict[str, Any]],
    metric: str,
    value: float,
    opt_range: tuple[float, float],
    label: str,
    borderline: bool = False,
    note: str | None = None,
) -> None:
    """Append a metric to the good_metrics list.

    ``borderline`` marks a value that sits just outside the optimal band
    but within the clinical deadband -- still reported as on-target, with
    an optional ``note`` the LLM copywriter can surface.
    """
    entry: dict[str, Any] = {
        "metric": metric,
        "value": round(value, 1),
        "range": list(opt_range),
        "label": label,
    }
    if borderline:
        entry["borderline"] = True
        if note:
            entry["note"] = note
    good_metrics.append(entry)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_action_plan(
    position: str,
    angle_statistics: dict[str, Any],
    sport_specific_metrics: dict[str, Any],
    technique_score: int,
    letter_grade: str,
    detected_issues: list[dict[str, Any]],
) -> ActionPlan:
    """Build a deterministic action plan from cycling analysis results.

    Checks metrics in strict fitting order:
      1. Saddle height (knee @ BDC)
      2. Saddle fore/aft (hip angle -- asymmetric scoring)
      3. Bar position (trunk angle)
      4. Crank length (knee @ TDC)
      5+ Aero metrics (informational)
    """
    ref = get_cycling_reference(position)
    terminology = _get_terminology(position)
    sm = sport_specific_metrics or {}

    diagnostics: list[Diagnostic] = []
    good_metrics: list[dict[str, Any]] = []
    medical_warnings: list[dict[str, str]] = []
    active_priorities: list[str] = []

    # ------------------------------------------------------------------
    # Priority 1: Saddle Height (knee @ BDC)
    # ------------------------------------------------------------------
    knee_bdc = _get_value("knee_at_bdc", sm)
    bdc_min, bdc_max = ref["knee_at_bdc"]
    saddle_has_raise = False

    if knee_bdc is not None:
        knee_class = _classify(knee_bdc, bdc_min, bdc_max, "knee_at_bdc")
        if knee_class == "out_low":
            diagnostics.append(Diagnostic(
                component="saddle_height",
                status="needs_adjustment",
                metric_name="knee_at_bdc",
                current_value=round(knee_bdc, 1),
                target_range=(bdc_min, bdc_max),
                action="raise_saddle",
                amount="5mm",
                reason=(
                    f"Knee angle at BDC is {knee_bdc:.0f} deg "
                    f"(optimal {bdc_min:.0f}-{bdc_max:.0f} deg). "
                    f"Insufficient extension -- raise saddle by 5mm, "
                    f"ride 30 min before reassessing."
                ),
                priority=1,
            ))
            saddle_has_raise = True
            active_priorities.append("saddle height")
        elif knee_class == "out_high":
            diagnostics.append(Diagnostic(
                component="saddle_height",
                status="needs_adjustment",
                metric_name="knee_at_bdc",
                current_value=round(knee_bdc, 1),
                target_range=(bdc_min, bdc_max),
                action="lower_saddle",
                amount="5mm",
                reason=(
                    f"Knee angle at BDC is {knee_bdc:.0f} deg "
                    f"(optimal {bdc_min:.0f}-{bdc_max:.0f} deg). "
                    f"Overextension -- lower saddle by 5mm, "
                    f"ride 30 min before reassessing."
                ),
                priority=1,
            ))
            active_priorities.append("saddle height")
        else:
            border = knee_class.startswith("borderline")
            _add_good_metric(
                good_metrics, "knee_at_bdc", knee_bdc,
                (bdc_min, bdc_max), "Saddle height (knee at BDC)",
                borderline=border,
                note=(
                    _borderline_note(
                        "high" if knee_class == "borderline_high" else "low",
                        bdc_min, bdc_max,
                    ) if border else None
                ),
            )

    # ------------------------------------------------------------------
    # Priority 2: Saddle Fore/Aft (hip angle -- ASYMMETRIC scoring)
    # ------------------------------------------------------------------
    hip_avg = _get_value("hip_angle", sm)
    hip_min, hip_max = ref["hip_angle_max"]

    if hip_avg is not None:
        hip_class = _classify(hip_avg, hip_min, hip_max, "hip_angle")
        if hip_class in ("out_high", "borderline_high"):
            # ASYMMETRIC: above optimal = more comfort, NOT a problem
            _add_good_metric(
                good_metrics, "hip_angle", hip_avg,
                (hip_min, hip_max), "Hip angle (open, comfortable)",
            )
        elif hip_class == "out_low":
            # Hip too closed -- check kinematic chain
            if saddle_has_raise:
                # Raising saddle will also open hip angle
                diagnostics.append(Diagnostic(
                    component="saddle_fore_aft",
                    status="needs_adjustment",
                    metric_name="hip_angle_avg",
                    current_value=round(hip_avg, 1),
                    target_range=(hip_min, hip_max),
                    action="reassess_after_saddle_height",
                    amount="",
                    reason=(
                        f"Hip angle is {hip_avg:.0f} deg "
                        f"(optimal {hip_min:.0f}-{hip_max:.0f} deg). "
                        f"This is linked to saddle height -- raising the "
                        f"saddle will also open the hip angle. Reassess "
                        f"after saddle height correction."
                    ),
                    priority=2,
                    linked_to="saddle_height",
                ))
            else:
                diagnostics.append(Diagnostic(
                    component="saddle_fore_aft",
                    status="needs_adjustment",
                    metric_name="hip_angle_avg",
                    current_value=round(hip_avg, 1),
                    target_range=(hip_min, hip_max),
                    action="move_saddle_back",
                    amount="5mm",
                    reason=(
                        f"Hip angle is {hip_avg:.0f} deg "
                        f"(optimal {hip_min:.0f}-{hip_max:.0f} deg). "
                        f"Hip is too closed -- move saddle back 5mm "
                        f"to open hip angle."
                    ),
                    priority=2,
                ))
            active_priorities.append("saddle fore/aft")
        else:
            # in_range or borderline_low (just-closed, within tolerance)
            border = hip_class == "borderline_low"
            _add_good_metric(
                good_metrics, "hip_angle", hip_avg,
                (hip_min, hip_max), "Hip angle",
                borderline=border,
                note=_borderline_note("low", hip_min, hip_max) if border else None,
            )

    # Medical warnings for hip angle
    if hip_avg is not None:
        med_warnings = get_medical_warnings(position, {
            "hip_angle_min": hip_avg,
            "hip_angle_max": hip_avg,
            "trunk_angle_avg": sm.get("trunk_angle_avg", 0),
        })
        medical_warnings.extend(med_warnings)

    # Triathlon-specific: hip flexor warning for run transition
    if position == "triathlon" and hip_avg is not None and hip_avg < 55:
        # Only add if not already covered by medical warnings
        has_hip_warning = any("hip flexor" in w.get("message", "").lower() for w in medical_warnings)
        if not has_hip_warning:
            medical_warnings.append({
                "type": "triathlon_hip_flexor",
                "message": (
                    f"Hip angle is {hip_avg:.0f} deg. For triathlon, a hip angle "
                    f"below 55 deg can compromise hip flexor function for the run "
                    f"leg. Consider a slightly more open position to preserve "
                    f"running ability off the bike."
                ),
                "source": "Triathlon bike fit best practices",
            })

    # ------------------------------------------------------------------
    # Priority 3: Bar Position (trunk angle)
    # ------------------------------------------------------------------
    trunk_avg = _get_value("trunk_angle", sm)
    trunk_min, trunk_max = ref["trunk_angle"]
    bar_height = terminology["bar_height_term"]
    bar_reach = terminology["bar_reach_term"]

    if trunk_avg is not None:
        trunk_class = _classify(trunk_avg, trunk_min, trunk_max, "trunk_angle")
        if trunk_class == "out_high":
            # Too upright -- need to lower bars or extend reach
            diagnostics.append(Diagnostic(
                component="bar_position",
                status="needs_adjustment",
                metric_name="trunk_angle",
                current_value=round(trunk_avg, 1),
                target_range=(trunk_min, trunk_max),
                action="lower_bars",
                amount="",
                reason=(
                    f"Trunk angle is {trunk_avg:.0f} deg "
                    f"(optimal {trunk_min:.0f}-{trunk_max:.0f} deg). "
                    f"Position is too upright -- lower {bar_height} "
                    f"or increase {bar_reach} to achieve a more "
                    f"forward position."
                ),
                priority=3,
            ))
            active_priorities.append("bar position")
        elif trunk_class == "out_low":
            # Too aggressive
            diagnostics.append(Diagnostic(
                component="bar_position",
                status="needs_adjustment",
                metric_name="trunk_angle",
                current_value=round(trunk_avg, 1),
                target_range=(trunk_min, trunk_max),
                action="raise_bars",
                amount="",
                reason=(
                    f"Trunk angle is {trunk_avg:.0f} deg "
                    f"(optimal {trunk_min:.0f}-{trunk_max:.0f} deg). "
                    f"Position is too aggressive -- raise {bar_height} "
                    f"for a more sustainable position."
                ),
                priority=3,
            ))
            active_priorities.append("bar position")
        else:
            border = trunk_class.startswith("borderline")
            _add_good_metric(
                good_metrics, "trunk_angle", trunk_avg,
                (trunk_min, trunk_max), "Trunk angle",
                borderline=border,
                note=(
                    _borderline_note(
                        "high" if trunk_class == "borderline_high" else "low",
                        trunk_min, trunk_max,
                    ) if border else None
                ),
            )

    # ------------------------------------------------------------------
    # Priority 4: Crank Length (knee @ TDC)
    # ------------------------------------------------------------------
    knee_tdc = _get_value("knee_at_tdc", sm)
    tdc_min, tdc_max = ref["knee_at_tdc"]

    if knee_tdc is not None:
        saddle_is_optimal = not any(
            d.component == "saddle_height" and d.status == "needs_adjustment"
            for d in diagnostics
        )
        if knee_tdc < 55 and saddle_is_optimal:
            diagnostics.append(Diagnostic(
                component="crank_length",
                status="needs_adjustment",
                metric_name="knee_at_tdc",
                current_value=round(knee_tdc, 1),
                target_range=(tdc_min, tdc_max),
                action="consider_shorter_cranks",
                amount="",
                reason=(
                    f"Knee angle at TDC is {knee_tdc:.0f} deg "
                    f"(optimal {tdc_min:.0f}-{tdc_max:.0f} deg). "
                    f"Very tight knee flexion at top of pedal stroke "
                    f"despite correct saddle height. Consider shorter "
                    f"cranks to improve clearance."
                ),
                priority=4,
            ))
        else:
            tdc_class = _classify(knee_tdc, tdc_min, tdc_max, "knee_at_tdc")
            if tdc_class != "out_low" and tdc_class != "out_high":
                border = tdc_class.startswith("borderline")
                _add_good_metric(
                    good_metrics, "knee_at_tdc", knee_tdc,
                    (tdc_min, tdc_max), "Knee flexion at TDC",
                    borderline=border,
                    note=(
                        _borderline_note(
                            "high" if tdc_class == "borderline_high" else "low",
                            tdc_min, tdc_max,
                        ) if border else None
                    ),
                )

    # ------------------------------------------------------------------
    # Aero & secondary metrics (informational). _get_value drops
    # implausible / phantom-zero readings, so each call reaches the
    # informational helper only when the analyzer measured a real value.
    # ------------------------------------------------------------------
    _check_informational_metric(
        good_metrics, "elbow_angle", _get_value("elbow_angle", sm),
        ref["elbow_angle"], "Elbow angle",
    )
    _check_informational_metric(
        good_metrics, "shoulder_angle", _get_value("shoulder_angle", sm),
        ref["shoulder_angle"], "Shoulder angle",
    )
    _check_informational_metric(
        good_metrics, "forearm_tilt", _get_value("forearm_tilt", sm),
        ref["forearm_tilt"], "Forearm tilt",
    )
    _check_informational_metric(
        good_metrics, "head_alignment", _get_value("head_alignment", sm),
        ref["head_alignment"], "Head alignment",
    )
    _check_informational_metric(
        good_metrics, "pelvic_ratio", _get_value("pelvic_ratio", sm),
        ref["pelvic_ratio"], "Pelvic rotation ratio",
    )

    # ------------------------------------------------------------------
    # Build fitting sequence note
    # ------------------------------------------------------------------
    if active_priorities:
        fitting_note = (
            f"Adjust in order: {' -> '.join(active_priorities)}. "
            f"Each adjustment: 5mm only, ride 30 minutes before reassessing."
        )
    else:
        fitting_note = (
            "All primary metrics are within optimal ranges. "
            "Maintain current setup and re-analyze after 4-6 weeks of training."
        )

    # Sort diagnostics by priority
    diagnostics.sort(key=lambda d: d.priority)

    return ActionPlan(
        position=position,
        position_label=get_position_label(position),
        terminology=terminology,
        technique_score=technique_score,
        letter_grade=letter_grade,
        diagnostics=diagnostics,
        good_metrics=good_metrics,
        medical_warnings=medical_warnings,
        fitting_sequence_note=fitting_note,
    )


def _check_informational_metric(
    good_metrics: list[dict[str, Any]],
    metric: str,
    value: float | None,
    opt_range: tuple[float, float],
    label: str,
) -> None:
    """Add metric to good_metrics if it's within the optimal range.

    ``value`` is ``None`` when the analyzer never measured it or the
    reading was outside :data:`_IMPLAUSIBLE_BOUNDS` -- in either case we
    must not emit a card.
    """
    if value is None:
        return
    opt_min, opt_max = opt_range
    # Apply the clinical deadband so a value marginally outside the band
    # (but within measurement tolerance) is still surfaced as a Strong
    # Point -- flagged borderline -- rather than silently dropped. Only
    # genuinely out-of-band (beyond the deadband) values are suppressed.
    cls = _classify(value, opt_min, opt_max, metric)
    if cls in ("out_low", "out_high"):
        return
    border = cls.startswith("borderline")
    _add_good_metric(
        good_metrics, metric, value, opt_range, label,
        borderline=border,
        note=(
            _borderline_note(
                "high" if cls == "borderline_high" else "low",
                opt_min, opt_max,
            ) if border else None
        ),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def action_plan_to_json(plan: ActionPlan) -> dict[str, Any]:
    """Serialize ActionPlan to a JSON-compatible dict for the LLM prompt."""
    return {
        "position": plan.position,
        "position_label": plan.position_label,
        "terminology": plan.terminology,
        "technique_score": plan.technique_score,
        "letter_grade": plan.letter_grade,
        "fitting_sequence_note": plan.fitting_sequence_note,
        "diagnostics": [
            {
                "component": d.component,
                "status": d.status,
                "metric_name": d.metric_name,
                "current_value": d.current_value,
                "target_range": list(d.target_range),
                "action": d.action,
                "amount": d.amount,
                "reason": d.reason,
                "priority": d.priority,
                "linked_to": d.linked_to,
            }
            for d in plan.diagnostics
        ],
        "good_metrics": plan.good_metrics,
        "medical_warnings": [
            {"type": w.get("type", ""), "message": w.get("message", "")}
            for w in plan.medical_warnings
        ],
    }
