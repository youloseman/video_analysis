"""Cycling position profiles with evidence-based optimal ranges.

Different bike positions have dramatically different optimal angles:
- Road bike on hoods: moderate trunk, bent elbows on brake hoods
- Road bike in drops: more aggressive trunk, still road geometry
- TT/Tri in aero: low trunk, elbows bent on aero pads (~90 deg)
- Triathlon: like TT but slightly more conservative (run preservation)
- Casual/commuter: upright trunk, relaxed arms

Ranges are sourced from Retul fit windows, Bini et al. 2011,
Burt 'Bike Fit' 2014, Fintelman et al. 2015, and wind tunnel studies.

Knee BDC is mostly saddle-height dependent (similar across positions).
Knee TDC varies: TT aero has a more closed hip angle at TDC.
"""

import math
from typing import Any


# ---------------------------------------------------------------------------
# Position-specific optimal angle ranges
# ---------------------------------------------------------------------------

CYCLING_POSITIONS: dict[str, dict[str, Any]] = {
    "road_hoods": {
        "label": "Road - Hoods",
        "category": "road",
        "trunk_angle": (40, 55),        # Retul recreational rider norms
        "elbow_angle": (145, 165),      # Retul road: slightly bent, not locked
        "knee_at_bdc": (135, 145),      # Retul: 35-45 deg extension = 135-145 internal
        "knee_at_tdc": (65, 75),        # Retul fit window
        "hip_angle_max": (55, 65),      # avg hip angle (above optimal = comfort/OK)
        "hip_at_tdc": (55, 70),         # hip angle at top of stroke (closed)
        "hip_at_bdc": (100, 120),       # hip angle at bottom of stroke (open)
        "shoulder_angle": (90, 120),
        "forearm_tilt": (5, 20),
        "head_alignment": (50, 100),    # less critical on road
        "pelvic_ratio": (1.5, 3.5),
    },
    "road_drops": {
        "label": "Road - Drops",
        "category": "road",
        "trunk_angle": (30, 45),        # competitive road in drops
        "elbow_angle": (130, 160),      # slightly more bent than hoods
        "knee_at_bdc": (135, 145),      # Retul; Bini et al. 2011
        "knee_at_tdc": (65, 75),        # Retul fit window
        "hip_angle_max": (50, 65),      # avg hip angle in drops (above = OK)
        "hip_at_tdc": (52, 68),
        "hip_at_bdc": (95, 118),
        "shoulder_angle": (85, 110),
        "forearm_tilt": (0, 15),
        "head_alignment": (60, 100),
        "pelvic_ratio": (1.8, 3.5),
    },
    "tt_aero": {
        "label": "TT - Aero",
        "category": "tt",
        "trunk_angle": (15, 30),        # Retul TT ~20 deg; track 4.5-8.6 deg
        "elbow_angle": (75, 110),       # elbows bent on aero pads, NOT straight
        "knee_at_bdc": (138, 145),      # Retul TT/Tri: 37-42 deg extension
        "knee_at_tdc": (60, 72),        # more closed hip angle in aero
        "hip_angle_max": (35, 55),      # TT avg hip angle; above 55 = comfort/OK
        "hip_at_tdc": (42, 60),         # aggressive closed hip at TDC
        "hip_at_bdc": (85, 110),        # open hip at BDC
        "shoulder_angle": (80, 100),    # surplus power optimal 88.9-105.3 deg
        "forearm_tilt": (5, 25),        # modern trend 10-20 deg upward tilt
        "head_alignment": (75, 100),    # head tuck saves ~4.6% drag
        "pelvic_ratio": (2.0, 4.0),    # forward rotation enables aero
    },
    "triathlon": {
        "label": "Triathlon - Aero",
        "category": "tt",
        "trunk_angle": (20, 30),        # more conservative than pure TT (~25 vs ~20)
        "elbow_angle": (75, 110),       # same pad setup as TT
        "knee_at_bdc": (138, 145),      # Retul TT/Tri fit window
        "knee_at_tdc": (60, 72),
        "hip_angle_max": (40, 55),      # tri avg hip; more conservative min for run
        "hip_at_tdc": (45, 62),
        "hip_at_bdc": (88, 112),
        "shoulder_angle": (80, 105),
        "forearm_tilt": (5, 25),
        "head_alignment": (70, 100),
        "pelvic_ratio": (2.0, 4.0),
    },
    "casual": {
        "label": "Casual / Commuter",
        "category": "casual",
        "trunk_angle": (45, 60),        # very upright, comfort focus
        "elbow_angle": (130, 165),      # nearly straight, relaxed
        "knee_at_bdc": (135, 150),      # wider window for casual
        "knee_at_tdc": (60, 80),
        "hip_angle_max": (65, 85),      # casual: wide open, comfort focus
        "hip_at_tdc": (68, 88),
        "hip_at_bdc": (110, 138),
        "shoulder_angle": (100, 140),
        "forearm_tilt": (-5, 15),
        "head_alignment": (40, 100),
        "pelvic_ratio": (1.5, 3.5),
    },
}


# ---------------------------------------------------------------------------
# Evidence metadata: sources, warning thresholds, medical risks
# Parallel dict keyed by position, then by metric name.
# ---------------------------------------------------------------------------

CYCLING_POSITIONS_META: dict[str, dict[str, dict[str, Any]]] = {
    "road_hoods": {
        "trunk_angle": {
            "source": "Retul; recreational rider norms",
            "warning_low": 35,
            "warning_high": 60,
        },
        "elbow_angle": {
            "source": "Retul standard road fit window",
            "warning_low": 140,
            "warning_high": 170,
        },
        "knee_at_bdc": {
            "source": "Retul fit window; Burt 'Bike Fit' 2014",
            "warning_low": 130,
            "warning_high": 150,
            "note": "Dynamic measurement ~8 deg greater than static (Bini & Hume 2016)",
        },
        "knee_at_tdc": {
            "source": "Retul fit window",
        },
        "hip_angle_max": {
            "source": "Retul recommended normal ranges",
            "warning_low": 50,
            "note": "ASYMMETRIC: above optimal = comfort (OK). Below optimal = closed hip (risk).",
        },
        "shoulder_angle": {
            "source": "General bike fit guidance; shoulders relaxed",
            "note": "Less critical for road. Focus on comfort over number.",
        },
    },
    "road_drops": {
        "trunk_angle": {
            "source": "Competitive road rider norms",
            "warning_low": 25,
            "warning_high": 50,
        },
        "elbow_angle": {
            "source": "Retul road fit window",
            "warning_high": 170,
        },
        "knee_at_bdc": {
            "source": "Retul; Bini et al. 2011",
            "warning_low": 130,
            "warning_high": 150,
        },
        "hip_angle_max": {
            "source": "Retul; derived from competitive rider trunk 30-45 deg",
            "warning_low": 45,
            "note": "ASYMMETRIC: above optimal = comfort (OK). Below optimal = closed hip.",
        },
    },
    "tt_aero": {
        "trunk_angle": {
            "source": "Retul TT fit; surplus power peak 4.5-8.6 deg (track)",
            "warning_low": 10,
            "warning_high": 35,
        },
        "elbow_angle": {
            "source": "Retul TT fit window: 90-100 deg",
            "note": "90 deg = structural support via skeleton. Locked-out eliminates shock absorption.",
        },
        "knee_at_bdc": {
            "source": "Retul TT/Triathlon fit window",
            "warning_low": 133,
            "warning_high": 148,
            "note": "Dynamic measurement ~8 deg greater than static (Bini & Hume 2016)",
        },
        "hip_angle_max": {
            "source": "Retul TT fit window; Fintelman et al. 2015",
            "warning_low": 30,
            "note": "ASYMMETRIC: above optimal = comfort (OK). Below 45 = medical risk.",
            "medical_warning": (
                "Hip angle <45 deg carries risk of iliac artery endofibrosis. "
                "Monitor for sudden power loss or leg numbness during hard efforts."
            ),
        },
        "shoulder_angle": {
            "source": "Wind tunnel: surplus power 88.9-105.3 deg; pure power 88-93 deg",
            "power_optimal": (88, 93),
            "note": "Elbow directly below shoulder (~90 deg) provides skeletal support.",
        },
        "forearm_tilt": {
            "source": "UCI regulations; modern aero trends (Steinmetz 2024)",
        },
        "pelvic_ratio": {
            "source": "Pelvic rotation studies",
            "note": "Forward rotation allows low trunk while keeping hip open.",
        },
        "head_alignment": {
            "source": "Track testing: head tuck saves ~4.6% drag (~4.17s over pursuit distance)",
        },
    },
    "triathlon": {
        "trunk_angle": {
            "source": "Triathlon position norms; ~25 deg recommended",
            "warning_low": 15,
            "warning_high": 35,
        },
        "hip_angle_max": {
            "source": "Retul; adjusted for triathlon (run preservation)",
            "warning_low": 35,
            "note": "ASYMMETRIC: above optimal = comfort (OK). Below 45 = medical risk. More conservative than TT for run.",
            "medical_warning": (
                "Hip angle <45 deg carries risk of iliac artery endofibrosis. "
                "Monitor for sudden power loss or leg numbness during hard efforts."
            ),
        },
        "knee_at_bdc": {
            "source": "Retul TT/Triathlon fit window",
            "warning_low": 133,
            "warning_high": 148,
        },
    },
    "casual": {
        "trunk_angle": {
            "source": "Recreational/commuter norms",
            "warning_high": 65,
        },
        "knee_at_bdc": {
            "source": "Retul; Burt 'Bike Fit'",
            "warning_low": 130,
            "warning_high": 155,
        },
    },
}


# ---------------------------------------------------------------------------
# Public API: range lookups
# ---------------------------------------------------------------------------

def get_cycling_reference(position: str | None = None) -> dict[str, tuple[float, float]]:
    """Get optimal ranges for a cycling position.

    Falls back to road_hoods if position is None or invalid.
    Returns dict matching CYCLING_REFERENCE format from sport_configs.py.
    """
    pos = CYCLING_POSITIONS.get(position or "road_hoods", CYCLING_POSITIONS["road_hoods"])
    return {
        "knee_at_bdc": pos["knee_at_bdc"],
        "knee_at_tdc": pos["knee_at_tdc"],
        "trunk_angle": pos["trunk_angle"],
        "elbow_angle": pos["elbow_angle"],
        "hip_angle_max": pos["hip_angle_max"],
        "hip_at_bdc": pos["hip_at_bdc"],
        "hip_at_tdc": pos["hip_at_tdc"],
        "shoulder_angle": pos["shoulder_angle"],
        "forearm_tilt": pos["forearm_tilt"],
        "head_alignment": pos["head_alignment"],
        "pelvic_ratio": pos["pelvic_ratio"],
    }


def get_position_label(position: str | None) -> str:
    """Get display label for a cycling position."""
    if position and position in CYCLING_POSITIONS:
        return CYCLING_POSITIONS[position]["label"]
    return "Road - Hoods"


def is_valid_position(position: str | None) -> bool:
    """Check if a cycling position string is valid."""
    return position is not None and position in CYCLING_POSITIONS


def get_position_meta(position: str | None, metric: str) -> dict[str, Any]:
    """Get evidence metadata for a specific metric in a position.

    Returns empty dict if no metadata is available.
    """
    pos_meta = CYCLING_POSITIONS_META.get(position or "road_hoods", {})
    return pos_meta.get(metric, {})


# ---------------------------------------------------------------------------
# Auto-detection: resolve cycling config from measured metrics
# ---------------------------------------------------------------------------

def determine_cycling_config(
    cycling_position: str | None,
    detected_metrics: dict[str, Any],
) -> tuple[str, str | None]:
    """Determine the best cycling config based on user selection + measured metrics.

    Returns (resolved_position, override_reason).
    override_reason is None if the user-selected position matches the data.
    """
    pos = cycling_position or "road_hoods"

    trunk = detected_metrics.get("trunk_angle_avg", 0)
    elbow = detected_metrics.get("elbow_angle_avg", 0)

    if trunk <= 0 or elbow <= 0:
        return pos, None

    # User said road but metrics clearly show TT position
    if pos in ("road_hoods", "road_drops", "casual"):
        if trunk < 30 and elbow < 115:
            return "tt_aero", (
                f"Measured trunk {trunk:.0f} deg and elbow {elbow:.0f} deg "
                f"suggest a TT/aero position rather than {get_position_label(pos)}."
            )
        if pos == "road_hoods" and trunk < 40:
            return "road_drops", (
                f"Measured trunk {trunk:.0f} deg is more aggressive than "
                f"typical hoods position -- using Drops ranges for more accurate scoring."
            )

    # User said TT but metrics show upright position
    if pos in ("tt_aero", "triathlon"):
        if trunk > 40 and elbow > 130:
            return "road_hoods", (
                f"Measured trunk {trunk:.0f} deg and elbow {elbow:.0f} deg "
                f"suggest a road position rather than {get_position_label(pos)}."
            )

    return pos, None


# ---------------------------------------------------------------------------
# Medical warnings based on measured metrics + position
# ---------------------------------------------------------------------------

def get_medical_warnings(
    position: str | None,
    metrics: dict[str, Any],
) -> list[dict[str, str]]:
    """Check measured metrics against medical risk thresholds.

    Returns list of warning dicts with type, message, source.
    """
    warnings: list[dict[str, str]] = []
    pos = position or "road_hoods"
    category = CYCLING_POSITIONS.get(pos, {}).get("category", "road")

    # Hip angle closure risk (primarily TT/triathlon)
    hip_min = metrics.get("hip_angle_min")
    if hip_min is None:
        hip_min = metrics.get("hip_angle_max")

    if hip_min is not None and hip_min < 45 and category == "tt":
        meta = get_position_meta(pos, "hip_angle_max")
        warnings.append({
            "type": "iliac_artery_risk",
            "message": meta.get(
                "medical_warning",
                "Hip angle <45 deg carries risk of iliac artery endofibrosis. "
                "Monitor for sudden power loss or leg numbness.",
            ),
            "source": "Burt 'Bike Fit'; Fintelman et al. 2015",
        })

    # Extreme trunk angle (sustainability concern)
    trunk = metrics.get("trunk_angle_avg", 0)
    if trunk > 0 and trunk < 15 and category == "tt":
        warnings.append({
            "type": "extreme_trunk",
            "message": (
                f"Trunk angle {trunk:.0f} deg is extremely aggressive. "
                "Sustained extreme positions cause fatigue limiting sustainability. "
                "Ensure you can hold this for the full event duration."
            ),
            "source": "Track cycling research; positional drift studies",
        })

    return warnings


# ---------------------------------------------------------------------------
# Position archetype classification (Steinmetz framework)
# ---------------------------------------------------------------------------

_ARCHETYPE_INFO: dict[str, dict[str, str]] = {
    "bar_chaser": {
        "label": "Bar Chaser",
        "description": (
            "Your position pattern is compact -- you tend to keep your "
            "shoulder angle tight and may shift forward on the saddle to "
            "reach the bars. This can work well but watch for closed hip "
            "angles at the top of the pedal stroke."
        ),
        "recommendation": (
            "If extending reach, preserve your shoulder angle and let the "
            "bars come to you rather than reaching for them."
        ),
    },
    "saddle_chaser": {
        "label": "Saddle Chaser",
        "description": (
            "Your position is extended -- you stay anchored on the saddle "
            "with open shoulder and arm angles. This 'superman' style can "
            "be very aero but ensure you feel stable on the front end."
        ),
        "recommendation": (
            "Modern trend (Steinmetz 2024): consider adding 10-20 deg bar "
            "tilt to create a 'pocket' for your head."
        ),
    },
    "balanced": {
        "label": "Balanced",
        "description": (
            "Your position shows a good balance between reach and "
            "compactness. This is similar to what we see in athletes like "
            "Magnus Ditlev and Sam Laidlow -- effective 'all-day' aero "
            "positions."
        ),
        "recommendation": (
            "Maintain this balance. Focus on comfort and sustainability "
            "at race power."
        ),
    },
}


def detect_position_archetype(
    shoulder_angle: float,
    elbow_angle: float,
    trunk_angle: float,
    hip_angle: float,
    cycling_position: str | None = None,
) -> dict[str, Any] | None:
    """Classify rider as bar_chaser, saddle_chaser, or balanced.

    Based on Mat Steinmetz's archetype framework:
    - Bar chasers preserve shoulder angle, shift forward on saddle
    - Saddle chasers stay seated, extend arm/shoulder angles (superman)
    - Balanced riders have good reach/compactness equilibrium

    Returns dict with type/label/confidence/description/recommendation,
    or None if inputs are insufficient or confidence is too low.
    """
    # Guard: need valid angle measurements
    for val in (shoulder_angle, elbow_angle, trunk_angle, hip_angle):
        if val == 0 or (isinstance(val, float) and math.isnan(val)):
            return None

    # Get position-specific optimal ranges for normalization
    pos = CYCLING_POSITIONS.get(
        cycling_position or "road_hoods", CYCLING_POSITIONS["road_hoods"],
    )
    sh_min, sh_max = pos["shoulder_angle"]
    el_min, el_max = pos["elbow_angle"]

    # Normalize each angle relative to the position's optimal range
    # 0 = at lower bound (compact), 1 = at upper bound (extended)
    sh_range = max(1, sh_max - sh_min)
    el_range = max(1, el_max - el_min)

    sh_norm = (shoulder_angle - sh_min) / sh_range
    el_norm = (elbow_angle - el_min) / el_range
    hip_norm = (hip_angle - 35) / 35  # general hip range 35-70 deg

    # Weighted extension score (0 = very compact, 1 = very extended)
    extension_score = (
        sh_norm * 0.4
        + el_norm * 0.3
        + hip_norm * 0.3
    )
    extension_score = max(0.0, min(1.0, extension_score))

    if extension_score < 0.35:
        archetype = "bar_chaser"
        # Further from center = more confident it's bar_chaser
        confidence = round((0.5 - extension_score) * 2, 2)
    elif extension_score > 0.65:
        archetype = "saddle_chaser"
        # Further from center = more confident it's saddle_chaser
        confidence = round((extension_score - 0.5) * 2, 2)
    else:
        archetype = "balanced"
        # Closer to center = more confident it's balanced
        confidence = round(1.0 - abs(extension_score - 0.5) * 2 / 0.3, 2)
        confidence = max(0.0, min(1.0, confidence))

    # Don't report if too uncertain
    if confidence < 0.3:
        return None

    info = _ARCHETYPE_INFO[archetype]
    return {
        "type": archetype,
        "label": info["label"],
        "confidence": confidence,
        "extension_score": round(extension_score, 3),
        "description": info["description"],
        "recommendation": info["recommendation"],
    }
