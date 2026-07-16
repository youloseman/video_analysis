"""Relative aero (CdA-zone) estimation from measured trunk angle.

This module turns the trunk angle we already measure dynamically into a
*relative* aerodynamic read-out, in the spirit of what studio tools like
Velogicfit surface -- but honestly scoped to what a phone-video pipeline
can actually claim.

What it is NOT
--------------
It is NOT an absolute CdA measurement. We have no frontal-area sensor
(no depth camera) and no wind-tunnel. Even Velogicfit -- which *does*
use a time-of-flight depth camera -- states in its own docs that its
aero read "isn't accurate as an absolute measure" and is only good for
detecting *relative* changes between positions. We claim strictly less:
a qualitative CdA *zone* keyed off torso angle, plus the drag/watts
*delta* of moving toward a flatter torso.

Methodology (kept deliberately transparent)
-------------------------------------------
1. Anchors. van Druenen & Blocken (2023) measured one rig in three
   postures at 15 m/s:
       dropped-high (upright) torso : CdA 0.266 m^2   (baseline 100%)
       dropped-low  torso           : CdA 0.231 m^2   (~87%)
       time-trial tuck              : CdA 0.213 m^2   (~80%)
   The study labels postures qualitatively, NOT by a numeric torso
   angle, so we do NOT pretend "42 deg -> 0.266". Instead we map our
   own position trunk-angle bands (cycling_positions.py) onto these
   qualitative zones, which is exactly how the bands were themselves
   derived (Retul / competitive norms).

2. Delta. Drag force scales linearly with CdA at fixed speed, and at a
   fixed speed the power to overcome aero drag scales with CdA too, so
       dP / P_aero  ==  dCdA / CdA .
   Reporting the *relative* saving sidesteps every absolute-CdA
   unknown (clothing, helmet, wind, actual speed). The watt figure is
   an illustrative point estimate at one reference speed, clearly
   labelled as such, using a typical rider aero-power baseline.

3. Guardrails. We only estimate for road / TT / tri positions. For a
   casual/commuter fit, aero is not the goal and we stay silent. We
   never recommend a torso lower than the position's own optimal
   floor -- flatter is only "better" if the rider can still make power
   and breathe, which is a fit constraint, not an aero one.

References
----------
- van Druenen, T. & Blocken, B. (2023), Computers & Fluids 257.
- Blocken, B. et al. (2018), J. Wind Eng. Ind. Aerodyn. 182
  (aero ~90% of resistance at 54 km/h).
- Burt, P. (2014), Bike Fit 2nd ed. (rider body = 70-80% of drag).
See content/academy/bike-aero-position.md for the rider-facing writeup.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# CdA zone anchors (van Druenen & Blocken 2023, one rig / one method)
# ---------------------------------------------------------------------------
# Ordered flattest-torso-last. Each zone owns the trunk-angle band from
# the matching position archetype. `cda` is the reference value; `rel`
# is drag relative to the upright baseline (0.266 -> 1.00).
AERO_ZONES: list[dict[str, Any]] = [
    {
        "key": "upright",
        "label": "Upright torso",
        "trunk_band": (45, 90),      # casual / hoods-high
        "cda": 0.266,
        "rel": 1.00,
    },
    {
        "key": "dropped_high",
        "label": "Dropped, high torso",
        "trunk_band": (38, 45),      # road on hoods
        "cda": 0.266,
        "rel": 1.00,
    },
    {
        "key": "dropped_low",
        "label": "Dropped, low torso",
        "trunk_band": (28, 38),      # road in drops / competitive
        "cda": 0.231,
        "rel": 0.87,
    },
    {
        "key": "tt_tuck",
        "label": "Time-trial tuck",
        "trunk_band": (0, 28),       # TT / tri aero
        "cda": 0.213,
        "rel": 0.80,
    },
]

# Illustrative reference point for the watt figure. Aero power to hold
# ~40 km/h on the flat for a typical road rider sits near this order of
# magnitude; it exists only to make the relative delta tangible and is
# labelled "approx" everywhere it surfaces.
REF_SPEED_KMH = 40.0
REF_AERO_WATTS = 190.0  # ballpark aero-only power @ ~40 km/h, road drops
# Speed band for the watt *range*. Aero power scales with speed^3, so the
# same %-drag saving is worth very different watts at 35 vs 45 km/h. We
# report the range instead of a single point so the figure is honest about
# that dependence (and the client can recompute at the rider's own speed).
REF_SPEED_LOW_KMH = 35.0
REF_SPEED_HIGH_KMH = 45.0


def _aero_watts_at(speed_kmh: float) -> float:
    """Ballpark aero-only power at ``speed_kmh`` (scales with speed^3)."""
    return REF_AERO_WATTS * (speed_kmh / REF_SPEED_KMH) ** 3


def watts_saved_at(rel_saving: float, speed_kmh: float) -> int:
    """Watts saved from a relative drag saving at a given speed, to nearest 5."""
    return int(round(rel_saving * _aero_watts_at(speed_kmh) / 5.0) * 5)

# Below this relative saving a "flatten your torso" suggestion is noise
# (same zone, rounding), so we suppress next_zone entirely.
_MIN_MEANINGFUL_SAVING = 0.02

# Positions where an aero read is meaningful. Casual is intentionally
# excluded -- comfort is the goal there, not drag.
_AERO_POSITIONS = {"road_hoods", "road_drops", "tt_aero", "triathlon"}

_DISCLAIMER = (
    "Relative estimate from torso angle only -- not an absolute CdA. "
    "Real drag also depends on clothing, helmet, speed and wind. "
    "Flatter is only faster if you can still make power and breathe there."
)


def _zone_for_trunk(trunk_angle: float) -> dict[str, Any]:
    """Pick the CdA zone whose trunk band contains ``trunk_angle``."""
    for zone in AERO_ZONES:
        lo, hi = zone["trunk_band"]
        if lo <= trunk_angle < hi:
            return zone
    # Above every band -> most upright; below every band -> deepest tuck.
    if trunk_angle >= AERO_ZONES[0]["trunk_band"][1]:
        return AERO_ZONES[0]
    return AERO_ZONES[-1]


def estimate_aero(
    trunk_angle: float | None,
    cycling_position: str | None,
    *,
    optimal_trunk_band: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Build a relative aero read-out from the measured trunk angle.

    Parameters
    ----------
    trunk_angle:
        Mean dynamic trunk angle in degrees from vertical (our
        ``trunk_angle_avg``). ``None`` -> no estimate.
    cycling_position:
        Resolved position key. Only road/TT/tri produce an estimate.
    optimal_trunk_band:
        The position's own optimal (lo, hi) trunk range, used to cap the
        recommended flatten target at the position's floor. When given,
        we never suggest flatter than ``lo``.

    Returns
    -------
    dict with the current zone, drag relative to upright, the next
    flatter zone (if any and reachable within the fit), the drag/watt
    delta of reaching it, and a disclaimer -- or ``None`` when an aero
    read is not applicable.
    """
    if trunk_angle is None or cycling_position not in _AERO_POSITIONS:
        return None
    if not (0.0 < trunk_angle < 90.0):
        return None

    current = _zone_for_trunk(trunk_angle)
    cur_idx = AERO_ZONES.index(current)

    result: dict[str, Any] = {
        "trunk_angle": round(float(trunk_angle), 1),
        "zone_key": current["key"],
        "zone_label": current["label"],
        "cda_reference": current["cda"],
        "drag_vs_upright_pct": round(current["rel"] * 100),
        "reference_speed_kmh": REF_SPEED_KMH,
        "disclaimer": _DISCLAIMER,
        "source": "van Druenen & Blocken 2023 (Computers & Fluids 257)",
        "next_zone": None,
    }

    # Is there a flatter zone, and does the fit allow reaching for it?
    next_zone = AERO_ZONES[cur_idx + 1] if cur_idx + 1 < len(AERO_ZONES) else None
    if next_zone is not None:
        # Target the boundary between current and next band (the top of
        # the next band), but never flatter than the position's optimal
        # floor -- aero must not push past a safe fit.
        target_trunk = float(next_zone["trunk_band"][1])
        floor = optimal_trunk_band[0] if optimal_trunk_band else None
        if floor is not None:
            target_trunk = max(target_trunk, float(floor))

        # Only offer it if the target is actually flatter than now AND
        # the saving is above rounding noise. The watt figure is rounded
        # to the nearest 5 W so it reads as an order-of-magnitude, not a
        # precise measurement.
        rel_saving = (current["rel"] - next_zone["rel"]) / current["rel"]
        if target_trunk < trunk_angle and rel_saving >= _MIN_MEANINGFUL_SAVING:
            watts = watts_saved_at(rel_saving, REF_SPEED_KMH)
            watts_low = watts_saved_at(rel_saving, REF_SPEED_LOW_KMH)
            watts_high = watts_saved_at(rel_saving, REF_SPEED_HIGH_KMH)
            result["next_zone"] = {
                "zone_key": next_zone["key"],
                "zone_label": next_zone["label"],
                "target_trunk_angle": round(target_trunk, 1),
                "drag_reduction_pct": round(rel_saving * 100, 1),
                # Point estimate at 40 km/h (kept for back-compat) + an honest
                # range across a typical race-speed band. The client can also
                # recompute at the rider's own speed from drag_reduction_pct
                # and ref_aero_watts (watts = saving * ref * (v/ref_speed)^3).
                "approx_watts_saved": watts,
                "watts_saved_range": [watts_low, watts_high],
                "speed_range_kmh": [REF_SPEED_LOW_KMH, REF_SPEED_HIGH_KMH],
            }

    # Expose the reference baseline so the client can personalise the watt
    # figure to the rider's own speed without another server round-trip.
    result["ref_aero_watts"] = REF_AERO_WATTS

    return result
