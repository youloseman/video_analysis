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
    # Below 28 deg the anchor study has one posture, so this used to be one
    # bucket -- and 10 of 14 Kona 2026 pros landed in it, the scale going
    # silent exactly where a triathlete looks for the next step. Fintelman's
    # thesis (Birmingham 2015, Table 3.2, n=19 TT riders and triathletes)
    # MEASURED frontal area at 24 / 16 / 8 / 0 deg: 0.335 / 0.318 / 0.302 /
    # 0.288 m^2. Each zone below is centred on one measured angle, its edges
    # halfway to the next, and steps down from the van Druenen TT anchor by
    # the measured frontal-area ratio. Frontal area, not drag area: the
    # conservative choice (Underwood 2011 found drag area falling faster,
    # ~16 % over 2-20 deg), and Maddocks & Walker 2025 warn the two need not
    # move together. The 0 deg point is left out: Fintelman found a flat back
    # cost power and metabolism significantly against every other position.
    {
        "key": "tt_tuck",
        "label": "Time-trial tuck",
        "trunk_band": (20, 28),      # centred on Fintelman's 24 deg
        "cda": 0.213,
        "rel": 0.80,
    },
    {
        "key": "tt_low",
        "label": "Low time-trial tuck",
        "trunk_band": (12, 20),      # centred on 16 deg: 0.318 / 0.335 frontal area
        "cda": round(0.213 * 0.318 / 0.335, 3),
        "rel": round(0.80 * 0.318 / 0.335, 3),
    },
    {
        "key": "tt_flat",
        "label": "Flat time-trial tuck",
        "trunk_band": (0, 12),       # centred on 8 deg: 0.302 / 0.335 frontal area
        "cda": round(0.213 * 0.302 / 0.335, 3),
        "rel": round(0.80 * 0.302 / 0.335, 3),
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


# Relative drag against trunk angle inside the TT range, on Fintelman's
# measured points (frontal area, scaled to the van Druenen TT anchor at 24
# deg). Linear between them, flat beyond: nothing measured above 24 in this
# series, and the 0 deg point is excluded (see AERO_ZONES).
_TT_REL_POINTS: tuple[tuple[float, float], ...] = (
    (8.0, 0.80 * 0.302 / 0.335),
    (16.0, 0.80 * 0.318 / 0.335),
    (24.0, 0.80),
    # The road-drops posture of the anchor study, at the middle of its band.
    # Linear from the TT posture to it: a rider at 29 deg sits between the
    # two measured postures, not at the step between them. Without this a
    # rider at 28.9 was told a degree flatter was worth 8 % (2026-10-13).
    (33.0, 0.87),
)
_TT_RANGE_TOP = 33.0


def _rel_for_trunk(trunk_angle: float) -> float:
    """Drag relative to upright at this trunk angle.

    Up to the road-drops posture (33 deg) it is interpolated between measured
    postures, so a two-degree change is worth two degrees: the zone-step figure
    would have credited a triathlete moving 22 -> 20 deg with the whole
    24 -> 16 deg saving (5 %) where the measured slope gives 1.3 %. Above it the
    zones are qualitative postures with no angles, and their step is all there
    is.
    """
    if trunk_angle >= _TT_RANGE_TOP:
        return float(_zone_for_trunk(trunk_angle)["rel"])
    pts = _TT_REL_POINTS
    if trunk_angle <= pts[0][0]:
        return pts[0][1]
    if trunk_angle >= pts[-1][0]:
        return pts[-1][1]
    for (a0, r0), (a1, r1) in zip(pts, pts[1:]):
        if a0 <= trunk_angle <= a1:
            return r0 + (r1 - r0) * (trunk_angle - a0) / (a1 - a0)
    return pts[-1][1]


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


# --- Secondary aero levers (head, elbow bend). Only meaningful in an
# aero-bar position, where the anchors below actually apply. Torso stays the
# quantified primary; these enrich the picture. ---
_AERO_BAR_POSITIONS = {"tt_aero", "triathlon"}


def helmet_lever(
    helmet_above_back: float | None,
    helmet_check: bool | None,
    position: str | None,
) -> dict[str, Any] | None:
    """The head lever, from the helmet reading (head_silhouette.py).

    This lever used to read the old head-alignment score -- an ear under a line
    through the joint centres, which every aero rider passed -- and told a rider
    sitting up with the head high (a $4 report, 2026-10-13) that the head was
    "well tucked". It reads the helmet above the back now, against the same
    Kona field and free band the score uses. Qualitative: the one study we
    have (Beaumont 2018, CFD) compares head up with head down, up to ~6.4 %
    drag, and gives no curve to turn a percentage of torso into watts.
    """
    if position not in _AERO_BAR_POSITIONS or helmet_above_back is None:
        return None
    from app.services.video_analysis.biomechanics.head_silhouette import (
        KONA_PRO_FREE_UP_TO,
        KONA_PRO_UNDER_FREE,
        graded_helmet,
    )
    # No helmet advice: from the side we cannot tell which helmet anyone is
    # wearing, and telling riders in aero helmets to put one on cost a review.
    if graded_helmet(helmet_above_back, helmet_check) is None:
        return {"key": "head", "label": "Head", "status": "unknown", "qualitative": True,
                "cue": "We could not tell where your back ends and the helmet begins on "
                       "this clip, so the head is not graded on it."}
    if helmet_above_back <= KONA_PRO_FREE_UP_TO:
        return {"key": "head", "label": "Head", "status": "ok",
                "cue": f"Helmet {helmet_above_back:+.0f}% of a torso above your back: "
                       f"where {KONA_PRO_UNDER_FREE} Kona pros rode."}
    return {
        "key": "head", "label": "Head", "status": "available", "qualitative": True,
        "cue": f"Your helmet sits {helmet_above_back:+.0f}% of a torso above your back, "
               f"higher than {KONA_PRO_UNDER_FREE} Kona pros (up to "
               f"+{KONA_PRO_FREE_UP_TO:.0f}%). Drop the head toward the line of your "
               "back, as far as you can still see the road: head up against head "
               "down changed drag by up to ~6% in a CFD study.",
    }


def with_helmet_lever(
    aero: dict[str, Any] | None,
    helmet_above_back: float | None,
    helmet_check: bool | None,
    position: str | None,
) -> dict[str, Any] | None:
    """The aero read-out with its head lever replaced by the helmet reading.

    The helmet is measured after the analyzer builds the aero read-out (it
    needs the clip's frames re-read through the segmentation model), so the
    runner calls this once it has the value.
    """
    if not isinstance(aero, dict):
        return aero
    levers = [lv for lv in aero.get("levers") or [] if lv.get("key") != "head"]
    lever = helmet_lever(helmet_above_back, helmet_check, position)
    if lever is not None:
        torso_at = next((i for i, lv in enumerate(levers) if lv.get("key") == "torso"), -1)
        levers.insert(torso_at + 1, lever)
    return {**aero, "levers": levers}


def _elbow_lever(
    elbow_angle: float | None,
    elbow_optimal_band: tuple[float, float] | None,
    position: str | None,
) -> dict[str, Any] | None:
    """Elbow-bend lever -- qualitative only (no clean per-degree drag anchor).

    Straight arms on the pads widen the frontal area; we cue a tuck toward
    the position's optimal elbow band without inventing a watt figure.
    """
    if position not in _AERO_BAR_POSITIONS:
        return None
    if elbow_angle is None or elbow_optimal_band is None:
        return None
    lo, hi = float(elbow_optimal_band[0]), float(elbow_optimal_band[1])
    if elbow_angle > hi + 10:
        return {
            "key": "elbow", "label": "Elbow bend", "status": "available",
            "qualitative": True,
            "cue": (
                f"Arms are quite straight (~{round(elbow_angle)}°). Bend your "
                f"elbows toward {int(lo)}-{int(hi)}° on the pads to narrow your "
                f"frontal area (smaller aero win, hard to quantify from 2D)."
            ),
        }
    return {"key": "elbow", "label": "Elbow bend", "status": "ok",
            "cue": "Elbows are well bent on the pads."}


def estimate_aero(
    trunk_angle: float | None,
    cycling_position: str | None,
    *,
    optimal_trunk_band: tuple[float, float] | None = None,
    elbow_angle: float | None = None,
    elbow_optimal_band: tuple[float, float] | None = None,
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
        "drag_vs_upright_pct": round(_rel_for_trunk(trunk_angle) * 100),
        "reference_speed_kmh": REF_SPEED_KMH,
        "disclaimer": _DISCLAIMER,
        "source": (
            "van Druenen & Blocken 2023 (Computers & Fluids 257)"
            + ("; below 28 deg: Fintelman 2015 (PhD, Birmingham), measured frontal area"
               if trunk_angle < 28.0 else "")
        ),
        "next_zone": None,
    }

    # Is there a flatter zone, and does the fit allow reaching for it?
    next_zone = AERO_ZONES[cur_idx + 1] if cur_idx + 1 < len(AERO_ZONES) else None
    if next_zone is not None:
        # Target the MIDDLE of the next zone, never flatter than the
        # position's optimal floor -- aero must not push past a safe fit.
        # It used to target the zone's top edge, and read the whole step
        # between zones as the saving: a rider at 28.9 deg was told to go
        # "toward 28.0 deg" for 8 % less drag and ~15 W -- a degree is not
        # worth that (a $4 report, 2026-10-13).
        lo_z, hi_z = next_zone["trunk_band"]
        target_trunk = (float(lo_z) + float(hi_z)) / 2.0
        floor = optimal_trunk_band[0] if optimal_trunk_band else None
        if floor is not None:
            target_trunk = max(target_trunk, float(floor))

        # Only offer it if the target is actually flatter than now AND
        # the saving is above rounding noise. The watt figure is rounded
        # to the nearest 5 W so it reads as an order-of-magnitude, not a
        # precise measurement.
        # The target sits ON the boundary of the next zone, and a boundary
        # belongs to the zone above it ([lo, hi)), so read the next zone's
        # own figure there rather than the current zone's again.
        rel_now = _rel_for_trunk(trunk_angle)
        rel_target = (_rel_for_trunk(target_trunk) if target_trunk < _TT_RANGE_TOP
                      else float(next_zone["rel"]))
        rel_saving = (rel_now - rel_target) / rel_now
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

    # ---- Multi-lever breakdown -------------------------------------------
    # Torso is the quantified primary; head/elbow are secondary levers that
    # only apply in an aero-bar position. Each lever carries its own cue and
    # (where we have an anchor) its drag/watt contribution.
    levers: list[dict[str, Any]] = []
    nz = result["next_zone"]
    if nz:
        levers.append({
            "key": "torso", "label": "Torso angle", "status": "available",
            "cue": f"Flatten your torso from {result['trunk_angle']}° toward "
                   f"{nz['target_trunk_angle']}°.",
            "drag_reduction_pct": nz["drag_reduction_pct"],
            "approx_watts_saved": nz["approx_watts_saved"],
            "watts_saved_range": nz["watts_saved_range"],
        })
    else:
        levers.append({"key": "torso", "label": "Torso angle", "status": "ok",
                       "cue": "Torso is well-placed for this position."})
    for lever in (
        _elbow_lever(elbow_angle, elbow_optimal_band, cycling_position),
    ):
        if lever is not None:
            levers.append(lever)
    result["levers"] = levers

    # Combined quantified opportunity. Drag reductions compound
    # multiplicatively (1 - product of remaining fractions), not additively.
    quant = [
        lv["drag_reduction_pct"] / 100.0
        for lv in levers if lv.get("drag_reduction_pct")
    ]
    if len(quant) >= 2:
        remaining = 1.0
        for d in quant:
            remaining *= (1.0 - d)
        total_drag = 1.0 - remaining
        result["total_drag_reduction_pct"] = round(total_drag * 100, 1)
        result["total_watts_saved"] = watts_saved_at(total_drag, REF_SPEED_KMH)
        result["total_watts_saved_range"] = [
            watts_saved_at(total_drag, REF_SPEED_LOW_KMH),
            watts_saved_at(total_drag, REF_SPEED_HIGH_KMH),
        ]

    return result
