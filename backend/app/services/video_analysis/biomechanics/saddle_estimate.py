"""Saddle height in millimetres, from the bike's own ruler.

The report has always known which way the saddle should go -- the knee at
the bottom of the stroke is either inside its band or not -- and has always
said "about 5 mm", because a side view has no length in it. It does have
one: the ankle traces the crank, and a crank is a known length (172.5 mm on
most bikes; the rider can tell us theirs). ``bilateral.summarize_side``
already measures the thigh, the shin and the hip-ankle chord at BDC in units
of that circle's radius, so every one of them is a real length the moment a
crank length is named.

From there the estimate is one triangle. At BDC the hip, knee and ankle form
a triangle with two known sides (thigh T, shin S) and the knee angle between
them; the third side is the hip-to-pedal distance d. Moving the knee from
its measured angle to the middle of the band changes d by the law of
cosines, and that change -- along the line from hip to pedal, which is near
enough the seat tube -- is the saddle change::

    d(k) = sqrt(T^2 + S^2 - 2 T S cos k)
    delta = (d(target) - d(now)) * crank_mm

Positive = raise (more extension wanted). ~2.5 mm per degree near 140 deg on
an adult leg, which is the fitters' rule of thumb arrived at from the other
end.

WHAT THE +/- IS. The knee reading itself is the largest term: the same clip
mirrored -- which cannot change an angle -- moves knee_at_bdc by up to 3 deg
(see fit_tradeoffs / bilateral notes), and a clip's own stroke-to-stroke
spread can be wider. That angle floor is carried through the same
derivative, and the chord's measured scatter across revolutions is added in
quadrature. On a drive-side clip the chainring sits on the ankle's bottom
arc and the crank reads ~12% small (the drive-side plan), so the scale is
12% uncertain too and the interval widens by that share.

WHAT IT REFUSES. Fewer than two clean revolutions (no ruler), no knee at
BDC (nothing to move), or a geometry the pipeline could not reduce. It never
refuses for being out of band -- in band it reports ``direction: "none"``
with the distance to the middle, which the page may choose not to print.
"""

from __future__ import annotations

import math
from typing import Any

from app.services.video_analysis.biomechanics.bilateral import SideGeometry

DEFAULT_CRANK_MM = 172.5
# The floor on the knee angle's own uncertainty: the mirror test.
ANGLE_FLOOR_DEG = 3.0
# Share the crank ruler reads small on a drive-side clip.
DRIVE_SIDE_SCALE_ERR = 0.12
MIN_REVOLUTIONS = 2
# The verdict's own tolerance past the band (cycling_analyzer): no direction
# is given inside it.
ASSESS_MARGIN_DEG = 5.0
# Below this the number is inside its own noise and the page says "keep".
MATERIAL_MM = 2.0


def _chord(thigh: float, shin: float, knee_deg: float) -> float:
    k = math.radians(knee_deg)
    return math.sqrt(max(0.0, thigh * thigh + shin * shin - 2.0 * thigh * shin * math.cos(k)))


def estimate_saddle_change(
    geometry: dict[str, Any] | None,
    knee_at_bdc: float | None,
    band: tuple[float, float],
    crank_length_mm: float | None = None,
    *,
    bdc_variability_deg: float | None = None,
    camera_side: str | None = None,
) -> dict[str, Any] | None:
    """The saddle change that would put the knee at BDC mid-band, in mm.

    ``geometry`` is the result's ``bilateral_geometry``; ``band`` the
    position's ``knee_at_bdc`` reference; ``crank_length_mm`` the rider's, or
    None for the default (and the result says which).
    """
    geom = SideGeometry.from_dict(geometry)
    if geom is None or geom.revolutions < MIN_REVOLUTIONS:
        return None
    if knee_at_bdc is None or not math.isfinite(float(knee_at_bdc)):
        return None
    thigh, shin = geom.thigh, geom.shin
    if not (thigh > 0 and shin > 0):
        return None
    lo, hi = float(band[0]), float(band[1])
    if not (lo < hi):
        return None
    crank = float(crank_length_mm) if crank_length_mm else DEFAULT_CRANK_MM
    knee = float(knee_at_bdc)
    target = (lo + hi) / 2.0

    d_now = _chord(thigh, shin, knee)
    d_target = _chord(thigh, shin, target)
    if d_now <= 0:
        return None
    delta_mm = (d_target - d_now) * crank

    # d(d)/dk = T S sin k / d, per radian -- the lever the angle floor rides.
    per_deg = thigh * shin * math.sin(math.radians(knee)) / d_now * math.pi / 180.0
    sigma_deg = max(ANGLE_FLOOR_DEG, float(bdc_variability_deg or 0.0))
    ci = abs(per_deg * sigma_deg) * crank
    ci = math.hypot(ci, float(geom.chord_sd or 0.0) * crank)
    drive_side = camera_side == "right"
    if drive_side:
        ci += abs(delta_mm) * DRIVE_SIDE_SCALE_ERR

    in_band = lo <= knee <= hi
    # The distance to the band's nearer EDGE, the smallest change that puts
    # the knee in range; ``delta_mm`` is the change to its middle.
    edge = lo if knee < lo else (hi if knee > hi else knee)
    to_band_mm = (_chord(thigh, shin, edge) - d_now) * crank if not in_band else 0.0
    # A direction only where the report's own verdict says the saddle is
    # wrong (cycling_analyzer._assess_saddle_height: 5 deg past the band).
    # Inside that margin the verdict is "acceptable" and a number that says
    # "raise 18 mm" under it would be the page arguing with itself.
    if in_band or abs(delta_mm) < MATERIAL_MM or (lo - ASSESS_MARGIN_DEG <= knee <= hi + ASSESS_MARGIN_DEG):
        direction = "none"
    else:
        direction = "raise" if delta_mm > 0 else "lower"
    return {
        "delta_mm": round(delta_mm, 1),
        "to_band_mm": round(to_band_mm, 1),
        "ci_mm": round(ci, 1),
        "direction": direction,
        "in_band": in_band,
        "knee_now": round(knee, 1),
        "knee_target": round(target, 1),
        "mm_per_deg": round(abs(per_deg) * crank, 2),
        "crank_length_mm": crank,
        "crank_source": "given" if crank_length_mm else "assumed",
        "drive_side": drive_side,
        "revolutions": geom.revolutions,
        # Where the number stops being honest: the plane of the near leg,
        # the hip landmark the model infers, one crank length.
        "basis": "hip-to-pedal distance at BDC from the crank circle; along the seat tube, near side only",
    }


def format_amount(est: dict[str, Any] | None, direction: str) -> str | None:
    """'≈8 mm (±5)' for the action plan, or None when the estimate does not
    back this direction (the caller keeps its generic amount)."""
    if not est or est.get("direction") != direction:
        return None
    return f"≈{abs(float(est['delta_mm'])):.0f} mm (±{float(est['ci_mm']):.0f})"
