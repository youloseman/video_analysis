"""Two clips, two sides, one body: combine a left-view and a right-view ride.

WHY THIS EXISTS. A bike side view can only measure the leg nearest the camera,
so a complete fit needs two clips. But filmed separately they disagree, and on
2026-08-26 that disagreement was traced to its root rather than papered over:

* Re-running detection on the SAME clip with the pixels mirrored -- which
  cannot change a real angle -- moved knee-at-bottom by 1.4-3.5 deg and the
  hip landmark by 3-7 cm. Pure model, no anatomy.
* Aligning two clips of one unchanged bike on the bottom bracket showed them
  agreeing on hip-to-ankle distance to 0.1% -- the quantity the FIT sets --
  while disagreeing by ~2% on thigh+shin, the quantity ANATOMY sets. Only one
  of those can be measurement error.
* Swapping one joint at a time, the hip alone carried 85% of the gap. The hip
  joint is inside the pelvis; no camera has ever seen it. The model guesses,
  and guesses differently depending on which side faces the lens.

The fix follows from the third point. The hip's error is mostly PERPENDICULAR
to the leg, and a length is insensitive to perpendicular error to second order
(40 mm sideways on a 775 mm leg moves the length by ~1 mm). So the hip-ankle
CHORD is robust where the direct vector angle is not -- the vector angle reads
the very direction that is wrong. Compute the knee from the chord and a pair
of bone lengths instead, estimate those bone lengths ONCE from both clips, and
the two views stop contradicting each other: measured on Artur's pair, the gap
between sides fell from 9.1 deg to 0.9 deg.

WHAT THIS BUYS BEYOND ONE NUMBER. Once both sides describe one body, the
residual difference between them is no longer dominated by the model, so it
becomes a usable asymmetry reading -- and metrics that MUST agree (the trunk
is a midline structure seen from both sides) turn into a direct measurement of
this session's error, for this rider, on this day. The pair calibrates itself.

WHAT IT CANNOT DO. Pooling two views removes the disagreement between them; it
does not remove a bias they share. The pooled leg length can still be off, and
near full extension the knee angle amplifies that error by ~4 deg per 1% of
leg length. So the combined value ships with an uncertainty, never bare.

AND IT REFUSES. The whole method rests on the two clips sharing a scale. Two
clips filmed at different distances, or one whose ankle orbit the chainring
corrupted, do not -- on the 24 Aug pair the crank-radius estimate was off 11%
and forcing a shared body there produced a 51 deg gap, worse than doing
nothing. ``combine_sides`` checks that first and declines rather than
returning a confident wrong answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Landmark indices, per side.
_HIP = {"left": 23, "right": 24}
_KNEE = {"left": 25, "right": 26}
_ANKLE = {"left": 27, "right": 28}
_SHOULDER = {"left": 11, "right": 12}

# A revolution needs enough bottom-of-orbit samples to have a median worth
# taking, and a clip needs enough revolutions for the spread to mean anything.
_MIN_BOTTOM_SAMPLES_PER_REV = 2
_MIN_REVOLUTIONS = 3
_MIN_MEASURED_FRAMES = 40

# "Bottom of the orbit": the ankle within this band of its circle, in radii.
# No phase, no rotation direction, no peak picking -- just the foot at the
# bottom, which is the one crank position both clips can agree on.
_BOTTOM_MIN_Y = 0.90
_BOTTOM_MAX_X = 0.25
# Frames whose ankle sits this far off the fitted circle are not on the pedal
# path at all (the tracker is on the chainring or the mat) and cannot define
# the bottom of the stroke.
_ON_PATH_BAND = (0.80, 1.25)

# The scale gate. Both clips see the same crank and the same fit, so the
# hip-ankle chord expressed in crank radii is the same number twice. When it
# is not, the crank-radius estimate is wrong in at least one clip and nothing
# downstream can be trusted. 2% is the line: the 26 Aug pair agreed to 0.13%,
# the 24 Aug pair (chainring-corrupted orbit) disagreed by 13%.
_SCALE_TOLERANCE_PCT = 2.0
# Two estimates of one rider's leg differing by more than this are not two
# views of the same person having a bad day -- something is wrong (wrong clip
# paired, different rider, tracking failure).
_LEG_DISAGREEMENT_LIMIT_PCT = 8.0

# Mirroring a clip's pixels cannot change a real angle, yet it moved
# knee-at-bottom by 1.4-3.5 deg. That is the model's own absolute wobble, and
# averaging two views does not cancel a bias they share, so it stays in the
# reported uncertainty as a floor.
_ABSOLUTE_FLOOR_DEG = 3.0


def _pt(frame: dict, idx: int, aspect: float) -> tuple[float, float] | None:
    """A landmark in image-plane units (x rescaled by the frame aspect).

    Normalized coordinates are stretched by the frame's aspect ratio -- on a
    portrait clip x and y are not the same unit, and an angle or a length read
    from them raw is wrong. See flapp's world-vs-image finding.
    """
    lms = frame.get("normalized_landmarks")
    if not lms or idx >= len(lms):
        return None
    lm = lms[idx]
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if x is None or y is None:
        return None
    x, y = float(x), float(y)
    if math.isnan(x) or math.isnan(y):
        return None
    return (x * aspect, y)


def _dist(a, b) -> float:
    if a is None or b is None:
        return float("nan")
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _kasa(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit (Kasa). Fast, and good enough to seed the trim."""
    A = np.column_stack([xs, ys, np.ones(len(xs))])
    b = xs**2 + ys**2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0] / 2.0, c[1] / 2.0
    return cx, cy, math.sqrt(max(c[2] + cx * cx + cy * cy, 1e-12))


def _robust_circle(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Circle fit that ignores the points the tracker dragged off the path.

    The ankle cloud is not a clean circle -- ankling makes an egg, and twice a
    revolution the tracker lands on the chainring or the trainer instead of
    the shoe. Trimming to the closest three quarters, three times, keeps the
    fit on the orbit rather than splitting the difference with the strays.
    """
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    for _ in range(3):
        if len(xs) < 12:
            break
        cx, cy, r = _kasa(xs, ys)
        d = np.abs(np.hypot(xs - cx, ys - cy) - r)
        keep = d <= np.percentile(d, 75)
        if keep.sum() < 12:
            break
        xs, ys = xs[keep], ys[keep]
    return _kasa(xs, ys)


def knee_from_chord(thigh: float, shin: float, chord: float) -> float | None:
    """Knee angle from two bone lengths and the hip-ankle distance.

    This is the whole point of the module: the law of cosines needs the LENGTH
    from hip to ankle, and never asks where the hip sits sideways -- which is
    exactly the coordinate the pose model gets wrong by centimetres and which
    the ordinary vector angle at the knee depends on most.
    """
    if not (thigh > 0 and shin > 0) or not math.isfinite(chord):
        return None
    cosine = (thigh * thigh + shin * shin - chord * chord) / (2.0 * thigh * shin)
    if not math.isfinite(cosine):
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def leg_length_sensitivity_deg(thigh: float, shin: float, chord: float) -> float:
    """Degrees of knee angle per 1% error in the pooled leg length.

    Scaling both bones by (1+e) moves cos(k) by 2e*D^2/(2TS), so the angle
    moves by 2Q/sin(k) radians per unit e. Near full extension sin(k) is small
    and the lever is brutal: ~4 deg per 1% at 150 deg against ~1.6 at 90. This
    is why the bottom of the stroke was the only phase where the two views
    ever disagreed, and why the answer must carry an uncertainty.
    """
    knee = knee_from_chord(thigh, shin, chord)
    if knee is None:
        return float("nan")
    sin_k = math.sin(math.radians(knee))
    if sin_k < 1e-6:
        return float("inf")
    q = (chord * chord) / (2.0 * thigh * shin)
    return math.degrees(2.0 * q / sin_k) * 0.01


@dataclass
class SideGeometry:
    """One clip's body-and-bike geometry, in units of that clip's crank radius.

    Everything is divided by the ankle orbit radius, so the numbers from two
    clips are comparable without knowing either camera's distance -- provided
    the orbit was measured cleanly in both, which ``combine_sides`` verifies.
    """

    camera_side: str
    thigh: float
    shin: float
    torso: float
    chord_bdc: float
    chord_sd: float
    revolutions: int
    measured_frames: int
    crank_radius_px: float

    @property
    def leg(self) -> float:
        return self.thigh + self.shin

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_side": self.camera_side,
            "thigh": round(self.thigh, 4),
            "shin": round(self.shin, 4),
            "leg": round(self.leg, 4),
            "torso": round(self.torso, 4),
            "chord_bdc": round(self.chord_bdc, 4),
            "chord_sd": round(self.chord_sd, 4),
            "revolutions": self.revolutions,
            "measured_frames": self.measured_frames,
        }


def summarize_side(
    frames: list[dict[str, Any]],
    camera_side: str,
    frame_aspect: float,
) -> SideGeometry | None:
    """Reduce one stabilized clip to the geometry the bilateral merge needs.

    Only frames the analyzer would MEASURE are used: a gate-filled frame
    carries a reconstructed leg drawn for the overlay's sake, and letting one
    into a length estimate would be measuring our own prediction.

    And "measured" here means NEITHER leg was gated, not just this one. That
    is stricter than the analyzer's own per-side rule, and it was chosen on
    evidence: when the gate fires on the far leg, the near leg's landmarks in
    that frame are usually disturbed too -- the tracker is losing the whole leg
    region, it just only flagged the side that crossed the threshold. Measured
    on the 26 Aug pair, admitting far-gated frames moved the crank-radius
    estimate by 3.9% and blew the two clips' agreement on hip-to-ankle from
    0.22% to 2.72%, which is the difference between the two sides landing 0.9
    deg apart and 12 deg apart. It costs frames (one clip kept 5 revolutions
    of 13) and ``_MIN_REVOLUTIONS`` refuses when too few survive -- the right
    trade when the alternative is a confident wrong answer.
    """
    if not frames or camera_side not in ("left", "right"):
        return None
    hip_i, knee_i = _HIP[camera_side], _KNEE[camera_side]
    ankle_i, sh_i = _ANKLE[camera_side], _SHOULDER[camera_side]

    n = len(frames)
    hips, knees, ankles, shoulders = [], [], [], []
    measured = np.zeros(n, dtype=bool)
    for i, fr in enumerate(frames):
        # Not `camera_side not in filled` -- see the docstring: a far-leg gate
        # is evidence this frame's near leg is unreliable too.
        measured[i] = not (fr.get("leg_gate_filled") or ())
        hips.append(_pt(fr, hip_i, frame_aspect))
        knees.append(_pt(fr, knee_i, frame_aspect))
        ankles.append(_pt(fr, ankle_i, frame_aspect))
        shoulders.append(_pt(fr, sh_i, frame_aspect))

    ax = np.array([p[0] if p else np.nan for p in ankles], dtype=float)
    ay = np.array([p[1] if p else np.nan for p in ankles], dtype=float)
    usable = measured & np.isfinite(ax) & np.isfinite(ay)
    if usable.sum() < _MIN_MEASURED_FRAMES:
        return None

    cx, cy, radius = _robust_circle(ax[usable], ay[usable])
    if not (radius > 1e-6):
        return None

    rel_x = (ax - cx) / radius
    rel_y = (ay - cy) / radius
    on_path = np.hypot(rel_x, rel_y)
    good = usable & (on_path >= _ON_PATH_BAND[0]) & (on_path <= _ON_PATH_BAND[1])

    thigh = np.array([_dist(hips[i], knees[i]) for i in range(n)], dtype=float)
    shin = np.array([_dist(knees[i], ankles[i]) for i in range(n)], dtype=float)
    torso = np.array([_dist(shoulders[i], hips[i]) for i in range(n)], dtype=float)
    chord = np.array([_dist(hips[i], ankles[i]) for i in range(n)], dtype=float)

    # The bottom of the orbit, grouped into revolutions so the chord's spread
    # across the ride is a real repeatability figure and not the frame rate.
    at_bottom = np.where(good & (rel_y > _BOTTOM_MIN_Y)
                         & (np.abs(rel_x) < _BOTTOM_MAX_X)
                         & np.isfinite(chord))[0]
    if len(at_bottom) < _MIN_BOTTOM_SAMPLES_PER_REV * _MIN_REVOLUTIONS:
        return None
    groups: list[list[int]] = [[int(at_bottom[0])]]
    for idx in at_bottom[1:]:
        if idx - groups[-1][-1] <= 4:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    per_rev = [float(np.nanmedian(chord[g])) for g in groups
               if len(g) >= _MIN_BOTTOM_SAMPLES_PER_REV]
    if len(per_rev) < _MIN_REVOLUTIONS:
        return None

    def _med(arr: np.ndarray) -> float:
        vals = arr[good & np.isfinite(arr)]
        return float(np.median(vals)) if len(vals) else float("nan")

    thigh_m, shin_m, torso_m = _med(thigh), _med(shin), _med(torso)
    if not (thigh_m > 0 and shin_m > 0):
        return None

    return SideGeometry(
        camera_side=camera_side,
        thigh=thigh_m / radius,
        shin=shin_m / radius,
        torso=(torso_m / radius) if math.isfinite(torso_m) else float("nan"),
        chord_bdc=float(np.median(per_rev)) / radius,
        chord_sd=float(np.std(per_rev)) / radius,
        revolutions=len(per_rev),
        measured_frames=int(good.sum()),
        crank_radius_px=float(radius),
    )


@dataclass
class BilateralFit:
    """The verdict of merging two sides -- or the reason there isn't one."""

    combined: bool
    reason: str | None = None
    knee_at_bdc: float | None = None
    uncertainty_deg: float | None = None
    per_side: dict[str, float] = field(default_factory=dict)
    raw_per_side: dict[str, float] = field(default_factory=dict)
    asymmetry_deg: float | None = None
    asymmetry_significant: bool | None = None
    asymmetry_floor_deg: float | None = None
    scale_disagreement_pct: float | None = None
    leg_disagreement_pct: float | None = None
    shared_thigh: float | None = None
    shared_shin: float | None = None

    def as_dict(self) -> dict[str, Any]:
        def _r(v, nd=1):
            return None if v is None else round(v, nd)
        return {
            "combined": self.combined,
            "reason": self.reason,
            "knee_at_bdc": _r(self.knee_at_bdc),
            "uncertainty_deg": _r(self.uncertainty_deg),
            "per_side": {k: round(v, 1) for k, v in self.per_side.items()},
            "raw_per_side": {k: round(v, 1) for k, v in self.raw_per_side.items()},
            "asymmetry_deg": _r(self.asymmetry_deg),
            "asymmetry_significant": self.asymmetry_significant,
            "asymmetry_floor_deg": _r(self.asymmetry_floor_deg),
            "scale_disagreement_pct": _r(self.scale_disagreement_pct, 2),
            "leg_disagreement_pct": _r(self.leg_disagreement_pct, 2),
        }


def _pct_gap(a: float, b: float) -> float:
    mean = (a + b) / 2.0
    if not (mean > 0):
        return float("inf")
    return abs(a - b) / mean * 100.0


def combine_sides(left: SideGeometry, right: SideGeometry) -> BilateralFit:
    """Merge a left-view and a right-view clip into one bike-fit verdict.

    Refuses unless the two clips share a scale, because everything after that
    point pools numbers from both and pooling across a broken scale is worse
    than not pooling at all.
    """
    if left is None or right is None:
        return BilateralFit(False, reason="missing_side")
    if left.camera_side == right.camera_side:
        return BilateralFit(False, reason="same_side")

    # --- gate: do the two clips agree on the one length the bike fixes? -----
    scale_gap = _pct_gap(left.chord_bdc, right.chord_bdc)
    if scale_gap > _SCALE_TOLERANCE_PCT:
        # Same rider, same unchanged bike: hip-to-ankle at the bottom of the
        # stroke is one number, so a disagreement here means the crank-radius
        # estimate (the shared ruler) is wrong in at least one clip -- filmed
        # from a different distance, or an ankle orbit the chainring corrupted.
        logger.info("BILATERAL_REFUSED", reason="scale", gap_pct=round(scale_gap, 2))
        return BilateralFit(False, reason="scale_mismatch",
                            scale_disagreement_pct=scale_gap)

    leg_gap = _pct_gap(left.leg, right.leg)
    if leg_gap > _LEG_DISAGREEMENT_LIMIT_PCT:
        logger.info("BILATERAL_REFUSED", reason="leg", gap_pct=round(leg_gap, 2))
        return BilateralFit(False, reason="leg_mismatch",
                            scale_disagreement_pct=scale_gap,
                            leg_disagreement_pct=leg_gap)

    # --- one body -----------------------------------------------------------
    thigh = (left.thigh + right.thigh) / 2.0
    shin = (left.shin + right.shin) / 2.0
    chord = (left.chord_bdc + right.chord_bdc) / 2.0

    knee = knee_from_chord(thigh, shin, chord)
    if knee is None:
        return BilateralFit(False, reason="degenerate_geometry")

    per_side: dict[str, float] = {}
    for side in (left, right):
        v = knee_from_chord(thigh, shin, side.chord_bdc)
        if v is not None:
            per_side[side.camera_side] = v
    raw_per_side = {
        side.camera_side: v
        for side in (left, right)
        if (v := knee_from_chord(side.thigh, side.shin, side.chord_bdc)) is not None
    }

    # --- what we are allowed to claim ---------------------------------------
    sens = leg_length_sensitivity_deg(thigh, shin, chord)   # deg per 1%
    if not math.isfinite(sens):
        return BilateralFit(False, reason="degenerate_geometry")
    # Pooling two leg estimates leaves each about half their disagreement away
    # from the pooled value; the chord's own scatter across revolutions adds to
    # it; and the model's absolute wobble (measured by mirroring a clip) is a
    # floor neither averaging nor anything else here can remove.
    from_leg = (leg_gap / 2.0) * sens
    chord_noise_pct = (
        (left.chord_sd + right.chord_sd) / 2.0 / chord * 100.0 if chord > 0 else 0.0
    )
    from_chord = chord_noise_pct * sens
    uncertainty = math.sqrt(from_leg**2 + from_chord**2 + _ABSOLUTE_FLOOR_DEG**2)

    # --- asymmetry, now that the model's share is out of it -----------------
    # With one body imposed, a difference between the sides can only come from
    # the chords, i.e. from how far each leg actually reaches. The floor is the
    # chord scatter -- what the same leg varies by between revolutions.
    asymmetry = None
    significant = None
    floor = None
    if len(per_side) == 2:
        asymmetry = per_side["left"] - per_side["right"]
        floor = max(2.0 * from_chord, 2.0)
        significant = abs(asymmetry) > floor

    logger.info(
        "BILATERAL_COMBINED",
        knee=round(knee, 1), uncertainty=round(uncertainty, 1),
        scale_gap_pct=round(scale_gap, 2), leg_gap_pct=round(leg_gap, 2),
        asymmetry=None if asymmetry is None else round(asymmetry, 1),
        significant=significant,
    )
    return BilateralFit(
        combined=True,
        knee_at_bdc=knee,
        uncertainty_deg=uncertainty,
        per_side=per_side,
        raw_per_side=raw_per_side,
        asymmetry_deg=asymmetry,
        asymmetry_significant=significant,
        asymmetry_floor_deg=floor,
        scale_disagreement_pct=scale_gap,
        leg_disagreement_pct=leg_gap,
        shared_thigh=thigh,
        shared_shin=shin,
    )
