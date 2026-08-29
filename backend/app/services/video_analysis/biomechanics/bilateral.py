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

WHAT IT CANNOT DO -- and this cost a rewrite to learn. Pooling two views
removes the disagreement between them; it does not remove a bias they share,
and near full extension the knee angle amplifies a 1% leg-length error into
about 4 deg. So the combined value ships with an uncertainty, never bare.

AND IT DOES NOT REPORT AN ASYMMETRY. The first version did, from the residual
difference between the two sides after imposing one body, and on Artur's pair
that residual was a tidy 0.9 deg. It was an artifact. Pooling needs the ratio
of the two clips' image scales, and measured across four real pairs the gap
between the sides always came out as the clips' disagreement about the
hip-ankle chord times the sensitivity constant -- ratio 4.1, 4.6, 3.6, 4.3,
5.4. In other words the per-side split is the scale choice restated, not a
second piece of evidence, and choosing the scale that makes the bike look
unchanged (which is the right thing to do) FORCES the legs to look equal.
Left-right difference and scale error are degenerate here; neither this
method nor any amount of care inside it can separate them, so the session
reports each clip's own reading and says plainly that the difference between
them is the instrument.

WHAT IT DOES NOT REFUSE. The first version declined whenever the crank ruler
disagreed by more than 2%, which rejected two of the three pairs ever filmed
in the wild. That was overcautious: the COMBINED angle barely moves with the
anchor (149.9 vs 150.0 on one pair, 149.9 vs 149.8 on another), because
scaling a whole triangle does not change its shape. Only the split it
refused to protect was anchor-dependent, and the split is gone. What is left
is a sanity check for input that is not a pair at all.
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

# Pooling bone lengths across two clips needs the ratio of their image
# scales, and there is more than one physical thing that is the same in both:
#
#   crank  -- the ankle orbit's radius. The bike's own ruler, but the
#             chainring sits on that orbit and can corrupt the fit of it.
#   torso  -- shoulder to hip. A body length, unaffected by the drivetrain,
#             but it leans on the hip landmark, the worst one we have.
#
# Neither is reliable alone: measured across four real pairs, the crank was
# the better ruler on one and off by 14-25% on the others. So both are
# computed and the one under which the two clips AGREE about the hip-ankle
# chord is used -- the chord is fixed by the fit, so a scale that makes the
# bike appear to change between two clips of one session is the wrong scale.
#
# An earlier version REFUSED whenever the crank ruler disagreed by more than
# 2%. That rejected two of the three real pairs ever filmed. It was also
# unnecessary: the combined knee angle turns out to be almost independent of
# which anchor is used (149.9 vs 150.0 on one pair, 149.9 vs 149.8 on
# another), because scaling one clip's whole triangle does not change its
# shape. What the anchor DOES decide is the per-side split -- see the note on
# `combine_sides` about why that is no longer published.
_SCALE_SANITY_PCT = 35.0
# Two estimates of one rider's leg differing by more than this, AFTER the
# best scale has been chosen, are not two views of the same person having a
# bad day -- something is wrong (wrong clip paired, different rider, tracking
# failure).
_LEG_DISAGREEMENT_LIMIT_PCT = 12.0

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

    # Lengths back in the clip's own image units. Needed to form a scale ratio
    # from an anchor OTHER than the crank -- everything above is already
    # divided by it, so the crank cannot be cross-checked in its own units.
    @property
    def thigh_px(self) -> float:
        return self.thigh * self.crank_radius_px

    @property
    def shin_px(self) -> float:
        return self.shin * self.crank_radius_px

    @property
    def chord_px(self) -> float:
        return self.chord_bdc * self.crank_radius_px

    @property
    def torso_px(self) -> float:
        return self.torso * self.crank_radius_px

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SideGeometry | None:
        """Rebuild from what a stored result carries. None when it can't be."""
        if not isinstance(d, dict):
            return None
        try:
            return cls(
                camera_side=str(d["camera_side"]),
                thigh=float(d["thigh"]), shin=float(d["shin"]),
                torso=float(d.get("torso") or float("nan")),
                chord_bdc=float(d["chord_bdc"]),
                chord_sd=float(d.get("chord_sd") or 0.0),
                revolutions=int(d.get("revolutions") or 0),
                measured_frames=int(d.get("measured_frames") or 0),
                crank_radius_px=float(d.get("crank_radius_px") or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            return None

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
    # What each clip measured ON ITS OWN. A fact about a clip, not about a
    # leg -- there is no per-side reconciled value any more, because it was
    # the scale choice wearing a measurement's clothes.
    raw_per_side: dict[str, float] = field(default_factory=dict)
    # Which invariant set the scale, and how far the alternative was from it.
    scale_anchor: str | None = None
    scale_chord_disagreement_pct: float | None = None
    scale_anchor_spread_deg: float | None = None
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
            "raw_per_side": {k: round(v, 1) for k, v in self.raw_per_side.items()},
            "scale_anchor": self.scale_anchor,
            "scale_chord_disagreement_pct": _r(self.scale_chord_disagreement_pct, 2),
            "scale_anchor_spread_deg": _r(self.scale_anchor_spread_deg, 1),
            "leg_disagreement_pct": _r(self.leg_disagreement_pct, 2),
        }


# Metrics that describe the rider as a whole rather than one leg, so two
# clips are two readings of ONE quantity and averaging them is the point. The
# trunk especially: it is a midline structure both cameras see, which is why
# it doubles as this session's error gauge.
_AVERAGED_METRICS = (
    "knee_at_tdc",
    "trunk_angle_avg",
    "hip_angle_avg",
    "elbow_angle_avg",
    "shoulder_angle_avg",
    "head_alignment_avg",
    "forearm_tilt_avg",
    "pelvic_ratio",
)
# How far apart two readings of a midline metric may sit before the pair
# stops being two views of one ride. Chosen from the same measurements as the
# compare table's floors: the 26 Aug pair agreed on the trunk to ~1 deg.
_MIDLINE_AGREEMENT_LIMIT_DEG = 8.0


def merge_summaries(
    summary_left: dict[str, Any],
    summary_right: dict[str, Any],
    fit: BilateralFit,
    cycling_position: str | None = None,
) -> dict[str, Any]:
    """One set of metrics for the rider, from two one-legged clips.

    Built on the side with more revolutions behind it so every key the scorer
    and the plan builder expect is present and self-consistent, then the
    quantities two clips genuinely measure twice are replaced by their merged
    values. Knee-at-bottom comes from ``combine_sides`` rather than an average
    -- that is the metric the shared body exists to fix.

    The per-side ``{side}_knee_at_bdc`` keys are overwritten too, deliberately:
    ``score_cycling`` reads those FIRST and would otherwise score the merged
    ride on one clip's unmerged leg, which is the contradiction this whole
    feature exists to end.
    """
    if not fit.combined:
        raise ValueError("merge_summaries needs a combined fit")
    base_side = summary_left if _rev_count(summary_left) >= _rev_count(summary_right) \
        else summary_right
    other = summary_right if base_side is summary_left else summary_left
    merged = dict(base_side)

    for key in _AVERAGED_METRICS:
        a, b = base_side.get(key), other.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            merged[key] = (float(a) + float(b)) / 2.0
        elif isinstance(b, (int, float)) and not isinstance(a, (int, float)):
            merged[key] = float(b)

    knee = fit.knee_at_bdc
    merged["knee_at_bdc"] = knee
    # Both per-side keys get the SAME merged value. Not laziness: the method
    # cannot separate the two legs from its own scale error, so publishing a
    # different number per side -- which ``score_cycling`` reads first -- would
    # be inventing the distinction back.
    for side in ("left", "right"):
        merged[f"{side}_knee_at_bdc"] = knee

    # The saddle verdict is derived from the knee, so it has to be re-derived
    # from the merged one -- carrying the base clip's verdict forward would
    # reintroduce exactly the disagreement we just resolved.
    merged["saddle_height_assessment"] = _assess_saddle(knee, cycling_position)

    merged["bilateral"] = fit.as_dict()
    merged["camera_side"] = "both"
    merged["camera_side_label"] = "Both sides"
    merged["frames_analyzed"] = (
        int(base_side.get("frames_analyzed") or 0)
        + int(other.get("frames_analyzed") or 0)
    )
    return merged


def _rev_count(summary: dict[str, Any]) -> int:
    geom = summary.get("bilateral_geometry") or {}
    return int(geom.get("revolutions") or 0)


def _assess_saddle(knee: float | None, cycling_position: str | None) -> str:
    """Saddle verdict for a merged knee angle, on the position's own band."""
    if knee is None:
        return "insufficient_data"
    from app.services.video_analysis.biomechanics.cycling_positions import (
        get_cycling_reference,
    )
    opt_min, opt_max = get_cycling_reference(cycling_position)["knee_at_bdc"]
    if knee < opt_min - 5:
        return "too_low"
    if knee > opt_max + 5:
        return "too_high"
    if opt_min <= knee <= opt_max:
        return "optimal"
    return "acceptable"


def midline_agreement(
    summary_left: dict[str, Any], summary_right: dict[str, Any],
) -> dict[str, Any]:
    """How far apart the two clips are on things that cannot differ by side.

    The trunk, the hip, the shoulder: one body seen from two sides. Whatever
    gap shows up here was produced by the method, not the rider, so it is the
    honest confidence figure for everything else in the session -- measured on
    this rider, on this day, instead of assumed from a constant.
    """
    gaps: dict[str, float] = {}
    for key in ("trunk_angle_avg", "hip_angle_avg", "shoulder_angle_avg",
                "elbow_angle_avg"):
        a, b = summary_left.get(key), summary_right.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            gaps[key] = abs(float(a) - float(b))
    if not gaps:
        return {"agree": None, "gaps": {}, "worst": None, "worst_metric": None}
    worst_metric = max(gaps, key=lambda k: gaps[k])
    worst = gaps[worst_metric]
    return {
        "agree": worst <= _MIDLINE_AGREEMENT_LIMIT_DEG,
        "gaps": {k: round(v, 1) for k, v in gaps.items()},
        "worst": round(worst, 1),
        "worst_metric": worst_metric,
        "limit_deg": _MIDLINE_AGREEMENT_LIMIT_DEG,
    }


def _pct_gap(a: float, b: float) -> float:
    mean = (a + b) / 2.0
    if not (mean > 0):
        return float("inf")
    return abs(a - b) / mean * 100.0


def _scale_candidates(a: SideGeometry, b: SideGeometry) -> list[tuple[str, float]]:
    """Ratios of b's image scale to a's, one per invariant both clips share."""
    out: list[tuple[str, float]] = []
    if a.crank_radius_px > 0 and b.crank_radius_px > 0:
        out.append(("crank", a.crank_radius_px / b.crank_radius_px))
    if math.isfinite(a.torso_px) and math.isfinite(b.torso_px) and b.torso_px > 0:
        out.append(("torso", a.torso_px / b.torso_px))
    return out


def combine_sides(left: SideGeometry, right: SideGeometry) -> BilateralFit:
    """Merge a left-view and a right-view clip into one bike-fit verdict.

    Pooling bone lengths needs the ratio of the two clips' image scales. Each
    shared invariant offers one -- see the module note -- and the chosen one is
    whichever makes the two clips agree about the hip-ankle chord, since that
    length is fixed by the bike and a scale that makes the bike appear to have
    changed mid-session is the wrong scale.

    The disagreement between the anchors is not discarded: the combined angle
    is computed under BOTH, and how far apart those answers land goes into the
    uncertainty. That is the honest cost of not knowing the scale exactly, and
    it replaces the old behaviour of refusing outright.
    """
    if left is None or right is None:
        return BilateralFit(False, reason="missing_side")
    if left.camera_side == right.camera_side:
        return BilateralFit(False, reason="same_side")

    candidates = _scale_candidates(left, right)
    if not candidates:
        return BilateralFit(False, reason="no_shared_scale")

    # Score each candidate on the one thing the bike guarantees: with the right
    # clip scaled into the left's units, both must report the same hip-ankle
    # chord.
    scored = []
    for name, k in candidates:
        if not (math.isfinite(k) and k > 0):
            continue
        gap = abs(left.chord_px - right.chord_px * k) / max(left.chord_px, 1e-9) * 100.0
        scored.append((gap, name, k))
    if not scored:
        return BilateralFit(False, reason="no_shared_scale")
    scored.sort()
    chord_gap, anchor, k = scored[0]

    def _merged(scale: float):
        """(knee, thigh, shin, chord) with the right clip scaled into the left's."""
        t = (left.thigh_px + right.thigh_px * scale) / 2.0
        sh = (left.shin_px + right.shin_px * scale) / 2.0
        ch = (left.chord_px + right.chord_px * scale) / 2.0
        return knee_from_chord(t, sh, ch), t, sh, ch

    knee, thigh, shin, chord = _merged(k)
    if knee is None:
        return BilateralFit(False, reason="degenerate_geometry")

    # A pair that cannot be made to agree on the rider's own leg is not two
    # views of one session -- wrong clip paired, or tracking that failed.
    leg_gap = _pct_gap(left.thigh_px + left.shin_px,
                       (right.thigh_px + right.shin_px) * k)
    if leg_gap > _LEG_DISAGREEMENT_LIMIT_PCT or chord_gap > _SCALE_SANITY_PCT:
        logger.info("BILATERAL_REFUSED", reason="not_one_session",
                    leg_gap_pct=round(leg_gap, 2), chord_gap_pct=round(chord_gap, 2))
        return BilateralFit(False, reason="leg_mismatch",
                            scale_anchor=anchor,
                            scale_chord_disagreement_pct=chord_gap,
                            leg_disagreement_pct=leg_gap)

    # How far the answer would have moved on the other ruler. Small in practice
    # -- scaling a whole triangle barely changes its shape -- but it is a real
    # unknown, and it belongs in the error bar rather than in a footnote.
    others = [v for nm, kk in candidates if nm != anchor
              and (v := _merged(kk)[0]) is not None]
    anchor_spread = max((abs(v - knee) for v in others), default=0.0)

    raw_per_side = {
        side.camera_side: v
        for side in (left, right)
        if (v := knee_from_chord(side.thigh, side.shin, side.chord_bdc)) is not None
    }

    sens = leg_length_sensitivity_deg(thigh, shin, chord)   # deg per 1%
    if not math.isfinite(sens):
        return BilateralFit(False, reason="degenerate_geometry")
    # Pooling two leg estimates leaves each about half their disagreement away
    # from the pooled value; the chord's own scatter across revolutions adds to
    # it; the choice of ruler adds its own; and the model's absolute wobble
    # (measured by mirroring a clip) is a floor none of this can remove.
    from_leg = (leg_gap / 2.0) * sens
    sd_px = (left.chord_sd * left.crank_radius_px
             + right.chord_sd * right.crank_radius_px * k) / 2.0
    chord_noise_pct = (sd_px / chord * 100.0) if chord > 0 else 0.0
    from_chord = chord_noise_pct * sens
    uncertainty = math.sqrt(
        from_leg**2 + from_chord**2 + anchor_spread**2 + _ABSOLUTE_FLOOR_DEG**2)

    logger.info(
        "BILATERAL_COMBINED",
        knee=round(knee, 1), uncertainty=round(uncertainty, 1),
        anchor=anchor, chord_gap_pct=round(chord_gap, 2),
        anchor_spread=round(anchor_spread, 1), leg_gap_pct=round(leg_gap, 2),
    )
    return BilateralFit(
        combined=True,
        knee_at_bdc=knee,
        uncertainty_deg=uncertainty,
        raw_per_side=raw_per_side,
        scale_anchor=anchor,
        scale_chord_disagreement_pct=chord_gap,
        scale_anchor_spread_deg=anchor_spread,
        leg_disagreement_pct=leg_gap,
        shared_thigh=thigh,
        shared_shin=shin,
    )
