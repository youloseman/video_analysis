"""One stride, five postures, one image -- the kinogram.

Coaches have been building these by hand for years: scrub the clip, find the
five frames that define a stride, screenshot each, paste them into a row. The
ALTIS Kinogram Method (McMillan, ALTIS) named the five and made them a shared
vocabulary. Doing it by hand takes twenty minutes per athlete; the gait phases
this analyzer already detects make it free.

The five positions, in cycle order for the NEAR (camera-facing) leg:

===============  ==========================================================
Touch-down       First frame of a confirmed ground contact.
Full support     The foot passes under the pelvis -- ALTIS plumbs the toe
                 against the ASIS; a hip landmark is the same line and is
                 the one we actually track, so we minimise the horizontal
                 hip-to-ankle offset across the contact.
Toe-off          Last frame before the foot leaves the ground.
Max vertical     The apex of flight. ALTIS identifies it by the two feet
                 being level, which is a proxy for the thing itself; the
                 hip is tracked far more reliably than the far foot, so we
                 take the highest hip of the flight directly.
Strike           The swing leg loaded, hamstring at maximum stretch. ALTIS
                 marks it by the OPPOSITE thigh passing vertical -- which
                 is the far leg, the half of the body a side view tracks
                 worst. We try that definition first and fall back to the
                 near thigh at maximum hip flexion, which by the symmetry
                 of gait lands on the same instant. Which one was used is
                 reported in ``strike_source`` rather than hidden.
===============  ==========================================================

A clip contains many strides. Rather than take the middle one on faith, every
complete cycle is scored on how well its five frames were actually tracked and
the best one wins -- see :func:`_score_cycle`.

When the flight phase cannot be resolved (a coarse sampling stride, a clip
where the gait signal broke down) the three stance positions are still worth
looking at, so a three-tile kinogram is emitted with a warning rather than
nothing at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis import overlay_style
from app.services.video_analysis.biomechanics.angle_calculator import (
    SPORT_LANDMARK_VISIBILITY,
    calculate_forward_sign,
)
from app.services.video_analysis.biomechanics.sport_configs import RUNNING_REFERENCE

logger = structlog.get_logger(__name__)

# Positions in the order they occur in the near leg's cycle. Rendering follows
# this order left to right (or right to left -- see ``RUN_RIGHT_TO_LEFT``).
POSITION_ORDER: tuple[str, ...] = (
    "touch_down", "full_support", "toe_off", "mvp", "strike",
)
STANCE_POSITIONS: tuple[str, ...] = ("touch_down", "full_support", "toe_off")

POSITION_LABELS: dict[str, str] = {
    "touch_down": "TOUCH-DOWN",
    "full_support": "FULL SUPPORT",
    "toe_off": "TOE-OFF",
    "mvp": "MAX VERTICAL",
    "strike": "STRIKE",
}

# Near/far landmark indices per camera side.
_SIDE_LANDMARKS: dict[str, dict[str, int]] = {
    "left":  {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31},
    "right": {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28, "heel": 30, "toe": 32},
}

# Landmarks whose visibility decides whether a chosen frame is worth showing.
# Near-side leg + torso: the parts a running kinogram is read from.
_QUALITY_LANDMARK_KEYS = ("shoulder", "hip", "knee", "ankle", "toe")

# A far-thigh reading is only trusted for the ALTIS Strike definition when the
# far hip and knee both clear this. Below it the near-side fallback is used --
# the far leg is exactly what a side view tracks worst, so a confident-looking
# far-thigh angle on a 0.3-visibility landmark is a number with nothing behind
# it.
_FAR_THIGH_MIN_VISIBILITY = 0.55

# Minimum frames of flight needed before MVP/Strike are attempted. Two frames
# cannot resolve an apex AND a distinct strike position.
_MIN_FLIGHT_FRAMES = 3

# Signals that make a kinogram meaningless, checked before anything is drawn.
#
# Every position here is defined relative to the NEAR leg's ground contact, so
# the picture is only as good as the analyzer's grip on (a) which leg is which
# and (b) when that leg was down. The rest of the report already withholds
# cadence, ground-contact and flight time when those fail -- a kinogram that
# publishes anyway is the one component contradicting the verdict, and it does
# it in the most persuasive form the report has: a photograph with a label on
# it. So it takes the same verdict.
#
# The thresholds are the confidence scorer's, not new ones. "low" is the bar at
# which that module says the model demonstrably could not hold left/right
# identity -- past it, "the near foot landed here" is a coin flip.
_TRUST_KEYS = ("leg_swap_pct", "flip_pct", "side_vote_disagreement_pct")


@dataclass
class KinogramMetric:
    """One value on a tile: a short label, a formatted value, a status colour."""

    label: str
    value: str
    status: str = "muted"   # good | warn | bad | muted


@dataclass
class KinogramPosition:
    """One tile: which analyzed frame, what it is called, what it reads."""

    key: str
    label: str
    analyzed_idx: int
    metrics: list[KinogramMetric] = field(default_factory=list)
    source: str = ""


@dataclass
class KinogramSelection:
    """The chosen cycle and its positions, plus how much to trust them."""

    positions: list[KinogramPosition]
    cycle_index: int
    first_idx: int
    last_idx: int
    camera_side: str
    forward_sign: float
    strike_source: str
    mean_visibility: float
    complete: bool
    warnings: list[str] = field(default_factory=list)

    def to_meta(self) -> dict[str, Any]:
        """JSON-safe description of what was picked, for the result payload."""
        return {
            "positions": [
                {
                    "key": p.key,
                    "label": p.label,
                    "analyzed_idx": p.analyzed_idx,
                    "source": p.source,
                    "metrics": [
                        {"label": m.label, "value": m.value, "status": m.status}
                        for m in p.metrics
                    ],
                }
                for p in self.positions
            ],
            "cycle_index": self.cycle_index,
            "camera_side": self.camera_side,
            "strike_source": self.strike_source,
            "mean_visibility": round(self.mean_visibility, 3),
            "complete": self.complete,
            "warnings": list(self.warnings),
            "method": "ALTIS Kinogram Method (McMillan) -- near-side adaptation",
        }


# ---------------------------------------------------------------------------
# Landmark access helpers. All of these tolerate the NaN coordinates the
# stabilizer writes for gated landmarks, and the missing-key case for an
# analyzer that never stashed its landmarks.
# ---------------------------------------------------------------------------

def _norm_landmarks(analyzer: Any, idx: int) -> Any | None:
    try:
        fr = analyzer.frame_results[idx]
    except (IndexError, TypeError):
        return None
    return (fr.extra_metrics or {}).get("_norm_landmarks")


def _world_landmarks(analyzer: Any, idx: int) -> Any | None:
    try:
        fr = analyzer.frame_results[idx]
    except (IndexError, TypeError):
        return None
    return (fr.extra_metrics or {}).get("_world_landmarks")


def _point(lms: Any, idx: int) -> tuple[float, float, float] | None:
    """``(x, y, visibility)`` for one landmark, or None if unusable."""
    if lms is None:
        return None
    try:
        lm = lms[idx]
    except (IndexError, TypeError, KeyError):
        return None
    x, y = getattr(lm, "x", None), getattr(lm, "y", None)
    if x is None or y is None:
        return None
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isnan(y):
        return None
    vis = getattr(lm, "visibility", 1.0)
    return (x, y, float(vis) if vis is not None else 1.0)


def _frame_visibility(lms: Any, side: str) -> float:
    """Mean visibility of the landmarks a running kinogram is actually read from.

    A landmark the stabilizer gated out (NaN coordinates) counts as 0 rather
    than being skipped -- a frame missing its ankle is a bad tile even if the
    landmarks that survived are crisp.
    """
    idxs = _SIDE_LANDMARKS[side]
    vals: list[float] = []
    for key in _QUALITY_LANDMARK_KEYS:
        pt = _point(lms, idxs[key])
        vals.append(pt[2] if pt else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _thigh_angle_from_vertical(
    lms: Any, hip_idx: int, knee_idx: int, forward: float,
) -> float:
    """Signed thigh angle from vertical: positive = knee ahead of the hip.

    Mirrors :func:`angle_calculator.calculate_signed_segment_to_vertical` but
    reads points already fetched here, and returns NaN rather than raising when
    the direction of travel is unknown.
    """
    if forward == 0.0:
        return float("nan")
    hip = _point(lms, hip_idx)
    knee = _point(lms, knee_idx)
    if hip is None or knee is None:
        return float("nan")
    dx = (knee[0] - hip[0]) * forward
    dy = abs(knee[1] - hip[1])
    if (abs(dx) + dy) < 1e-6:
        return float("nan")
    return math.degrees(math.atan2(dx, dy))


# ---------------------------------------------------------------------------
# Position finders. Each returns an analyzed-frame index, or None when the
# window it was given holds nothing usable.
# ---------------------------------------------------------------------------

def _find_full_support(
    analyzer: Any, side: str, start: int, end: int,
) -> int | None:
    """Frame of the contact where the ankle sits most nearly under the hip.

    Normalized (image-plane) coordinates: this is a "which frame" question
    about a vertical line in the picture, not a metric measurement, so the
    image plane is the right frame of reference and no scale is needed.
    """
    idxs = _SIDE_LANDMARKS[side]
    best_idx, best_dx = None, float("inf")
    for i in range(start, end + 1):
        lms = _norm_landmarks(analyzer, i)
        hip = _point(lms, idxs["hip"])
        ankle = _point(lms, idxs["ankle"])
        if hip is None or ankle is None:
            continue
        dx = abs(ankle[0] - hip[0])
        if dx < best_dx:
            best_idx, best_dx = i, dx
    return best_idx


def _find_mvp(analyzer: Any, start: int, end: int) -> int | None:
    """Frame of the flight where the pelvis is highest (smallest y).

    Uses the mid-hip rather than one side: the pelvis centre is the closest
    single point to the centre of mass that pose estimation offers, and
    averaging the two hips halves the noise of either.
    """
    best_idx, best_y = None, float("inf")
    for i in range(start, end + 1):
        lms = _norm_landmarks(analyzer, i)
        left = _point(lms, 23)
        right = _point(lms, 24)
        pts = [p for p in (left, right) if p is not None]
        if not pts:
            continue
        y = float(np.mean([p[1] for p in pts]))
        if y < best_y:
            best_idx, best_y = i, y
    return best_idx


def _find_strike(
    analyzer: Any, side: str, start: int, end: int, forward: float,
) -> tuple[int | None, str]:
    """Frame where the swing leg is loaded, plus which definition found it.

    Preference order:

    1. ``far_thigh_vertical`` -- the ALTIS definition, the opposite thigh
       passing perpendicular to the ground. Requires the far hip and knee to
       clear ``_FAR_THIGH_MIN_VISIBILITY`` on at least one frame of the window.
    2. ``near_max_hip_flexion`` -- the near thigh at its furthest forward.
       Gait is periodic and near-symmetric, so the instant the far thigh
       passes vertical is the instant the near thigh is at maximum flexion;
       this reads the same moment off the half of the body a side view can
       actually see.
    """
    far_side = "right" if side == "left" else "left"
    far = _SIDE_LANDMARKS[far_side]
    near = _SIDE_LANDMARKS[side]

    best_idx, best_dev = None, float("inf")
    for i in range(start, end + 1):
        lms = _norm_landmarks(analyzer, i)
        hip = _point(lms, far["hip"])
        knee = _point(lms, far["knee"])
        if hip is None or knee is None:
            continue
        if min(hip[2], knee[2]) < _FAR_THIGH_MIN_VISIBILITY:
            continue
        dy = abs(knee[1] - hip[1])
        dx = abs(knee[0] - hip[0])
        if (dx + dy) < 1e-6:
            continue
        dev = abs(math.degrees(math.atan2(dx, dy)))
        if dev < best_dev:
            best_idx, best_dev = i, dev
    if best_idx is not None:
        return best_idx, "far_thigh_vertical"

    best_idx, best_flex = None, -float("inf")
    for i in range(start, end + 1):
        lms = _norm_landmarks(analyzer, i)
        flex = _thigh_angle_from_vertical(lms, near["hip"], near["knee"], forward)
        if math.isnan(flex):
            continue
        if flex > best_flex:
            best_idx, best_flex = i, flex
    return best_idx, "near_max_hip_flexion" if best_idx is not None else "none"


# ---------------------------------------------------------------------------
# Per-tile readings.
# ---------------------------------------------------------------------------

def _fmt_deg(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.0f}°"


def _graded(
    label: str, value: float | None, ref_key: str, *, min_margin: float = 3.0,
    fmt: str = "deg",
) -> KinogramMetric:
    """A reading coloured against a published band from ``RUNNING_REFERENCE``."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return KinogramMetric(label, "--", "muted")
    band = RUNNING_REFERENCE.get(ref_key)
    text = _fmt_deg(value) if fmt == "deg" else f"{value:.2f}"
    if not band:
        return KinogramMetric(label, text, "muted")
    status = overlay_style.status_for(value, band[0], band[1], min_margin=min_margin)
    return KinogramMetric(label, text, status)


def _plain(label: str, value: float | None, fmt: str = "deg") -> KinogramMetric:
    """A reading with no published band -- shown, not graded.

    Colouring a number against a band that does not exist would be inventing
    authority; the athlete still gets the measurement.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return KinogramMetric(label, "--", "muted")
    return KinogramMetric(
        label, _fmt_deg(value) if fmt == "deg" else f"{value:.2f}", "muted",
    )


def _angle_at(analyzer: Any, name: str, idx: int) -> float | None:
    """One filtered angle at one frame, or None when it was never measured."""
    values = analyzer.angle_history.get(name)
    if not values or idx >= len(values):
        return None
    val = values[idx]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def _overstride_at(analyzer: Any, side: str, idx: int) -> float | None:
    """Foot-ahead-of-hip over leg length at one frame.

    Same construction as ``RunningAnalyzer._compute_overstride_ratio`` -- world
    landmarks, leg length via hip->knee->ankle so a bent knee at contact does
    not shrink the denominator -- but for a single frame instead of a median
    over every contact. The tile is describing the frame it shows.
    """
    wl = _world_landmarks(analyzer, idx)
    if wl is None:
        return None
    idxs = _SIDE_LANDMARKS[side]
    vis_thresh = SPORT_LANDMARK_VISIBILITY.get("run", 0.7)
    try:
        hip, knee, ankle = wl[idxs["hip"]], wl[idxs["knee"]], wl[idxs["ankle"]]
    except (IndexError, TypeError):
        return None
    if min(
        getattr(hip, "visibility", 0.0), getattr(ankle, "visibility", 0.0),
    ) < vis_thresh:
        return None
    try:
        leg_len = (
            math.dist((hip.x, hip.y), (knee.x, knee.y))
            + math.dist((knee.x, knee.y), (ankle.x, ankle.y))
        )
        horiz = abs(ankle.x - hip.x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(leg_len) or leg_len < 1e-3 or not math.isfinite(horiz):
        return None
    return horiz / leg_len


def _metrics_for(
    analyzer: Any, key: str, idx: int, side: str, forward: float,
    summary: dict[str, Any] | None = None,
) -> list[KinogramMetric]:
    """The two readings shown on one tile.

    Graded against a published band where one exists in ``RUNNING_REFERENCE``
    (all sourced -- Heiderscheit 2011, Folland 2017), plain otherwise.

    A reading whose whole-clip counterpart the summary REFUSED to publish is
    shown as ``--`` here too. ``compute_summary`` drops trunk lean and
    overstride when they fail their plausibility gates, and a tile printing a
    confident red "TRUNK 75deg" beside a metric table that says the trunk could
    not be measured is the report arguing with itself -- in a form the reader
    has no way to adjudicate.
    """
    s = summary or {}
    knee = _angle_at(analyzer, "knee", idx)
    trunk = _angle_at(analyzer, "trunk", idx) if s.get("trunk_lean") is not None else None
    idxs = _SIDE_LANDMARKS[side]
    lms = _norm_landmarks(analyzer, idx)

    if key == "touch_down":
        overstride = (
            _overstride_at(analyzer, side, idx)
            if s.get("overstride_ratio") is not None else None
        )
        return [
            _graded("KNEE", knee, "knee_at_initial_contact"),
            _graded(
                "OVERSTRIDE", overstride,
                "overstride_ratio", min_margin=0.05, fmt="ratio",
            ),
        ]
    if key == "full_support":
        return [
            _graded("KNEE", knee, "knee_at_midstance"),
            _graded("TRUNK", trunk, "trunk_lean"),
        ]
    if key == "toe_off":
        thigh = _thigh_angle_from_vertical(lms, idxs["hip"], idxs["knee"], forward)
        # Behind vertical at toe-off, so the sign is flipped into a positive
        # "how much hip extension" reading.
        hip_ext = -thigh if not math.isnan(thigh) else float("nan")
        return [
            _plain("HIP EXT", hip_ext),
            _graded("TRUNK", trunk, "trunk_lean"),
        ]
    if key == "mvp":
        return [
            _plain("KNEE", knee),
            _graded("TRUNK", trunk, "trunk_lean"),
        ]
    if key == "strike":
        thigh = _thigh_angle_from_vertical(lms, idxs["hip"], idxs["knee"], forward)
        return [
            _plain("KNEE", knee),
            _plain("THIGH", thigh),
        ]
    return []


def stride_trust_block(
    summary: dict[str, Any] | None,
    tracking_stability: dict[str, Any] | None,
) -> str | None:
    """Why this clip must not get a kinogram, or None if it may have one.

    See ``_TRUST_KEYS`` for the argument. Returns a short machine-readable
    reason so the refusal is debuggable rather than a silent absence.
    """
    from app.services.video_analysis.biomechanics.confidence_scorer import THRESHOLDS

    s = summary or {}
    t = tracking_stability or {}

    # Cadence is the analyzer's own verdict on whether it could time the
    # stride. When it is withheld, ground-contact and flight time are withheld
    # with it -- and those ARE the kinogram's five positions.
    if s.get("cadence_spm") is None:
        return "cadence_unmeasured"

    framing = t.get("framing") or {}
    if framing.get("verdict") == "tiny":
        return "athlete_too_small_in_frame"

    for key in _TRUST_KEYS:
        value = t.get(key)
        limit = THRESHOLDS.get(f"{key}_low")
        if value is None or limit is None:
            continue
        try:
            if float(value) >= float(limit):
                return f"unstable_tracking:{key}"
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Cycle assembly and scoring.
# ---------------------------------------------------------------------------

def _build_cycle(
    analyzer: Any, side: str, forward: float,
    touch_down: int, toe_off: int, next_touch_down: int, cycle_index: int,
    summary: dict[str, Any] | None = None,
) -> KinogramSelection | None:
    """Turn one stance run plus the flight after it into a selection."""
    warnings: list[str] = []
    found: list[tuple[str, int, str]] = []

    full_support = _find_full_support(analyzer, side, touch_down, toe_off)
    if full_support is None:
        return None

    found.append(("touch_down", touch_down, "stance_run_start"))
    found.append(("full_support", full_support, "min_hip_ankle_offset"))
    found.append(("toe_off", toe_off, "stance_run_end"))

    strike_source = "none"
    flight_start, flight_end = toe_off + 1, next_touch_down - 1
    flight_len = flight_end - flight_start + 1

    if flight_len >= _MIN_FLIGHT_FRAMES:
        mvp = _find_mvp(analyzer, flight_start, flight_end)
        if mvp is not None:
            found.append(("mvp", mvp, "highest_mid_hip"))
            # Strike sits after the apex, on the way back down to the ground.
            strike_from = min(mvp + 1, flight_end)
            strike, strike_source = _find_strike(
                analyzer, side, strike_from, flight_end, forward,
            )
            if strike is not None:
                found.append(("strike", strike, strike_source))

    if "mvp" not in {k for k, _, _ in found}:
        warnings.append(
            "The flight phase between contacts was too short to resolve -- "
            "showing the ground-contact positions only. A higher frame rate "
            "(60 fps or more) puts the airborne positions back."
        )
    elif "strike" not in {k for k, _, _ in found}:
        warnings.append(
            "The strike position could not be located in this stride: the far "
            "thigh was not tracked well enough for the ALTIS definition and "
            "the direction of travel was unreadable, so the near-side fallback "
            "had nothing to work from."
        )

    # Drop duplicates: two positions landing on the same frame would print the
    # same picture twice under different names, which reads as a bug.
    seen: set[int] = set()
    positions: list[KinogramPosition] = []
    for key, idx, source in sorted(found, key=lambda t: POSITION_ORDER.index(t[0])):
        if idx in seen:
            warnings.append(
                f"{POSITION_LABELS[key]} fell on the same frame as the previous "
                "position and was dropped."
            )
            continue
        seen.add(idx)
        positions.append(KinogramPosition(
            key=key, label=POSITION_LABELS[key], analyzed_idx=idx,
            metrics=_metrics_for(analyzer, key, idx, side, forward, summary),
            source=source,
        ))

    if len(positions) < len(STANCE_POSITIONS):
        return None

    # Counted AFTER the de-duplication, not before: a cycle that found all five
    # events and then lost one to a frame collision is a four-tile kinogram,
    # and calling it complete is how the caller ends up promising five.
    complete = len(positions) == len(POSITION_ORDER)

    vis = [
        _frame_visibility(_norm_landmarks(analyzer, p.analyzed_idx), side)
        for p in positions
    ]
    return KinogramSelection(
        positions=positions,
        cycle_index=cycle_index,
        first_idx=touch_down,
        last_idx=next_touch_down,
        camera_side=side,
        forward_sign=forward,
        strike_source=strike_source,
        mean_visibility=float(np.mean(vis)) if vis else 0.0,
        complete=complete,
        warnings=warnings,
    )


def _score_cycle(sel: KinogramSelection, total_frames: int) -> float:
    """How good a kinogram this cycle would make. Higher wins.

    Tracking quality dominates: a stride whose five frames were cleanly tracked
    is worth more than one in a nominally better part of the clip. Completeness
    is worth a lot because a five-tile kinogram is the product and a three-tile
    one is the consolation. Centrality is a tiebreak only -- the opening and
    closing strides of a clip are where the athlete is entering or leaving the
    frame, so all else equal the middle of the clip is the safer pick.
    """
    score = sel.mean_visibility * 100.0
    if sel.complete:
        score += 40.0
    if sel.strike_source == "far_thigh_vertical":
        score += 5.0   # the real ALTIS definition, not the substitute
    if total_frames > 0:
        midpoint = (sel.first_idx + sel.last_idx) / 2.0
        centrality = 1.0 - abs(midpoint - total_frames / 2.0) / (total_frames / 2.0)
        score += max(0.0, centrality) * 10.0
    return score


def select_run_kinogram(
    analyzer: Any, *, summary: dict[str, Any] | None = None, min_run: int = 3,
) -> KinogramSelection | None:
    """Pick the best stride in the clip and its kinogram positions.

    ``summary`` is the analyzer's own aggregate. It is used only to suppress
    per-tile readings whose whole-clip value the summary refused to publish --
    see :func:`_metrics_for`.

    Returns None when the clip holds no usable cycle -- too short, gait phases
    never resolved, or landmarks stashed by a different analyzer. Callers treat
    that as "no kinogram", never as an error.
    """
    if getattr(analyzer, "sport_type", None) != "run":
        return None
    if not getattr(analyzer, "frame_results", None):
        return None
    if not hasattr(analyzer, "stance_runs"):
        return None

    runs = analyzer.stance_runs(min_run)
    if len(runs) < 2:
        logger.info("KINOGRAM_SKIP", reason="fewer_than_two_contacts", runs=len(runs))
        return None

    side = getattr(analyzer, "camera_side", None) or "left"
    if side not in _SIDE_LANDMARKS:
        side = "left"

    forward = _median_forward_sign(analyzer)
    total = len(analyzer.frame_results)

    best: KinogramSelection | None = None
    best_score = -float("inf")
    for k in range(len(runs) - 1):
        touch_down, toe_off = runs[k]
        next_touch_down = runs[k + 1][0]
        sel = _build_cycle(
            analyzer, side, forward, touch_down, toe_off, next_touch_down, k,
            summary,
        )
        if sel is None:
            continue
        s = _score_cycle(sel, total)
        if s > best_score:
            best, best_score = sel, s

    if best is None:
        logger.info("KINOGRAM_SKIP", reason="no_usable_cycle", runs=len(runs))
        return None

    if forward == 0.0:
        best.warnings.append(
            "Direction of travel could not be read from the feet -- the "
            "hip-extension and thigh angles are omitted."
        )

    logger.info(
        "KINOGRAM_SELECTED",
        cycle=best.cycle_index,
        positions=len(best.positions),
        complete=best.complete,
        strike_source=best.strike_source,
        visibility=round(best.mean_visibility, 2),
        frames=[p.analyzed_idx for p in best.positions],
    )
    return best


def _median_forward_sign(analyzer: Any) -> float:
    """Direction of travel, voted across the clip.

    ``calculate_forward_sign`` reads one frame; a foot occluded on that frame
    returns 0 and would silently drop every direction-dependent angle. Sampling
    across the clip and taking the majority costs nothing and cannot be undone
    by one bad frame.
    """
    votes: list[float] = []
    n = len(analyzer.frame_results)
    step = max(1, n // 60)
    for i in range(0, n, step):
        lms = _norm_landmarks(analyzer, i)
        if lms is None:
            continue
        sign = calculate_forward_sign(lms)
        if sign != 0.0:
            votes.append(sign)
    if not votes:
        return 0.0
    return 1.0 if sum(votes) > 0 else -1.0


# ===========================================================================
# Rendering: the selection above, composed into one shareable image.
# ===========================================================================

# A tile is portrait because a runner is. 520 px tall keeps the athlete legible
# on a phone at five-across, and keeps a five-tile sheet under ~1900 px wide --
# past that the JPEG stops being something you can paste into a chat.
TILE_H = 520
TILE_ASPECT = 0.70          # tile width / tile height
TILE_GAP = 8
OUTER_PAD = 18
HEADER_H = 58
FOOTER_H = 34

# How much room to leave around the athlete inside a tile. 1.0 would clip the
# skeleton at the crown and the toe; the extra 40% is what makes the athlete
# look like they are running rather than filling a box.
CROP_MARGIN = 1.40

# Canvas colour. Deliberately near-black rather than the chip fill, so the
# frames themselves are the brightest thing in the picture.
_CANVAS_BGR = (20, 17, 14)

# Landmarks that define the athlete's extent in a frame. Head through toes on
# both sides -- the crop must contain the whole body even when only the near
# half gets drawn, or the far arm hangs outside the tile.
_EXTENT_LANDMARKS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)


def _athlete_box(
    lms: Any, width: int, height: int,
) -> tuple[float, float, float, float] | None:
    """``(cx, cy, w, h)`` of the athlete in frame pixels, or None."""
    xs: list[float] = []
    ys: list[float] = []
    for idx in _EXTENT_LANDMARKS:
        pt = _point(lms, idx)
        if pt is None or pt[2] < 0.3:
            continue
        xs.append(pt[0] * width)
        ys.append(pt[1] * height)
    if len(xs) < 4:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def _crop_window(
    box: tuple[float, float, float, float], crop_h: float,
    width: int, height: int,
) -> tuple[int, int, int, int]:
    """A fixed-size window around the athlete, shifted (never shrunk) to fit.

    Every tile uses the SAME crop size so the athlete does not change scale
    from frame to frame -- a kinogram whose tiles are differently zoomed is
    unreadable as a sequence. When the window runs off the edge of the frame it
    is slid back in rather than clipped; when the frame is smaller than the
    window the excess is padded by the caller.
    """
    cx, cy, _, _ = box
    crop_w = crop_h * TILE_ASPECT
    x0 = cx - crop_w / 2.0
    y0 = cy - crop_h / 2.0
    x0 = max(0.0, min(x0, width - crop_w)) if width > crop_w else (width - crop_w) / 2.0
    y0 = max(0.0, min(y0, height - crop_h)) if height > crop_h else (height - crop_h) / 2.0
    return (int(round(x0)), int(round(y0)), int(round(crop_w)), int(round(crop_h)))


def _read_frames(cv2_mod: Any, video_path: str, wanted: list[int]) -> dict[int, Any]:
    """Decode the video once, keeping only the frames asked for.

    Sequential rather than ``CAP_PROP_POS_FRAMES`` seeking on purpose: with
    long-GOP phone footage a seek can land on a neighbouring frame, and a
    kinogram whose skeleton is one frame off the picture it is drawn on looks
    broken in a way a thumbnail never would. Reading stops at the last wanted
    frame, so the cost is a partial decode, not a full one.
    """
    if not wanted:
        return {}
    targets = set(wanted)
    last = max(targets)
    out: dict[int, Any] = {}
    cap = cv2_mod.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return {}
        idx = 0
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in targets:
                out[idx] = frame.copy()
                if len(out) == len(targets):
                    break
            idx += 1
    finally:
        cap.release()
    return out


def render_kinogram(
    video_path: str,
    selection: KinogramSelection,
    frame_data_list: list[dict[str, Any]],
    analyzer: Any,
    *,
    technique_score: int = 0,
    letter_grade: str = "--",
    hide_values: bool = False,
    quality: int = 88,
) -> str | None:
    """Compose the selected positions into one annotated JPEG data URI.

    Returns None on any failure -- a kinogram is a bonus artifact and must
    never take an analysis down with it.
    """
    import base64

    import cv2

    from app.services.video_analysis.pipeline import (
        build_skeleton_geometry,
        landmarks_to_pixels,
    )

    positions = selection.positions
    if not positions:
        return None

    # analyzed index -> index into the ORIGINAL video
    video_idx: dict[int, int] = {}
    for p in positions:
        if p.analyzed_idx >= len(frame_data_list):
            logger.warning("KINOGRAM_INDEX_OOB", idx=p.analyzed_idx)
            return None
        video_idx[p.analyzed_idx] = frame_data_list[p.analyzed_idx]["frame_idx"]

    frames = _read_frames(cv2, video_path, sorted(set(video_idx.values())))
    if len(frames) < len(positions):
        logger.warning(
            "KINOGRAM_FRAMES_MISSING",
            wanted=len(positions), got=len(frames),
        )
        return None

    # One crop size for every tile, from the tallest athlete among them.
    boxes: dict[int, tuple[float, float, float, float]] = {}
    for p in positions:
        frame = frames[video_idx[p.analyzed_idx]]
        h, w = frame.shape[:2]
        box = _athlete_box(_norm_landmarks(analyzer, p.analyzed_idx), w, h)
        if box is None:
            logger.warning("KINOGRAM_NO_BOX", idx=p.analyzed_idx)
            return None
        boxes[p.analyzed_idx] = box
    crop_h = max(b[3] for b in boxes.values()) * CROP_MARGIN
    if crop_h <= 1:
        return None

    tile_w = int(round(TILE_H * TILE_ASPECT))
    n = len(positions)
    canvas_w = OUTER_PAD * 2 + tile_w * n + TILE_GAP * (n - 1)
    canvas_h = OUTER_PAD * 2 + HEADER_H + TILE_H + FOOTER_H
    canvas = np.full((canvas_h, canvas_w, 3), _CANVAS_BGR, dtype=np.uint8)

    tile_tops = OUTER_PAD + HEADER_H
    tile_lefts: list[int] = []

    for k, p in enumerate(positions):
        frame = frames[video_idx[p.analyzed_idx]]
        fh, fw = frame.shape[:2]
        x0, y0, cw, ch = _crop_window(boxes[p.analyzed_idx], crop_h, fw, fh)

        # Cut the window, padding whatever falls outside the frame so every
        # tile is exactly the same shape regardless of where in the picture
        # the athlete was.
        tile_src = np.full((ch, cw, 3), _CANVAS_BGR, dtype=np.uint8)
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(fw, x0 + cw), min(fh, y0 + ch)
        if sx1 > sx0 and sy1 > sy0:
            tile_src[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame[sy0:sy1, sx0:sx1]

        scale = TILE_H / float(ch)
        tile = cv2.resize(
            tile_src, (tile_w, TILE_H),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4,
        )

        # Skeleton, in tile coordinates. Same builder the overlay video and the
        # keyframe use, so all three draw the same body.
        lms = _norm_landmarks(analyzer, p.analyzed_idx)
        if lms is not None:
            pix = landmarks_to_pixels(
                lms, fw, fh, offset=(x0, y0), scale=scale,
            )
            segments, dots = build_skeleton_geometry(pix, "run", selection.camera_side)
            overlay_style.draw_glow_skeleton(
                cv2, tile, segments, dots, glow=True, line_w=3, dot_r=4,
            )

        left = OUTER_PAD + k * (tile_w + TILE_GAP)
        tile_lefts.append(left)
        canvas[tile_tops:tile_tops + TILE_H, left:left + tile_w] = tile
        cv2.rectangle(
            canvas, (left, tile_tops), (left + tile_w - 1, tile_tops + TILE_H - 1),
            (70, 62, 54), 1,
        )

    # -- text layer: one PIL pass for the header, titles, chips and brand --
    chips = overlay_style.ChipLayer(canvas)
    status = "good" if technique_score >= 75 else "warn" if technique_score >= 60 else "bad"
    chips.header(
        (OUTER_PAD, OUTER_PAD), "RUN", f"{technique_score}/100", letter_grade, status,
        right_text="KINOGRAM · ONE STRIDE", frame_w=canvas_w, scale=1.15,
    )

    for k, p in enumerate(positions):
        left = tile_lefts[k]
        chips.caption(
            (left + 8, tile_tops + 8), f"{k + 1} · {p.label}", scale=1.0,
        )
        # Readings stack up from the bottom of the tile.
        y = tile_tops + TILE_H - 26
        for m in reversed(p.metrics):
            value, m_status = ("LOCKED", "muted") if hide_values else (m.value, m.status)
            chips.metric_chip(
                (left + 8, y), m.label, value, m_status, scale=0.82,
            )
            y -= 34

    chips.caption(
        (OUTER_PAD, canvas_h - OUTER_PAD - FOOTER_H + 6),
        "POSITIONS AFTER THE ALTIS KINOGRAM METHOD",
        scale=0.85, status="muted", plate=False,
    )
    chips.brand(
        (canvas_w - OUTER_PAD, canvas_h - OUTER_PAD), "FLAPP",
        "FREE" if hide_values else "", scale=1.1,
    )
    canvas = chips.flush()

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    logger.info(
        "KINOGRAM_RENDERED",
        tiles=n, width=canvas_w, height=canvas_h, kb=round(len(buf) / 1024),
    )
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def build_run_kinogram(
    video_path: str,
    analyzer: Any,
    frame_data_list: list[dict[str, Any]],
    *,
    summary: dict[str, Any] | None = None,
    tracking_stability: dict[str, Any] | None = None,
    technique_score: int = 0,
    letter_grade: str = "--",
    hide_values: bool = False,
) -> tuple[str | None, dict[str, Any] | None]:
    """Select and render in one call: ``(data_uri, meta)``.

    Either half may be None. A clip whose stride timing the analyzer does not
    trust gives ``(None, {"refused": reason})``; no usable stride gives
    ``(None, None)``; a stride that was found but could not be drawn gives
    ``(None, meta)`` so the result can still say what was picked and why the
    picture is missing.
    """
    blocked = stride_trust_block(summary, tracking_stability)
    if blocked:
        logger.info("KINOGRAM_REFUSED", reason=blocked)
        return None, {"refused": blocked}

    selection = select_run_kinogram(analyzer, summary=summary)
    if selection is None:
        return None, None
    meta = selection.to_meta()
    try:
        uri = render_kinogram(
            video_path, selection, frame_data_list, analyzer,
            technique_score=technique_score, letter_grade=letter_grade,
            hide_values=hide_values,
        )
    except Exception as e:  # noqa: BLE001 -- a bonus artifact, never fatal
        logger.warning("KINOGRAM_RENDER_FAILED", err=str(e))
        return None, meta
    return uri, meta
