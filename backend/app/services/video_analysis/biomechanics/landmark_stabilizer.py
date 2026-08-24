"""Post-processing stabilizer for MediaPipe landmarks.

Three passes over raw frame results:
1. Visibility gating -- low-confidence landmarks are replaced with NaN so
   downstream consumers (angle calculator, Butterworth) skip them instead
   of smoothing noise into the signal. Threshold is keyed by
   (sport_type, camera_angle) so swim above/under water get different floors.
2. Anti-flip correction -- detects when MediaPipe swaps left/right sides
   and swaps them back using Z-depth consistency of hips.
3. Landmark smoothing on (x, y, z) per landmark to reduce jitter. Routed
   per sport (see ``_use_butterworth_landmarks``): offline side-view sports
   (bike, run) and swim under-water use a zero-phase Butterworth (no lag);
   the rest use the causal One Euro filter with velocity-adaptive cutoff.

Applied BETWEEN MediaPipe detection and sport-specific analysis.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import structlog

from app.services.video_analysis.biomechanics.one_euro import OneEuro

logger = structlog.get_logger(__name__)

# All left/right landmark pairs to swap when a flip is detected
_SWAP_PAIRS = [
    (11, 12),  # shoulders
    (13, 14),  # elbows
    (15, 16),  # wrists
    (17, 18),  # pinkies
    (19, 20),  # index fingers
    (21, 22),  # thumbs
    (23, 24),  # hips
    (25, 26),  # knees
    (27, 28),  # ankles
    (29, 30),  # heels
    (31, 32),  # foot index
    (1, 4),    # eye inner
    (2, 5),    # eye
    (3, 6),    # eye outer
    (7, 8),    # ears
    (9, 10),   # mouth
]


# Leg landmark pairs handled by the leg-level anti-swap pass (legs only --
# whole-body flips are _fix_flips' job).
_LEG_SWAP_PAIRS = [(25, 26), (27, 28), (29, 30), (31, 32)]

# _fix_leg_swaps tuning (normalized image units).
LEG_SWAP_MARGIN = 0.25          # swapped match must beat identity by 25%
LEG_COLLAPSE_DIST = 0.02        # ankles closer than this = legs collapsed onto one
LEG_MIN_DECIDABLE_DIST = 0.012  # skip the decision when predicted legs coincide
LEG_MAX_VELOCITY = 0.12         # per-frame velocity clamp (kills post-gap spikes)
# Re-seed (accept MediaPipe's labels, don't decide) after a tracking gap
# longer than this. Past ~1/3 of a leg cycle (~240 ms at 165 spm) the legs
# may have physically exchanged places, so a continuity decision against
# stale predictions is a coin flip -- and a wrong one self-perpetuates.
LEG_GAP_RESEED_FRAMES = 3       # consecutive unmeasurable frames
LEG_GAP_RESEED_MS = 200.0       # wall-clock gap between measured frames

# --- _gate_leg_identity_breaks tuning (bike) ------------------------------
#
# Thresholds are fractions of the distance between the two ankles, because on a
# trainer that distance IS a physical length: two pedals half a revolution
# apart, i.e. the crank diameter (0.098 in normalized units on IMG_9981).
#
# The bar is set from the measured distribution of how far an ankle lands from
# its own predicted position, not from a guess. On IMG_9981, in fractions of
# that separation:
#
#     near (right) ankle : p50 0.06   p90 0.31   p95 0.66   p98 0.94
#     far  (left)  ankle : p50 0.22   p90 0.86   p95 1.21   p98 2.07
#
# The near leg's own tail IS the breaks, so a bar just under its p98 blanks
# about the rate of real identity events and leaves ordinary pedalling alone. A
# hop to the other foot displaces the ankle by roughly one separation, so this
# also sits just below the thing it is meant to catch. The far leg loses far
# more frames at the same bar, which is the honest reading of a landmark whose
# median deviation is 3.5x the near leg's -- and it costs nothing, because the
# far side is neither drawn nor measured on a side view.
LEG_BREAK_FRAC = 0.85
LEG_BREAK_MIN_SEPARATION = 0.02  # below this the two feet are one blob: give up
# The bar does NOT widen while a leg is held, and that is deliberate. The first
# attempt widened it, reasoning that an aging prediction deserves more slack --
# but a foot sitting on the WRONG pedal is wrong by about one separation every
# frame, so any slack past ~1.2x lets it straight back in, which is the one
# thing this must not do. The cascade that motivated the widening had a
# different cause: the velocity was being DECAYED while held, which aimed the
# prediction at a foot standing still. Carrying the velocity instead (a
# pedalling foot keeps going round at much the same rate) fixes the cascade
# without opening the door.
#
# A leg cannot stay suspect forever. After this many consecutive blanked frames
# MediaPipe's current labels are accepted and the track restarts. Re-seeds are
# COUNTED and reported: a clip with several is one where identity was lost for
# long stretches, which is a different (worse) situation from a few blinks.
#
# The number is set by how long the PREDICTOR stays honest, not by how long an
# excursion lasts. Carrying a velocity forward is a straight line through what
# is really a circle: at ~40 frames a revolution, five frames is 45 degrees of
# crank and a chord-vs-arc error still well inside the bar, while forty frames
# is most of a revolution and the prediction is meaningless. Swept on IMG_9981
# (bar held at 0.85), which shows exactly that -- patience buys nothing and
# then destroys the measurement it was meant to protect:
#
#     patience  blanked   BDC variability   saddle verdict
#            3     3.8%              2.8    acceptable
#            5     5.0%              2.7    acceptable
#            8     6.8%              3.4    acceptable
#           12     9.1%              5.5    optimal      <- verdict flips
#           20    13.9%             14.4    optimal
#           40    25.6%             21.2    optimal      <- 9 strokes, not 10
#
# The re-seed count stayed at 3 throughout, which is the tell: those excursions
# are not resolving on their own even at 40 frames, so the extra patience is
# spent blanking good frames rather than covering a longer break. Raising this
# is not the way to handle a long excursion -- re-acquiring the foot by the
# geometry (the two ankles straddle the bottom bracket) is, and that needs a
# validation set first.
LEG_BREAK_RESEED_FRAMES = 5


# Visibility gate -- below this, landmark coordinates become NaN.
# Swim above-water is strict: glare/splash makes low-conf = hallucination.
# Swim under-water is lenient: distortion depresses confidence but points
# are usually valid.
MIN_VISIBILITY: dict[tuple[str, str | None], float] = {
    ("run",  None):          0.3,
    ("bike", None):          0.4,
    ("swim", "above_water"): 0.6,
    ("swim", "under_water"): 0.3,
    ("swim", None):          0.5,  # fallback
}


# One Euro params per (sport, camera_angle).
# Above-water: low min_cutoff + low beta -> aggressive smoothing, tolerates
# lag to kill splash jitter.
# Under-water: high min_cutoff + higher beta -> responsive, captures fast
# catch kinematics.
ONE_EURO_PARAMS: dict[tuple[str, str | None], dict[str, float]] = {
    ("run",  None):          dict(min_cutoff=1.7, beta=0.10, d_cutoff=1.0),
    ("bike", None):          dict(min_cutoff=1.5, beta=0.05, d_cutoff=1.0),
    ("swim", "above_water"): dict(min_cutoff=0.6, beta=0.01, d_cutoff=1.0),
    ("swim", "under_water"): dict(min_cutoff=3.0, beta=0.70, d_cutoff=1.0),
    ("swim", None):          dict(min_cutoff=1.0, beta=0.05, d_cutoff=1.0),
}


def _lookup(
    table: dict[tuple[str, str | None], Any],
    sport: str,
    camera_angle: str | None,
    default: Any,
) -> Any:
    if (sport, camera_angle) in table:
        return table[(sport, camera_angle)]
    if (sport, None) in table:
        return table[(sport, None)]
    return default


def _to_mutable(landmark: Any) -> SimpleNamespace:
    """Convert a MediaPipe landmark (possibly protobuf) to a mutable object."""
    return SimpleNamespace(
        x=landmark.x,
        y=landmark.y,
        z=landmark.z,
        visibility=getattr(landmark, "visibility", 1.0),
    )


def _ensure_mutable(frame_results: list[dict[str, Any]]) -> None:
    """Convert all landmarks in frame_results to mutable SimpleNamespace objects.

    MediaPipe Tasks API returns protobuf objects that don't support setattr.
    This converts them once so all downstream code can freely mutate coordinates.
    """
    for frame in frame_results:
        for key in ("world_landmarks", "normalized_landmarks"):
            landmarks = frame[key]
            if landmarks and isinstance(landmarks[0], SimpleNamespace):
                continue
            frame[key] = [_to_mutable(lm) for lm in landmarks]


def _use_butterworth_landmarks(
    sport_type: str, camera_angle: str | None, camera_view: str | None,
) -> bool:
    """Decide which landmark smoother applies for this clip.

    Butterworth (zero-phase, post-hoc) is chosen when the whole signal
    is available up front and skeleton stability beats real-time
    responsiveness. One Euro (causal, adaptive) is the default elsewhere.

    Cases routed to Butterworth:
      - swim under-water: One Euro lags the fast catch transient.
      - bike side-view: visualizer landmark jitter — need landmark-level
        smoothing, not just the angle Butterworth that runs later.
        Excludes bike rear-view, which has its own 1.2 Hz Butterworth
        inside ``PelvicStabilityAnalyzer`` and would over-smooth if
        filtered twice.
      - run side-view: One Euro's causal phase lag left the skeleton
        visibly trailing the athlete on real-speed 60 fps clips (the lag
        is ~constant in ms, so at real-time limb speeds it spans several
        frames; slo-mo hid it). Zero-phase removes the lag by
        construction.
    """
    if sport_type == "swim" and camera_angle == "under_water":
        return True
    if sport_type == "bike" and camera_view != "rear":
        return True
    if sport_type == "run" and camera_view != "rear":
        return True
    return False


def stabilize_landmarks(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None = None,
    fps: float = 30.0,
    context: dict[str, Any] | None = None,
    camera_view: str | None = None,
) -> list[dict[str, Any]]:
    """Stabilize landmark sequence: gate low-visibility, fix flips, smooth.

    Args:
        frame_results: List of frame dicts with 'world_landmarks' and
            'normalized_landmarks' keys (from pipeline _iterate_video_frames).
        sport_type: "run" | "bike" | "swim".
        camera_angle: For swim, "above_water" or "under_water". None for other sports.
        fps: Video frame rate, used by the One Euro / Butterworth filter.
        context: Optional mutable dict — when provided, the function
            populates ``context["butterworth_meta"]`` with cutoff
            diagnostics so the caller (pipeline.py) can surface warnings
            without changing this function's return type.
        camera_view: For bike/run, "side" or "rear". Used to keep bike
            rear-view on One Euro (PelvicStabilityAnalyzer applies its
            own 1.2 Hz Butterworth). None for swim and for legacy callers.

    Returns:
        The same list, mutated in place.
    """
    if len(frame_results) < 3:
        return frame_results

    _ensure_mutable(frame_results)
    dropped = _gate_by_visibility(frame_results, sport_type, camera_angle)
    flips = _fix_flips(frame_results, sport_type)

    # Leg-level anti-swap (run only) MUST run before the smoothing pass:
    # a zero-phase filter applied across identity swaps blends the two
    # legs' series into each other, and no later correction can undo that.
    leg_swaps, leg_swap_pct, leg_collapse_pct = (0, None, None)
    leg_identity_diag: dict[str, Any] | None = None
    if sport_type == "run":
        # Whole-clip identity resolution (see leg_identity.py for why the
        # greedy pass it replaces could not be tuned into correctness). The
        # greedy corrector remains as the fallback: losing identity repair to
        # an exception in the new path would be strictly worse than the old
        # behaviour.
        try:
            from app.services.video_analysis.biomechanics.leg_identity import (
                resolve_run_leg_identity,
            )

            leg_swaps, leg_swap_pct, leg_collapse_pct, leg_identity_diag = (
                resolve_run_leg_identity(frame_results)
            )
            if (leg_identity_diag or {}).get("method") == "skipped_coarse_sampling":
                # Frames too far apart for the whole-clip evidence to mean
                # anything (see leg_identity._MAX_MEDIAN_SPACING_MS). The
                # greedy pass is no better placed, but it is the known
                # behaviour -- and the diag says out loud that identity on
                # this clip is unrepaired rather than resolved.
                leg_swaps, leg_swap_pct, leg_collapse_pct = _fix_leg_swaps(
                    frame_results,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("LEG_IDENTITY_DP_FAILED", err=str(e))
            leg_swaps, leg_swap_pct, leg_collapse_pct = _fix_leg_swaps(frame_results)
            leg_identity_diag = None

    # Bike gets the same whole-clip resolver first -- for the one failure it
    # CAN fix there: a mutual label exchange, where both legs' points swap
    # indices and swap back (the IMG_9981 frame-21 hop). Relabelling recovers
    # those frames instead of losing them to the gate. Everything one-sided
    # -- the near index landing on the far shoe while the far index stays
    # put -- carries no far-leg truth to relabel from, so the gate still
    # blanks it afterwards: blink, never jump. ``leg_swap_pct`` deliberately
    # stays None for bike: the far leg is neither drawn nor measured on a
    # side view, and an input-noise share dominated by its invented
    # landmarks would cry wolf (see the runner's near-leg-only warning).
    leg_gate: dict[str, Any] = {}
    if sport_type == "bike":
        try:
            from app.services.video_analysis.biomechanics.leg_identity import (
                resolve_run_leg_identity,
            )

            # On coarse sampling the resolver declines and relabels nothing;
            # the gate alone is then the known behaviour, and the diag says
            # identity went unrepaired. No greedy fallback here -- bike never
            # had one, and that is the point of the gate.
            # swap_one_sided=False: the far leg is visibility-blanked on
            # large parts of a bike side view, and relabelling a frame where
            # only one leg has data trades the near leg's real points for
            # far-leg NaNs (91 destroyed frames on a real clip, seen as the
            # skeleton vanishing once per revolution).
            _, _, _, leg_identity_diag = resolve_run_leg_identity(
                frame_results, swap_one_sided=False,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("LEG_IDENTITY_DP_FAILED", err=str(e), sport="bike")
            leg_identity_diag = None
        leg_gate = _gate_leg_identity_breaks(frame_results)

    if context is not None:
        # Surfaced so the pipeline can judge tracking stability: a clip where
        # a large share of frames needed a left/right swap-back is one where
        # MediaPipe kept flipping the skeleton (classic backlit-silhouette
        # failure) -- the confidence scorer downgrades on these signals.
        context["flips_corrected"] = flips
        context["flip_pct"] = round(flips / max(len(frame_results), 1) * 100, 1)
        context["leg_swaps_corrected"] = leg_swaps
        context["leg_swap_pct"] = leg_swap_pct
        context["leg_collapse_pct"] = leg_collapse_pct
        context["leg_identity"] = leg_identity_diag
        context["leg_identity_gate"] = leg_gate or None
    if _use_butterworth_landmarks(sport_type, camera_angle, camera_view):
        smoothed, butter_meta = _apply_butterworth_landmarks(
            frame_results, sport_type, camera_angle, fps,
        )
        if context is not None:
            context["butterworth_meta"] = butter_meta
    else:
        smoothed = _apply_one_euro(
            frame_results, sport_type, camera_angle, fps,
        )

    logger.info(
        "LANDMARK_STABILIZER",
        sport=sport_type,
        camera_angle=camera_angle,
        camera_view=camera_view,
        frames=len(frame_results),
        gated_out=dropped,
        flips_corrected=flips,
        leg_swaps_corrected=leg_swaps,
        leg_swap_pct=leg_swap_pct,
        leg_collapse_pct=leg_collapse_pct,
        leg_identity_gate=leg_gate or None,
        smoothed_landmarks=smoothed,
    )

    return frame_results


def _gate_by_visibility(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None,
) -> int:
    """Mark low-confidence landmarks as NaN.

    Downstream angle calculation and Butterworth filtering already handle
    NaN (see butterworth_filter.restore_nans) -- gating here propagates
    gaps through cleanly instead of letting the smoother blend noise into
    the signal.
    """
    thr = _lookup(MIN_VISIBILITY, sport_type, camera_angle, 0.5)
    dropped = 0
    for frame in frame_results:
        for key in ("world_landmarks", "normalized_landmarks"):
            for lm in frame[key]:
                if getattr(lm, "visibility", 1.0) < thr:
                    lm.x = math.nan
                    lm.y = math.nan
                    lm.z = math.nan
                    dropped += 1
    return dropped


def _fix_flips(frame_results: list[dict[str, Any]], sport_type: str) -> int:
    """Detect and correct left/right skeleton flips.

    MediaPipe can swap left and right sides between frames, especially in
    profile views (cycling, running). We detect this by checking Z-depth
    consistency of hips: in a stable side-view, the near-side hip should
    always have a smaller Z than the far-side hip.

    DISABLED FOR BIKE: anti-flip relies on world_landmarks Z, which is
    unreliable when the far-side hip is occluded by the torso (always the
    case for bike side-view). Strict Z comparison without hysteresis flips
    landmarks frame-to-frame on noise, corrupting knee_angle histories and
    tripping the pedal-stroke quality gate. Riders don't physically flip
    during a recording, so anti-flip provides no value here. See
    diagnostic notes 2026-04-29.
    """
    if sport_type == "bike":
        logger.info("LANDMARK_STABILIZER anti-flip skipped (bike)")
        return 0

    if len(frame_results) < 5:
        return 0

    calibration_frames = min(10, len(frame_results))
    left_closer_votes = 0

    for i in range(calibration_frames):
        wl = frame_results[i]["world_landmarks"]
        try:
            z_left, z_right = wl[23].z, wl[24].z
            if math.isnan(z_left) or math.isnan(z_right):
                continue
            if z_left < z_right:
                left_closer_votes += 1
        except (IndexError, AttributeError):
            continue

    expect_left_closer = left_closer_votes > calibration_frames / 2
    flip_count = 0

    for i in range(len(frame_results)):
        wl = frame_results[i]["world_landmarks"]

        try:
            z_left, z_right = wl[23].z, wl[24].z
            if math.isnan(z_left) or math.isnan(z_right):
                continue
            hip_left_closer = z_left < z_right
        except (IndexError, AttributeError):
            continue

        if hip_left_closer == expect_left_closer:
            continue

        # Confirm with shoulders to reduce false positives
        try:
            zs_left, zs_right = wl[11].z, wl[12].z
            if not (math.isnan(zs_left) or math.isnan(zs_right)):
                sh_left_closer = zs_left < zs_right
                if sh_left_closer == expect_left_closer:
                    continue
        except (IndexError, AttributeError):
            pass

        for key in ("world_landmarks", "normalized_landmarks"):
            lm = frame_results[i][key]
            for left_idx, right_idx in _SWAP_PAIRS:
                try:
                    lm[left_idx], lm[right_idx] = lm[right_idx], lm[left_idx]
                except IndexError:
                    continue
        flip_count += 1

    return flip_count


def _fix_leg_swaps(frame_results: list[dict[str, Any]]) -> tuple[int, float, float]:
    """Correct per-frame left/right LEG identity swaps via track continuity.

    MediaPipe assigns limb identity per frame from appearance cues; on
    backlit / silhouette clips it re-assigns the leg indices freely, so the
    near-side knee/ankle series alternates between the two physical legs.
    The torso-level anti-flip (``_fix_flips``) cannot see this -- hips and
    shoulders stay put -- and neither can any visibility-based quality
    check, because the swapped landmarks are confidently detected.

    Each leg is tracked by continuity in IMAGE space (normalized coords --
    depth is exactly the signal that is unreliable here): predict this
    frame's knee+ankle per leg from the previous frame plus velocity, then
    keep whichever assignment (identity vs swapped) matches the predictions
    better. The swap must win by ``LEG_SWAP_MARGIN``, and the decision is
    skipped while the predicted legs nearly coincide, so legitimate scissor
    crossings don't oscillate -- velocity carries each track through the
    crossing. Swaps are mirrored onto world landmarks so angles and drawing
    stay consistent.

    Run side-view only. Bike is still skipped -- but NOT for the reason this
    used to give, and the difference matters to whoever picks this up next.

    The old rationale was that "on a trainer the 2D ankles ride close together
    through whole pedal circles, so the decision would be permanently
    ambiguous". Measured on the repo's bike clip (IMG_9981, 503 frames), that is
    simply false: the two feet are two pedals half a revolution apart, so their
    separation has a floor rather than a crossing. Median separation 0.098
    normalized, and only 6 frames of 503 fall below
    ``LEG_MIN_DECIDABLE_DIST``. Cycling is the EASY case for separability; it
    is running, where the legs actually scissor past each other, that is hard.

    What does block it is the other half of that sentence. The far leg is
    occluded by the near leg and the crank for a large part of every
    revolution: both ankles are measurable on 98.8% of frames but both KNEES
    only on 78.1%, so the all-four-points entry condition above reseeds
    constantly. Worse, the cost is symmetric -- it sums how well BOTH legs match
    their predictions -- so the far leg's invented coordinates land in
    ``cost_keep`` and can outvote the near leg, which is the only one the bike
    report reads or draws. Enabling this pass as-is on IMG_9981 swapped 16
    already-correct frames and pushed the largest near-knee discontinuity from
    56 to 81 degrees: a net loss.

    So the gap is real and unprotected -- the near-side foot demonstrably hops
    to the far shoe on real bike clips (IMG_9981 frames 21-22, and 7 such events
    in an 11 s user clip) and nothing here catches it. Fixing it needs a
    near-anchored rule that never lets far-leg noise decide, validated on more
    than one clip: a first attempt at that cut near-ankle breaks 9 -> 6 but
    introduced new ones by drifting its own track onto the far foot, which is
    exactly the self-perpetuating failure the reseed logic below exists to
    avoid. Do not turn this on for bike without that validation set.

    Returns ``(swaps_corrected, leg_swap_pct, leg_collapse_pct)`` where the
    percentages are over decidable frame pairs / measured frames.
    """
    n = len(frame_results)
    if n < 3:
        return 0, 0.0, 0.0

    keys = ("lk", "rk", "la", "ra")
    indices = {"lk": 25, "rk": 26, "la": 27, "ra": 28}

    def _pt(frame: dict[str, Any], idx: int) -> tuple[float, float] | None:
        lm = frame["normalized_landmarks"][idx]
        x, y = lm.x, lm.y
        if x is None or y is None:
            return None
        if (isinstance(x, float) and math.isnan(x)) or (
            isinstance(y, float) and math.isnan(y)
        ):
            return None
        return (float(x), float(y))

    def _clamp_vel(v: tuple[float, float]) -> tuple[float, float]:
        return (
            max(-LEG_MAX_VELOCITY, min(LEG_MAX_VELOCITY, v[0])),
            max(-LEG_MAX_VELOCITY, min(LEG_MAX_VELOCITY, v[1])),
        )

    prev: dict[str, tuple[float, float]] | None = None
    vel: dict[str, tuple[float, float]] = {k: (0.0, 0.0) for k in keys}
    swaps = 0
    decided = 0
    collapse_frames = 0
    measured_frames = 0
    gap_frames = 0
    last_ts: float | None = None

    for frame in frame_results:
        cur = {k: _pt(frame, indices[k]) for k in keys}

        la, ra = cur["la"], cur["ra"]
        if la is not None and ra is not None:
            measured_frames += 1
            if math.dist(la, ra) < LEG_COLLAPSE_DIST:
                collapse_frames += 1

        if any(v is None for v in cur.values()):
            # Unmeasurable frame: keep the tracks, decay velocity so a long
            # gap doesn't extrapolate the prediction off the body.
            gap_frames += 1
            vel = {k: (vx * 0.5, vy * 0.5) for k, (vx, vy) in vel.items()}
            continue

        cur_ts = frame.get("timestamp_ms")
        gap_ms = (
            cur_ts - last_ts
            if isinstance(cur_ts, (int, float)) and isinstance(last_ts, (int, float))
            else None
        )
        last_ts = cur_ts if isinstance(cur_ts, (int, float)) else last_ts

        if prev is None:
            prev = cur  # type: ignore[assignment]
            gap_frames = 0
            continue

        # After a real gap the legs may have physically exchanged places, so
        # matching against the stale prediction is a coin flip whose wrong
        # outcome self-perpetuates for the rest of the clip. Re-seed instead:
        # accept MediaPipe's labels for the first post-gap frame. The
        # timestamp check also covers hard gaps -- undetected frames never
        # enter frame_results at all, so adjacent entries can sit far apart
        # with zero unmeasurable frames in between.
        if gap_frames > LEG_GAP_RESEED_FRAMES or (
            gap_ms is not None and gap_ms > LEG_GAP_RESEED_MS
        ):
            prev = cur  # type: ignore[assignment]
            vel = {k: (0.0, 0.0) for k in keys}
            gap_frames = 0
            continue
        gap_frames = 0

        pred = {
            k: (prev[k][0] + vel[k][0], prev[k][1] + vel[k][1]) for k in keys
        }
        cost_keep = (
            math.dist(cur["la"], pred["la"]) + math.dist(cur["ra"], pred["ra"])
            + math.dist(cur["lk"], pred["lk"]) + math.dist(cur["rk"], pred["rk"])
        )
        cost_swap = (
            math.dist(cur["la"], pred["ra"]) + math.dist(cur["ra"], pred["la"])
            + math.dist(cur["lk"], pred["rk"]) + math.dist(cur["rk"], pred["lk"])
        )
        decided += 1

        pred_sep = math.dist(pred["la"], pred["ra"])
        if (
            pred_sep >= LEG_MIN_DECIDABLE_DIST
            and cost_swap < cost_keep * (1.0 - LEG_SWAP_MARGIN)
        ):
            for key in ("world_landmarks", "normalized_landmarks"):
                lms = frame[key]
                for li, ri in _LEG_SWAP_PAIRS:
                    lms[li], lms[ri] = lms[ri], lms[li]
            swaps += 1
            cur = {"lk": cur["rk"], "rk": cur["lk"], "la": cur["ra"], "ra": cur["la"]}

        vel = {
            k: _clamp_vel((cur[k][0] - prev[k][0], cur[k][1] - prev[k][1]))
            for k in keys
        }
        prev = cur  # type: ignore[assignment]

    leg_swap_pct = round(swaps / max(decided, 1) * 100, 1)
    leg_collapse_pct = round(collapse_frames / max(measured_frames, 1) * 100, 1)
    return swaps, leg_swap_pct, leg_collapse_pct


_LEG_LANDMARKS = {
    "left": (25, 27, 29, 31),    # knee, ankle, heel, foot index
    "right": (26, 28, 30, 32),
}
_LEG_ANKLE = {"left": 27, "right": 28}


def _gate_leg_identity_breaks(
    frame_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Blank a leg on the frames where it left its own track. Bike only.

    The bike path has no working defence against MediaPipe handing a leg's
    index to the other leg's foot (see :func:`_fix_leg_swaps` for what was
    tried and measured). Every correction attempt failed the same way: they all
    need the FAR leg as evidence, and the far leg's landmarks are partly
    invented -- median knee visibility 0.47 on a clean trainer clip.

    So this does not correct anything. It only refuses to measure. When an
    ankle lands further from its own predicted position than a leg could
    plausibly move, that frame is EXCLUDED from measurement: the leg's world
    landmarks go NaN and the frame is marked in ``frame["leg_gate_filled"]``,
    which the cycling analyzer reads to NaN that side's leg angles. The
    DISPLAY landmarks are treated differently -- a blinking skeleton read as
    an app error to athletes replaying their video -- so the normalized
    points keep the leg's last good shape carried to the predicted ankle
    position: the overlay stays on the near leg through the break (bounded
    by the re-seed patience, ~5 frames), while no number is computed from a
    frame we could not identify. Before any good shape exists the old NaN
    blank remains, honesty over cosmetics.

    The asymmetry is the point. The worst case here is discarding good frames
    (and drawing a briefly coasting leg); it cannot put the reported series
    on the wrong leg, which is exactly what every swap-based attempt risked
    and what happens today.

    Both legs are tested independently, so this needs no camera-side decision
    (which is not available this early anyway). Visibility is left untouched --
    it records what MediaPipe reported, and the detection-quality metrics read
    it -- so blanking here shows up as missing measurements, not as a quietly
    downgraded confidence score.

    Returns a diagnostics dict with PER-LEG counts. Per-leg matters: the far
    leg is expected to be a mess and blanking it costs nothing, so a caller
    judging whether this clip is trustworthy has to read the near leg's numbers
    specifically -- which side that is, is not known this early.
    """
    out: dict[str, Any] = {
        "frames": len(frame_results),
        "blanked": {"left": 0, "right": 0},
        "reseeds": {"left": 0, "right": 0},
        "ankle_separation": None,
    }
    n = len(frame_results)
    if n < 5:
        return out

    def _pt(frame: dict[str, Any], idx: int) -> tuple[float, float] | None:
        lm = frame["normalized_landmarks"][idx]
        x, y = lm.x, lm.y
        if x is None or y is None:
            return None
        if (isinstance(x, float) and math.isnan(x)) or (
            isinstance(y, float) and math.isnan(y)
        ):
            return None
        return (float(x), float(y))

    # The clip's own scale: how far apart the two feet sit. Median over the
    # frames where both were measured, so occlusion and the odd collapsed
    # frame do not set it.
    seps = [
        math.dist(a, b)
        for a, b in (
            (_pt(f, 27), _pt(f, 28)) for f in frame_results
        )
        if a is not None and b is not None
    ]
    if len(seps) < 5:
        return out
    separation = float(np.median(seps))
    out["ankle_separation"] = round(separation, 4)
    if separation < LEG_BREAK_MIN_SEPARATION:
        # The two feet never resolve as two feet. Nothing here can be judged,
        # and blanking on a bad scale would erase the whole clip.
        logger.info("LEG_IDENTITY_GATE_SKIPPED", reason="feet_unresolved",
                    separation=round(separation, 4))
        return out

    threshold = LEG_BREAK_FRAC * separation

    for side, ankle_idx in _LEG_ANKLE.items():
        prev: tuple[float, float] | None = None
        vel = (0.0, 0.0)
        held = 0
        blanked = 0
        reseeds = 0
        # Last good positions of this leg's points RELATIVE to its ankle,
        # so an excluded frame can still show the leg (shape carried to the
        # predicted ankle) instead of blinking out.
        shape: dict[int, tuple[float, float]] | None = None

        for frame in frame_results:
            cur = _pt(frame, ankle_idx)
            if cur is None:
                # The leg is missing entirely (blanked upstream). Within the
                # same patience the display is filled the same way a break
                # is: predicted ankle, last good shape. The world landmarks
                # are already NaN, so nothing is measured off the fill.
                if (
                    prev is not None and shape is not None
                    and held < LEG_BREAK_RESEED_FRAMES
                ):
                    pred = (
                        prev[0] + vel[0] * (held + 1),
                        prev[1] + vel[1] * (held + 1),
                    )
                    for i in _LEG_LANDMARKS[side]:
                        if i in shape:
                            nlm = frame["normalized_landmarks"][i]
                            nlm.x = pred[0] + shape[i][0]
                            nlm.y = pred[1] + shape[i][1]
                            nlm.z = 0.0
                    frame.setdefault("leg_gate_filled", set()).add(side)
                held += 1
                continue
            if prev is None:
                prev, held = cur, 0
                continue

            # Velocity is CARRIED across held frames, not decayed: a pedalling
            # foot keeps going round at much the same rate, so extrapolating it
            # forward is a fair guess, while decaying it would aim the
            # prediction at a foot standing still and blank the next frame too.
            pred = (prev[0] + vel[0] * (held + 1), prev[1] + vel[1] * (held + 1))
            if math.dist(cur, pred) > threshold:
                if held >= LEG_BREAK_RESEED_FRAMES:
                    prev, vel, held = cur, (0.0, 0.0), 0
                    reseeds += 1
                    continue
                for i in _LEG_LANDMARKS[side]:
                    wlm = frame["world_landmarks"][i]
                    wlm.x = math.nan
                    wlm.y = math.nan
                    wlm.z = math.nan
                    nlm = frame["normalized_landmarks"][i]
                    if shape is not None and i in shape:
                        nlm.x = pred[0] + shape[i][0]
                        nlm.y = pred[1] + shape[i][1]
                        nlm.z = 0.0
                    else:
                        nlm.x = math.nan
                        nlm.y = math.nan
                        nlm.z = math.nan
                frame.setdefault("leg_gate_filled", set()).add(side)
                blanked += 1
                held += 1
                continue

            vel = _clamp_leg_velocity((
                (cur[0] - prev[0]) / (held + 1), (cur[1] - prev[1]) / (held + 1),
            ))
            prev, held = cur, 0
            new_shape: dict[int, tuple[float, float]] = {}
            for i in _LEG_LANDMARKS[side]:
                p = _pt(frame, i)
                if p is not None:
                    new_shape[i] = (p[0] - cur[0], p[1] - cur[1])
            if new_shape:
                shape = new_shape

        out["blanked"][side] = blanked
        out["reseeds"][side] = reseeds

    return out


def _clamp_leg_velocity(v: tuple[float, float]) -> tuple[float, float]:
    return (
        max(-LEG_MAX_VELOCITY, min(LEG_MAX_VELOCITY, v[0])),
        max(-LEG_MAX_VELOCITY, min(LEG_MAX_VELOCITY, v[1])),
    )


def _apply_one_euro(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None,
    fps: float,
) -> int:
    """Apply One Euro Filter to landmark (x, y, z) across frames.

    One filter instance per (landmark_set, landmark_idx, coord). NaN inputs
    pass through as the previous estimate so visibility-gated gaps are
    preserved without breaking the recursion.

    Bike is strictly 2D sagittal-plane: z is unused downstream and the
    far-side hip occlusion makes z noise meaningless. Skip z smoothing for
    bike to avoid spending filter state on a coordinate we then ignore.
    Run and swim continue smoothing all three coordinates as before.
    """
    if len(frame_results) < 2:
        return 0

    params = _lookup(
        ONE_EURO_PARAMS,
        sport_type,
        camera_angle,
        dict(min_cutoff=1.0, beta=0.05, d_cutoff=1.0),
    )

    first_wl = frame_results[0]["world_landmarks"]
    first_nl = frame_results[0]["normalized_landmarks"]
    n_landmarks = min(len(first_wl), len(first_nl), 33)

    smooth_z = sport_type != "bike"

    filters = {
        "world_landmarks": [
            [OneEuro(freq=fps, **params) for _ in range(3)]
            for _ in range(n_landmarks)
        ],
        "normalized_landmarks": [
            [OneEuro(freq=fps, **params) for _ in range(3)]
            for _ in range(n_landmarks)
        ],
    }

    for frame in frame_results:
        for key in ("world_landmarks", "normalized_landmarks"):
            lms = frame[key]
            for i in range(n_landmarks):
                lm = lms[i]
                lm.x = filters[key][i][0](lm.x)
                lm.y = filters[key][i][1](lm.y)
                if smooth_z:
                    lm.z = filters[key][i][2](lm.z)

    return n_landmarks


# Zero-phase Butterworth params for landmark coordinate smoothing.
# Keyed by (sport, camera_angle). swim_under and bike side-view use this
# path; other modes keep the causal One Euro filter.
BUTTER_LANDMARK_CUTOFF_HZ: dict[tuple[str, str | None], float] = {
    # 8 Hz lets the sharp catch-phase elbow flexion and wrist-y peaks
    # through. Relies on SPORT_SAMPLE_RATES["swim"] == 1 so effective
    # fps is 30 Hz -> Nyquist 15 Hz, leaving headroom above the clamp.
    ("swim", "under_water"): 8.0,
    # Bike side-view: pedal cadence is ~1.5 Hz at 90 RPM; harmonic
    # content extends to ~3-4 Hz at TDC/BDC direction reversals. 6 Hz
    # cutoff preserves real motion (4x fundamental) while removing
    # MediaPipe re-detection jitter (typical 8-15 Hz). The fps-adaptive
    # cap (0.2 * effective_fps) usually binds first at 30 fps -> 6 Hz,
    # at 60 fps the target wins (cap 12, target 6).
    ("bike", None): 6.0,
    # Run side-view: step frequency is ~1.4-1.5 Hz per leg (~170-180 spm
    # combined); swing-leg and foot-strike kinematics carry meaningful
    # content up to ~8 Hz (standard gait-lab low-pass range is 6-10 Hz).
    # At 30 fps the 0.2*fps adaptive cap tightens this to 6 Hz -- the
    # classic Winter cutoff -- so both common phone framerates land on
    # defensible values.
    ("run", None): 8.0,
}
BUTTER_LANDMARK_ORDER = 4
# filtfilt needs enough samples beyond the padlen (default 3*order*2 for
# SOS) to be numerically stable. 24 matches what the existing angle
# Butterworth helper uses as its minimum.
MIN_BUTTER_SAMPLES = 24

# Filter-stability ceiling: cutoff must stay well below Nyquist to avoid
# ringing at the edge. 0.85 is empirically safe for swim-catch energy.
NYQUIST_SAFETY_FACTOR = 0.85

# Adaptive cutoff cap: cutoff should not exceed FPS_ADAPTIVE_FRACTION *
# fps. This ties smoothing to the actual sampling frequency so that a
# 30 fps video gets proportionally more smoothing than a 60 fps one.
# Rationale: at 30 fps the fixed 8 Hz target is 53% of Nyquist -- that
# passes almost all high-frequency content through, so the skeleton
# appears to jitter frame-to-frame. Capping at 0.2*fps gives a
# consistent smoothness profile across source framerates.
#
# With the default constants, the adaptive cap is always tighter than
# the Nyquist safety ceiling (0.2*fps < 0.85*fps/2 = 0.425*fps), so
# "fps_adaptive" is the reason that fires in practice. "nyquist_safety"
# remains in the reason-selection logic for non-default call sites that
# pass a larger fraction.
FPS_ADAPTIVE_FRACTION = 0.2


def _compute_safe_butterworth_cutoff(
    target_cutoff_hz: float,
    effective_fps: float,
    nyquist_safety_factor: float = NYQUIST_SAFETY_FACTOR,
    fps_adaptive_fraction: float = FPS_ADAPTIVE_FRACTION,
) -> dict[str, Any]:
    """Compute a clamped Butterworth cutoff and quantify any degradation.

    The actual cutoff is the minimum of three bounds:
      - ``target_cutoff_hz`` (the ideal cutoff for the sport/mode)
      - ``fps_adaptive_fraction * effective_fps`` (fps-adaptive cap that
        keeps the smoothness profile consistent across source
        framerates -- stops a 30 fps clip from looking jittery while
        leaving 60 fps untouched)
      - ``nyquist_safety_factor * nyquist`` (hard stability ceiling
        against ringing at the Nyquist edge)

    Returns a metadata dict with the actual cutoff, reduction
    percentage, which bound was binding (``reduction_reason``), and a
    human-readable warning when the reduction exceeds 20%.
    """
    nyquist = effective_fps / 2.0
    safety_ceiling = nyquist_safety_factor * nyquist
    adaptive_cap = fps_adaptive_fraction * effective_fps

    actual = min(target_cutoff_hz, adaptive_cap, safety_ceiling)
    reduction_pct = (
        round((target_cutoff_hz - actual) / target_cutoff_hz * 100, 1)
        if target_cutoff_hz > 0 else 0.0
    )

    reduction_reason: str | None = None
    if actual < target_cutoff_hz:
        # If both caps are below target, the tighter one wins; report
        # that as the reason so triage knows which bound to tune.
        reduction_reason = (
            "fps_adaptive" if adaptive_cap <= safety_ceiling
            else "nyquist_safety"
        )

    warning: str | None = None
    if reduction_pct > 20:
        warning = (
            f"Effective fps {effective_fps:.1f} is low. Smoothing cutoff "
            f"reduced from {target_cutoff_hz:.1f} to {actual:.1f} Hz "
            f"({reduction_pct:.0f}%) to maintain skeleton stability."
        )
    return {
        "target_cutoff_hz": target_cutoff_hz,
        "actual_cutoff_hz": round(actual, 2),
        "effective_fps": round(effective_fps, 2),
        "nyquist_hz": round(nyquist, 2),
        "reduction_pct": reduction_pct,
        "reduction_reason": reduction_reason,
        "warning": warning,
    }


def _apply_butterworth_landmarks(
    frame_results: list[dict[str, Any]],
    sport_type: str,
    camera_angle: str | None,
    fps: float,
) -> tuple[int, dict[str, Any]]:
    """Zero-phase 4th-order Butterworth on every (landmark, axis) series.

    Non-causal (sosfiltfilt = forward + backward) so there is no phase
    lag through fast transients like the underwater swim catch. NaN gaps
    from the P0 visibility gate are linearly interpolated before
    filtering and re-masked afterwards so gated frames stay hidden
    downstream.

    Returns (count_of_filtered_series, butterworth_metadata_dict).
    Falls back to _apply_one_euro when the frame count is below the
    stable filtfilt minimum (returns 0 metadata in that case).
    """
    n = len(frame_results)
    if n < MIN_BUTTER_SAMPLES:
        # NOTE: This fallback path is defensive as of 2026-04-17.
        # After the 2-second minimum duration check in validator.py, no
        # normal swim upload can reach this branch (MIN_BUTTER_SAMPLES=24
        # ≈ 0.8 s at 30 fps). If this fires in production, something
        # upstream failed to validate.
        if sport_type == "swim" and camera_angle == "under_water":
            logger.warning(
                "BUTTERWORTH_FALLBACK",
                sport=sport_type,
                camera_angle=camera_angle,
                frame_count=n,
                min_required=MIN_BUTTER_SAMPLES,
                msg=(
                    "Butterworth fallback triggered for swim_under. "
                    "This should not happen after the 2s minimum duration "
                    "validation -- investigate upstream."
                ),
            )
        count = _apply_one_euro(frame_results, sport_type, camera_angle, fps)
        fallback_meta: dict[str, Any] = {
            "fallback_triggered": True,
            "fallback_reason": "insufficient_samples",
            "frame_count": n,
            "min_required": MIN_BUTTER_SAMPLES,
            "warning": (
                "Video was too short for optimal filtering. Analysis used "
                "a fallback smoother -- results may be less precise. "
                "Consider a longer clip (3+ seconds)."
            ) if sport_type == "swim" and camera_angle == "under_water" else None,
        }
        return count, fallback_meta

    from scipy.signal import butter, sosfiltfilt

    try:
        from app.services.video_analysis.pipeline import SPORT_SAMPLE_RATES
        sample_rate = SPORT_SAMPLE_RATES.get(sport_type, 1)
    except Exception:
        sample_rate = 1
    effective_fps = max(fps / max(sample_rate, 1), 1.0)
    target_cutoff = BUTTER_LANDMARK_CUTOFF_HZ.get((sport_type, camera_angle), 4.0)
    cutoff_info = _compute_safe_butterworth_cutoff(target_cutoff, effective_fps)
    cutoff_hz = cutoff_info["actual_cutoff_hz"]
    nyquist = cutoff_info["nyquist_hz"]
    wn = cutoff_hz / nyquist
    sos = butter(BUTTER_LANDMARK_ORDER, wn, btype="low", output="sos")

    first_wl = frame_results[0]["world_landmarks"]
    first_nl = frame_results[0]["normalized_landmarks"]
    n_lm = min(len(first_wl), len(first_nl), 33)

    # Bike never reads z downstream (cycling_analyzer projects z=0 via
    # _strip_z, and One Euro also skips z for bike). Skip Butterworth on
    # z too to keep behavior consistent and avoid wasted filter work.
    axes = ("x", "y") if sport_type == "bike" else ("x", "y", "z")

    filtered_series = 0
    for key in ("world_landmarks", "normalized_landmarks"):
        for i in range(n_lm):
            for attr in axes:
                series = np.array(
                    [getattr(frame_results[f][key][i], attr) for f in range(n)],
                    dtype=float,
                )
                nan_mask = np.isnan(series)
                valid = ~nan_mask
                if valid.sum() < MIN_BUTTER_SAMPLES:
                    continue
                # Linear-interp NaN gaps so filtfilt sees a finite input
                idx = np.arange(n)
                series[nan_mask] = np.interp(
                    idx[nan_mask], idx[valid], series[valid]
                )
                try:
                    out = sosfiltfilt(sos, series)
                except ValueError:
                    continue  # padlen mismatch -- leave original
                # Restore NaN so visibility-gated positions stay hidden
                out[nan_mask] = np.nan
                for f in range(n):
                    setattr(frame_results[f][key][i], attr, float(out[f]))
                filtered_series += 1

    cutoff_info["series_filtered"] = filtered_series
    cutoff_info["frames"] = n
    cutoff_info["order"] = BUTTER_LANDMARK_ORDER
    cutoff_info["fallback_triggered"] = False
    cutoff_info["fallback_reason"] = None

    logger.info(
        "LANDMARK_BUTTERWORTH",
        sport=sport_type,
        camera_angle=camera_angle,
        **cutoff_info,
    )
    return n_lm, cutoff_info
