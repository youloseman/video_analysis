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
#
# This is the NORMAL-SPEED value of what is really a crank-angle horizon:
# 45 degrees, cycle/8. The stabilizer measures the clip's own revolution
# period and passes the scaled value in -- a slow-motion clip spans ~4x the
# frames per revolution, and holding patience at 5 fixed frames there cuts
# the horizon to ~10 degrees, which is how the blinking came back on the
# first slo-mo upload. The sweep above stays authoritative in DEGREES.
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
            if leg_identity_diag is not None:
                # Did identity actually hold? Gait fixes the answer: the two
                # ankles swap vertical order twice per stride and no oftener,
                # so the excess counts the times the labels traded legs. The
                # resolver cannot check its own work -- this can, and the
                # report is entitled to know when it should not be trusted.
                from app.services.video_analysis.biomechanics.stride_consistency import (
                    measure_leg_identity_stability,
                )
                leg_identity_diag["stability"] = measure_leg_identity_stability(
                    frame_results, fps or 30.0,
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
    bike_cycle_frames: float | None = None
    if sport_type == "bike":
        # The crank clock: every frame-count constant below encodes crank
        # rotation, so a slow-motion clip (same fit, ~4x the frames per
        # revolution) needs them re-expressed in the clip's own period.
        # Foot-cluster hallucinations (chords through the pedal circle) are
        # reconstructed first: they poison the identity resolver's ankle
        # costs and every knee angle alike. After this pass the display foot
        # rides the calibrated path and the flagged frames are measurement-
        # excluded via leg_gate_filled.
        ankle_path = _gate_ankle_off_path(frame_results)
        bike_cycle_frames = _estimate_cycle_frames(frame_results, fps)
        patience = LEG_BREAK_RESEED_FRAMES
        if bike_cycle_frames:
            # 45 degrees of crank, the horizon the straight-line predictor
            # stays honest over (see the LEG_BREAK_RESEED_FRAMES sweep) --
            # which is 5-6 frames at normal speed, unchanged.
            patience = int(min(45, max(
                LEG_BREAK_RESEED_FRAMES, round(bike_cycle_frames / 8),
            )))
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
        leg_gate = _gate_leg_identity_breaks(frame_results, patience=patience)
        # After the gate so it never runs on frames the gate re-filled from a
        # prediction, and before the smoothing pass so the restored ankle is
        # what gets filtered.
        leg_gate["shin_restored"] = _enforce_shin_length(frame_results)
        leg_gate["cycle_frames"] = bike_cycle_frames
        leg_gate["patience"] = patience
        leg_gate["ankle_off_path"] = ankle_path

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
        cutoff_override: float | None = None
        if sport_type == "bike" and bike_cycle_frames:
            cycle_hz = fps / bike_cycle_frames
            if cycle_hz < 1.0:
                # Slow motion (or sub-60 rpm): the shipped 6 Hz cutoff was
                # picked as ~4x a normal pedalling fundamental; at 0.16 Hz
                # container-time motion it sits 40x above the signal and
                # removes nothing the eye can see. Scale it with the clip's
                # own fundamental; at any normal cadence this branch never
                # runs and the shipped cutoff stands.
                cutoff_override = max(1.0, 6.0 * cycle_hz)
        smoothed, butter_meta = _apply_butterworth_landmarks(
            frame_results, sport_type, camera_angle, fps,
            target_cutoff_override=cutoff_override,
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
    readable_frames = 0

    for i in range(calibration_frames):
        wl = frame_results[i]["world_landmarks"]
        try:
            z_left, z_right = wl[23].z, wl[24].z
            if math.isnan(z_left) or math.isnan(z_right):
                continue
            readable_frames += 1
            if z_left < z_right:
                left_closer_votes += 1
        except (IndexError, AttributeError):
            continue

    # The quorum is the frames that could actually SEE the hips. Dividing by
    # the calibration window let NaN frames vote for the status quo: five
    # readable frames unanimously saying "left closer" lost 5 > 5, and zero
    # readable frames produced an expectation from no evidence at all --
    # after which every frame disagreeing with that null expectation got a
    # full-body swap. No quorum, no flipping.
    if readable_frames < 3:
        logger.info(
            "ANTI_FLIP_SKIPPED", reason="calibration_unreadable",
            readable=readable_frames, window=calibration_frames,
        )
        return 0
    expect_left_closer = left_closer_votes > readable_frames / 2
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

# --- the crank clock (bike) -------------------------------------------------
# Every frame-count constant in the bike path encodes an amount of CRANK
# ROTATION, calibrated at normal speed (~40-50 frames a revolution): the
# gate's patience is really "45 degrees of crank", the smoothing cutoff is
# really "a few times the pedalling fundamental". A slow-motion clip breaks
# all of them at once -- the same fit filmed in slo-mo spans ~180 frames a
# revolution, so a fixed 5-frame patience covers 10 degrees instead of 45
# (fills stop reaching across breaks: the blinking came back), and a 6 Hz
# cutoff sits 40x above the 0.16 Hz motion (jitter sails through untouched).
# The bike carries its own clock: the ankles' vertical oscillation. Measure
# the revolution period once per clip and express the constants in crank
# degrees; at normal speed everything reduces to the shipped values.
# No human pedals faster than ~150 rpm, so a real revolution is at least
# this long -- in SECONDS, converted to frames from the fps the series
# actually has. The first version fixed this at 10 analysed frames, which
# on an adaptively decimated long clip (stride 4 -> 7.5 analysed fps at
# 90 rpm, true period ~5 frames) put the floor ABOVE the true period, and
# the first admissible autocorrelation peak was the two-revolution lag:
# the kinogram's +/-45 deg tiles landed at +/-90 deg and the gate's
# patience miscalibrated by the same factor.
_CYCLE_MIN_LAG_S = 0.35
_CYCLE_MIN_LAG_FLOOR = 3     # frames; guards degenerate fps values
_CYCLE_MIN_AUTOCORR = 0.4    # the oscillation must actually repeat
_CYCLE_MIN_REPEATS = 2.0     # need at least this many revolutions in clip


def _estimate_cycle_frames(
    frame_results: list[dict[str, Any]], fps: float = 30.0,
) -> float | None:
    """Frames per crank revolution, from the ankles' own oscillation."""
    min_lag = max(_CYCLE_MIN_LAG_FLOOR, int(round(_CYCLE_MIN_LAG_S * fps)))
    estimates = []
    for ankle_idx in (27, 28):
        ys = []
        for f in frame_results:
            lm = f["normalized_landmarks"][ankle_idx]
            y = lm.y
            ys.append(
                float("nan")
                if y is None or (isinstance(y, float) and math.isnan(y))
                else float(y)
            )
        arr = np.array(ys)
        good = ~np.isnan(arr)
        if good.sum() < min_lag * 3:
            continue
        v = arr - np.nanmedian(arr)
        v[~good] = 0.0
        ac = np.correlate(v, v, "full")[len(v) - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        limit = int(len(ac) / _CYCLE_MIN_REPEATS)
        for lag in range(min_lag, max(min_lag, limit - 1)):
            if (
                ac[lag] > _CYCLE_MIN_AUTOCORR
                and ac[lag] >= ac[lag - 1]
                and ac[lag] >= ac[lag + 1]
            ):
                estimates.append(float(lag))
                break
    if not estimates:
        return None
    return float(np.median(estimates))


# --- ankle path gate (bike) -------------------------------------------------
# The foot is bolted to the pedal and the pedal rides a circle around the
# bottom bracket, so each ankle's honest track is a thin closed band around
# that circle -- an egg, not a circle: ankling raises it at the top and
# widens it at the bottom, so the band is LEARNED per clip and per phase,
# never assumed. What MediaPipe actually produces, twice per revolution on
# every fixture clip, is a CHORD: the whole foot cluster (ankle, heel, toe)
# lets go of the shoe and cuts through the middle of the circle -- onto the
# crank, the chainring (the drive side is the worst: the chainring disc sits
# exactly on the ankle's bottom arc), or the trainer mat. Measured by manual
# joint marking on gridded frames: the drive-side fixture's reported BDC
# stood ~7-8 deg above its true knee angle because of exactly this. Neither
# the identity gate (a drift, not a teleport), the visibility floor (0.85+
# on hallucinated points), nor the shin prior (the chord keeps shin length
# near median) can see it; only the pedal's own geometry can.
#
# What happens to a flagged frame follows the codebase's split: the DISPLAY
# foot is RECONSTRUCTED on the learned path (phase interpolated in time from
# honest neighbours -- steady cadence makes that solid), so the athlete sees
# a foot on the pedal instead of a blink; the MEASUREMENT is excluded, via
# the same ``leg_gate_filled`` flag the identity gate uses -- the analyzer
# refuses leg angles on flagged frames, so no number stands on a
# reconstruction. A first version that NaN'd the cluster instead was
# measured re-opening draw holes and wrecking the slow-motion fixture's
# variability; reconstruction has no frame-count constants to mis-scale.
_ANKLE_PATH_MIN_POINTS = 60      # need most of a revolution to learn a path
_ANKLE_PATH_BINS = 16
_ANKLE_PATH_BIN_MIN = 4          # honest samples a bin needs to testify
_ANKLE_PATH_HONEST_R = (0.75, 1.35)   # of fitted R: the learning band
_ANKLE_PATH_INTERIOR_R = 0.70    # inside this, it is a chord, no debate
_ANKLE_PATH_TOL = 0.18           # of R, around the learned per-phase radius
_ANKLE_PATH_MIN_HONEST_SHARE = 0.5    # else the fit itself is not trusted


def _gate_ankle_off_path(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct off-path foot clusters on the pedal path. Bike only.

    Returns per-side counts of reconstructed frames. Flagged frames carry
    ``leg_gate_filled`` so the analyzer excludes their leg angles.
    """
    out: dict[str, Any] = {"left": 0, "right": 0}
    if not frame_results:
        return out
    fw = frame_results[0].get("frame_width") or 1
    fh = frame_results[0].get("frame_height") or 1
    aspect = fw / fh

    def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        a = np.c_[2 * x, 2 * y, np.ones(len(x))]
        b = x * x + y * y
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        cx, cy = float(sol[0]), float(sol[1])
        return cx, cy, math.sqrt(max(float(sol[2]) + cx * cx + cy * cy, 1e-12))

    for side, (ankle_i, heel_i, toe_i) in (
        ("left", (27, 29, 31)), ("right", (28, 30, 32)),
    ):
        pts: list[tuple[float, float] | None] = []
        for f in frame_results:
            lm = f["normalized_landmarks"][ankle_i]
            x, y = lm.x, lm.y
            bad = (
                x is None or y is None
                or (isinstance(x, float) and math.isnan(x))
                or (isinstance(y, float) and math.isnan(y))
            )
            pts.append(None if bad else (float(x) * aspect, float(y)))
        finite_idx = [i for i, p in enumerate(pts) if p is not None]
        if len(finite_idx) < _ANKLE_PATH_MIN_POINTS:
            continue

        xs = np.array([pts[i][0] for i in finite_idx])
        ys = np.array([pts[i][1] for i in finite_idx])
        fx, fy = xs, ys
        for _ in range(3):
            cx, cy, radius = _fit(fx, fy)
            d = np.abs(np.hypot(fx - cx, fy - cy) - radius)
            keep = d < np.percentile(d, 75)
            if keep.sum() < _ANKLE_PATH_MIN_POINTS // 2:
                break
            fx, fy = fx[keep], fy[keep]
        cx, cy, radius = _fit(fx, fy)
        if not (0.005 < radius < 0.5):
            continue  # no crank-sized orbit here; leave the clip alone

        # Learn the per-phase radius from the honest band only, so a sector
        # where the chords OUTNUMBER the honest samples cannot teach the
        # profile that the chord is normal (a blind per-bin median was
        # measured reading 0.45R in corrupted sectors).
        lo = _ANKLE_PATH_HONEST_R[0] * radius
        hi = _ANKLE_PATH_HONEST_R[1] * radius
        honest = 0
        samples: list[list[float]] = [[] for _ in range(_ANKLE_PATH_BINS)]
        for i in finite_idx:
            p = pts[i]
            r = math.hypot(p[0] - cx, p[1] - cy)
            if lo <= r <= hi:
                honest += 1
                th = math.atan2(p[1] - cy, p[0] - cx)
                b = int((th + math.pi) / (2 * math.pi) * _ANKLE_PATH_BINS) \
                    % _ANKLE_PATH_BINS
                samples[b].append(r)
        if honest < _ANKLE_PATH_MIN_HONEST_SHARE * len(finite_idx):
            logger.info(
                "ANKLE_PATH_DISTRUSTED", side=side,
                honest=honest, total=len(finite_idx),
            )
            continue
        prof: list[float | None] = [
            float(np.median(s)) if len(s) >= _ANKLE_PATH_BIN_MIN else None
            for s in samples
        ]

        def _radius_at(th: float) -> float:
            b = int((th + math.pi) / (2 * math.pi) * _ANKLE_PATH_BINS) \
                % _ANKLE_PATH_BINS
            r = prof[b]
            return r if r is not None else radius

        tol = _ANKLE_PATH_TOL * radius
        flagged: list[int] = []
        theta: dict[int, float] = {}
        for i in finite_idx:
            p = pts[i]
            r = math.hypot(p[0] - cx, p[1] - cy)
            th = math.atan2(p[1] - cy, p[0] - cx)
            bad = r < _ANKLE_PATH_INTERIOR_R * radius
            if not bad:
                b = int((th + math.pi) / (2 * math.pi) * _ANKLE_PATH_BINS) \
                    % _ANKLE_PATH_BINS
                if prof[b] is not None and abs(r - prof[b]) > tol:
                    bad = True
            if bad:
                flagged.append(i)
            else:
                theta[i] = th
        if not flagged:
            continue

        # Phase for a flagged frame: unwrap the honest neighbours' angles and
        # interpolate in FRAME TIME across the episode. A corrupted point's
        # own angle is exactly what cannot be trusted.
        hon_idx = sorted(theta)
        unwrapped = np.unwrap([theta[i] for i in hon_idx])
        th_of = dict(zip(hon_idx, unwrapped))

        # Foot shape memory: heel/toe offsets relative to the ankle from the
        # last honest frame, carried into the reconstruction.
        def _shape(i: int) -> dict[int, tuple[float, float]] | None:
            f = frame_results[i]
            a = f["normalized_landmarks"][ankle_i]
            outv: dict[int, tuple[float, float]] = {}
            for idx in (heel_i, toe_i):
                lm = f["normalized_landmarks"][idx]
                x, y = lm.x, lm.y
                if x is None or (isinstance(x, float) and math.isnan(x)):
                    continue
                outv[idx] = (float(x) - float(a.x), float(y) - float(a.y))
            return outv or None

        for i in flagged:
            pos = np.searchsorted(hon_idx, i)
            prev_i = hon_idx[pos - 1] if pos > 0 else None
            next_i = hon_idx[pos] if pos < len(hon_idx) else None
            if prev_i is not None and next_i is not None:
                w = (i - prev_i) / max(next_i - prev_i, 1)
                th = th_of[prev_i] * (1 - w) + th_of[next_i] * w
            elif prev_i is not None:
                th = th_of[prev_i]
            elif next_i is not None:
                th = th_of[next_i]
            else:
                continue
            r = _radius_at(math.atan2(math.sin(th), math.cos(th)))
            ax = cx + r * math.cos(th)
            ay = cy + r * math.sin(th)
            new_ankle = (ax / aspect, ay)   # back to normalized units

            shape = None
            if prev_i is not None:
                shape = _shape(prev_i)
            if shape is None and next_i is not None:
                shape = _shape(next_i)

            f = frame_results[i]
            alm = f["normalized_landmarks"][ankle_i]
            alm.x, alm.y = new_ankle
            for idx in (heel_i, toe_i):
                lm = f["normalized_landmarks"][idx]
                if shape and idx in shape:
                    lm.x = new_ankle[0] + shape[idx][0]
                    lm.y = new_ankle[1] + shape[idx][1]
                else:
                    lm.x = math.nan
                    lm.y = math.nan
            # Measurement refuses the reconstruction: same channel as the
            # identity gate's fills. World stays whatever it was -- nothing
            # measures the bike off world landmarks any more, and the legacy
            # path's own NaN handling covers the rest.
            f.setdefault("leg_gate_filled", set()).add(side)
            out[side] += 1
    return out


# --- shin-length prior (bike) ----------------------------------------------
# The knee-to-ankle segment is a bone; on a side view its projected length is
# near-constant through the pedal stroke (+/-5% from perspective and slight
# out-of-plane travel). MediaPipe disagrees once per revolution: near TDC it
# slides the ankle UP the shin -- measured on a real left-side clip the shin
# "shrank" to 76% of its median at every TDC (the heel floats even higher,
# while the TOE stays planted on the shoe). The knee ANGLE barely notices --
# the ankle slides along the shin, and angles read directions -- but the
# drawn foot hangs in the air above the shoe and the ankle-vertex angle is
# junk on those frames. Frames whose shin is shorter than this fraction of
# the clip's median get the ankle re-projected to the median length along
# the measured direction, the heel translated with it, and the toe left
# where it was detected (it was the one landmark that stayed honest).
SHIN_MIN_FRAC = 0.90


def _enforce_shin_length(frame_results: list[dict[str, Any]]) -> dict[str, int]:
    """Restore bone length where MediaPipe shortened the shin. Bike only."""
    out = {"left": 0, "right": 0}
    if not frame_results:
        return out
    fw = frame_results[0].get("frame_width") or 1
    fh = frame_results[0].get("frame_height") or 1
    aspect = fw / fh

    def pt(frame: dict[str, Any], idx: int) -> tuple[float, float] | None:
        lm = frame["normalized_landmarks"][idx]
        x, y = lm.x, lm.y
        if x is None or y is None:
            return None
        if (isinstance(x, float) and math.isnan(x)) or (
            isinstance(y, float) and math.isnan(y)
        ):
            return None
        return (float(x), float(y))

    def length(a: tuple[float, float], b: tuple[float, float]) -> float:
        # Isotropic length: normalized x spans the width, y the height.
        return math.hypot((a[0] - b[0]) * aspect, a[1] - b[1])

    for side, (knee_i, ankle_i, heel_i, _toe_i) in _LEG_LANDMARKS.items():
        lengths = []
        for f in frame_results:
            k, a = pt(f, knee_i), pt(f, ankle_i)
            if k and a:
                lengths.append(length(k, a))
        if len(lengths) < 10:
            continue
        median = float(np.median(lengths))
        if median <= 1e-6:
            continue
        floor = SHIN_MIN_FRAC * median
        for f in frame_results:
            if side in f.get("leg_gate_filled", set()):
                continue  # display-only reconstruction; leave it be
            k, a = pt(f, knee_i), pt(f, ankle_i)
            if not (k and a):
                continue
            cur = length(k, a)
            if cur >= floor or cur <= 1e-9:
                continue
            scale = median / cur
            new_a = (k[0] + (a[0] - k[0]) * scale, k[1] + (a[1] - k[1]) * scale)
            dx, dy = new_a[0] - a[0], new_a[1] - a[1]
            for idx in (ankle_i, heel_i):
                lm = f["normalized_landmarks"][idx]
                p = pt(f, idx)
                if p is None:
                    continue
                lm.x = p[0] + dx
                lm.y = p[1] + dy
            out[side] += 1
    return out


def _gate_leg_identity_breaks(
    frame_results: list[dict[str, Any]],
    patience: int = LEG_BREAK_RESEED_FRAMES,
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

    # Anatomical leash: an ankle cannot sit further from its own hip than a
    # straightened leg. The velocity clamp alone is far too loose over a
    # whole patience window (0.12/frame x 5 frames = more than a leg), and a
    # prediction that has run away both draws off the body AND blanks good
    # frames for the crime of disagreeing with nonsense -- measured on a
    # real clip: four display fills off the frame edge while the raw ankle
    # sat within normal reach the whole time.
    _LEG_HIP = {"left": 23, "right": 24}
    reach_limit: dict[str, float | None] = {}
    for side, ankle_idx in _LEG_ANKLE.items():
        reaches = [
            math.dist(h, a)
            for h, a in (
                (_pt(f, _LEG_HIP[side]), _pt(f, ankle_idx))
                for f in frame_results
            )
            if h is not None and a is not None
        ]
        reach_limit[side] = (
            1.35 * float(np.median(reaches)) if len(reaches) >= 10 else None
        )

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
        limit = reach_limit[side]
        hip_idx = _LEG_HIP[side]

        def _leash(pred, frame):
            hip = _pt(frame, hip_idx)
            if hip is None or limit is None:
                return pred
            d = math.dist(pred, hip)
            if d <= limit or d <= 1e-9:
                return pred
            s = limit / d
            return (hip[0] + (pred[0] - hip[0]) * s,
                    hip[1] + (pred[1] - hip[1]) * s)

        for frame in frame_results:
            cur = _pt(frame, ankle_idx)
            if cur is None:
                # The leg is missing entirely (blanked upstream). Within the
                # same patience the display is filled the same way a break
                # is: predicted ankle, last good shape. The world landmarks
                # are already NaN, so nothing is measured off the fill.
                if (
                    prev is not None and shape is not None
                    and held < patience
                ):
                    pred = _leash((
                        prev[0] + vel[0] * (held + 1),
                        prev[1] + vel[1] * (held + 1),
                    ), frame)
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
            pred = _leash(
                (prev[0] + vel[0] * (held + 1), prev[1] + vel[1] * (held + 1)),
                frame,
            )
            if math.dist(cur, pred) > threshold:
                if held >= patience:
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
    target_cutoff_override: float | None = None,
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
    target_cutoff = (
        target_cutoff_override
        if target_cutoff_override is not None
        else BUTTER_LANDMARK_CUTOFF_HZ.get((sport_type, camera_angle), 4.0)
    )
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
