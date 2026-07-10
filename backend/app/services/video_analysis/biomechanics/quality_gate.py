"""Partial Analysis quality gate for swim videos.

Three trigger paths, ANY of which switches an analysis into
"partial" mode:

1. ``unknown_phase_pct >= 70``      -- phase detection broke down.
2. Majority of tracked angles       -- landmark breakdown.
   have ``nan_pct > 60``.
3. ``valid_frames / frames < 40%``  -- wholesale frame failure.

Partial mode hides the technique score and LLM coaching report,
classifies metrics into reliable / unreliable buckets, and filters
detected issues by reliability.

This module is a pure-function gate; the pipeline wires the result
into the summary dict.
"""

from __future__ import annotations

from typing import Any

# Quality gate thresholds per camera angle.
#
# Above-water: strict. Head + limbs are usually visible, so if a
# majority (>50%) of the tracked angles is noisy that signals a
# real tracking failure and we should gate.
#
# Under-water: relaxed on landmark-quality triggers. Underwater
# clips routinely have noisy lower-body angles (legs in splash,
# flip-turn motion, feet leaving frame) while upper-body
# tracking -- which is what the under-water metrics actually
# depend on -- stays clean. Only gate if 80%+ of angles are
# noisy, OR valid_frames_ratio is extremely low.
#
# Default (missing/unknown camera angle): fall back to the
# stricter above-water thresholds. A misconfigured call should
# err on the side of gating rather than silently letting junk
# through.
#
# Unknown-phase and per-angle NaN thresholds are identical across
# modes; only the aggregation thresholds (majority ratio, minimum
# frame ratio) differ.
#
# TODO: move to per-sport critical-angle lists. For under-water
# swim the elbow / shoulder angles actually determine metric
# reliability; leg noise is secondary. A weighted/critical-angle
# gate would be a cleaner fix than a global ratio bump.
#
# Threshold rationale for ``min_lower_body_ratio``:
#
# As of 2026-04-30, bike side-view and run side-view use
# UNILATERAL lower-body measurement (camera-side knee + ankle
# only -- 2 indices instead of 4). This matches the unilateral
# analyzer architecture and avoids false-triggering the gate on
# legitimate clips where the far-side leg is occluded by the
# rider's body / bike frame.
#
# With unilateral measurement, per-frame match probability is
# substantially higher than with the previous 4-index AND-gate.
# Side-view thresholds were therefore raised:
#   - bike_default: 0.4 -> 0.6
#   - bike_tt_aero: 0.5 -> 0.55
#   - run_side:     0.5 -> 0.6
#
# Empirical expectations (to be validated post-deploy):
#   - Healthy clean side-view clip:        0.80 - 0.95 (passes)
#   - Marginal clip (slight crop, blur):   0.40 - 0.70 (some
#     legit clips may sit near threshold; calibrate)
#   - Bad clip (camera too far, cropped):  < 0.40 (gate fires)
#
# Bilateral sports are unaffected -- swim and the rear-view
# profiles still use the 4-index AND-gate.
#
# If real-world deploy shows legitimate clips false-triggering
# the gate, lower thresholds in 0.05 increments. If garbage clips
# slip through to full-mode analysis, raise. Document final
# values inline.
QUALITY_GATE_THRESHOLDS: dict[str, dict[str, float]] = {
    "above_water": {
        "unknown_phase_pct": 70.0,
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.5,
        "min_valid_frames_ratio": 0.4,
        # Upper-body floor for swim. Strict for above-water because
        # arm tracking is the primary signal (catch, recovery,
        # entry are all upper-body) and noise here invalidates the
        # bulk of the report.
        "min_upper_body_ratio": 0.4,
    },
    "under_water": {
        "unknown_phase_pct": 70.0,
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.8,
        "min_valid_frames_ratio": 0.3,
        # Slightly relaxed for under-water because submerged clips
        # naturally have more occlusion/distortion. 30% still
        # catches the pathological cases (a 2.3% upper-body video
        # with phantom strokes was the production trigger for this
        # criterion).
        "min_upper_body_ratio": 0.3,
    },
    # Running side-view profile (Sprint R2). The "phase" channel is
    # gait phase classification rather than swim stroke phases;
    # current detector never returns UNKNOWN so the criterion sits
    # idle today, but it stays wired in for future detectors that
    # may emit an "uncertain" gait phase. Critical-region channel
    # for running is the LOWER body (knees + ankles drive cadence
    # and swing metrics) -- min_lower_body_ratio replaces the swim
    # min_upper_body_ratio.
    "run_side": {
        "unknown_phase_pct": 60.0,
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.5,
        "min_valid_frames_ratio": 0.4,
        # Unilateral lower-body measurement (camera-side knee +
        # ankle only). Raised from 0.5 because the 2-index
        # detection probability is structurally higher than the
        # old 4-index AND-gate. See dict-level comment above.
        "min_lower_body_ratio": 0.6,
    },
    "run_rear": {
        # Rear-view running uses pelvic-stability metrics that need
        # fewer phases / less full-stride coverage than side-view
        # cadence. Looser thresholds across the board.
        "unknown_phase_pct": 70.0,
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.6,
        "min_valid_frames_ratio": 0.4,
        "min_lower_body_ratio": 0.4,
    },
    # Cycling side-view profiles (Sprint B2). Bike has no per-frame
    # phase classification (BDC/TDC is detected once per pedal
    # stroke), so unknown_phase_pct is unused. Critical region is
    # the LOWER body -- knee/hip drive every fitting metric.
    # Upper body sits relatively static and is not a measurement
    # quality signal, so min_upper_body_ratio is intentionally
    # absent.
    "bike_default": {
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.6,
        "min_valid_frames_ratio": 0.4,
        # Unilateral lower-body measurement (camera-side knee +
        # ankle only). Raised from 0.4 because the 2-index
        # detection probability is structurally higher than the
        # old 4-index AND-gate. See dict-level comment above.
        "min_lower_body_ratio": 0.6,
    },
    "bike_tt_aero": {
        # Tucked aero positions (tt_aero / triathlon) hide the
        # upper body behind the cockpit, but only the lower body
        # determines knee/hip/saddle measurements. Looser
        # angles-noise bar reflects that more far-side limb
        # noise is normal here.
        #
        # min_lower_body_ratio is unilateral (camera-side knee +
        # ankle), raised from 0.5 -> 0.55. Slightly below
        # bike_default because tt_aero clips already have more
        # ambient pose noise (forward fold, helmet-down view) so
        # the gate stays one notch more permissive on the legs
        # too -- consistent with the looser majority_angles_ratio
        # this profile already runs.
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.7,
        "min_valid_frames_ratio": 0.4,
        "min_lower_body_ratio": 0.55,
    },
    "default": {
        "unknown_phase_pct": 70.0,
        "nan_pct_per_angle": 60.0,
        "majority_angles_ratio": 0.5,
        "min_valid_frames_ratio": 0.4,
        "min_upper_body_ratio": 0.4,
    },
}


def _resolve_profile_key(
    sport: str | None,
    camera_angle: str | None,
    camera_view: str | None,
    cycling_position: str | None = None,
) -> str:
    """Pick the threshold profile key for (sport, camera, position).

    Swim resolves on ``camera_angle`` (above_water / under_water).
    Run resolves on ``camera_view`` (rear -> run_rear; anything
    else, including None, -> run_side).
    Bike resolves on ``cycling_position``: tt_aero / triathlon
    (tucked aero) -> bike_tt_aero; anything else (road_hoods,
    road_drops, casual, None) -> bike_default. Bike rear-view
    (PelvicStabilityAnalyzer) is not currently gate-managed and
    the caller is expected to skip evaluate_quality_gate for it.
    Anything else falls back to ``default`` (strict swim-style).
    """
    if sport == "swim":
        return camera_angle if camera_angle in QUALITY_GATE_THRESHOLDS else "default"
    if sport == "run":
        return "run_rear" if camera_view == "rear" else "run_side"
    if sport == "bike":
        return (
            "bike_tt_aero"
            if cycling_position in ("tt_aero", "triathlon")
            else "bike_default"
        )
    # Backward compat: callers that still pass only camera_angle
    # (older swim-only call sites).
    if camera_angle and camera_angle in QUALITY_GATE_THRESHOLDS:
        return camera_angle
    return "default"


def _format_unknown_phase_reason(unknown_phase_pct: float, _angle: str | None) -> str:
    """Why a high unknown-phase rate triggered the gate.

    Stroke phase is an internal classification (catch / pull / push
    / recovery / entry). Users don't think in those terms; they
    think in "stroke movements". Mode-agnostic copy is fine here
    because the failure mode -- BlazePose can't decide where the
    arm is in the cycle -- happens for the same reasons whether
    the camera is above or below water.
    """
    return (
        f"{unknown_phase_pct:.0f}% of stroke movements couldn't be "
        f"clearly identified -- this can happen with unusual "
        f"swimming styles or when the body position is hard to read."
    )


def _format_landmark_breakdown_reason(
    high_nan_count: int, total_angles: int, _angle: str | None,
) -> str:
    """Why the majority-of-angles trigger fired."""
    return (
        f"Body tracking was inconsistent -- {high_nan_count} of "
        f"{total_angles} body angles had too much noise to measure "
        f"reliably."
    )


def _format_frame_failure_reason(
    valid_ratio: float,
    _angle: str | None = None,
    sport: str | None = None,
) -> str:
    """Why valid-frames-ratio dropped below the floor.

    Sport-aware so the noun matches what the user uploaded.
    Pre-Sprint-B2 callers passed only ``camera_angle`` and got the
    swim-flavoured copy; ``sport`` is now the canonical signal --
    ``camera_angle`` is retained as a positional alias for backward
    compat but is unused.
    """
    pct = round(valid_ratio * 100)
    if sport == "bike":
        return (
            f"The rider was clearly visible in only {pct}% of the "
            f"video -- too little for reliable measurements."
        )
    if sport == "run":
        return (
            f"The runner was clearly visible in only {pct}% of the "
            f"video -- too little for reliable measurements."
        )
    if sport == "swim":
        return (
            f"The swimmer was clearly visible in only {pct}% of the "
            f"video -- too little for reliable measurements."
        )
    return (
        f"The subject was clearly visible in only {pct}% of the "
        f"video -- too little for reliable measurements."
    )


def _format_lower_body_reason(
    ratio: float,
    camera_view: str | None,
    sport: str | None = None,
) -> str:
    """Why lower-body detection fell below the floor.

    Sport-aware: bike adds a "rider / bike frame" framing because
    the cycling near-side analyzer's measurement gap is structural
    (far-side leg occluded by the bike). Run keeps its original
    side / rear-view branching. Swim's lower-body floor is rare
    but covered for completeness. Unknown sport falls back to
    generic "subject" copy so the message is still grammatical.
    """
    pct = round(ratio * 100)

    if sport == "bike":
        return (
            f"Your knees and ankles were clearly visible in only "
            f"{pct}% of the video. The rider may have been too far "
            f"from the camera, or the legs were partially occluded "
            f"by the bike frame."
        )

    if sport == "run":
        if camera_view == "rear":
            return (
                f"Your knees and ankles were clearly visible in only "
                f"{pct}% of the video. The view may have been partly "
                f"blocked by the runner's torso, or the camera was "
                f"too low to see the lower legs."
            )
        return (
            f"Your knees and ankles were clearly visible in only "
            f"{pct}% of the video. The runner may have been too far "
            f"from the camera, or the lower legs were cropped out "
            f"of frame."
        )

    if sport == "swim":
        return (
            f"Your legs were clearly visible in only {pct}% of the "
            f"video. The swimmer may have been too far from the "
            f"camera, or the legs partially obscured."
        )

    return (
        f"The lower body was clearly visible in only {pct}% of the "
        f"video. The subject may have been too far from the camera "
        f"or partially occluded."
    )


def _format_upper_body_reason(
    ratio: float, camera_angle: str | None,
) -> str:
    """Why upper-body detection fell below the floor.

    Mode-aware because the typical *cause* differs:
        - under_water: distance, bubbles, swimmer leaving frame.
        - above_water: surface refraction obscuring the body, or
          distance.
    The user already chose their camera angle at upload; calling
    out the relevant cause is more useful than re-stating that
    "stroke metrics need arm tracking" (which they already know,
    that's why they uploaded the video).
    """
    pct = round(ratio * 100)
    if camera_angle == "under_water":
        return (
            f"Your arms were clearly visible in only {pct}% of "
            f"the video. The swimmer may have been too far from "
            f"the camera, partially obscured by bubbles, or moving "
            f"out of frame."
        )
    if camera_angle == "above_water":
        return (
            f"Your arms were clearly visible in only {pct}% of "
            f"the video. This often happens when the view is "
            f"partially obscured by the water surface, or the "
            f"swimmer is too far from the camera."
        )
    return (
        f"Your arms were clearly visible in only {pct}% of the "
        f"video, which limits the measurements we can take."
    )


def evaluate_quality_gate(
    unknown_phase_pct: float | None,
    angle_statistics: dict[str, dict[str, Any]],
    valid_frames: int,
    frames_processed: int,
    camera_angle: str | None = None,
    upper_body_detection_ratio: float | None = None,
    sport: str | None = None,
    camera_view: str | None = None,
    lower_body_detection_ratio: float | None = None,
    cycling_position: str | None = None,
    bdc_present: bool | None = None,
    tdc_present: bool | None = None,
) -> dict[str, Any]:
    """Decide whether the Partial Analysis gate fires.

    Args:
        unknown_phase_pct: fraction (0-100) of frames whose stroke
            phase (swim) or gait phase (run) was ``unknown``.
            ``None`` means phase data wasn't available and this
            trigger path is skipped.
        angle_statistics: per-angle stats dict with ``nan_pct``
            entries, as produced by
            ``SportAnalyzer.compute_angle_statistics``.
        valid_frames: frames that produced at least one trusted
            angle (used for the wholesale-failure trigger).
        frames_processed: total frames the analyzer processed.
        camera_angle: swim only -- ``"above_water"`` /
            ``"under_water"``. Selects the swim threshold profile.
        upper_body_detection_ratio: fraction (0-1) of frames where
            shoulders+elbows+wrists were all visible. Critical
            channel for swim stroke metrics. ``None`` skips the
            criterion.
        sport: ``"swim"`` / ``"run"`` / ``"bike"`` / ``None``.
            With ``camera_view`` selects the run threshold
            profile (``run_side`` / ``run_rear``). Optional for
            backward compat -- pre-Sprint-R2 callers passed only
            ``camera_angle`` and got swim/default behaviour.
        camera_view: ``"side"`` / ``"rear"`` for run.
        lower_body_detection_ratio: fraction (0-1) of frames
            where knees+ankles were all visible. Critical channel
            for run cadence + swing metrics. ``None`` skips the
            criterion.

    Returns dict with:
        ``triggered``    -- bool.
        ``reasons``      -- user-friendly strings per trigger.
        ``criteria``     -- machine-readable values for diagnostics.
    """
    profile_key = _resolve_profile_key(
        sport, camera_angle, camera_view, cycling_position,
    )
    thresholds = QUALITY_GATE_THRESHOLDS.get(
        profile_key, QUALITY_GATE_THRESHOLDS["default"],
    )
    reasons: list[str] = []

    # --- criterion 1: phase-detection breakdown -----------------
    phase_triggered = False
    if isinstance(unknown_phase_pct, (int, float)):
        if unknown_phase_pct >= thresholds["unknown_phase_pct"]:
            phase_triggered = True
            reasons.append(_format_unknown_phase_reason(
                unknown_phase_pct, camera_angle,
            ))

    # --- criterion 2: landmark breakdown ------------------------
    total_angles = 0
    high_nan_count = 0
    for stats in angle_statistics.values():
        nan_pct = stats.get("nan_pct") if isinstance(stats, dict) else None
        if not isinstance(nan_pct, (int, float)):
            continue
        total_angles += 1
        if nan_pct > thresholds["nan_pct_per_angle"]:
            high_nan_count += 1

    landmark_triggered = False
    if total_angles > 0:
        ratio = high_nan_count / total_angles
        if ratio >= thresholds["majority_angles_ratio"]:
            landmark_triggered = True
            reasons.append(_format_landmark_breakdown_reason(
                high_nan_count, total_angles, camera_angle,
            ))

    # --- criterion 3: wholesale frame failure -------------------
    frame_triggered = False
    if frames_processed > 0:
        valid_ratio = valid_frames / frames_processed
        if valid_ratio < thresholds["min_valid_frames_ratio"]:
            frame_triggered = True
            reasons.append(_format_frame_failure_reason(
                valid_ratio, camera_angle, sport=sport,
            ))
    else:
        valid_ratio = 0.0

    # --- criterion 4: upper-body landmark visibility (swim) -----
    # Stroke metrics (catch, EVF, entry, recovery) all depend on
    # arms being tracked. When upper-body detection is below the
    # threshold the report is fundamentally untrustworthy
    # regardless of what the per-angle stats say.
    upper_body_triggered = False
    if isinstance(upper_body_detection_ratio, (int, float)):
        floor = thresholds.get("min_upper_body_ratio")
        if floor is not None and upper_body_detection_ratio < floor:
            upper_body_triggered = True
            reasons.append(_format_upper_body_reason(
                upper_body_detection_ratio, camera_angle,
            ))

    # --- criterion 5: lower-body landmark visibility (run / bike) ---
    # Cadence detection + swing-phase knee metrics for run, and
    # knee/hip/saddle metrics for bike, need knees and ankles
    # tracked across most of the clip. The 30-50% floor catches
    # "athlete is too distant or cropped" without rejecting clips
    # that lose a foot for a few frames.
    lower_body_triggered = False
    if isinstance(lower_body_detection_ratio, (int, float)):
        floor = thresholds.get("min_lower_body_ratio")
        if floor is not None and lower_body_detection_ratio < floor:
            lower_body_triggered = True
            reasons.append(_format_lower_body_reason(
                lower_body_detection_ratio, camera_view, sport=sport,
            ))

    # --- criterion 6: pedal-stroke detection failure (bike) -----
    # Bike has no per-frame phase classification; the analog
    # signal is BDC/TDC detection. When neither knee_at_bdc nor
    # knee_at_tdc was measurable, the analyzer never observed a
    # complete pedal stroke -- fitting metrics that depend on
    # those values are ALL phantom. Only fires when we have
    # explicit "both absent" evidence; passing None for both
    # means "caller doesn't know" and skips this criterion.
    bdc_tdc_triggered = False
    if bdc_present is False and tdc_present is False:
        bdc_tdc_triggered = True
        reasons.append(
            "Pedal stroke detection failed -- couldn't measure "
            "knee extension at the top or bottom of the pedal "
            "cycle. The rider may not be pedaling, or the frame "
            "range is too short for stable measurement."
        )

    triggered = (
        phase_triggered
        or landmark_triggered
        or frame_triggered
        or upper_body_triggered
        or lower_body_triggered
        or bdc_tdc_triggered
    )

    return {
        "triggered": triggered,
        "reasons": reasons,
        "criteria": {
            "unknown_phase_pct": unknown_phase_pct,
            "angles_high_nan_count": high_nan_count,
            "total_angles": total_angles,
            "valid_frames_ratio": round(valid_ratio, 3),
            "upper_body_detection_ratio": (
                round(upper_body_detection_ratio, 3)
                if isinstance(upper_body_detection_ratio, (int, float))
                else None
            ),
            "lower_body_detection_ratio": (
                round(lower_body_detection_ratio, 3)
                if isinstance(lower_body_detection_ratio, (int, float))
                else None
            ),
            "camera_angle": camera_angle,
            "camera_view": camera_view,
            "cycling_position": cycling_position,
            "bdc_present": bdc_present,
            "tdc_present": tdc_present,
            "sport": sport,
            "threshold_profile": profile_key,
        },
    }


def compute_landmark_quality_pct(summary: dict[str, Any]) -> float:
    """Extract the overall landmark detection percentage from
    summary, falling back to 0 when absent.

    The pipeline writes ``summary["landmark_quality"] = {
    "overall_pct": ..., "confidence": ..., "regions": {...}}`` at
    line 658. This helper just reads it.
    """
    lq = summary.get("landmark_quality")
    if isinstance(lq, dict):
        val = lq.get("overall_pct")
        if isinstance(val, (int, float)):
            return float(val)
    return 0.0


# Every detected-issue type produced by SwimmingAnalyzer, mapped to
# its primary source metric. Used to filter issues when the source
# metric has been reclassified as unreliable.
ISSUE_TO_METRIC: dict[str, str] = {
    # Under-water
    "dropped_elbow": "elbow_at_catch_avg",
    "poor_evf": "evf_angle_avg",
    "poor_body_line": "body_line_angle_avg",
    # Above-water
    "entry_too_narrow": "entry_angle_avg",
    "entry_too_wide": "entry_angle_avg",
    "straight_arm_recovery": "recovery_elbow_angle_avg",
    "head_too_high": "head_alignment_avg",
    "strongly_unilateral_breathing": "breathing_side",
    # Both
    "poor_streamline": "streamline_avg",
    "kick_too_wide": "kick_amplitude_avg",
    "kick_too_narrow": "kick_amplitude_avg",
}


def filter_issues_by_reliability(
    issues: list[dict[str, Any]],
    reliable_keys: set[str],
) -> list[dict[str, Any]]:
    """Drop issues whose source metric isn't in the reliable set.

    .. deprecated::
        Superseded by :func:`filter_visually_verifiable_issues` in
        the Partial Analysis v2 pipeline. Kept in module surface
        for tests and any future FULL-mode use. The new gated
        pipeline calls ``filter_visually_verifiable_issues``
        instead, which uses empirical, permissive thresholds
        instead of a reliability-class mapping.

    Temporal fatigue issues (``render_hint == "temporal"``,
    ``type`` suffix ``_declines_under_fatigue``) are always dropped
    in partial mode -- their early/mid/late decomposition depends
    on both reliable per-frame aggregation AND stable phase
    anchoring, and the copy is harder to qualify briefly.
    """
    kept: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        itype = issue.get("type", "")
        if issue.get("render_hint") == "temporal" or itype.endswith("_declines_under_fatigue"):
            # Temporal issues go through a richer pipeline; skip in
            # partial mode rather than risk misleading framing.
            continue
        source_metric = ISSUE_TO_METRIC.get(itype)
        if source_metric is None:
            # Unknown issue type -- conservative: keep it only if the
            # type itself appears to reference a reliable metric name.
            if any(key in itype for key in reliable_keys):
                kept.append(issue)
            continue
        if source_metric in reliable_keys:
            kept.append(issue)
    return kept


# Permissive thresholds for "visually obvious" findings. These are
# calibrated so that at the listed value the qualitative finding is
# correct regardless of BlazePose noise -- e.g. a head pitched 30°
# above horizontal is unmistakable on the video, even if the exact
# number is off by 10°. See Partial Analysis v2 spec, Change A3.
VISUALLY_VERIFIABLE_THRESHOLDS: dict[str, float] = {
    # Swim
    "head_alignment_avg": 30.0,  # head_too_high
    "streamline_avg": 15.0,      # poor_streamline
    # Run
    "trunk_lean_avg": 15.0,      # excessive_forward_lean
    "knee_min": 130.0,           # insufficient_knee_drive (knee_min HIGH = no drive)
}


# Bike thresholds use both min and max because the same source
# metric can trigger different findings in opposite directions
# (e.g. saddle_too_low = knee_at_bdc LOW, saddle_too_high = knee
# HIGH). Two-sided guards stay numerically conservative -- the
# qualitative finding ("rider's leg looks bent at bottom of
# stroke" / "leg locks out") only holds at extreme deviations
# regardless of measurement noise.
BIKE_VISUALLY_VERIFIABLE_THRESHOLDS: dict[str, dict[str, float]] = {
    "saddle_too_low":      {"metric": "knee_at_bdc", "max": 110.0},  # knee bent at BDC
    "saddle_too_high":     {"metric": "knee_at_bdc", "min": 165.0},  # knee locks out
    "trunk_too_aggressive":{"metric": "trunk_angle_avg", "max": 15.0},
    "trunk_too_upright":   {"metric": "trunk_angle_avg", "min": 70.0},
}

# Map issue type -> (source metric key, threshold). Only issues in
# this map are even considered for the visually-verifiable filter;
# everything else is suppressed under the gate. The semantic is
# "value >= threshold" (the qualitative finding holds when the
# measurement is clearly above the floor regardless of noise).
# Cadence and vertical oscillation are deliberately NOT in this
# map: they are precise measurements not verifiable from the
# video alone, so they're suppressed entirely under the gate.
_VISUALLY_VERIFIABLE_ISSUE_RULES: dict[str, tuple[str, float]] = {
    # Swim
    "head_too_high": ("head_alignment_avg", VISUALLY_VERIFIABLE_THRESHOLDS["head_alignment_avg"]),
    "poor_streamline": ("streamline_avg", VISUALLY_VERIFIABLE_THRESHOLDS["streamline_avg"]),
    # Run
    "excessive_forward_lean": ("trunk_lean_avg", VISUALLY_VERIFIABLE_THRESHOLDS["trunk_lean_avg"]),
    "insufficient_knee_drive": ("knee_min", VISUALLY_VERIFIABLE_THRESHOLDS["knee_min"]),
}


def filter_visually_verifiable_issues(
    issues: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Gate-aware issue filter for Partial Analysis v2.

    Under the quality gate the user can't trust precise angle
    values, but some qualitative findings remain obvious from the
    video alone. Keep only those: an issue survives iff its source
    metric exists in the summary and clears a permissive threshold
    (e.g. head pitch >= 30°, streamline deviation >= 15°). Every
    other issue type is dropped, including all temporal/fatigue
    issues which depend on aggregation precision we don't have.

    Args:
        issues: raw list produced by ``analyzer.detect_issues()``.
        summary: swim summary dict, before reliable/unreliable
            split. Must carry the source metric values.

    Returns:
        Filtered list, preserving the original dict shape. Callers
        should also strip per-issue ``value`` / ``early_value`` /
        ``late_value`` before display, since we don't stand behind
        the numbers even when the qualitative finding holds.
    """
    kept: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        itype = issue.get("type", "")
        if issue.get("render_hint") == "temporal" or itype.endswith("_declines_under_fatigue"):
            continue

        # Bike rules use {metric, min?, max?} schema because some
        # issue types fire on the LOW side of a metric and others
        # on the HIGH side of the same metric (e.g. saddle_too_low
        # vs saddle_too_high both key off knee_at_bdc).
        bike_rule = BIKE_VISUALLY_VERIFIABLE_THRESHOLDS.get(itype)
        if bike_rule is not None:
            metric_key = bike_rule["metric"]
            value = summary.get(metric_key)
            if not isinstance(value, (int, float)):
                continue
            lo = bike_rule.get("min")
            hi = bike_rule.get("max")
            # "min" semantics = surface only when value >= min
            # (e.g. knee >= 165 -> saddle clearly too high).
            # "max" semantics = surface only when value <= max
            # (e.g. knee <= 110 -> saddle clearly too low).
            if lo is not None and value >= lo:
                kept.append(issue)
                continue
            if hi is not None and value <= hi:
                kept.append(issue)
                continue
            continue

        rule = _VISUALLY_VERIFIABLE_ISSUE_RULES.get(itype)
        if rule is None:
            continue
        metric_key, threshold = rule
        value = summary.get(metric_key)
        if not isinstance(value, (int, float)):
            continue
        if value >= threshold:
            kept.append(issue)
    return kept
