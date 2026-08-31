"""What the clip itself did to the measurement, said in numbers the athlete can act on.

The analysis already knows a great deal about how the recording went -- how many
pixels tall the athlete was, whether the pose model could hold left from right,
whether the timebase looked like slow motion. Until now that knowledge reached
the reader as at most two sentences of prose, fired only at the extreme end of
each scale, in an order nobody chose.

This turns it into an ordered report: what was measured, what it should have
been, and the one change that fixes it. Ordered by measured impact, so the first
line is the thing actually worth re-filming for.

Why this is worth more than another metric
------------------------------------------
A controlled experiment on one clip (2026-08-21): the same footage, the same
code, cropped tighter around the athlete.

    athlete height   255 px  ->  452 px
    leg_swap_pct     45.6%   ->   1.8%
    cadence          199.6   ->  154.8 spm  (the first was nonsense)

Nothing else in this codebase recovers a factor of twenty-five in tracking
quality. The pose model was never the weak link; the framing was. So framing
leads the report, and the advice is specific enough to act on ("zoom in, you
were at a third of the target") rather than a category ("good lighting").

That experiment is also why the "good" bar for framing sits where it does. The
clip that read 255 px was rated OK by the old threshold while its legs were
being swapped on half the frames -- a false all-clear on the single measurement
that mattered most. Two crops of one clip is calibration, not proof, so the bar
is set where the evidence points and the numbers behind it are stated here
rather than buried.
"""

from __future__ import annotations

from typing import Any

from app.services.video_analysis.biomechanics.tracking_diagnostics import (
    FRAMING_SMALL_PX,
    FRAMING_TINY_PX,
)

# The long-edge cap every clip is shrunk to before detection. Imported rather
# than restated: it is what decides whether a file had enough pixels to spend.
try:  # pragma: no cover - the runner owns this constant
    from app.services.video_analysis.runner import DETECT_MAX_LONG_EDGE
except Exception:  # noqa: BLE001 - runner pulls in the ML stack
    DETECT_MAX_LONG_EDGE = 1280

# Athlete height, in pixels of the frame the DETECTOR sees (after the long-edge
# cap -- a 4K clip and a 720p clip of the same framing score the same, which is
# the point: megapixels buy nothing here, filling the frame buys everything).
# The bars are the diagnostics module's, imported rather than restated so the
# verdict it logs and the report the athlete reads cannot drift apart.
FRAMING_GOOD_PX = FRAMING_SMALL_PX
FRAMING_WARN_PX = FRAMING_TINY_PX
# What to tell them to aim for. Above the "good" bar on purpose: a target set at
# the pass mark produces clips that sit on the pass mark.
FRAMING_TARGET_PX = 550

# Seconds of usable movement. Below MIN there are too few strides to choose a
# representative one; above LONG the frame budget forces decimation, which costs
# the time-based metrics their resolution.
DURATION_MIN_S = 3.0
DURATION_GOOD_S = 5.0
DURATION_LONG_S = 20.0

# Share of sampled frames the pose model actually found somebody in. Below
# this, the clip's LENGTH stops describing how much movement it holds -- the
# athlete is out of shot, or unfindable in it, for the rest. A little slack is
# normal at the ends of any clip, so this is not set near 1.
TRACKED_RATIO_WARN = 0.75

_IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}
_STATUS_ORDER = {"bad": 0, "warn": 1, "good": 2, "unknown": 3}


def _check(
    check_id: str, label: str, status: str, impact: str,
    measured: str, target: str, action: str = "",
) -> dict[str, Any]:
    return {
        "id": check_id, "label": label, "status": status, "impact": impact,
        "measured": measured, "target": target, "action": action,
    }


def _framing_check(
    framing: dict[str, Any] | None, frame_height: int | None = None,
) -> dict[str, Any]:
    """How many pixels of athlete the pose model got, and which fault cost them.

    Two very different faults land on the same number, and they have opposite
    fixes. Standing too far away wastes the frame you have; sending a
    downscaled export wastes the frame you had. Telling somebody to zoom in
    when their framing was fine and their FILE is 554 px tall is advice they
    cannot act on, so the two are separated by asking what the same framing
    would have measured at full detector resolution.
    """
    px = (framing or {}).get("subject_height_px")
    frac = (framing or {}).get("subject_height_frac")
    if not px:
        return _check(
            "framing", "How much of the frame you fill", "unknown", "high",
            "could not be measured", f"about {FRAMING_TARGET_PX} px tall",
        )
    pct = f" ({frac * 100:.0f}% of the frame height)" if frac else ""
    target = (
        f"{FRAMING_TARGET_PX} px or more -- roughly two thirds of the frame height"
    )
    if px >= FRAMING_GOOD_PX:
        return _check(
            "framing", "How much of the frame you fill", "good", "high",
            f"{px} px tall{pct}", target,
        )

    status = "bad" if px < FRAMING_WARN_PX else "warn"

    # Would this framing have been enough on a full-resolution file? If so the
    # athlete did nothing wrong and the pixels were lost before we saw them.
    low_res = bool(
        frac and frame_height
        and frame_height < DETECT_MAX_LONG_EDGE
        and frac * DETECT_MAX_LONG_EDGE >= FRAMING_GOOD_PX
    )
    if low_res:
        return _check(
            "framing", "Resolution of the file", status, "high",
            f"only {px} px of athlete{pct}, from a {frame_height} px tall file",
            f"{FRAMING_TARGET_PX} px or more of athlete",
            "Your framing is fine -- the file itself is small. This is a "
            "downscaled export rather than the original recording, and no "
            "amount of zooming can put the detail back. Send the clip straight "
            "from the camera roll, and avoid anything that re-compresses it on "
            "the way (messaging apps and 'optimised' exports are the usual "
            "culprits).",
        )

    zoom = FRAMING_TARGET_PX / float(px)
    return _check(
        "framing", "How much of the frame you fill", status, "high",
        f"{px} px tall{pct}", target,
        f"Zoom in about {zoom:.1f}x, or step closer, so you span most of the "
        "frame from head to feet. This does more for accuracy than any other "
        "change -- at your size the two legs are only a few pixels apart for "
        "much of the stride, which is what makes the skeleton jump between them.",
    )


def _camera_view_check(view: dict[str, Any] | None) -> dict[str, Any] | None:
    """Whether the clip was actually filmed from the side.

    Every angle in this pipeline assumes the camera is perpendicular to the
    movement; until now nothing checked, and nothing said. This row states the
    assumption and, when the evidence is there, confirms it.

    It NEVER warns, and the reason is honesty about calibration rather than
    caution for its own sake. The "side" verdict is validated on real footage
    -- both reference clips sit at a depth ratio of 0.89 against a 0.35
    threshold, on every one of their frames. The "not side" verdict is not
    validated on anything, because no clip filmed from behind exists to
    calibrate it; the only evidence is a synthetically rotated body, and
    tuning a threshold against a fake to then accuse an athlete of misfilming
    is the exact move this codebase refuses everywhere else.

    So: confirm when sure, stay quiet otherwise. The day a real off-axis clip
    exists, the negative half of ``detect_camera_view`` can be calibrated and
    this row can start warning.
    """
    if not view or view.get("view") is None:
        return None
    verdict, ratio = view.get("view"), view.get("ratio")
    target = "filmed square-on to your side, so the joints move across the lens"
    if verdict == "side":
        return _check(
            "camera_view", "The angle you filmed from", "good", "high",
            f"side on (depth ratio {ratio})", target,
        )
    measured = (
        f"could not be judged ({view.get('reason')})" if ratio is None
        else f"unclear -- depth ratio {ratio}, and a clean side view reads about 0.9"
    )
    return _check(
        "camera_view", "The angle you filmed from", "unknown", "high",
        measured, target,
    )


def _orientation_check(
    width: int | None, height: int | None, framing_ok: bool,
) -> dict[str, Any]:
    if not width or not height:
        return _check(
            "orientation", "Which way you held the phone", "unknown", "medium",
            "could not be read", "portrait (tall)",
        )
    portrait = height > width
    shape = "portrait" if portrait else ("square" if height == width else "landscape")
    measured = f"{shape} ({width}x{height})"
    target = "portrait -- a tall frame puts nearly twice the pixels on your legs"
    if portrait:
        return _check(
            "orientation", "Which way you held the phone", "good", "medium",
            measured, target,
        )
    if framing_ok:
        # Landscape is not wrong, it is wasteful -- and they did not waste it.
        return _check(
            "orientation", "Which way you held the phone", "good", "medium",
            measured + ", and you filled it anyway", target,
        )
    return _check(
        "orientation", "Which way you held the phone", "warn", "medium",
        measured, target,
        "Turn the phone upright. We shrink every clip to a fixed long edge "
        "before measuring, so a tall frame leaves about 1280 px of height for "
        "you to fill instead of 720 -- roughly double the detail on your legs, "
        "for free.",
    )


def _leg_identity_check(
    leg_swap_pct: float | None, sport_type: str,
) -> dict[str, Any] | None:
    """Whether the pose model could hold left leg apart from right.

    This is the SYMPTOM of bad framing rather than an independent fault, and it
    is reported as such -- but it is the symptom that decides whether cadence,
    ground contact and every stride metric mean anything, so it is worth its own
    line. Bike locks the side up front and never runs the leg pass, so there is
    nothing to report there.
    """
    if sport_type != "run" or leg_swap_pct is None:
        return None
    from app.services.video_analysis.biomechanics.confidence_scorer import THRESHOLDS

    warn = THRESHOLDS.get("leg_swap_pct_medium", 6.0)
    bad = THRESHOLDS.get("leg_swap_pct_low", 15.0)
    measured = f"legs confused on {leg_swap_pct:.1f}% of frames"
    target = "under 5% -- on a well-framed clip this sits near zero"
    if leg_swap_pct < warn:
        return _check(
            "leg_identity", "Telling your left leg from your right", "good",
            "high", measured, target,
        )
    return _check(
        "leg_identity", "Telling your left leg from your right",
        "bad" if leg_swap_pct >= bad else "warn", "high", measured, target,
        # Deliberately points at "how few pixels of you there were" rather than
        # at "framing": the row above blames the framing or the file depending
        # on which lost the pixels, and this line has to read correctly under
        # either.
        "This is almost always a symptom of how few pixels of you there were "
        "(the row above) rather than a fault of its own. If that was fine, the "
        "other two causes are a body reading as a dark silhouette -- keep the "
        "light behind the camera, not behind the athlete -- and a view almost "
        "exactly side-on, where a few degrees of angle helps separate the legs.",
    )


def _duration_check(
    duration_s: float | None, sampling_degraded: Any,
    tracked_ratio: float | None = None,
    slow_motion_factor: Any = None,
) -> dict[str, Any]:
    """How much USABLE movement the clip holds, which is not its length.

    Two things separate the two. The file can run ten seconds while the athlete
    is only in shot, and only findable, for four of them -- they enter late,
    leave early, or read as a silhouette the pose model loses. And a clip shot
    in slow motion holds a fraction of the running its duration advertises: at
    8x, twelve seconds of playback is a second and a half of stride. Every
    stride metric is built from real movement, so a check reading the
    container's duration and calling it generous is measuring the wrong thing
    twice over.
    """
    if not duration_s:
        return _check(
            "duration", "How long the clip runs", "unknown", "medium",
            "could not be read", f"{DURATION_GOOD_S:.0f}-10 s of steady movement",
        )
    target = f"{DURATION_GOOD_S:.0f}-10 s of steady movement"

    try:
        slow = float(slow_motion_factor or 1) or 1.0
    except (TypeError, ValueError):
        slow = 1.0
    real_s = duration_s / slow if slow > 1 else duration_s

    if slow > 1 and real_s < DURATION_GOOD_S:
        status = "bad" if real_s < DURATION_MIN_S else "warn"
        return _check(
            "duration", "How much running the clip holds", status, "medium",
            f"{duration_s:.1f} s of playback, but filmed at {slow:.0f}x slow "
            f"motion -- {real_s:.1f} s of actual running",
            target,
            "Slow motion stretches the picture, not the stride: there are only "
            f"{real_s:.1f} s of movement in here however long it takes to "
            "watch. Film at normal speed and the same recording length becomes "
            f"{slow:.0f} times the strides to measure.",
        )

    if tracked_ratio is not None and tracked_ratio < TRACKED_RATIO_WARN:
        tracked_s = real_s * tracked_ratio
        status = "bad" if tracked_s < DURATION_GOOD_S else "warn"
        return _check(
            "duration", "How much of the clip is usable", status, "medium",
            f"{duration_s:.1f} s long, but you were only trackable in "
            f"{tracked_s:.1f} s of it ({tracked_ratio * 100:.0f}% of frames)",
            target,
            "The rest is footage with nobody findable in it: either you are "
            "outside the frame, or the pose model lost you. Start recording "
            "once you are already in shot and running steadily, and keep the "
            "light behind the camera -- a body lit from behind reads as a "
            "silhouette, and a silhouette has no joints in it.",
        )

    duration_s = real_s
    measured = (
        f"{real_s:.1f} s of running" if slow > 1 else f"{duration_s:.1f} s"
    )
    if duration_s < DURATION_MIN_S:
        return _check(
            "duration", "How long the clip runs", "bad", "medium",
            measured, target,
            "Too short to contain several strides, so there is nothing to "
            "choose a representative one from and the timing metrics rest on "
            "one or two contacts.",
        )
    if duration_s < DURATION_GOOD_S:
        return _check(
            "duration", "How long the clip runs", "warn", "medium",
            measured, target,
            "Enough to measure, but not enough to average. A few more seconds "
            "makes every timing metric steadier.",
        )
    if sampling_degraded or duration_s > DURATION_LONG_S:
        return _check(
            "duration", "How long the clip runs", "warn", "medium",
            measured, target,
            "Long clips are decimated to stay inside the frame budget, which "
            "costs the time-based metrics their resolution. Trim to the best "
            f"{DURATION_GOOD_S:.0f}-10 s.",
        )
    return _check(
        "duration", "How long the clip runs", "good", "medium", measured, target,
    )


def _camera_motion_check(motion: dict[str, Any] | None) -> dict[str, Any] | None:
    """Whether the operator's own movement is inside the athlete's reading.

    Reported only when it was measured. The harm is specific: vertical
    oscillation is the rise and fall of the hips in the picture, and a camera
    held by someone standing still bobs at about the frequency a runner's steps
    do -- so it lands in the same number and no detrend removes it.

    The figure is the camera's bounce as a share of the hip movement, because
    a pixel count means nothing without knowing how big the athlete was.
    """
    if not motion:
        return None
    share = motion.get("vertical_share_of_hip_motion")
    if share is None:
        return None
    status = motion.get("verdict", "unknown")
    measured = (
        f"camera moved {share * 100:.0f}% as much as your hips did "
        f"({motion.get('vertical_bounce_px')} px)"
    )
    target = "under 15% -- a camera on a tripod or a fence post reads near zero"
    if status == "good":
        return _check(
            "camera_motion", "How steady the camera was", "good", "medium",
            measured, target,
        )
    return _check(
        "camera_motion", "How steady the camera was", status, "medium",
        measured, target,
        "Rest the phone on something, or lean against a wall. Vertical "
        "oscillation is measured as your hips rising and falling in the frame, "
        "so a camera that rises and falls with the operator's breathing or "
        "footsteps adds itself to the reading -- and it does it at roughly your "
        "step frequency, which is why it cannot simply be filtered out.",
    )


def _time_base_check(
    time_base_uncertain: Any, legs_unreliable: bool = False,
) -> dict[str, Any] | None:
    """Slow motion -- but only when nothing else already explains the symptom.

    The flag is raised by "cadence was unmeasurable while the frame-counted
    metrics computed", and slow motion is the usual cause. It is not the only
    one: legs the model could not tell apart destroy the cadence signal just as
    thoroughly. Reporting slow motion in that case sends somebody to check a
    setting that was never on, while the real fault sits two rows above with
    its own fix.
    """
    if not time_base_uncertain:
        return None
    if legs_unreliable:
        return _check(
            "time_base", "Timing metrics", "warn", "medium",
            "cadence could not be measured, so ground contact and flight time "
            "are withheld",
            "measurable cadence",
            "Most likely a consequence of the leg tracking above rather than a "
            "separate fault -- a stride the model cannot follow has no cadence "
            "to read. Worth checking slow motion is off, but fix the framing "
            "first.",
        )
    return _check(
        "time_base", "Playback speed", "bad", "medium",
        "reads as slow motion (or a very slow pace)",
        "normal-speed recording",
        "Cadence could not be measured, so ground-contact and flight time are "
        "withheld rather than shown wrong. Re-film with slow-motion off -- your "
        "joint angles are unaffected either way.",
    )


def build_capture_report(
    *,
    sport_type: str,
    duration_s: float | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    framing: dict[str, Any] | None = None,
    tracking_stability: dict[str, Any] | None = None,
    time_base_uncertain: Any = None,
    sampling_degraded: Any = None,
    camera_motion: dict[str, Any] | None = None,
    tracked_ratio: float | None = None,
    slow_motion_factor: Any = None,
    camera_view: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """An ordered, quantified account of how the recording limited the analysis.

    Every check names what was measured, what it should have been, and the one
    change that fixes it. Checks are ordered worst-first within impact, so the
    first entry is the thing worth re-filming for; ``headline`` is that entry's
    action, pulled out for a caller with room for a single sentence.
    """
    tracking = tracking_stability or {}
    framing = framing if framing is not None else (tracking.get("framing") or {})

    framing_check = _framing_check(framing, frame_height)
    framing_ok = framing_check["status"] == "good"
    legs = _leg_identity_check(tracking.get("leg_swap_pct"), sport_type)
    legs_unreliable = bool(legs and legs["status"] in ("warn", "bad"))

    checks = [
        framing_check,
        _camera_view_check(camera_view),
        _orientation_check(frame_width, frame_height, framing_ok),
        legs,
        _camera_motion_check(camera_motion),
        _duration_check(
            duration_s, sampling_degraded, tracked_ratio, slow_motion_factor,
        ),
        _time_base_check(time_base_uncertain, legs_unreliable),
    ]
    checks = [c for c in checks if c is not None]
    checks.sort(key=lambda c: (
        _STATUS_ORDER.get(c["status"], 9), _IMPACT_ORDER.get(c["impact"], 9),
    ))

    worst = checks[0]["status"] if checks else "unknown"
    verdict = {"bad": "poor", "warn": "fair", "good": "good"}.get(worst, "unknown")
    headline = next(
        (c["action"] for c in checks if c["status"] in ("bad", "warn") and c["action"]),
        "",
    )
    return {
        "verdict": verdict,
        "headline": headline,
        "checks": checks,
        "problem_count": sum(1 for c in checks if c["status"] in ("bad", "warn")),
    }


__all__ = [
    "DURATION_GOOD_S",
    "FRAMING_GOOD_PX",
    "FRAMING_TARGET_PX",
    "FRAMING_WARN_PX",
    "build_capture_report",
]
