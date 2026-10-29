"""Single-photo body position analysis with scoring and annotated thumbnail.

Unlike video analysis (multi-frame, temporal, background task, DB storage),
photo analysis is:
- Single frame (static_image_mode=True in MediaPipe)
- Synchronous (returns JSON directly)
- No DB storage (instant feedback)
- Reuses the same angle calculation functions from the video pipeline
"""

import base64
import io
import math
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import structlog
from PIL import Image, ImageOps

from app.services.video_analysis import overlay_style

from app.services.video_analysis.biomechanics.angle_calculator import (
    MIN_LANDMARK_VISIBILITY,
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_body_rotation,
    calculate_forearm_tilt_2d,
    calculate_forward_sign,
    calculate_segment_to_vertical,
    calculate_shank_foot_angle_2d,
    calculate_signed_segment_to_vertical,
)
from app.services.video_analysis.biomechanics.cycling_positions import (
    AERO_POSITIONS,
    AERO_TRUNK_EXTREME_DEG,
    HIP_MEDICAL_MIN_DEG,
    AERO_TRUNK_LOW_CONCERN,
    SHOULDER_CLOSED_FORGIVENESS,
    SHOULDER_CLOSED_OK_POSITIONS,
    get_cycling_reference,
    get_medical_warnings,
    get_position_label,
)
from app.services.video_analysis.biomechanics.head_silhouette import (
    BACK_TOP_POINT,
    HELMET_TOP_POINT,
    KONA_PRO_FREE_UP_TO,
    KONA_PRO_MEDIAN,
    KONA_PRO_RANGE,
    apply_point_corrections,
    graded_helmet,
    helmet_above_back,
    top_contour,
)
from app.services.video_analysis.pipeline import _draw_dashed_line
from app.services.video_analysis.biomechanics.sport_configs import (
    RUNNING_REFERENCE,
    SWIMMING_REFERENCE,
)
from app.core.config import settings

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Angle landmark triplets per sport
# ---------------------------------------------------------------------------

# Running: near-side only, unprefixed keys (matches running_analyzer.py)
RUNNING_PHOTO_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "left": {
        "knee": (23, 25, 27),   # LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
        "hip": (11, 23, 25),    # LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE
        "elbow": (11, 13, 15),  # LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
    },
    "right": {
        "knee": (24, 26, 28),
        "hip": (12, 24, 26),
        "elbow": (12, 14, 16),
    },
}

# Run ankle is NOT a vertex triplet: it is the angle between the shank axis
# (ankle -> knee) and the foot axis (heel -> toe). Neutral = 90 deg,
# midstance dorsiflexion ~70, toe-off plantarflexion ~115. Axis form is
# robust to BlazePose drifting the ankle landmark up the shin on bulky
# running shoes (matches RUNNING_ANKLE_LANDMARKS in running_analyzer.py).
# Tuple: (knee, ankle, heel, toe).
RUNNING_PHOTO_ANKLE: dict[str, tuple[int, int, int, int]] = {
    "left":  (25, 27, 29, 31),
    "right": (26, 28, 30, 32),
}

RUNNING_TRUNK_LANDMARKS: dict[str, tuple[int, int]] = {
    "left": (11, 23),
    "right": (12, 24),
}

# Cycling: near-side only, unprefixed keys (matches cycling_analyzer.py)
CYCLING_PHOTO_ANGLES: dict[str, dict[str, tuple[int, int, int]]] = {
    "left": {
        "knee": (23, 25, 27),
        "hip": (11, 23, 25),
        "ankle": (25, 27, 31),   # KNEE, ANKLE, FOOT_INDEX
        "elbow": (11, 13, 15),
        "shoulder": (13, 11, 23),  # ELBOW -> SHOULDER -> HIP
    },
    "right": {
        "knee": (24, 26, 28),
        "hip": (12, 24, 26),
        "ankle": (26, 28, 32),
        "elbow": (12, 14, 16),
        "shoulder": (14, 12, 24),  # ELBOW -> SHOULDER -> HIP
    },
}

# Swimming: both sides, prefixed keys, uses 3D angles
SWIMMING_PHOTO_ANGLES: dict[str, tuple[int, int, int]] = {
    "left_shoulder": (23, 11, 13),   # HIP -> SHOULDER -> ELBOW
    "right_shoulder": (24, 12, 14),
    "left_elbow": (11, 13, 15),      # SHOULDER -> ELBOW -> WRIST
    "right_elbow": (12, 14, 16),
    "left_knee": (23, 25, 27),       # HIP -> KNEE -> ANKLE
    "right_knee": (24, 26, 28),
}

# ---------------------------------------------------------------------------
# Human-readable labels
# ---------------------------------------------------------------------------

_ANGLE_LABELS: dict[str, dict[str, str]] = {
    "run": {
        "knee": "Knee Angle",
        "hip": "Hip Angle",
        "ankle": "Ankle Angle",
        "elbow": "Elbow Angle",
        "trunk": "Trunk Lean",
    },
    "bike": {
        "knee": "Knee Angle",
        "hip": "Hip Angle",
        "ankle": "Ankle Angle",
        "elbow": "Elbow Angle",
        "trunk": "Trunk Angle",
        "shoulder": "Shoulder Angle",
        "forearm_tilt": "Forearm Tilt",
        "head_alignment": "Head Tuck",
        "helmet_above_back": "Helmet above back, % torso",
        "pelvic_ratio": "Pelvic Rotation",
    },
    "swim": {
        "left_shoulder": "Left Shoulder",
        "right_shoulder": "Right Shoulder",
        "left_elbow": "Left Elbow",
        "right_elbow": "Right Elbow",
        "left_knee": "Left Knee",
        "right_knee": "Right Knee",
        "body_rotation": "Body Rotation",
        "streamline": "Streamline",
    },
}

# ---------------------------------------------------------------------------
# Optimal ranges per sport
# ---------------------------------------------------------------------------


# Hip band per detected gait phase, in the measured convention (internal
# shoulder-hip-knee angle, 180 = trunk and thigh in line). ``None`` = a still
# cannot support a verdict at this instant, so the joint is measured and shown
# but never scored -- the same contract the bike side uses for a hip whose
# crank position could not be resolved.
#
# The band inverts across the cycle, which is the whole reason this exists: a
# single (150, 180) band applied to every frame is a TOE-OFF band, and judging
# a knee-drive frame against it reported ~123 deg -- textbook peak hip flexion
# -- as "hip too closed, restricts propulsion, do high-knee drills".
_RUN_HIP_BANDS: dict[str, tuple[float, float] | None] = {
    # Thigh behind the body. This is the instant hip extension is defined at,
    # and the one the old always-on band was actually written for. Novacheck
    # 1998: peak extension ~10-20 deg past vertical; the measured angle folds
    # back below 180 because the vertex angle is unsigned.
    "toe_off": (150.0, 180.0),
    # Foot reaching ahead on a near-straight knee. Heiderscheit 2011: ~30 deg
    # hip flexion at contact -> ~150 internal. A markedly more closed hip here
    # means the foot lands well ahead of the centre of mass (overstride),
    # which is a real and well-evidenced fault -- so this phase IS scored.
    "initial_contact": (138.0, 165.0),
    # Thigh under the body: the hip reads ~180 by construction, whatever the
    # athlete does. Nothing to score.
    "midstance": None,
    # Thigh ahead with the heel tucked. The hip angle here is a pure function
    # of how far through the swing the shutter caught the leg -- peak knee
    # drive and mid-swing are ~40 deg apart and a still cannot tell them
    # apart. Measured, shown, not scored.
    "swing": None,
    "uncertain": None,
}

# Why a phase carries no hip verdict -- surfaced in the result table and in the
# capture warnings so "not scored" reads as a deliberate limit, not an error.
_RUN_HIP_UNSCORED_NOTES: dict[str, str] = {
    "swing": (
        "Thigh is mid-swing — at this instant the hip angle is set by when the "
        "shutter fired, not by technique. Use a frame at footstrike or at "
        "toe-off for a hip verdict."
    ),
    "midstance": (
        "Thigh is under the body, where the hip reads ~180° whatever you do. "
        "Use a frame at footstrike or at toe-off for a hip verdict."
    ),
    "uncertain": (
        "Could not tell which part of the stride this frame caught, so the hip "
        "angle is measured but not scored."
    ),
}

_RUN_HIP_SCORED_NOTES: dict[str, str] = {
    "toe_off": "Thigh is behind you — scored against the hip-extension band.",
    "initial_contact": (
        "Foot is reaching ahead — scored against the footstrike band, where a "
        "closed hip means the foot lands ahead of your centre of mass."
    ),
}


# Why the elbow carries no verdict on a photo. Behind ``contact_metrics``.
_RUN_ELBOW_UNSCORED_NOTE = (
    "The elbow closes as the arm swings forward and opens as it swings back, "
    "so one photo catches it wherever the swing happened to be. Shown, not "
    "scored; about 90° through the middle of the swing is the usual carry."
)


# Why the knee carries no verdict on a swing frame. Behind ``contact_metrics``.
_RUN_KNEE_SWING_NOTE = (
    "Leg is mid-swing — a tightly folded knee here is what a good stride "
    "looks like (it shortens the lever the hip has to swing), so this "
    "frame cannot fault it. Use a frame at footstrike for a knee verdict."
)


def _get_running_optimal_ranges(
    gait_phase: str = "uncertain",
    phased_knee: bool = False,
) -> dict[str, tuple[float, float]]:
    """Optimal ranges for a running photo at the detected gait phase.

    A joint with no meaningful band at this instant is simply absent from the
    dict. ``_score_photo_angles`` skips any key it has no band for and
    renormalises the remaining weights, and ``analyze_photo`` reports it as
    ``phase_dependent`` rather than dropping it from the table.
    """
    ranges: dict[str, tuple[float, float]] = {
        "knee": (80, 175),   # Wide -- photo captures one unknown gait instant
        "trunk": RUNNING_REFERENCE["trunk_lean"],       # (2, 10)
        # Full-cycle carry range, NOT RUNNING_REFERENCE["elbow_angle"] (85,
        # 100). That figure is the ~90 deg mid-cycle carry, but the elbow
        # closes through the front of the swing (~75-90) and opens through the
        # backswing (~115-130), so the mid-cycle band faults a perfectly
        # normal arm caught at either end of its travel. ~90 deg stays the
        # coaching target (Napier); what a single still can honestly detect is
        # the actual fault -- arms carried straight, or clamped shut.
        #
        # Unlike the hip this is widened rather than phase-split: the arm
        # swing is small, its band does not invert across the cycle, and the
        # one fault worth naming is visible without knowing the phase.
        "elbow": (70, 135),
        # Shank-axis vs foot-axis convention (neutral = 90): the healthy
        # sweep across a gait cycle is ~70 (midstance dorsiflexion) to
        # ~115 (toe-off plantarflexion). Wide band, same reasoning as the
        # knee -- a photo captures one unknown gait instant.
        "ankle": (65, 115),
    }
    band = _RUN_HIP_BANDS.get(gait_phase)
    if band is not None:
        ranges["hip"] = band
    # ``phased_knee`` (behind contact_metrics): the knee's one wide band was
    # written when the phase was unknown, and its lower edge (80) faulted the
    # one thing a swing frame shows -- how far the leg folds. More fold is the
    # good direction (Folland 2017: a shorter swing lever), and elite runners at
    # speed fold well past it: a world-class 10,000 m runner's swing frame came
    # back "needs work" at 63 deg. The phase detector only calls a frame swing
    # when the knee is already folded (<= 115), so there is no under-folded
    # swing for a band to catch either. Measured and shown; not scored.
    #
    # Footstrike keeps the existing band: the detector's contact window starts
    # at 140, so (80, 175) is already one-sided there -- it faults only a knee
    # past 175, which is the near-locked landing RUNNING_REFERENCE names
    # (>~176 => overstriding).
    if phased_knee and gait_phase == "swing":
        ranges.pop("knee", None)
    # Behind the same flag the elbow is shown, not scored: even the full-cycle
    # band faulted elites caught with the arm at the front of its swing (a
    # Chicago marathoner at 43 deg, another at 50, 2026-10-28). Where the arm
    # is in its swing is set by when the shutter fired, as with the swing knee.
    if phased_knee:
        ranges.pop("elbow", None)
    return ranges


# The crank positions a still can be scored at. "uncertain" is not a position:
# it is the honest answer when the photo does not say where the crank was.
BIKE_PHOTO_PHASES = ("bdc", "tdc", "uncertain")
_LEGACY_PHASE = {"near_bdc": "bdc", "near_tdc": "tdc", "mid_stroke": "uncertain"}
_BIKE_PHASE_NOTES = {
    "knee": ("The knee straightens at the bottom of the stroke and bends at the "
             "top. Mark where the pedal was to score it — at the bottom it is "
             "the saddle-height reading."),
    "hip": ("Hip opens at the bottom of the stroke and closes at the top — pick "
            "the pedal position to score it."),
}


_WITHHELD_MESSAGES = {
    "leg_not_measured": (
        "No score: the leg nearest the camera could not be measured, so this "
        "photo cannot say anything about the fit. Usually another person or the "
        "bike is in front of the leg, or the model found someone else in the "
        "frame. One rider, full side view, leg against a plain background."
    ),
    # Same rule and thresholds as the video's position_mismatch.
    "not_aero": (
        "No score: on the aero bars the torso sits low and the elbows bent, and "
        "this photo reads neither. Usually the pose model did not find the rider "
        "properly -- a screenshot, a rider small in the frame, or text and "
        "watermarks across the body -- and sometimes the photo is on the base "
        "bar. Upload the original photo, side-on, with the rider filling most of "
        "it, or pick the position you rode."
    ),
    # Running. Same rule as the bike's, in the sport's own words.
    "run_leg_not_measured": (
        "No score: the leg nearest the camera could not be measured, so this "
        "photo cannot say anything about the stride. Usually another runner or "
        "an object is in front of it, or the model found someone else in the "
        "frame. One runner, full side view, both legs clear of the background."
    ),
    # The legs could not be named at all: no direction of travel, or a foot
    # not seen. Chicago 2026-10-28: a pacer group shot 3/4 from the front was
    # told it sat "between the moments" -- true of a frame in flight, not of
    # one where nobody could tell which way the runner was going.
    "run_legs_unreadable": (
        "No score: we could not tell which way you are running or which foot "
        "is on the ground. That usually means the photo was taken from the "
        "front or at an angle, or other runners overlap you. One runner, "
        "side-on, both feet in view."
    ),
    "stride_phase_uncertain": (
        "No score: this frame was caught between the moments a stride can be "
        "judged at. The angles depend on which instant of the cycle the camera "
        "froze, and this one is neither a foot landing nor a push-off. A frame "
        "at the instant the foot touches down is the one that says most."
    ),
}


def _bike_photo_withheld(
    angles: dict[str, float], cycling_position: str | None = None,
) -> str | None:
    """Why a bike photo gets no score at all, or None if it gets one.

    A bike-fit photo is about the leg. Without a measured knee the score was
    built from the torso and arms alone -- and on a Kona race photo with a
    camera motorbike in frame, the torso and arms were the motorbike
    passenger's: trunk 83 deg, printed as 38/100. The rule the video has kept
    since the quality gate: when we cannot measure the thing, no number.
    """
    knee = angles.get("knee")
    if knee is None or math.isnan(knee):
        return "leg_not_measured"
    # An aero photo that reads like sitting up is, far more often than not, a
    # skeleton on the wrong body parts: a phone screenshot of a watermarked race
    # photo read trunk 79 deg and was graded 44/100 (feedback #15, 2026-10-25).
    # The video has withheld this since position_mismatch; the photo had not.
    from app.services.video_analysis.biomechanics.cycling_positions import AERO_POSITIONS
    from app.services.video_analysis.biomechanics.plausibility import (
        AERO_ELBOW_MAX,
        AERO_TRUNK_MAX,
    )

    if cycling_position in AERO_POSITIONS:
        trunk, elbow = angles.get("trunk"), angles.get("elbow")
        if (trunk is not None and not math.isnan(trunk) and trunk > AERO_TRUNK_MAX) or (
            elbow is not None and not math.isnan(elbow) and elbow > AERO_ELBOW_MAX
        ):
            return "not_aero"
    return None


def _run_photo_withheld(
    angles: dict[str, float], gait_phase: str, legs: dict[str, Any] | None = None,
) -> str | None:
    """Why a running photo gets no score at all, or None if it gets one.

    The bike path has refused a number without a measured knee since the
    quality gate; the run path had no such rule, so a frame the detector
    could not place in the stride was still handed a score built on bands
    chosen for a phase nobody could name. Two refusals, in order of what the
    athlete can do about them:

    * no knee -- nothing about the stride can be said, same as the bike;
    * phase ``uncertain`` -- the knee was measured but the frame sits between
      the instants a stride is judged at, and every band would be a guess.
      The hip is already left unscored here; the score as a whole now is too.
      When ``legs`` says the legs could not be named at all (no direction of
      travel, a foot not seen -- ``stance`` None), the reason is that, not the
      instant: ``run_legs_unreadable``.
    """
    knee = angles.get("knee")
    if knee is None or math.isnan(knee):
        return "run_leg_not_measured"
    if gait_phase == "uncertain":
        if legs is not None and legs.get("stance") is None:
            return "run_legs_unreadable"
        return "stride_phase_uncertain"
    return None


def _bike_phase_bands(cycling_position: str | None) -> dict[str, dict[str, list[float]]]:
    """The knee and hip bands at each dead centre, for one position.

    Both joints sweep through 50-80 deg every revolution, so a band means
    something only at a named crank position. These are the reference bands
    themselves -- the ones the video grades against and the table prints. The
    allowance for a frame caught a little off dead centre is applied by the
    scorer (``_KNEE_SLACK``), not by widening what the rider is shown.
    """
    ref = get_cycling_reference(cycling_position)
    return {
        "knee": {"bdc": list(ref["knee_at_bdc"]), "tdc": list(ref["knee_at_tdc"])},
        "hip": {"bdc": list(ref["hip_at_bdc"]), "tdc": list(ref["hip_at_tdc"])},
    }


# Degrees of knee reading forgiven before the score starts falling, per side
# (low, high), at each dead centre. Two parts, both measured rather than felt:
#
#   crank position: modelled on a 0.45/0.45 m leg over a 170 mm crank, a frame
#     taken within 10 deg of bottom dead centre reads the knee 3 deg more bent
#     on one side of it and 2 deg straighter on the other (the knee peaks just
#     off BDC and is flat there). At the top it mirrors: off-TDC reads OPEN.
#   instrument: the same 2 deg the video forgives (SCORE_ANGLE_TOLERANCE).
#
# The old allowance was a -12/+8 widened band -- +/-30 deg of crank, and on
# the same model 12 deg of knee is ~4.5 cm of saddle height left unflagged.
_CRANK_SLACK_DEG = {"bdc": (3.0, 2.0), "tdc": (2.0, 3.0)}
_INSTRUMENT_SLACK_DEG = 2.0
_KNEE_SLACK = {
    p: (lo + _INSTRUMENT_SLACK_DEG, hi + _INSTRUMENT_SLACK_DEG)
    for p, (lo, hi) in _CRANK_SLACK_DEG.items()
}


def _knee_falloff(band: tuple[float, float]) -> float:
    """Points lost per degree past the slack: the video's slope
    (``technique_scorer.score_in_range``: 100 / (2 x band width)), so a photo
    and a clip of one saddle cannot disagree about how bad it is. On the TT
    BDC band (138-145) that is 7.1 per degree, against the flat 2.0 every photo
    metric used to share."""
    width = max(band[1] - band[0], 1.0)
    return 100.0 / (2.0 * width)


def _get_cycling_optimal_ranges(
    cycling_position: str | None,
    pedal_phase: str = "bdc",
) -> dict[str, tuple[float, float]]:
    """The bands a single bike photo is scored against, at one crank position.

    Upper-body bands do not depend on the crank and always apply. The knee and
    hip apply only at a named dead centre. Three readings are never scored from
    a still:

    - the knee and hip at an uncertain crank position. The old fallback scored
      the knee against 60-155 there, and because the position was guessed FROM
      the knee, a saddle far too low -- knee 118 at the bottom of the stroke --
      was filed as "mid-stroke" and scored 100. A still could only ever fault a
      saddle that was too high.
    - the pelvic ratio. It is hip / trunk, so it inherits the hip's sweep, and
      its 2-4 band was set on the stroke-AVERAGE hip the video measures. On a
      single frame it read 5.7 on a world-class position and cost it 33 points.
    - the ankle (2D ankle from a still is too noisy to judge).
    """
    phase = _LEGACY_PHASE.get(pedal_phase, pedal_phase)
    ref = get_cycling_reference(cycling_position)
    ranges: dict[str, tuple[float, float]] = {
        "trunk": ref["trunk_angle"],
        "elbow": ref["elbow_angle"],
        "shoulder": ref["shoulder_angle"],
        "forearm_tilt": ref["forearm_tilt"],
        "head_alignment": ref["head_alignment"],
    }
    if phase in ("bdc", "tdc"):
        bands = _bike_phase_bands(cycling_position)
        ranges["knee"] = tuple(bands["knee"][phase])
        ranges["hip"] = tuple(bands["hip"][phase])
    return ranges


def _get_swimming_optimal_ranges() -> dict[str, tuple[float, float]]:
    return {
        "left_shoulder": (120, 180),
        "right_shoulder": (120, 180),
        "left_elbow": SWIMMING_REFERENCE["elbow_at_catch"],   # (90, 120)
        "right_elbow": SWIMMING_REFERENCE["elbow_at_catch"],
        "left_knee": (150, 180),
        "right_knee": (150, 180),
        "body_rotation": SWIMMING_REFERENCE["body_rotation"],  # (40, 60)
        "streamline": SWIMMING_REFERENCE["streamline"],        # (0, 10)
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

PHOTO_RUNNING_WEIGHTS = {
    "knee": 0.30, "trunk": 0.25, "elbow": 0.20, "hip": 0.15, "ankle": 0.10,
}
# Hip and ankle sweep through a huge range across the pedal stroke (open at the
# bottom, closed at the top), exactly like the knee -- but unlike the knee we
# have no per-phase reference band for them, so a single-photo value can't be
# scored against the stroke-*average* band without faulting good positions (a
# near-BDC hip reads ~95deg, well above the average band). They are therefore
# measured + shown but NOT scored or judged from a still. (See the interactive
# pedal-phase picker for the follow-up that scores them against the right band.)
# The knee joins them at an uncertain crank position, and the pelvic ratio --
# hip over trunk, so it sweeps with the hip -- is never scored from a still.
_PHASE_DEPENDENT_BIKE = frozenset({"knee", "hip", "ankle", "pelvic_ratio"})

# No head term in any of these. The old "Head position" score (ear below the
# hip-shoulder joint line) came back 100 for every aero rider -- all fifteen
# Kona 2026 pros -- so its 11-13 % of the weight was free points. Its
# replacement, helmet above the back line, is measured but not scored: no
# source gives a target for it (head_silhouette.py). 2026-10-11.
PHOTO_CYCLING_WEIGHTS = {
    "knee": 0.24, "trunk": 0.20, "shoulder": 0.13,
    "elbow": 0.11, "forearm_tilt": 0.05, "helmet_above_back": 0.06,
}
# Near TDC/BDC from a single photo: the pedal phase is only a guess, so trust
# the knee less than a true motion-capture dead-centre -- shift its weight to
# the pedal-phase-independent trunk/shoulder metrics.
#
# Hip IS scored here, unlike the other profiles. The exclusion in
# _PHASE_DEPENDENT_BIKE exists because a stroke-average band faults good
# positions -- but at a detected dead centre we have the actual per-phase band
# (hip_at_tdc / hip_at_bdc) to score against. Leaving it out meant the report
# could show "Hip angle: needs work" in the table and 100/100 as the score, and
# a closed hip is the one cycling metric with a medical consequence.
PHOTO_CYCLING_WEIGHTS_PHASED = {
    "knee": 0.14, "trunk": 0.24, "shoulder": 0.14,
    "elbow": 0.11, "forearm_tilt": 0.05, "hip": 0.12, "helmet_above_back": 0.06,
}
# Crank position unknown: the score is the upper body only, and says so. The
# knee used to stay in at 0.06 against a 60-155 band -- free points, since
# almost any knee lands in that band.
PHOTO_CYCLING_WEIGHTS_UNCERTAIN = {
    "trunk": 0.27, "shoulder": 0.15,
    "elbow": 0.12, "forearm_tilt": 0.05, "helmet_above_back": 0.06,
}
# The helmet term (2026-10-11, Artur's call) only exists in a score when the
# reading is trustworthy and the position is an aero one -- see
# _with_helmet_band. It costs nothing up to where twenty-eight of the thirty-one
# Kona pros sat (+23 % of torso length) and is amber, never red, above it: the only study (Beaumont
# 2018, CFD) compares head up with head down -- up to 6.4 % drag -- and gives
# no curve to hang a red line on. Half the old head weight.
_HELMET_BAND = (-100.0, KONA_PRO_FREE_UP_TO)
_HELMET_SCORE_FLOOR = 70
_BIKE_PHASE_WEIGHTS = {
    "bdc": PHOTO_CYCLING_WEIGHTS_PHASED,
    "tdc": PHOTO_CYCLING_WEIGHTS_PHASED,
    "uncertain": PHOTO_CYCLING_WEIGHTS_UNCERTAIN,
}
PHOTO_SWIMMING_WEIGHTS = {
    "left_elbow": 0.20, "right_elbow": 0.10,
    "body_rotation": 0.25, "streamline": 0.25,
    "left_shoulder": 0.10, "right_shoulder": 0.10,
}

_PHOTO_WEIGHTS = {
    "run": PHOTO_RUNNING_WEIGHTS,
    "bike": PHOTO_CYCLING_WEIGHTS,
    "swim": PHOTO_SWIMMING_WEIGHTS,
}


# Not every scored metric is an angle in degrees. The defaults below -- a 3.0
# "acceptable" margin and 2.0 score points lost per unit outside the band -- are
# degree-scaled. Applied to pelvic rotation, whose ENTIRE optimal band is 2.0
# wide, they make the metric unfalsifiable: a ratio of 0.0 still classifies as
# "acceptable", and 30% below the band costs one point. Format:
#   key -> (acceptable_margin, score points lost per unit outside the band)
_NON_DEGREE_SCALES: dict[str, tuple[float, float]] = {
    "pelvic_ratio": (0.2, 60.0),
    # Leg lengths ahead of the hip at footstrike. 250 points per unit: just
    # past the band (0.20) is amber at ~87, a clear reach (0.35, measured on a
    # real touchdown) lands at ~50. See _RUN_OVERSTRIDE_BAND.
    "overstride": (0.05, 250.0),
    # Degrees, but a wider amber than the 3-degree default: the shin band is
    # read off 13 photos, not a norm, so -10 to -16 is "worth a look", and
    # past it each degree costs 5 (-16 -> 70, -24 -> 30). See _RUN_SHIN_BAND.
    "shin": (6.0, 5.0),
}

# Overstride on a running photo at footstrike: the SAME band the video scores
# against (RUNNING_REFERENCE, Heiderscheit 2011; Souza 2016), so a photo and a
# clip of one runner cannot disagree about the same reach. Weighted level with
# the knee: at footstrike it is the reading the instant exists for, and the
# angles alone never caught it -- a 0.35 reach came back 100/100 "Excellent"
# because knee 172 and hip 156 both sat inside their bands.
_RUN_OVERSTRIDE_BAND = RUNNING_REFERENCE["overstride_ratio"]
_RUN_OVERSTRIDE_WEIGHT = 0.30

# Since 2026-10-29 the photo grades the touchdown by the SHIN, not by the
# reach, and the reach above is no longer scored on a photo. On all 13 race and
# landing photos Artur marked as a touchdown overstride sat outside the band --
# 0.18-0.51, median 0.29, Kiptum's 0.37 among them -- and it could hardly do
# otherwise: a frame is only called a touchdown once the foot is
# _RUN_CONTACT_AHEAD (0.10) ahead of the hip, which left 0.10-0.15 to be green
# in. The reach also grows with speed, and a still has no speed. The shin's
# lean at contact is far less tied to pace, and it already split the set: the
# plain reaching heel landings read -19 to -24 deg, every other touchdown -13
# to +13. Overstride is still measured and drawn, as information; the video
# keeps scoring it against RUNNING_REFERENCE.
#
# Band from those 13 photos, not from a published norm: up to 10 deg of
# backward lean is free, amber to -16 (_NON_DEGREE_SCALES), red past it; a
# shin leaning forward is no fault at contact. Weighted under the old
# overstride's 0.30, so one amber reading does not sink a good stride.
_RUN_SHIN_BAND = (-10.0, 20.0)
_RUN_SHIN_WEIGHT = 0.20
_RUN_OVERSTRIDE_INFO = (
    "Shown, not scored: how far ahead of the hip the foot lands grows with "
    "speed, and a photo does not know how fast you were going. The shin angle "
    "is the touchdown reading that is scored."
)

_DEFAULT_MARGIN_FLOOR = 3.0
_DEFAULT_FALLOFF = 2.0

# How far the overall score is pulled from the weighted mean toward the weakest
# scored metric. 0.0 = pure mean (a single serious fault disappears into the
# average), 1.0 = the score IS the worst metric (one noisy reading tanks an
# otherwise good position). 0.35 keeps a clean position at 100 while letting one
# real problem cost double digits.
_WORST_PULL = 0.35

# Ceiling applied when a measurement crosses a medical threshold: the top of the
# "Good" band, so the number and the grade label cannot contradict the warning
# banner shown next to them.
_MEDICAL_SCORE_CAP = 89


def _metric_margin(metric: str | None, optimal_min: float, optimal_max: float) -> float:
    """Width of the 'acceptable' band on either side of the optimal range."""
    scale = _NON_DEGREE_SCALES.get(metric or "")
    floor = scale[0] if scale else _DEFAULT_MARGIN_FLOOR
    return max((optimal_max - optimal_min) * 0.10, floor)


def _score_single_angle(
    value: float, optimal_min: float, optimal_max: float,
    low_tolerance: float = 1.0, high_tolerance: float = 1.0,
    falloff: float | None = None,
    slack: tuple[float, float] = (0.0, 0.0),
) -> int | None:
    """Score one angle 0-100 by distance from the optimal range.

    Linear falloff (not a step function) so a value 0.1 deg out of range
    doesn't cliff-drop 20 points -- the score degrades smoothly at
    ``FALLOFF_PER_DEG`` points per degree outside the band, floored at 20.

    ``low_tolerance`` / ``high_tolerance`` scale the penalty on each side
    (values > 1.0 = more lenient). Used to make a metric asymmetric, e.g.
    a deep aero trunk being *below* the optimal min is far less of a fault
    than being above it, so we forgive the low side more gently.

    ``slack`` (low, high): degrees past the band forgiven outright before any
    penalty -- the measurement's own uncertainty, not leniency about the fault.
    """
    if math.isnan(value):
        return None

    if optimal_min <= value <= optimal_max:
        return 100

    # Points lost per unit outside the range. Degrees by default; ratios and
    # other non-degree metrics pass their own scale (see _NON_DEGREE_SCALES).
    per_unit = falloff if falloff is not None else _DEFAULT_FALLOFF
    FLOOR = 20

    if value < optimal_min:
        distance = max(0.0, optimal_min - value - slack[0]) / max(low_tolerance, 0.1)
    else:
        distance = max(0.0, value - optimal_max - slack[1]) / max(high_tolerance, 0.1)

    score = 100.0 - distance * per_unit
    return int(round(max(FLOOR, min(100.0, score))))


def _assign_photo_grade(score: int) -> str:
    """Convert 0-100 score to grade label."""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Fair"
    else:
        return "Needs Work"


# Asymmetric scoring tolerances for aggressive aero positions. A trunk or hip
# angle *below* the optimal min means a flatter back / more closed hip -- which
# is the aerodynamic GOAL, not a fault -- so penalize the low side ~3x more
# gently. Being above the max (too upright / too open = draggy) keeps the full
# penalty. Format: angle_key -> (low_tolerance, high_tolerance).
_AERO_TOLERANCES = {
    # flat aero back below min: no points at all (AERO_TRUNK_LOW_CONCERN is
    # shown instead); too upright: full penalty
    "trunk": (math.inf, 1.0),
    "hip":   (2.5, 1.0),   # closed aero hip below min: mostly forgiven
    # closed shoulder on aero bars: the athlete's choice (Empfield 2007) -- see
    # cycling_positions.SHOULDER_CLOSED_OK_POSITIONS
    "shoulder": (SHOULDER_CLOSED_FORGIVENESS, 1.0),
}
_AERO_POSITIONS = AERO_POSITIONS

_SHOULDER_CLOSED_NOTE = (
    "Upper arm closer to the torso than the 90-degree fitters aim for. On aero "
    "bars that is the athlete's choice, not a fault -- check you can still "
    "breathe and push at race power, and that the arms carry you without a shrug."
)

_HELMET_NOTE = (
    "How far the top of the helmet rises above a level line through the top of "
    "your back -- level because the wind is -- as a share of torso length. "
    f"The Kona 2026 pros ranged from {KONA_PRO_RANGE[0]:+.0f}% to "
    f"{KONA_PRO_RANGE[1]:+.0f}%, median {KONA_PRO_MEDIAN:+.0f}%. Some helmet above "
    "the back is normal -- you have to see the road."
)

_HELMET_HIGH_NOTE = (
    "Higher than twenty-eight of the thirty-one Kona pros. A head up against a head down "
    "cost up to 6.4 % more drag in a CFD study of time-trial helmets (Beaumont "
    "2018). Amber, a small penalty: drop the head if you can still see the "
    "road and hold it."
)


def _with_helmet_band(
    ranges: dict[str, tuple[float, float]], scored: bool,
) -> dict[str, tuple[float, float]]:
    """The bike bands plus the helmet band when the helmet term is scored."""
    if not scored:
        return ranges
    return {**ranges, "helmet_above_back": _HELMET_BAND}


# Two different riders trip the flag: an aero helmet whose tail lies flush on
# the back (the point can land on the helmet), and no helmet at all -- a
# trainer session filmed bare-headed, where the neck rises into the head with
# no edge to find. On Artur's own trainer clips the point was right and the
# flag fired anyway. Both get the same honest answer: not graded until a
# person has placed the point, and the reading compares with pros in aero
# helmets, so it means most with the race helmet on.
_HELMET_CHECK_WARNING = (
    "We couldn't find a clear edge where your back ends and your helmet begins, "
    "so the helmet reading may be low and is not graded. If the green point sits "
    "on the helmet, use Adjust joints to drag it to the highest point of your back."
)


def _photo_next_steps(
    sport: str, cycling_position: str | None, pedal_phase: str | None,
    withheld: bool, camera_side: str | None, helmet: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Where to go from this photo -- present whatever the score.

    A clean photo used to end the report: 100, "Nothing to fix", and a rider
    with nowhere to go (and no reason to come back). What a single still cannot
    tell, each with the one thing that would -- the pedal position, a clip, the
    other side of the bike, the pro field -- said every time (never a helmet:
    its type cannot be told from the side),
    in the order a fitter would take them.
    """
    steps: list[dict[str, str]] = []
    aero = sport == "bike" and cycling_position in _AERO_POSITIONS
    if sport == "bike" and pedal_phase == "uncertain" and not withheld:
        steps.append({"key": "pedal", "title": "Mark the pedal position",
                      "text": "Under the angles, say whether the crank was at the bottom or the "
                              "top, or shoot a frame at the bottom of the stroke: the knee there is "
                              "the saddle-height reading, the one fit number everything else is set "
                              "from. Until then the score is your upper body."})
    if sport == "bike" and camera_side == "right":
        steps.append({"key": "side", "title": "Film from the non-drive side",
                      "text": "This photo is from the chainring side. The chainring sits behind the "
                              "ankle at the bottom of the stroke and disturbs the knee reading; the "
                              "other side of the bike measures the leg cleanly."})
    steps.append({"key": "clip", "title": "Film a 10-second clip",
                  "text": ("A photo is one instant. A side-on clip measures the whole stroke: the "
                           "knee at the top and the bottom, the hip through the revolution, how "
                           "steady the position is. That is what the fit is judged on."
                           if sport == "bike" else
                           "A photo is one instant of the stride. A side-on clip measures cadence, "
                           "contact time, overstride and every joint through the whole cycle.")})
    if aero:
        steps.append({"key": "pros", "title": "Put your numbers beside the Kona pros",
                      "text": "The Kona 2026 column in the table shows where 32 pros sat for each "
                              "angle. A number inside its band can still sit a long way from the "
                              "field; that gap is the next place to look."})
    return steps

_AERO_TRUNK_EXTREME_NOTE = (
    "Below 12 degrees the question is no longer aero but whether the body can "
    "do the work there at all -- a flat back was the one position that changed "
    "power and metabolic cost significantly against every other (Fintelman 2015)."
)


def _score_photo_angles(
    angles: dict[str, float],
    optimal_ranges: dict[str, tuple[float, float]],
    sport: str,
    weights_override: dict[str, float] | None = None,
    cycling_position: str | None = None,
    pedal_phase: str | None = None,
) -> dict[str, Any]:
    """Score all angles and compute weighted overall score.

    ``pedal_phase`` ("bdc"/"tdc") puts the bike knee on the video's slope with
    the dead-centre slack; at the bottom it is the saddle-height reading, the
    one fit number every other contact point is set from.
    """
    weights = weights_override or _PHOTO_WEIGHTS.get(sport, PHOTO_RUNNING_WEIGHTS)
    is_aero = sport == "bike" and cycling_position in _AERO_POSITIONS
    per_angle: dict[str, dict[str, Any]] = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for angle_key, weight in weights.items():
        value = angles.get(angle_key)
        if value is None or angle_key not in optimal_ranges:
            continue

        opt_min, opt_max = optimal_ranges[angle_key]
        low_tol, high_tol = _AERO_TOLERANCES.get(angle_key, (1.0, 1.0)) if is_aero else (1.0, 1.0)
        # The aero forgiveness on a closed hip assumes closed = deliberate. Past
        # the iliac-artery threshold that stops being true, so the discount ends
        # there -- otherwise the one hip reading with a medical consequence is
        # the one the scorer treats most leniently.
        if angle_key == "hip" and value < HIP_MEDICAL_MIN_DEG:
            low_tol = 1.0
        scale = _NON_DEGREE_SCALES.get(angle_key)
        falloff = scale[1] if scale else None
        slack = (0.0, 0.0)
        if sport == "bike" and angle_key == "knee" and pedal_phase in _KNEE_SLACK:
            falloff = _knee_falloff((opt_min, opt_max))
            slack = _KNEE_SLACK[pedal_phase]
        score = _score_single_angle(
            value, opt_min, opt_max, low_tol, high_tol,
            falloff=falloff, slack=slack,
        )
        if score is None:
            continue
        if angle_key == "helmet_above_back":
            score = max(score, _HELMET_SCORE_FLOOR)  # amber at worst, see _HELMET_BAND

        per_angle[angle_key] = {
            "score": score,
            "weight": weight,
            "weighted": round(score * weight, 1),
        }
        weighted_sum += score * weight
        total_weight += weight

    if total_weight > 0:
        mean = weighted_sum / total_weight
        # Pull the result toward the weakest metric. A plain weighted mean over
        # eight metrics cannot represent one serious fault: six good readings
        # outvote two bad ones, which is how a position with a hip past the
        # iliac-artery threshold scored 95/100.
        #
        # This also makes the number RESPOND. Under the mean, fixing the single
        # worst thing in your position moved the score a point or two, so the
        # score carried no information about whether the fix worked.
        worst = min(a["score"] for a in per_angle.values())
        overall = int(round(mean - _WORST_PULL * (mean - worst)))
    else:
        overall = 50

    overall = max(0, min(100, overall))

    # A position that crosses a documented medical threshold must not be able to
    # come back rated "Excellent", however clean the rest of it is.
    capped_by = None
    hip_val = angles.get("hip")
    if (sport == "bike" and cycling_position in _AERO_POSITIONS
            and hip_val is not None and not math.isnan(hip_val)
            and hip_val < HIP_MEDICAL_MIN_DEG
            and overall > _MEDICAL_SCORE_CAP):
        overall = _MEDICAL_SCORE_CAP
        capped_by = "hip_medical_threshold"

    result: dict[str, Any] = {
        "overall_score": overall,
        "grade": _assign_photo_grade(overall),
        "per_angle": per_angle,
    }
    if capped_by:
        result["capped_by"] = capped_by
    return result


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

# Skeleton connections (from pipeline.py)
_POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (15, 17), (16, 18), (15, 19), (16, 20),
    (15, 21), (16, 22), (17, 19), (18, 20),
    (19, 21), (20, 22),
    (27, 29), (28, 30), (27, 31), (28, 32),
    (29, 31), (30, 32),
]

# The annotated image is returned INLINE as a base64 data URI on a synchronous
# request, so its encoding is a payload decision as much as a quality one. PNG
# on a photograph is pathological: a full-size phone upload (3024x4032) encoded
# to a ~15 MB data URI, which is a broken experience on a phone however good the
# annotation looks. JPEG at this quality is ~1 MB for the same picture with no
# visible loss on the chips or the skeleton, and matches what the video path
# already ships for its keyframe (video_visualizer.render_keyframe).
_THUMB_JPEG_QUALITY = 92
_THUMB_MIME = "image/jpeg"

_LEFT_SIDE = {1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_SIDE = {4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
_MIDLINE = {0, 9, 10}
_VIS_THRESHOLD = 0.5

# Label positioning: landmark index + offset direction per angle key
_LABEL_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "run": {
        "left": {
            "knee":  {"idx": 25, "offset_dir": "left"},
            "hip":   {"idx": 23, "offset_dir": "up-left"},
            "trunk": {"idx": 11, "offset_dir": "up"},
            "elbow": {"idx": 13, "offset_dir": "up-left"},
            "ankle": {"idx": 27, "offset_dir": "down-left"},
        },
        "right": {
            "knee":  {"idx": 26, "offset_dir": "right"},
            "hip":   {"idx": 24, "offset_dir": "up-right"},
            "trunk": {"idx": 12, "offset_dir": "up"},
            "elbow": {"idx": 14, "offset_dir": "up-right"},
            "ankle": {"idx": 28, "offset_dir": "down-right"},
        },
    },
    "bike": {
        "left": {
            "knee":         {"idx": 25, "offset_dir": "left"},
            "hip":          {"idx": 23, "offset_dir": "up-left"},
            "trunk":        {"idx": 11, "offset_dir": "up"},
            "elbow":        {"idx": 13, "offset_dir": "up-left"},
            "ankle":        {"idx": 27, "offset_dir": "down-left"},
            "shoulder":     {"idx": 11, "offset_dir": "right"},
            "forearm_tilt": {"idx": 15, "offset_dir": "down-right"},
        },
        "right": {
            "knee":         {"idx": 26, "offset_dir": "right"},
            "hip":          {"idx": 24, "offset_dir": "up-right"},
            "trunk":        {"idx": 12, "offset_dir": "up"},
            "elbow":        {"idx": 14, "offset_dir": "up-right"},
            "ankle":        {"idx": 28, "offset_dir": "down-right"},
            "shoulder":     {"idx": 12, "offset_dir": "left"},
            "forearm_tilt": {"idx": 16, "offset_dir": "down-left"},
        },
    },
    "swim": {
        "any": {
            "left_shoulder":  {"idx": 11, "offset_dir": "up"},
            "right_shoulder": {"idx": 12, "offset_dir": "up-right"},
            "left_elbow":     {"idx": 13, "offset_dir": "left"},
            "right_elbow":    {"idx": 14, "offset_dir": "right"},
            "left_knee":      {"idx": 25, "offset_dir": "down-left"},
            "right_knee":     {"idx": 26, "offset_dir": "down-right"},
        },
    },
}


def _is_near_side(idx: int, camera_side: str | None) -> bool:
    """Check if a landmark index belongs to the near side."""
    if idx in _MIDLINE:
        return True
    if camera_side == "left":
        return idx in _LEFT_SIDE
    elif camera_side == "right":
        return idx in _RIGHT_SIDE
    return True


# angles_with_context status -> overlay chip status
_ROW_TO_CHIP = {
    "optimal": "good",
    "acceptable": "warn",
    "needs_work": "bad",
    "phase_dependent": "muted",
    "insufficient_visibility": "muted",
}


def _score_to_color(score: int | None) -> tuple[int, int, int]:
    """Map score to BGR color for thumbnail drawing."""
    if score is None:
        return (180, 180, 180)  # Gray for N/A
    if score >= 80:
        return (0, 220, 0)     # Green
    elif score >= 60:
        return (0, 200, 255)   # Orange
    else:
        return (0, 0, 255)     # Red


def _draw_arc(
    cv2_mod, frame, pixel_coords: list[tuple[int, int, float]],
    prox_idx: int, vert_idx: int, dist_idx: int,
    color: tuple[int, int, int], body_height_px: float,
) -> None:
    """Draw angle arc at joint vertex (same algorithm as pipeline._draw_angle_arc)."""
    max_idx = max(prox_idx, vert_idx, dist_idx)
    if max_idx >= len(pixel_coords):
        return

    px, py, pv = pixel_coords[prox_idx]
    vx, vy, vv = pixel_coords[vert_idx]
    dx, dy, dv = pixel_coords[dist_idx]

    if min(pv, vv, dv) < _VIS_THRESHOLD:
        return

    v1x, v1y = px - vx, py - vy
    v2x, v2y = dx - vx, dy - vy

    ang1 = math.degrees(math.atan2(v1y, v1x))
    ang2 = math.degrees(math.atan2(v2y, v2x))

    start = min(ang1, ang2)
    end = max(ang1, ang2)
    if end - start > 180:
        start, end = end, start + 360

    radius = max(12, int(body_height_px * 0.08))
    cv2_mod.ellipse(
        frame, (vx, vy), (radius, radius), 0,
        start, end, color, 1, cv2_mod.LINE_AA,
    )


def _draw_run_ankle_arc(
    cv2_mod, frame, pixel_coords: list[tuple[int, int, float]],
    camera_side: str, color: tuple[int, int, int], body_height_px: float,
) -> None:
    """Arc for the axis-based run ankle angle (see RUNNING_PHOTO_ANKLE).

    The measured angle is shank axis (ankle -> knee) vs foot axis
    (heel -> toe), so the arc must span those two directions -- a vertex
    arc through the heel would show a different angle than the number on
    the chip. Anchored at the ankle dot, with a dashed guide along the
    foot axis so the athlete can see what the number is measured against.
    """
    knee_i, ankle_i, heel_i, toe_i = RUNNING_PHOTO_ANKLE[camera_side]
    if max(knee_i, ankle_i, heel_i, toe_i) >= len(pixel_coords):
        return
    kx, ky, kv = pixel_coords[knee_i]
    ax, ay, av = pixel_coords[ankle_i]
    hx, hy, hv = pixel_coords[heel_i]
    tx, ty, tv = pixel_coords[toe_i]
    if min(kv, av, hv, tv) < _VIS_THRESHOLD:
        return

    v1x, v1y = kx - ax, ky - ay          # shank axis
    v2x, v2y = tx - hx, ty - hy          # foot axis, translated to the ankle
    if (abs(v1x) + abs(v1y) < 1) or (abs(v2x) + abs(v2y) < 1):
        return

    ang1 = math.degrees(math.atan2(v1y, v1x))
    ang2 = math.degrees(math.atan2(v2y, v2x))
    start = min(ang1, ang2)
    end = max(ang1, ang2)
    if end - start > 180:
        start, end = end, start + 360

    radius = max(12, int(body_height_px * 0.08))
    cv2_mod.ellipse(
        frame, (ax, ay), (radius, radius), 0,
        start, end, color, 1, cv2_mod.LINE_AA,
    )

    # Dashed guide slightly past heel and toe so it reads as an axis, not a bone.
    ext = 0.25
    _draw_dashed_line(
        cv2_mod, frame,
        (int(hx - v2x * ext), int(hy - v2y * ext)),
        (int(tx + v2x * ext), int(ty + v2y * ext)),
        color, 1,
    )


def _generate_photo_thumbnail(
    cv2_mod, image, normalized_landmarks,
    angles: dict[str, float],
    score_data: dict[str, Any],
    camera_side: str,
    sport: str,
    arc_triplets: dict[str, tuple[int, int, int]],
    cycling_position: str | None = None,
    hide_angle_values: bool = False,
    optimal_ranges: dict[str, tuple[float, float]] | None = None,
    corrected: list[int] | None = None,
    row_status: dict[str, str] | None = None,
    show_header: bool = True,
    helmet: dict[str, Any] | None = None,
    overstride: dict[str, Any] | None = None,
    run_pelvis_hip: int | None = None,
    shin: dict[str, Any] | None = None,
) -> bytes:
    """Generate annotated thumbnail with skeleton, angle labels, and score badge.

    Returns encoded image bytes (see ``_THUMB_MIME``). ``hide_angle_values``
    (free-tier teaser): keep the skeleton + arcs but mask the numeric angle
    labels and burn a watermark. ``corrected``: landmark indices the athlete
    placed by hand, ringed so they read as a person's statement rather than
    a detection. ``overstride`` (run, at touchdown): the contact block, drawn
    as the reach it measures. ``shin`` (run, at touchdown): the contact block
    when the shin was graded -- a chip on the shin in its row's colour.
    ``run_pelvis_hip``: the visible hip, when the
    measured leg hangs from the hidden one -- the trunk is drawn to it and
    the hidden hip as a hollow ring, an estimate.
    """
    # Draw on a canvas at least as big as the overlay geometry expects. Chip
    # type has a hard minimum size, so on a small photo -- a frame cropped out
    # of a video, a screenshot -- the annotation does not scale down with the
    # picture and swamps it. Landmarks are normalized, so every pixel position
    # below derives from this frame's w/h and follows the upscale for free.
    frame = overlay_style.fit_render_canvas(cv2_mod, image)
    frame = image.copy() if frame is image else frame
    h, w = frame.shape[:2]
    optimal_ranges = optimal_ranges or {}

    # Convert normalized landmarks to pixel coordinates
    pixel_coords: list[tuple[int, int, float]] = []
    for lm in normalized_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        vis = getattr(lm, "visibility", 1.0)
        pixel_coords.append((px, py, float(vis)))

    # --- 1. DRAW SKELETON (neon + soft glow, shared style) ---
    side_filter = camera_side if sport in ("run", "bike") else None
    trunk_sh, stance_hip = RUNNING_TRUNK_LANDMARKS.get(camera_side, (11, 23))
    hidden_hip = run_pelvis_hip is not None and len(pixel_coords) > 32

    _segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for start_idx, end_idx in _POSE_CONNECTIONS:
        if start_idx >= len(pixel_coords) or end_idx >= len(pixel_coords):
            continue
        if hidden_hip and {start_idx, end_idx} == {trunk_sh, stance_hip}:
            continue
        sx, sy, sv = pixel_coords[start_idx]
        ex, ey, ev = pixel_coords[end_idx]
        if sv < _VIS_THRESHOLD or ev < _VIS_THRESHOLD:
            continue
        if side_filter and not (
            _is_near_side(start_idx, side_filter)
            and _is_near_side(end_idx, side_filter)
        ):
            continue
        _segments.append(((sx, sy), (ex, ey)))

    _dots: list[tuple[int, int]] = []
    for i, (px, py, vis) in enumerate(pixel_coords):
        if vis < _VIS_THRESHOLD:
            continue
        if side_filter and not _is_near_side(i, side_filter):
            continue
        if hidden_hip and i == stance_hip:
            continue
        _dots.append((px, py))
    if hidden_hip:
        _segments.append((pixel_coords[trunk_sh][:2], pixel_coords[run_pelvis_hip][:2]))
        _dots.append(pixel_coords[run_pelvis_hip][:2])

    # scale the skeleton weight to the image so it reads on both small and large photos
    _sk = max(1.0, min(2.4, w / 900))
    overlay_style.draw_glow_skeleton(
        cv2_mod, frame, _segments, _dots,
        glow=True, line_w=max(2, int(2 * _sk)), dot_r=max(3, int(4 * _sk)),
    )
    if hidden_hip:
        # The stance leg's own hip is behind the body: a hollow ring, joined
        # to the visible hip, says it is the model's estimate.
        hs, hp = pixel_coords[stance_hip][:2], pixel_coords[run_pelvis_hip][:2]
        _draw_dashed_line(cv2_mod, frame, hp, hs, (255, 255, 255), max(1, int(_sk)))
        cv2_mod.circle(frame, hs, max(4, int(5 * _sk)), (110, 255, 61),
                       max(2, int(2 * _sk)), cv2_mod.LINE_AA)
        # The hip arc opens between the thigh and the trunk as measured --
        # the trunk laid onto this hip (see _run_hip_angle).
        sx, sy, sv = pixel_coords[trunk_sh]
        pixel_coords.append((hs[0] + sx - hp[0], hs[1] + sy - hp[1], sv))
        if "hip" in arc_triplets:
            arc_triplets = {**arc_triplets,
                            "hip": (len(pixel_coords) - 1, *arc_triplets["hip"][1:])}
    if corrected:
        overlay_style.draw_corrected_marks(
            cv2_mod, frame,
            [pixel_coords[i][:2] for i in corrected
             if 0 <= i < len(pixel_coords) and pixel_coords[i][2] >= _VIS_THRESHOLD],
            r=max(7, int(9 * _sk)),
        )
    chips = overlay_style.ChipLayer(frame)

    # Header first: it reserves the top strip so no metric chip lands under it.
    _sport_labels = {"run": "RUN", "bike": "BIKE", "swim": "SWIM"}
    _overall = score_data.get("overall_score")
    _grade = score_data.get("grade") or ""
    if _overall is None:
        # Withheld: no number anywhere, the picture included.
        _hdr_status, _score_text = "muted", "not scored"
    else:
        _hdr_status = "good" if _overall >= 75 else "warn" if _overall >= 60 else "bad"
        _score_text = f"{_overall}/100"
        if score_data.get("partial"):
            # The bike's partial score carries no badge text of its own and
            # keeps the words it always had; the run's names its own.
            _grade = f"{_grade} · {score_data.get('partial_badge', 'upper body')}"
    _title = (
        "AERODYNAMIC PROFILE"
        if (sport == "bike" and cycling_position in ("tt_aero", "triathlon"))
        else ("CYCLING PROFILE" if sport == "bike" else "RUNNING PROFILE")
    )
    _pad = int(max(10, h * 0.018))
    # ``show_header=False``: the skeleton and its angles without a score --
    # for figures that describe a position rather than grade one.
    if show_header:
        chips.header((_pad, _pad), _sport_labels.get(sport, sport.upper()),
                     _score_text, _grade, _hdr_status,
                     right_text=_title, frame_w=w, scale=_sk)

    # --- 2. ANGLE ARCS + LABELS ---
    if len(pixel_coords) > 25:
        # Body size reference for adaptive scaling
        s11x, s11y, _ = pixel_coords[11]
        h23x, h23y, _ = pixel_coords[23]
        body_height_px = max(50.0, abs(s11y - h23y))
        # The floor keeps chips off the body when the athlete is small in
        # frame; it has to track chip size, or on a large photo the chips grow
        # (via _sk) while the leader that is supposed to clear them does not.
        offset_px = max(int(70 * _sk), int(body_height_px * 0.5))

        offset_vectors = {
            "left":       (-offset_px, 0),
            "right":      (offset_px, 0),
            "up":         (0, -offset_px),
            "down":       (0, offset_px),
            "up-left":    (-offset_px, -int(offset_px * 0.7)),
            "up-right":   (offset_px, -int(offset_px * 0.7)),
            "down-left":  (-offset_px, int(offset_px * 0.7)),
            "down-right": (offset_px, int(offset_px * 0.7)),
        }

        per_angle = score_data.get("per_angle", {})
        labels = _ANGLE_LABELS.get(sport, {})

        # Get label configs for this sport + camera side
        if sport == "swim":
            label_cfg = _LABEL_CONFIGS.get("swim", {}).get("any", {})
        else:
            label_cfg = _LABEL_CONFIGS.get(sport, {}).get(camera_side, {})

        for angle_key, angle_value in angles.items():
            if math.isnan(angle_value):
                continue
            cfg = label_cfg.get(angle_key)
            if not cfg:
                continue

            lm_idx = cfg["idx"]
            if lm_idx >= len(pixel_coords):
                continue
            _, _, lm_vis = pixel_coords[lm_idx]
            if lm_vis < _VIS_THRESHOLD:
                continue

            # Color based on score
            angle_score_data = per_angle.get(angle_key)
            score_val = angle_score_data["score"] if angle_score_data else None
            color = _score_to_color(score_val)

            # Draw arc at joint. Run ankle is axis-based (no vertex triplet)
            # and gets its own arc between the shank and foot axes.
            if sport == "run" and angle_key == "ankle":
                _draw_run_ankle_arc(
                    cv2_mod, frame, pixel_coords,
                    camera_side, color, body_height_px,
                )
            else:
                triplet = arc_triplets.get(angle_key)
                if triplet:
                    _draw_arc(
                        cv2_mod, frame, pixel_coords,
                        *triplet, color, body_height_px,
                    )

            # Leader line + neon chip (label + big coloured value)
            jx, jy, _ = pixel_coords[lm_idx]
            dx, dy = offset_vectors.get(cfg["offset_dir"], (offset_px, 0))
            # chips on the left of the joint are right-aligned so they hug the leader
            align = "right" if dx < 0 else "left"
            lx = max(5, min(w - 5, jx + dx))
            ly = max(20, min(h - 20, jy + dy))

            label_name = labels.get(angle_key, angle_key.replace("_", " ").title()).upper()
            if hide_angle_values:
                value_txt, status = "LOCKED", "muted"
            else:
                if angle_key == "pelvic_ratio":
                    value_txt = f"{angle_value:.1f}x"
                elif angle_key == "head_alignment":
                    value_txt = f"{angle_value:.0f}/100"
                else:
                    value_txt = f"{angle_value:.0f}°"
                # colour by the same zone rule the score/UI use; ratios need a
                # smaller margin floor than the degree default. Joints with no
                # band here (bike ankle always; hip when the crank position is
                # uncertain) render neutral -- not judged from a still.
                # The table's own verdict when there is one, so the picture and
                # the rows under it cannot disagree. The picture used to grade
                # every chip symmetrically on its own: a flat aero trunk and a
                # closed aero shoulder showed red over rows that said amber and
                # cost no points, and the knee ignored its dead-centre slack.
                row = (row_status or {}).get(angle_key)
                if row in _ROW_TO_CHIP:
                    status = _ROW_TO_CHIP[row]
                else:
                    opt = optimal_ranges.get(angle_key)
                    floor = 0.3 if angle_key == "pelvic_ratio" else 3.0
                    status = (
                        overlay_style.status_for(angle_value, *opt, min_margin=floor)
                        if opt else "muted"
                    )

            status_rgb = overlay_style.STATUS_COLORS.get(status, overlay_style.INK_SOFT)
            overlay_style.draw_leader(cv2_mod, frame, (jx, jy), (lx, ly), status_rgb)
            chips.metric_chip((lx, ly), label_name, value_txt, status,
                              scale=_sk, align=align)

    # --- 2b. HELMET ABOVE THE BACK LINE (bike only) ---
    # A level line through the top of the back, out to the helmet, and a tick
    # from it up to the top of the helmet: the picture of exactly what the
    # number measures (head_silhouette.py). Neutral colour --
    # it is shown, not judged. The old "HEAD POSITION 100/100" chip is gone
    # with the score it drew (2026-10-11).
    if sport == "bike" and helmet:
        (bx, by), (fx, fy) = helmet["line"]
        hx_, hy_ = helmet["head_point"]
        p0 = (int(bx * w), int(by * h))
        p1 = (int(fx * w), int(fy * h))
        ph = (int(hx_ * w), int(hy_ * h))
        _draw_dashed_line(cv2_mod, frame, p0, p1, (180, 180, 255), max(1, int(2 * _sk)))
        cv2_mod.line(frame, p1, ph, overlay_style.INK_SOFT, max(1, int(2 * _sk)), cv2_mod.LINE_AA)
        cv2_mod.circle(frame, ph, max(3, int(4 * _sk)), overlay_style.INK_SOFT, -1, cv2_mod.LINE_AA)
        # The top of the back is where the reading starts; on a flush helmet
        # tail it is the point most worth checking, so it is drawn, in the
        # same green the editor offers it in.
        cv2_mod.circle(frame, p0, max(4, int(5 * _sk)), (110, 255, 61), -1, cv2_mod.LINE_AA)
        cv2_mod.circle(frame, p0, max(4, int(5 * _sk)), (255, 255, 255), 1, cv2_mod.LINE_AA)
        placed = [pt for pid, pt in ((BACK_TOP_POINT, p0), (HELMET_TOP_POINT, ph))
                  if pid in (helmet.get("moved") or [])]
        if placed:
            overlay_style.draw_corrected_marks(cv2_mod, frame, placed, r=max(7, int(9 * _sk)))
        if hide_angle_values:
            h_text = "LOCKED"
        else:
            h_text = f"{helmet['value']:+.0f}%"
        # Out in front of the face: above the helmet sit the trunk and
        # shoulder chips, and there the placer ran out of room and stacked
        # this one on top of the trunk's value.
        nose_x = pixel_coords[0][0] if len(pixel_coords) > 0 else ph[0]
        # Facing from the back toward the helmet; the nose can sit straight
        # under the helmet's top and give no direction at all.
        ahead = 1 if p1[0] >= p0[0] else -1
        hlx = max(5, min(w - 5, (max(ph[0], nose_x) if ahead > 0 else min(ph[0], nose_x))
                         + ahead * int(60 * _sk)))
        hly = max(20, min(h - 20, ph[1]))
        # Coloured by the table's own verdict: green up to the highest Kona
        # pro, amber above; grey while the reading is unscored.
        h_status = "muted" if hide_angle_values else _ROW_TO_CHIP.get(
            (row_status or {}).get("helmet_above_back"), "muted")
        h_rgb = overlay_style.STATUS_COLORS.get(h_status, overlay_style.INK_SOFT)
        overlay_style.draw_leader(cv2_mod, frame, ph, (hlx, hly), h_rgb)
        chips.metric_chip((hlx, hly), "HELMET ABOVE BACK", h_text, h_status, scale=_sk,
                          align="left" if ahead > 0 else "right")

    # --- 2c. OVERSTRIDE (run, at touchdown) ---
    # A dashed plumb line from the hip down to the ground and a bar from it out
    # to the landing ankle: the reach the number is, |ankle_x - hip_x| over leg
    # length. Neutral: on a photo it is information, not a verdict.
    if sport == "run" and overstride and len(pixel_coords) > 32:
        _, hip_i, _ = RUNNING_PHOTO_ANGLES[camera_side]["hip"]
        _, ankle_i, _, _ = RUNNING_PHOTO_ANKLE[camera_side]
        hx_, hy_, hv = pixel_coords[hip_i]
        ax_, ay_, av = pixel_coords[ankle_i]
        if min(hv, av) >= _VIS_THRESHOLD and ay_ > hy_:
            o_status = "muted" if hide_angle_values else _ROW_TO_CHIP.get(
                overstride.get("overstride_status"), "muted")
            o_rgb = overlay_style.STATUS_COLORS.get(o_status, overlay_style.INK_SOFT)
            thick = max(2, int(2 * _sk))
            _draw_dashed_line(cv2_mod, frame, (hx_, hy_), (hx_, ay_),
                              (255, 255, 255), max(1, int(2 * _sk)))
            cv2_mod.line(frame, (hx_, ay_), (ax_, ay_), overlay_style._bgr(o_rgb),
                         thick, cv2_mod.LINE_AA)
            for x_ in (hx_, ax_):
                cv2_mod.line(frame, (x_, ay_ - int(6 * _sk)), (x_, ay_ + int(6 * _sk)),
                             overlay_style._bgr(o_rgb), thick, cv2_mod.LINE_AA)
            o_text = "LOCKED" if hide_angle_values else f"{overstride['overstride_ratio']:.2f}x"
            # Under the hip end of the bar, opening away from the direction of
            # travel: the ankle's own chip already sits ahead of the foot.
            ahead = 1 if ax_ >= hx_ else -1
            olx = max(5, min(w - 5, hx_ - ahead * int(12 * _sk)))
            # Feet at the bottom edge leave no room below: go above the bar.
            below = ay_ + int(48 * _sk)
            oly = below if below < h - int(40 * _sk) else max(20, ay_ - int(40 * _sk))
            overlay_style.draw_leader(cv2_mod, frame, (hx_, ay_), (olx, oly), o_rgb)
            chips.metric_chip((olx, oly), "OVERSTRIDE", o_text, o_status, scale=_sk,
                              align="right" if ahead > 0 else "left")

    # --- 2d. SHIN AT TOUCHDOWN (run) ---
    # A fifth of the score, and the one reading that decides a touchdown, so
    # it is on the picture like the joints: a chip at mid-shin, behind the
    # leg (the knee and ankle chips sit ahead of it), coloured by its row.
    if sport == "run" and shin and len(pixel_coords) > 32:
        _, knee_i, ankle_i = RUNNING_PHOTO_ANGLES[camera_side]["knee"]
        hip_i = RUNNING_PHOTO_ANGLES[camera_side]["hip"][1]
        kx_, ky_, kv = pixel_coords[knee_i]
        ax_, ay_, av = pixel_coords[ankle_i]
        if min(kv, av) >= _VIS_THRESHOLD:
            s_status = "muted" if hide_angle_values else _ROW_TO_CHIP.get(
                shin.get("shin_status"), "muted")
            s_rgb = overlay_style.STATUS_COLORS.get(s_status, overlay_style.INK_SOFT)
            s_text = "LOCKED" if hide_angle_values else f"{shin['shin_angle']:+.0f}°"
            mx, my = (kx_ + ax_) // 2, (ky_ + ay_) // 2
            ahead = 1 if ax_ >= pixel_coords[hip_i][0] else -1
            slx = max(5, min(w - 5, mx - ahead * int(40 * _sk)))
            overlay_style.draw_leader(cv2_mod, frame, (mx, my), (slx, my), s_rgb)
            chips.metric_chip((slx, my), "SHIN", s_text, s_status, scale=_sk,
                              align="right" if ahead > 0 else "left")

    # No pelvic-ratio chip. It is a ratio of two pelvis landmarks the model
    # places by inference, drawn beside a hip that already carries a chip;
    # on the frame it read as a second, unexplained verdict about the same
    # joint. The number stays in the report's tiles and the score. Artur's
    # call, 2026-09-17.

    # --- 3. BRANDING, then paint the header + every chip in one PIL pass ---
    chips.brand((w - _pad, h - _pad), "FLAPP",
                "FREE" if hide_angle_values else "", scale=_sk)

    frame = chips.flush()

    # Encode (see _THUMB_JPEG_QUALITY -- the format is a payload decision)
    success, buf = cv2_mod.imencode(
        ".jpg", frame, [cv2_mod.IMWRITE_JPEG_QUALITY, _THUMB_JPEG_QUALITY],
    )
    if not success:
        raise ValueError("Failed to encode thumbnail image")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_camera_side(world_landmarks) -> str:
    """Detect which body side faces the camera (single frame).

    Same logic as base_analyzer.detect_camera_side():
    Z increases AWAY from camera; smaller Z = closer.
    """
    left_z = (world_landmarks[11].z + world_landmarks[23].z) / 2
    right_z = (world_landmarks[12].z + world_landmarks[24].z) / 2
    return "left" if left_z < right_z else "right"


def _classify_angle_status(
    value: float, optimal_min: float, optimal_max: float,
    metric: str | None = None,
    slack: tuple[float, float] | None = None,
) -> str:
    """Classify a measurement: optimal / acceptable / needs_work.

    ``metric`` selects the tolerance scale -- degrees by default, or the
    metric's own units for ratios (see _NON_DEGREE_SCALES). Without it a
    ratio inherits a 3.0 margin and can never leave "acceptable".

    ``slack`` (low, high) replaces the margin where the scorer forgives a
    measured uncertainty (the bike knee at a dead centre): "acceptable" is
    exactly the zone the score does not penalise, so a red row always costs
    points and an amber one never does.
    """
    if math.isnan(value):
        return "insufficient_visibility"

    if optimal_min <= value <= optimal_max:
        return "optimal"

    if slack is not None:
        lo_m, hi_m = slack
    else:
        lo_m = hi_m = _metric_margin(metric, optimal_min, optimal_max)

    if (optimal_min - lo_m) <= value <= (optimal_max + hi_m):
        return "acceptable"

    return "needs_work"


def _auto_detect_cycling_position(trunk_angle: float) -> str:
    """Guess cycling position from trunk angle."""
    if math.isnan(trunk_angle):
        return "road_hoods"

    if trunk_angle < 22:
        return "tt_aero"
    elif trunk_angle < 33:
        return "road_drops"
    elif trunk_angle < 46:
        return "road_hoods"
    else:
        return "casual"


def _estimate_pedal_phase(knee_angle: float) -> str:
    """A first guess at the crank position, from the knee alone.

    Returns 'bdc', 'tdc' or 'uncertain'. It is a GUESS and the result says so:
    the rider can overrule it, and every score is computed for every position
    so overruling it is instant.

    Why the middle is "uncertain" and not "mid-stroke": the same knee angle is
    the crank half-way round on a well-set saddle AND the bottom of the stroke
    on a saddle that is too low. One still cannot tell those apart -- the
    midpoint of the two ankles would locate the bottom bracket, but on 14 Kona
    race photos the far ankle was visible on 5 and contradicted the knee on one
    of those. Calling the frame "mid-stroke" asserted the first reading, and
    then the knee went unjudged exactly when it mattered.
    """
    if math.isnan(knee_angle):
        return "uncertain"
    if knee_angle >= 125:
        return "bdc"
    if knee_angle <= 85:
        return "tdc"
    return "uncertain"


# Gait-phase thresholds, in degrees of thigh deviation from vertical
# (+ = knee ahead of the hip) and internal knee angle.
#
# The forward/back thresholds are deliberately asymmetric: the thigh swings far
# further forward (~60-70 deg at peak drive) than back (~20 deg at peak
# extension), so a symmetric pair would file most of terminal stance under
# "midstance". The dead zone between them is where the thigh is close enough to
# vertical that neither hip band means anything anyway.
_RUN_THIGH_FORWARD_DEG = 18.0
_RUN_THIGH_BACK_DEG = -8.0
# Knee angle separating a leg reaching out to land from one folded up in swing.
# The gap between them is left unclassified rather than split down the middle.
_RUN_KNEE_EXTENDED_DEG = 140.0
_RUN_KNEE_FOLDED_DEG = 115.0


def _image_plane(nl: Any, image: np.ndarray) -> list[SimpleNamespace]:
    """Normalized landmarks with x rescaled by the frame's aspect, z dropped.

    Normalized x spans the width and y the height, so on any non-square frame
    they are different units; multiplying x by width/height puts both in
    "frame heights" and an angle read off them is the angle in the picture.
    The bike reads every angle off this since commit 219c20c -- MediaPipe's
    world skeleton was ~8 deg off in opposite directions from the two sides.
    """
    h, w = image.shape[:2]
    aspect = (w / h) if h else 1.0
    return [
        SimpleNamespace(
            x=(lm.x * aspect if lm.x is not None else None),
            y=lm.y,
            z=0.0,
            visibility=getattr(lm, "visibility", 1.0),
        )
        for lm in nl
    ]


# --------------------------------------------------------------------------
# Running photo legs, named by where they are, not by the model's left/right.
#
# On a single side-on frame MediaPipe's left/right leg labels are a coin toss.
# Measured on 13 treadmill photos of one runner (2026-10-19): the leg it called
# "near" was the REAR leg in 12 of them, while the tattooed calf showed the
# physical legs plainly alternating front and back. So the phase and every leg
# number were read off whichever leg happened to carry the label -- and not
# one of the 13 came back as a touchdown, though one plainly was.
#
# A video settles identity over time (leg_identity DP); a photo has no time.
# What a photo does have is position: the STANCE leg is the one whose foot is
# lowest, and where that foot sits against the hip says which instant this
# is. So the stance leg is the measured leg, whatever the model called it.
# --------------------------------------------------------------------------
_LEG_CHAINS = {"left": (25, 27, 29, 31), "right": (26, 28, 30, 32)}  # knee, ankle, heel, toe
# The lowest foot must be at least this much lower (in leg lengths) to count
# as the one on the ground; closer than that is flight or an unreadable frame.
_RUN_FOOT_CLEAR = 0.03
# Stance ankle against the hip midpoint, in leg lengths along the direction
# of travel (+ = ahead). Far enough ahead is a touchdown whatever the knee;
# ahead at all with a near-straight knee is too (a runner who lands under the
# hip still lands on a straight-ish leg -- midstance knees sit ~135-145).
_RUN_CONTACT_AHEAD = 0.10
_RUN_CONTACT_KNEE = 155.0
_RUN_TOEOFF_BEHIND = -0.12


def _foot_low_point(pl: Any, chain: tuple[int, int, int, int]) -> float:
    """Image y (down = larger) of the lowest visible point of one foot, or NaN."""
    ys = [pl[i].y for i in chain[1:]
          if getattr(pl[i], "visibility", 1.0) >= _VIS_THRESHOLD and pl[i].y is not None]
    return max(ys) if ys else math.nan


def _name_run_legs(nl: Any, wl: Any, image: Any, camera_side: str) -> tuple[Any, Any, dict[str, Any]]:
    """Pick the stance leg by position and make it the camera-side leg.

    Returns (nl, wl, legs). When the stance leg carries the far label, the two
    leg chains (knee, ankle, heel, toe) are swapped in copies of both landmark
    lists, so every later step -- angles, contact readings, the picture --
    measures and draws the stance leg through the ordinary camera-side
    indices. Each hip goes with its leg: the thigh is measured from its own
    hip. Read from the near hip, a far stance leg came out reaching further
    and bent more than it was -- race photos are taken from slightly ahead,
    and there the model puts the far hip 0.12-0.17 leg lengths in front of
    the near one (Chicago 2026: images.jpg overstride 0.32 from the near hip,
    0.18 from its own; knee 117 vs 131). The trunk stays on the visible hip
    (``_run_pelvis_hip``).

    ``legs``: ``stance`` (front / under / rear), ``dx`` (stance ankle vs the
    hip midpoint, leg lengths, + = ahead), ``clear`` (one foot is plainly the
    lowest), ``swapped``, ``swing_knee`` (the other leg's knee). ``clear`` is
    False and the rest None when the direction of travel or the feet cannot be
    read -- the phase is then uncertain and the run is not scored.
    """
    pl = _image_plane(nl, image)
    forward = _running_forward_sign(pl)
    lows = {s: _foot_low_point(pl, c) for s, c in _LEG_CHAINS.items()}
    hip = SimpleNamespace(x=(pl[23].x + pl[24].x) / 2, y=(pl[23].y + pl[24].y) / 2)

    def leg_len(side: str) -> float:
        k, a = _LEG_CHAINS[side][:2]
        h = 23 if side == "left" else 24
        return math.dist((pl[h].x, pl[h].y), (pl[k].x, pl[k].y)) + \
            math.dist((pl[k].x, pl[k].y), (pl[a].x, pl[a].y))

    blank = {"stance": None, "dx": None, "clear": False, "swapped": False, "swing_knee": None,
             "knee_gap": None}
    if forward == 0 or any(math.isnan(v) for v in lows.values()):
        return nl, wl, blank
    length = (leg_len("left") + leg_len("right")) / 2
    if not length or math.isnan(length):
        return nl, wl, blank
    stance = max(lows, key=lows.get)
    other = "right" if stance == "left" else "left"
    clear = (lows[stance] - lows[other]) / length >= _RUN_FOOT_CLEAR
    dx = (pl[_LEG_CHAINS[stance][1]].x - hip.x) * forward / length

    swapped = stance != camera_side
    if swapped:
        nl, wl = _swap_leg_chains(nl), _swap_leg_chains(wl)
        pl = _image_plane(nl, image)
    # After a swap the stance leg sits on the camera side; the other is swing.
    far = "right" if camera_side == "left" else "left"
    fk, fa = _LEG_CHAINS[far][:2]
    fh = 23 if far == "left" else 24
    swing_knee, _ = calculate_angle_2d(pl, fh, fk, fa)
    nk = _LEG_CHAINS[camera_side][0]
    knee_gap = math.dist((pl[nk].x, pl[nk].y), (pl[fk].x, pl[fk].y)) / length
    return nl, wl, {
        "stance": "front" if dx > 0.05 else ("rear" if dx < -0.05 else "under"),
        "dx": round(dx, 3), "clear": bool(clear), "swapped": swapped,
        "swing_knee": _safe_round(swing_knee),
        "knee_gap": round(knee_gap, 3),
    }


# Both knees within this many leg lengths of each other: the swing thigh
# crosses the stance thigh and the model has two knees to choose from. On the
# runphoto2026 study (27 photos checked by hand, 2026-10-28) every midstance
# frame sat at 0.09-0.19 and every touchdown at 0.27 or more. Of the eight
# crossed frames, one (p07) carried the knee 18 deg off; the rest were within
# 3 deg -- so the knee stays scored, and the athlete is asked to look at it.
# Which leg is near the camera cannot be told from one frame (the model's
# near/far label was wrong on 12 of 13 treadmill photos), so overlap is the
# signal, not "far leg".
_RUN_KNEES_CROSS = 0.20

_RUN_KNEE_CHECK_NOTE = (
    "The two knees overlap at this instant, so the knee point can land on the "
    "wrong leg. Check on the picture that it sits on the leg on the ground; "
    "if it does not, drag it there with Adjust joints under the picture."
)
_RUN_KNEE_CHECK_WARNING = (
    "Knee: your two knees overlap in this frame, so check on the picture that "
    "the knee point sits on the leg on the ground (Adjust joints moves it). A "
    "frame at the instant the foot lands keeps the legs apart."
)


def _run_knees_cross(legs: dict[str, Any] | None) -> bool:
    gap = (legs or {}).get("knee_gap")
    return gap is not None and gap < _RUN_KNEES_CROSS


# A landmark's visibility is the model's opinion of its LABEL: having called
# one leg "far", it rates that leg's knee hidden (0.37-0.48 on the treadmill
# photos, every other leg point 0.89-0.99) wherever the knee actually is. Once
# the legs are named by position, a point that low is still read, down to this.
_SWAPPED_MIN_VIS = 0.3


# A leg moved across takes its hip with it (see _name_run_legs).
_LEG_SWAP_PAIRS = ((23, 24),) + tuple(zip(_LEG_CHAINS["left"], _LEG_CHAINS["right"]))


def _swap_leg_chains(landmarks: Any) -> list[Any]:
    """A copy with the left and right hip/knee/ankle/heel/toe exchanged; the
    moved points keep their position and are read down to _SWAPPED_MIN_VIS."""
    out = list(landmarks)
    for a, b in _LEG_SWAP_PAIRS:
        for dst, src in ((a, landmarks[b]), (b, landmarks[a])):
            vis = getattr(src, "visibility", 1.0)
            if vis is not None and _SWAPPED_MIN_VIS <= vis < MIN_LANDMARK_VISIBILITY:
                vis = MIN_LANDMARK_VISIBILITY
            out[dst] = SimpleNamespace(x=src.x, y=src.y, z=getattr(src, "z", 0.0), visibility=vis)
    return out


def _run_pelvis_hip(wl: Any, camera_side: str) -> int:
    """The hip the trunk is read from: whichever is nearer the camera.

    Normally the camera-side hip. After _name_run_legs has moved a far stance
    leg across, its hidden hip sits on the camera-side index and the visible
    one on the other; the swap carries world depth with each point, so depth
    still tells them apart -- in a re-measure from the signed pose too, where
    nothing else records that a swap happened. On the runphoto2026 study the
    near hip was the shallower one on 32 of 33 photos, by 0.11-0.27 m; the
    33rd was seven runners with no score.
    """
    near = 23 if camera_side == "left" else 24
    far = 47 - near
    try:
        zn, zf = float(wl[near].z), float(wl[far].z)
    except (AttributeError, IndexError, TypeError, ValueError):
        return near
    if math.isnan(zn) or math.isnan(zf):
        return near
    return far if zf < zn else near


def _run_hip_angle(pl: Any, sh: int, pelvis: int, hip: int, knee: int) -> float:
    """Trunk against thigh, when the thigh hangs from a different hip point
    than the trunk: the trunk (shoulder -> visible hip) laid onto the stance
    hip. Shoulder -> hidden hip would tilt the trunk by the two hips' offset."""
    S, P, H, K = pl[sh], pl[pelvis], pl[hip], pl[knee]
    shifted = SimpleNamespace(
        x=H.x + (S.x - P.x), y=H.y + (S.y - P.y), z=0.0,
        visibility=min(getattr(S, "visibility", 1.0), getattr(P, "visibility", 1.0)),
    )
    val, _ = calculate_angle_2d([shifted, H, K], 0, 1, 2)
    return val


def _run_phase_by_position(legs: dict[str, Any], knee: float) -> str:
    """The stride instant from where the stance foot is, and the stance knee.

    Replaces the thigh-and-knee guess (``_estimate_run_gait_phase``) behind
    the flag: that one read whichever leg carried the near label, and had no
    name for a thigh behind the body with the knee folded (early swing), so it
    filed those frames as midstance or toe-off.
    """
    if not legs.get("clear") or legs.get("dx") is None:
        return "uncertain"
    dx = legs["dx"]
    if dx > _RUN_CONTACT_AHEAD or (dx > 0 and not math.isnan(knee) and knee >= _RUN_CONTACT_KNEE):
        return "initial_contact"
    if dx < _RUN_TOEOFF_BEHIND:
        return "toe_off"
    return "midstance"


def _running_forward_sign(landmarks) -> float:
    """Direction of travel, at the photo path's own visibility threshold."""
    return calculate_forward_sign(landmarks, min_visibility=_VIS_THRESHOLD)


def _thigh_deviation_deg(landmarks, hip_idx: int, knee_idx: int, forward: float) -> float:
    """Signed thigh angle from vertical: + = knee ahead of the hip (flexion).

    This -- not the hip angle -- is what identifies the gait phase. The
    measured hip angle is an unsigned vertex angle, so it folds at 180 deg:
    a thigh 10 deg behind vertical and one 10 deg ahead of it produce nearly
    the same reading with opposite meanings. The signed thigh deviation does
    not fold.
    """
    # The helper's sign is "first index ahead of second along the direction of
    # travel", and it takes abs() of the vertical component -- so which of the
    # two points is physically higher does not matter. (knee, hip) therefore
    # gives "+ = knee ahead of the hip" even though the knee is the lower one.
    val = calculate_signed_segment_to_vertical(
        landmarks, knee_idx, hip_idx, forward, min_visibility=_VIS_THRESHOLD,
    )
    return val if math.isnan(val) else round(val, 1)


# Foot strike from the foot's angle to the GROUND at contact: heel -> toe
# against horizontal, positive when the toe is higher than the heel.
#
# The first version read it off the shank-foot angle and got it backwards on
# exactly the frame it exists for. Shank-foot is measured against the SHIN,
# and on an overstride the shin leans back -- so the shin's lean and the foot's
# angle add up. Measured on a real touchdown: shin -15.9 deg, toe +9.4 deg above
# the heel, shank-foot ~111 deg, read as "forefoot". The landmarks said heel,
# plainly, and the overstride beside it (0.35) said the same. Foot strike is
# defined relative to the ground; so is this.
#
# Thresholds from Altman & Davis 2012 (Gait & Posture), who classify by foot
# angle at contact: > 8.1 deg rearfoot, -1.6 to 8.1 midfoot, < -1.6 forefoot.
# Theirs is relative to a standing calibration frame a single photo does not
# have, and the image's horizontal is only the ground's if the camera is level
# -- so the category is provisional, travels with the angle it came from, and
# carries a note saying it is a reading, not a verdict.
_FOOT_STRIKE_HEEL_ABOVE_DEG = 8.1
_FOOT_STRIKE_FORE_BELOW_DEG = -1.6


def _foot_strike_from_ground_angle(foot_angle: float) -> str | None:
    if foot_angle is None or math.isnan(foot_angle):
        return None
    if foot_angle > _FOOT_STRIKE_HEEL_ABOVE_DEG:
        return "heel"
    if foot_angle < _FOOT_STRIKE_FORE_BELOW_DEG:
        return "forefoot"
    return "midfoot"


def _foot_angle_to_ground(
    nl, heel_i: int, toe_i: int, forward: float, img_w: int, img_h: int,
) -> float:
    """Heel -> toe against the image horizontal, in degrees; + = toe up.

    Image-plane landmarks scaled to pixels, so the angle is the one a person
    would measure on the photo -- normalized x and y are not the same unit on
    a non-square frame. Signed along the direction of travel, so a runner going
    left reads the same as one going right. NaN when either landmark is not
    visible enough, the direction is unknown, or the toe is not ahead of the
    heel (a foot read backwards is a misread foot, not a reading).
    """
    if forward == 0.0 or max(heel_i, toe_i) >= len(nl):
        return math.nan
    heel, toe = nl[heel_i], nl[toe_i]
    if min(getattr(heel, "visibility", 1.0),
           getattr(toe, "visibility", 1.0)) < _VIS_THRESHOLD:
        return math.nan
    along = (toe.x - heel.x) * img_w * forward
    if along <= 0:
        return math.nan
    rise = (heel.y - toe.y) * img_h            # image y grows downwards
    return math.degrees(math.atan2(rise, along))


def _run_contact_metrics(
    wl, camera_side: str, forward: float, angles: dict[str, float],
    nl=None, img_w: int = 0, img_h: int = 0,
) -> dict[str, Any]:
    """What one frame at initial contact can say that no other instant can.

    Overstride, shin angle and foot strike are all DEFINED at the moment the
    foot meets the ground; at toe-off or mid-swing they are not approximate,
    they are meaningless. So the caller only asks for this when the phase
    detector has placed the frame at ``initial_contact``, and every value here
    is NaN or None rather than a number whenever its landmarks cannot be read.

    overstride_ratio
        ``|ankle_x - hip_x| / leg length``, leg length as hip->knee->ankle so a
        bent knee does not shorten it. Ported from the video analyzer's
        ``_compute_overstride_ratio`` -- the same quantity at one contact
        instead of the median over several. Under the hip reads ~0.0-0.15;
        past ~0.20 the foot is reaching out ahead, which is the braking
        heel-first landing the word describes. World landmarks (metres), as
        the video uses, so it is independent of body size and camera distance.

    shin_angle
        The knee->ankle segment from vertical, SIGNED by the direction of
        travel: positive is the knee ahead of the ankle, negative is the ankle
        ahead of the knee -- a shin leaning back towards the camera, which is
        what a reaching foot looks like from the side. Vertical is the
        textbook contact.

    foot_strike
        Heel / midfoot / forefoot, from the foot's angle to the ground (see
        ``_foot_angle_to_ground`` and why it is not the shank-foot angle).
        Needs the image-plane landmarks and the frame size; without them it is
        None rather than a guess.
    """
    _, hip_i, knee_i = RUNNING_PHOTO_ANGLES[camera_side]["hip"]
    _, ankle_i, heel_i, toe_i = RUNNING_PHOTO_ANKLE[camera_side]

    overstride = math.nan
    if max(hip_i, knee_i, ankle_i) < len(wl):
        hip, knee, ankle = wl[hip_i], wl[knee_i], wl[ankle_i]
        if min(getattr(hip, "visibility", 1.0),
               getattr(ankle, "visibility", 1.0)) >= _VIS_THRESHOLD:
            leg_len = (math.dist((hip.x, hip.y), (knee.x, knee.y))
                       + math.dist((knee.x, knee.y), (ankle.x, ankle.y)))
            if leg_len > 1e-3:
                overstride = abs(ankle.x - hip.x) / leg_len

    shin = calculate_signed_segment_to_vertical(
        wl, knee_i, ankle_i, forward, min_visibility=_VIS_THRESHOLD,
    )
    foot_angle = (
        _foot_angle_to_ground(nl, heel_i, toe_i, forward, img_w, img_h)
        if nl is not None and img_w and img_h else math.nan
    )

    return {
        "overstride_ratio": _safe_round(overstride, 3),
        "shin_angle": _safe_round(shin),
        "foot_strike": _foot_strike_from_ground_angle(foot_angle),
        "foot_angle_to_ground_deg": _safe_round(foot_angle),
        "foot_strike_note": (
            "Read from the foot's angle to the ground at contact, so it assumes "
            "the camera was level; a category, not a verdict. Heel-first is "
            "not a fault in itself -- it is the reach that comes with it that "
            "costs, and that is the overstride reading."
        ),
    }


def _estimate_run_gait_phase(thigh_dev: float, knee_angle: float) -> str:
    """Which instant of the stride a single running frame caught.

    Returns 'toe_off', 'initial_contact', 'swing', 'midstance' or 'uncertain'.
    The running analogue of :func:`_estimate_pedal_phase`.

    Thigh position alone separates extension from flexion. It cannot separate
    a leg reaching forward to land from one folded forward in swing -- both
    have the thigh ahead of the hip -- so the knee angle resolves that: nearly
    straight means the foot is reaching for the ground, folded means the heel
    is still tucked. Anything between the two is left 'uncertain' rather than
    guessed, because guessing wrong picks the wrong hip band.
    """
    if math.isnan(thigh_dev):
        return "uncertain"
    if thigh_dev <= _RUN_THIGH_BACK_DEG:
        return "toe_off"
    if thigh_dev >= _RUN_THIGH_FORWARD_DEG:
        if math.isnan(knee_angle):
            return "uncertain"
        if knee_angle >= _RUN_KNEE_EXTENDED_DEG:
            return "initial_contact"
        if knee_angle <= _RUN_KNEE_FOLDED_DEG:
            return "swing"
        return "uncertain"
    return "midstance"


def _safe_round(value: float, decimals: int = 1) -> float:
    """Round a value, returning NaN as-is."""
    if math.isnan(value):
        return value
    return round(value, decimals)


def _build_arc_triplets(
    sport: str, camera_side: str,
) -> dict[str, tuple[int, int, int]]:
    """Build arc triplet dict for thumbnail drawing."""
    if sport == "run":
        return dict(RUNNING_PHOTO_ANGLES[camera_side])
    elif sport == "bike":
        return dict(CYCLING_PHOTO_ANGLES[camera_side])
    elif sport == "swim":
        return dict(SWIMMING_PHOTO_ANGLES)
    return {}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def _preprocess_image_bytes(image_bytes: bytes) -> bytes:
    """Preprocess image: convert HEIC/HEIF to JPEG, apply EXIF orientation.

    Returns processed bytes suitable for cv2.imdecode.
    """
    # Check for HEIC/HEIF (ftyp box at offset 4)
    is_heic = (
        len(image_bytes) > 12
        and image_bytes[4:8] == b"ftyp"
    )

    if is_heic:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            logger.warning("HEIC_UNSUPPORTED", reason="pillow-heif not installed")
            return image_bytes

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    # For all other formats: apply EXIF auto-orientation if needed
    try:
        img = Image.open(io.BytesIO(image_bytes))
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            buf = io.BytesIO()
            fmt = img.format or "JPEG"
            if fmt.upper() == "MPO":
                fmt = "JPEG"
            transposed.save(buf, format=fmt, quality=95)
            return buf.getvalue()
    except Exception:
        logger.debug("Image EXIF auto-orient failed, using original", exc_info=True)

    return image_bytes


def decode_photo(image_bytes: bytes) -> Any:
    """The uploaded bytes as a BGR image -- HEIC converted, EXIF orientation
    applied. Split from detection so a re-analysis can decode the same photo
    without running the model. Raises ``ValueError`` on an undecodable file."""
    import cv2

    prepared = _preprocess_image_bytes(image_bytes)
    image = cv2.imdecode(np.frombuffer(prepared, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Could not decode image. Ensure it is a valid JPEG, PNG, WebP, or HEIC file."
        )
    return image


def detect_pose_in_image(image_bytes: bytes) -> dict[str, Any]:
    """Find one body pose in one photograph.

    Everything that is true of any single still, regardless of what the photo
    is *of*: HEIC and EXIF handling, decoding, the Tasks-API-then-legacy
    MediaPipe fallback, which side of the body faces the camera, and the three
    capture warnings that apply to every still (too small, shot from the front,
    too dark to see the joints).

    Split out of ``analyze_photo`` so the mobility screens can use it. A
    mobility photo is not a sport photo -- it is not scored, not graded and not
    compared to a fit window -- but getting a skeleton out of a JPEG is the
    same problem, and having two copies of the MediaPipe fallback ladder would
    mean fixing every future model change twice.

    Returns ``world`` (metric landmarks), ``normalized`` (image-space, carries
    ``visibility``), ``image`` (BGR), ``camera_side`` and ``warnings``.

    Raises ValueError when the image cannot be decoded or holds no pose.
    """
    import cv2
    import mediapipe as mp

    image = decode_photo(image_bytes)

    warnings: list[str] = []

    h_img, w_img = image.shape[:2]
    if min(h_img, w_img) < 480:
        warnings.append(
            "Low resolution image detected. "
            "Higher resolution photos provide more precise measurements."
        )

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    wl = None
    nl = None
    silhouette_top: list[float | None] | None = None
    # Set when the pose was only found by the last-resort low-threshold pass.
    # Travels with the result so the warnings can say so and the analytics can
    # count it -- a marginal reading that looks identical to a confident one is
    # the thing this whole ladder must not produce.
    low_confidence = False

    try:
        from pathlib import Path as _Path
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            PoseLandmarker,
            PoseLandmarkerOptions,
            RunningMode,
        )

        _model_paths = [
            settings.model_path,  # Flapp: backend/models/ (see core/config.py)
            _Path(__file__).parent / "models" / "pose_landmarker_heavy.task",
            _Path("/app/models/pose_landmarker_heavy.task"),
            _Path("pose_landmarker_heavy.task"),
        ]
        _model_path = None
        for _p in _model_paths:
            if _p.exists():
                _model_path = str(_p)
                break

        if _model_path:
            want_mask = _segmentation_mask_supported()

            def _attempt(rgb, conf: float):
                options = PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=_model_path),
                    running_mode=RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=conf,
                    min_pose_presence_confidence=conf,
                    output_segmentation_masks=want_mask,
                )
                landmarker = PoseLandmarker.create_from_options(options)
                try:
                    return landmarker.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                finally:
                    landmarker.close()

            def _smaller(rgb):
                k = _RETRY_MAX_SIDE / max(h_img, w_img)
                return cv2.resize(rgb, (round(w_img * k), round(h_img * k)),
                                  interpolation=cv2.INTER_AREA)

            # The ladder, in the order it is tried. Order is the whole design:
            #
            # 1. Full size at the normal threshold -- what every photo that
            #    works today gets, unchanged.
            # 2. Downscaled, same threshold. A clean side view at 3050 px came
            #    back "no pose" and was found at 1400 (Kona photo, 2026-10-11):
            #    the detector misses large frames it handles fine smaller.
            #    Landmarks are normalized and the silhouette edge is a fraction
            #    of the height, so a reading off the smaller copy maps straight
            #    back onto the full photo.
            # 3 and 4. The same two at a LOW threshold, and only ever last.
            #
            # The low pass is quarantined at the end because it is not a more
            # sensitive version of the same answer -- it is a different one.
            # Measured on the photo set: the detector is deterministic (two
            # runs at 0.5 agree to 0.0000), and dropping to 0.2 moves joints by
            # up to a quarter of the frame on photos that already worked. So it
            # must never be reached by a photo the normal threshold can handle;
            # the only thing it is allowed to replace is a dead end.
            #
            # What it risks is locking onto the wrong body -- the failure
            # `_bike_photo_withheld` exists for, after a race photo was scored
            # off a camera motorbike's passenger. That is why the result is
            # marked: the gate still withholds, and the athlete is told the
            # reading was marginal rather than being handed a confident-looking
            # number off a weak detection.
            # The normal threshold, full size then smaller. One low-threshold
            # attempt after them, and only one: the low pass is a single
            # mechanism, so a reading it produces and the stability check
            # rejects must end the search. Trying it again on a resized copy
            # would be asking the same unreliable question until it answered
            # the way we wanted -- which is exactly how a skeleton drawn on a
            # bicycle frame got through the first version of this.
            ladder = [("full", image_rgb, _POSE_CONFIDENCE)]
            if max(h_img, w_img) > _RETRY_MAX_SIDE:
                ladder.append(("smaller", _smaller(image_rgb), _POSE_CONFIDENCE))
            ladder.append(("low-confidence", image_rgb, _POSE_CONFIDENCE_LAST_RESORT))

            result = None
            for step, (label, rgb, conf) in enumerate(ladder):
                result = _attempt(rgb, conf)
                if not result.pose_world_landmarks:
                    result = None
                    continue
                if conf < _POSE_CONFIDENCE and not _agrees_across_scale(
                    result.pose_landmarks[0], rgb, conf, _attempt,
                ):
                    # Found something, and it moves when the image is resized.
                    # That is not a body (see _LOW_CONFIDENCE_AGREEMENT).
                    logger.info("PHOTO_POSE_UNSTABLE", attempt=label)
                    result = None
                    break
                if step:
                    logger.info(
                        "PHOTO_POSE_FOUND_ON_RETRY", attempt=label,
                        side=max(h_img, w_img), confidence=conf,
                    )
                low_confidence = conf < _POSE_CONFIDENCE
                break
            else:
                logger.info("PHOTO_POSE_NOT_FOUND", side=max(h_img, w_img),
                            attempts=len(ladder))

            if result is not None and result.pose_world_landmarks:
                wl = result.pose_world_landmarks[0]
                nl = result.pose_landmarks[0]
                if want_mask and result.segmentation_masks:
                    silhouette_top = top_contour(result.segmentation_masks[0].numpy_view())
    except Exception as tasks_err:
        logger.warning("Tasks API failed for photo, trying legacy", err=str(tasks_err))

    # Fallback: Legacy Solutions API (mediapipe < 0.10.31)
    if wl is None:
        try:
            mp_pose = mp.solutions.pose
            with mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            ) as pose:
                result = pose.process(image_rgb)
            if result.pose_world_landmarks:
                wl = result.pose_world_landmarks.landmark
                nl = result.pose_landmarks.landmark
        except AttributeError:
            pass  # mp.solutions removed in mediapipe >= 0.10.31

    if wl is None:
        raise ValueError(
            "No body pose detected in the image. "
            "Ensure the full body is visible and well-lit."
        )

    camera_side = _detect_camera_side(wl)

    # Warning: front/back view (not ideal side view)
    left_z = (wl[11].z + wl[23].z) / 2   # left shoulder + hip
    right_z = (wl[12].z + wl[24].z) / 2   # right shoulder + hip
    if abs(left_z - right_z) < 0.05:
        warnings.append(
            "Side view works best for position analysis. "
            "The photo appears to be from the front or back, "
            "which may give less accurate angle measurements."
        )

    # Warning: poor landmark visibility (bad lighting / occlusion)
    key_indices = [11, 12, 23, 24, 25, 26, 27, 28]
    avg_vis = sum(nl[i].visibility for i in key_indices) / len(key_indices)
    if avg_vis < 0.5:
        warnings.append(
            "Image quality appears low. "
            "Try a clearer, well-lit photo for more accurate results."
        )

    if low_confidence:
        # Said plainly and first, because this reading is weaker than the ones
        # the rest of the report is written as though it were. Without it the
        # athlete cannot tell a marginal detection from a confident one, and
        # the numbers look equally certain either way.
        warnings.insert(0, (
            "We could only just make out your body in this photo, so these "
            "measurements are less reliable than usual. A sharper side-on "
            "shot with your whole body in frame would be worth retaking."
        ))

    return {
        "world": wl,
        "normalized": nl,
        "image": image,
        "camera_side": camera_side,
        "avg_visibility": avg_vis,
        "low_confidence": low_confidence,
        "warnings": warnings,
        # The silhouette's top edge, for the helmet-above-back reading (see
        # biomechanics/head_silhouette.py). None on the legacy path or a
        # MediaPipe that cannot hand the mask over safely.
        "silhouette_top": silhouette_top,
    }


# Longest side of the downscaled copy a large photo is retried on when the
# detector finds no pose at full size (see detect_pose_in_image).
_RETRY_MAX_SIDE = 1400

# The detector's confidence threshold, and the one used only after every
# attempt at the normal threshold has come back empty.
#
# These are two different questions, not one knob with two settings. Measured
# on the stored photo set: the detector is deterministic -- the same image at
# 0.5 twice gives landmarks identical to four decimal places -- and the same
# image at 0.2 can put joints a QUARTER of the frame away from where 0.5 put
# them. So the low threshold does not find the same skeleton more readily; it
# finds a different skeleton. Running everything at 0.2 would silently change
# the angles of every photo that works today.
#
# It earns its place only where the alternative is nothing at all: a photo the
# normal threshold cannot see is currently a dead end that costs the athlete
# their attempt and tells them to try a better photo. A marginal reading they
# are warned about, and which the score gate can still withhold, is worth more
# than that.
_POSE_CONFIDENCE = 0.5
_POSE_CONFIDENCE_LAST_RESORT = 0.2

# A low-threshold detection is believed only if the same photo, fed at a
# different size, puts the joints in the same place.
#
# The low pass does not simply find bodies more readily -- it also assembles
# them out of whatever is lying around. Measured on the stored photos: on a
# race shot it put a correct skeleton on the rider; on another it draped one
# over the frame and the saddle while the rider's own legs, plainly visible,
# got nothing. Average landmark visibility does not separate those two (0.99
# and 0.83), so believing the flag alone would turn some dead ends into
# confident reports measured off a bicycle.
#
# What does separate them is stability. A body is in the same place whatever
# resolution it is found at; a pose assembled out of frame tubes is an
# artefact of one particular rasterisation and moves when that changes.
# Measured, median joint displacement between two scales:
#
#     correct skeleton on the rider      0.010, 0.020
#     skeleton on the bicycle frame      0.062
#     skeleton across a photo collage    0.166, 0.286
#
# Nothing lands between 0.020 and 0.062, so the cut sits in that gap. Six
# photos is a small sample for a threshold, which is the other reason the
# accepted reading is still flagged, still warned about, and still subject to
# the score gate rather than trusted outright.
_LOW_CONFIDENCE_AGREEMENT = 0.035
# Verified against SEVERAL sizes, and judged on the worst of them. One
# comparison is not enough: a skeleton draped over a bicycle frame agreed
# with itself between 1252 px and 900 px and disagreed at 1000 px, so a single
# check is a coin flip. Across three it has nowhere to hide -- the numbers in
# the comment above are each the worst of three.
_LOW_CONFIDENCE_VERIFY_SIDES = (1000, 800, 640)
# Shoulders, elbows, hips, knees, ankles -- the joints the analysis measures.
_AGREEMENT_JOINTS = (11, 12, 13, 14, 23, 24, 25, 26, 27, 28)


def _agrees_across_scale(found, rgb, conf: float, attempt) -> bool:
    """Whether the same pose turns up when the photo is fed at another size.

    ``found`` is the normalized landmark list already detected in ``rgb``;
    ``attempt`` runs the detector. Each size is compared on the MEDIAN joint
    displacement -- so one landmark on a limb the model is genuinely unsure
    about cannot veto an otherwise solid reading -- and the verdict is the
    WORST of those medians, because an artefact only has to hold still once to
    look convincing.

    Answers True when the check cannot be made at all: every verification size
    is larger than the photo, or no second pass found anything. A guard that
    cannot run should not silently become a rejection; the reading is flagged
    and gated regardless.
    """
    import cv2  # imported lazily here, as everywhere else in this module

    h, w = rgb.shape[:2]
    worst = 0.0
    checks = 0
    for side in _LOW_CONFIDENCE_VERIFY_SIDES:
        if max(h, w) <= side:
            continue
        k = side / max(h, w)
        other = attempt(
            cv2.resize(rgb, (round(w * k), round(h * k)),
                       interpolation=cv2.INTER_AREA),
            conf,
        )
        if not other.pose_landmarks:
            continue
        them = other.pose_landmarks[0]
        checks += 1
        worst = max(worst, float(np.median([
            np.hypot(found[i].x - them[i].x, found[i].y - them[i].y)
            for i in _AGREEMENT_JOINTS
        ])))
    if not checks:
        return True
    if worst > _LOW_CONFIDENCE_AGREEMENT:
        logger.info(
            "PHOTO_POSE_SCALE_DISAGREEMENT", spread=round(worst, 4), checks=checks,
        )
        return False
    return True


def _segmentation_mask_supported() -> bool:
    """Whether this MediaPipe can return the person mask without dying.

    0.10.35 aborts the whole process (a C++ CHECK, not an exception) the
    moment a float segmentation mask is read -- on Windows and on Linux alike,
    verified 2026-10-11. 1.x reads it. A CHECK cannot be caught, so the mask is
    never even requested from an older build.
    """
    import mediapipe as mp

    try:
        return int(str(mp.__version__).split(".")[0]) >= 1
    except (ValueError, AttributeError):
        return False


def analyze_photo(
    image_bytes: bytes,
    sport: str,
    cycling_position: str | None = None,
    hide_angle_values: bool = False,
    contact_metrics: bool = False,
) -> dict[str, Any]:
    """Analyze a single photo for body position feedback.

    This is a BLOCKING function (MediaPipe CPU work).
    The API endpoint wraps it in asyncio.to_thread().

    Args:
        image_bytes: Raw image file bytes (JPEG/PNG/WebP)
        sport: "run", "bike", or "swim"
        cycling_position: Optional cycling position for bike

    Returns:
        Dict matching PhotoAnalysisResponse schema.

    Raises:
        ValueError: If image invalid or no pose detected.
    """
    start_time = time.time()

    # 1-3. Decode, find the pose, work out which side faces the camera, and
    # collect the warnings that apply to any still (see detect_pose_in_image).
    pose_result = detect_pose_in_image(image_bytes)

    # 4-8. Everything measured, from the pose alone -- see analyze_from_pose.
    response = analyze_from_pose(
        pose_result["image"], pose_result["world"], pose_result["normalized"],
        pose_result["camera_side"], list(pose_result["warnings"]),
        sport, cycling_position, hide_angle_values=hide_angle_values,
        silhouette_top=pose_result.get("silhouette_top"),
        contact_metrics=contact_metrics,
    )
    response["processing_time_seconds"] = round(time.time() - start_time, 3)

    if sport == "bike":
        # The detected pose, signed, so the athlete can hand it back with
        # their corrections and have the photo re-measured without a second
        # model pass -- see services/photo_corrections.
        from app.services.photo_corrections import build_pose_blob

        response["pose"] = build_pose_blob(
            image_bytes, pose_result, helmet_points="helmet_points" in response,
        )
    elif sport == "run" and contact_metrics:
        # The same, for a run read behind the flag -- but with the legs named
        # by position first, so the leg the athlete drags is the stance leg
        # the report measured (see build_pose_blob).
        from app.services.photo_corrections import build_pose_blob

        nl, wl, _ = _name_run_legs(
            pose_result["normalized"], pose_result["world"],
            pose_result["image"], pose_result["camera_side"],
        )
        response["pose"] = build_pose_blob(
            image_bytes, {**pose_result, "normalized": nl, "world": wl}, sport="run",
        )
    return response


def analyze_from_pose(
    image: Any,
    wl: Any,
    nl: Any,
    camera_side: str,
    warnings: list[str],
    sport: str,
    cycling_position: str | None = None,
    *,
    hide_angle_values: bool = False,
    corrections: list[dict[str, Any]] | None = None,
    render_thumbnail: bool = True,
    silhouette_top: list[float | None] | None = None,
    helmet_offsets: dict[int, tuple[float, float]] | None = None,
    contact_metrics: bool = False,
) -> dict[str, Any]:
    """The measuring half of :func:`analyze_photo`: angles, score, picture.

    ``contact_metrics`` (running only): also apply the run "no score" rule
    and, on a frame at initial contact, measure overstride, shin angle and
    foot strike (``_run_contact_metrics``). Off by default so that nothing an
    athlete sees changes until the readings have been checked on real frames
    -- the field-study tool and admin accounts pass True. With it off this
    function is byte-for-byte what it was.

    Takes what :func:`detect_pose_in_image` found -- the decoded image, the
    world and image-plane landmarks, the camera side and the capture warnings
    -- and needs no model. That is what lets a photo be re-measured after the
    athlete moves a joint point (``services/photo_corrections``), and what
    lets this half be tested on a hand-built pose.

    ``corrections``: the athlete's adjustments already applied to ``nl`` by
    the caller. Echoed into the result and ringed on the thumbnail, never
    applied here. ``render_thumbnail=False`` skips the picture (a baseline
    measurement nobody will look at). ``silhouette_top``: the person mask's top
    edge from :func:`detect_pose_in_image`, for the helmet-above-back reading;
    without it that row is simply absent. ``helmet_offsets``: the athlete's
    own top-of-back / top-of-helmet placement (head_silhouette.BACK_TOP_POINT
    / HELMET_TOP_POINT -> dx, dy), applied to the silhouette's guess.
    """
    import cv2  # for the annotated thumbnail

    start_time = time.time()
    warnings = list(warnings)

    # 4. Compute angles (sport-specific)
    angles: dict[str, float] = {}
    helmet: dict[str, Any] | None = None  # bike only: see head_silhouette.py
    helmet_scored = False
    pedal_phase: str | None = None  # Only set for cycling
    gait_phase: str = "uncertain"   # Only meaningful for running
    thigh_dev: float = math.nan     # Only measured for running
    # Running, behind ``contact_metrics``: the contact-instant readings, or
    # why there are none. Stays None for every other sport and for callers
    # that did not ask.
    contact: dict[str, Any] | None = None
    # Running, behind the flag: the visible hip's index when the stance leg's
    # own hip is a different point (a far stance leg), else None.
    run_pelvis_hip: int | None = None

    if sport == "run":
        # Behind the flag, the run reads the picture like the bike does
        # (_image_plane): the world skeleton's per-side bias moves the knee,
        # and the stride phase is decided on the knee and the thigh, so the
        # bias could file a frame under the wrong phase and judge it against
        # the wrong bands. Off, the world skeleton stays -- users' numbers do
        # not move until the flag does.
        #
        # And behind the same flag the measured leg is the STANCE leg, found by
        # position (see _name_run_legs); its landmarks are moved onto the
        # camera-side indices, so everything below -- and the picture --
        # follows it without knowing.
        legs: dict[str, Any] | None = None
        if contact_metrics:
            nl, wl, legs = _name_run_legs(nl, wl, image, camera_side)
        rl = _image_plane(nl, image) if contact_metrics else wl
        for name, (a, b, c) in RUNNING_PHOTO_ANGLES[camera_side].items():
            val, _ = calculate_angle_2d(rl, a, b, c)
            angles[name] = _safe_round(val)

        # Ankle: shank axis vs foot axis (see RUNNING_PHOTO_ANKLE).
        k_i, a_i, h_i, t_i = RUNNING_PHOTO_ANKLE[camera_side]
        ankle_val, _ = calculate_shank_foot_angle_2d(rl, k_i, a_i, h_i, t_i)
        angles["ankle"] = _safe_round(ankle_val)

        # Direction of travel first: both the trunk sign and the stride phase
        # are meaningless without it.
        forward_sign = _running_forward_sign(rl)

        # Signed, so leaning BACK is distinguishable from leaning forward. The
        # unsigned reading reported both as the same positive number, which let
        # a backward lean land mid-band and score as optimal. NaN when the
        # direction is unknown (feet occluded): the trunk then drops out of the
        # score rather than being judged against a band it cannot be placed on,
        # and the partial-body warning below already tells the athlete why.
        sh_idx, hp_idx = RUNNING_TRUNK_LANDMARKS[camera_side]
        if contact_metrics:
            # A far stance leg brought its own (hidden) hip across; the trunk
            # stays on the visible one, and the hip angle is trunk vs thigh.
            pelvis = _run_pelvis_hip(wl, camera_side)
            if pelvis != hp_idx:
                run_pelvis_hip = pelvis
                angles["hip"] = _safe_round(_run_hip_angle(
                    rl, sh_idx, pelvis, hp_idx, RUNNING_PHOTO_ANGLES[camera_side]["hip"][2]))
                hp_idx = pelvis
        trunk = calculate_signed_segment_to_vertical(
            rl, sh_idx, hp_idx, forward_sign, min_visibility=_VIS_THRESHOLD,
        )
        angles["trunk"] = _safe_round(trunk)

        # Which instant of the stride this frame caught. Has to be resolved
        # before any band is picked: the hip band inverts across the cycle
        # (see _RUN_HIP_BANDS), so choosing one without the phase is what made
        # a hand-picked knee-drive frame come back as "hip too closed".
        _, run_hip_idx, run_knee_idx = RUNNING_PHOTO_ANGLES[camera_side]["hip"]
        thigh_dev = _thigh_deviation_deg(rl, run_hip_idx, run_knee_idx, forward_sign)
        if legs is not None:
            gait_phase = _run_phase_by_position(legs, angles.get("knee", math.nan))
        else:
            gait_phase = _estimate_run_gait_phase(
                thigh_dev, angles.get("knee", math.nan),
            )
        logger.info(
            "PHOTO_RUN_GAIT_PHASE",
            gait_phase=gait_phase,
            thigh_deviation=thigh_dev,
            knee_angle=angles.get("knee"),
            legs=legs,
        )

        optimal_ranges = _get_running_optimal_ranges(
            gait_phase, phased_knee=contact_metrics,
        )

        if contact_metrics:
            if gait_phase == "initial_contact":
                img_h, img_w = image.shape[:2]
                contact = {"phase": gait_phase, "measured": True,
                           **_run_contact_metrics(rl, camera_side, forward_sign, angles,
                                                  nl, img_w, img_h)}
                # The verdict the score uses, so the report's touchdown block
                # cannot colour it differently. The reach is information only
                # (see _RUN_SHIN_BAND).
                shin = contact.get("shin_angle")
                if shin is not None and not math.isnan(shin):
                    contact["shin_band"] = list(_RUN_SHIN_BAND)
                    contact["shin_status"] = _classify_angle_status(
                        shin, *_RUN_SHIN_BAND, metric="shin",
                    )
                ratio = contact.get("overstride_ratio")
                if ratio is not None and not math.isnan(ratio):
                    contact["overstride_note"] = _RUN_OVERSTRIDE_INFO
            else:
                # Not an approximation of the contact readings -- there are
                # none on this frame. Say which instant it is and what would
                # have them.
                contact = {
                    "phase": gait_phase, "measured": False,
                    "overstride_ratio": None, "shin_angle": None,
                    "foot_strike": None,
                    "note": (
                        "Overstride, shin angle and foot strike exist only at "
                        "the instant the foot lands. This frame is "
                        f"{'between phases' if gait_phase == 'uncertain' else 'at ' + gait_phase.replace('_', ' ')}"
                        "; a frame at touchdown would have them."
                    ),
                }
            # Which leg the numbers are from, said by position: the model's
            # left/right is not trustworthy on one frame (see _name_run_legs).
            contact["legs"] = legs

    elif sport == "bike":
        # Image-plane landmarks, the same source the VIDEO bike path moved to
        # (commit 219c20c): MediaPipe's world skeleton read the same fit ~8
        # deg too straight from the left and ~8 deg too bent from the right,
        # and the photo path kept that bias -- a photo and a video of one
        # bike could disagree about the rider's own knee. The preview already
        # draws the normalized landmarks; now the printed angles come from
        # the same body. Normalized x spans the width and y the height, so x
        # is rescaled by the aspect before any angle is read.
        _h_img, _w_img = image.shape[:2]
        _aspect = (_w_img / _h_img) if _h_img else 1.0
        pl = _image_plane(nl, image)

        for name, (a, b, c) in CYCLING_PHOTO_ANGLES[camera_side].items():
            val, _ = calculate_angle_2d(pl, a, b, c)
            angles[name] = _safe_round(val)

        # The NEAR shoulder and hip, like every other bike angle here. This read
        # 11/23 (left) whatever side faced the camera, so on a right-side photo
        # the trunk came from the far, half-hidden side -- and went NaN, taking
        # the pelvic ratio with it, whenever that side fell under the
        # visibility threshold.
        _sh, _hp = (11, 23) if camera_side == "left" else (12, 24)
        trunk_from_vert = calculate_segment_to_vertical(pl, _sh, _hp)
        trunk = 90.0 - trunk_from_vert  # Convert to from-horizontal (bike fitting convention)
        angles["trunk"] = _safe_round(trunk)

        # Forearm tilt (atan2-based, not standard 3-point angle)
        if camera_side == "left":
            forearm_tilt, _ = calculate_forearm_tilt_2d(pl, 13, 15)
        else:
            forearm_tilt, _ = calculate_forearm_tilt_2d(pl, 14, 16)
        angles["forearm_tilt"] = _safe_round(forearm_tilt)

        # Helmet above the top of the back, off the silhouette (head_silhouette.py).
        # It replaces the "Head position" score: ear below a line through the
        # hip and shoulder joint centres, a line that runs inside the body, so
        # every aero rider got 100/100 -- all fifteen Kona pros did -- and the
        # helmet, the part that actually sits in the wind, was never looked at.
        # Measured and shown, not scored: no fit source sets a target for it.
        if silhouette_top:
            helmet = helmet_above_back(
                silhouette_top, (nl[_sh].x, nl[_sh].y), (nl[_hp].x, nl[_hp].y),
                (nl[0].x, nl[0].y), _aspect,
            )
        if helmet is not None and helmet_offsets:
            helmet = apply_point_corrections(helmet, helmet_offsets)
        if helmet is not None:
            angles["helmet_above_back"] = helmet["value"]
            if helmet.get("check"):
                warnings.append(_HELMET_CHECK_WARNING)

        # Pelvic ratio (derived: hip / trunk)
        hip_val = angles.get("hip")
        trunk_val = angles.get("trunk")
        if (hip_val is not None and trunk_val is not None
                and trunk_val >= 5
                and not math.isnan(hip_val) and not math.isnan(trunk_val)):
            angles["pelvic_ratio"] = round(hip_val / trunk_val, 2)

        # The crank position: a guess from the knee, overrulable by the rider.
        # Everything that depends on it -- knee and hip bands, the score --
        # is computed for all three positions below, so the client can switch
        # without asking the server again.
        knee_val = angles.get("knee")
        pedal_phase = _estimate_pedal_phase(knee_val) if knee_val is not None else "uncertain"

        if not cycling_position:
            cycling_position = _auto_detect_cycling_position(trunk)
            logger.info(
                "PHOTO_CYCLING_AUTODETECT",
                detected_position=cycling_position,
                trunk_angle=trunk,
            )

        # The helmet term joins the score only on an aero position and only
        # when the reading can be trusted: no "check the point" flag, or a
        # point the athlete placed. A flagged reading can be 10 points low.
        helmet_scored = (
            helmet is not None and cycling_position in _AERO_POSITIONS
            and graded_helmet(helmet.get("value"), helmet.get("check")) is not None
        )
        optimal_ranges = _with_helmet_band(
            _get_cycling_optimal_ranges(cycling_position, pedal_phase), helmet_scored,
        )
        phase_bands = _bike_phase_bands(cycling_position)
        hip_phase = pedal_phase

    elif sport == "swim":
        for name, (a, b, c) in SWIMMING_PHOTO_ANGLES.items():
            val, _ = calculate_angle_3d(wl, a, b, c)
            angles[name] = _safe_round(val)

        body_rot = abs(calculate_body_rotation(wl))
        angles["body_rotation"] = _safe_round(body_rot)

        streamline = calculate_segment_to_vertical(wl, 11, 27)
        angles["streamline"] = _safe_round(streamline)

        optimal_ranges = _get_swimming_optimal_ranges()

    else:
        raise ValueError(
            f"Unsupported sport: {sport}. Must be run, bike, or swim."
        )

    # Warning: partial body (some angles are NaN)
    nan_count = sum(1 for v in angles.values() if math.isnan(v))
    if 0 < nan_count < len(angles):
        measured = len(angles) - nan_count
        warnings.append(
            f"Partial body detected: {measured} of {len(angles)} angles could be measured. "
            "Ensure full body is visible for complete analysis."
        )

    # 5. Build angles_with_context
    labels = _ANGLE_LABELS.get(sport, {})
    angles_with_context: dict[str, dict[str, Any]] = {}

    for angle_name, angle_value in angles.items():
        lbl = labels.get(angle_name, angle_name.replace("_", " ").title())
        # Bike knee and hip: interactive phase verdict. Ship both BDC/TDC bands
        # + the guessed phase so the client can re-score if the rider corrects
        # it; score now against the guessed band, or leave unscored if unsure.
        if sport == "bike" and angle_name in ("knee", "hip"):
            bands = phase_bands[angle_name]
            entry: dict[str, Any] = {
                "value": angle_value, "label": lbl,
                "bands": bands, "phase": pedal_phase,
                "note": _BIKE_PHASE_NOTES[angle_name],
            }
            knee_slack = angle_name == "knee"
            if knee_slack:
                entry["slack"] = {p: list(s) for p, s in _KNEE_SLACK.items()}
            if pedal_phase in ("bdc", "tdc"):
                lo, hi = bands[pedal_phase]
                entry["optimal_min"], entry["optimal_max"] = lo, hi
                entry["status"] = _classify_angle_status(
                    angle_value, lo, hi,
                    slack=_KNEE_SLACK[pedal_phase] if knee_slack else None,
                )
            else:
                entry["optimal_min"] = entry["optimal_max"] = None
                entry["status"] = "phase_dependent"
                entry["unscored_reason"] = "pedal position"
            angles_with_context[angle_name] = entry
            continue
        # Bike pelvic ratio: hip over trunk, so it sweeps with the hip, and its
        # band is a stroke average. Shown, never judged from a still.
        if sport == "bike" and angle_name == "pelvic_ratio":
            angles_with_context[angle_name] = {
                "value": angle_value, "label": lbl,
                "optimal_min": None, "optimal_max": None,
                "status": "phase_dependent",
                "unscored_reason": "needs a clip",
                "note": ("Hip angle over torso angle. The hip opens and closes with "
                         "every pedal stroke, so one frame cannot place it against "
                         "the range -- a clip measures it over whole revolutions."),
            }
            continue
        # Helmet above the back line: measured, never judged -- no fit source
        # names a target. The rider gets the field it was calibrated on.
        if sport == "bike" and angle_name == "helmet_above_back":
            lo, hi = KONA_PRO_RANGE
            if helmet_scored:
                free = KONA_PRO_FREE_UP_TO
                high = angle_value > free
                angles_with_context[angle_name] = {
                    "value": angle_value, "label": lbl,
                    "optimal_min": None, "optimal_max": free,
                    "status": "acceptable" if high else "optimal",
                    "note": _HELMET_NOTE + (" " + _HELMET_HIGH_NOTE if high else ""),
                }
            else:
                flagged = bool(helmet and helmet.get("check"))
                angles_with_context[angle_name] = {
                    "value": angle_value, "label": lbl,
                    "optimal_min": None, "optimal_max": None,
                    "status": "phase_dependent",
                    "unscored_reason": ("check the green point" if flagged
                                        else f"Kona pros {lo:+.0f} to {hi:+.0f}%"),
                    "note": _HELMET_NOTE + (" " + _HELMET_CHECK_WARNING if flagged else ""),
                }
            continue
        # Bike ankle: measured, but never judged from a still (2D ankle too noisy).
        if sport == "bike" and angle_name == "ankle":
            angles_with_context["ankle"] = {
                "value": angle_value, "label": lbl,
                "optimal_min": None, "optimal_max": None,
                "status": "phase_dependent",
                "unscored_reason": "pedal position",
                "note": "Depends on the pedal position — not scored from a single photo.",
            }
            continue
        # Run hip: scored only at the two instants where the band means
        # something (footstrike, toe-off). Everywhere else it is measured and
        # shown, but carries no verdict -- see _RUN_HIP_BANDS.
        if sport == "run" and angle_name == "hip":
            entry = {
                "value": angle_value, "label": lbl, "phase": gait_phase,
                "thigh_deviation": thigh_dev if not math.isnan(thigh_dev) else None,
            }
            band = optimal_ranges.get("hip")
            if band is not None:
                entry["optimal_min"], entry["optimal_max"] = band
                entry["status"] = _classify_angle_status(angle_value, *band)
                entry["note"] = _RUN_HIP_SCORED_NOTES.get(gait_phase, "")
            else:
                entry["optimal_min"] = entry["optimal_max"] = None
                entry["status"] = "phase_dependent"
                entry["unscored_reason"] = "stride position"
                entry["note"] = _RUN_HIP_UNSCORED_NOTES.get(
                    gait_phase, _RUN_HIP_UNSCORED_NOTES["uncertain"],
                )
            angles_with_context["hip"] = entry
            continue
        # Run knee on a swing frame (phased_knee): shown, never judged. Without
        # this branch the generic path below would DROP the row, and the
        # athlete would lose the number rather than just the verdict.
        if sport == "run" and angle_name == "knee" and "knee" not in optimal_ranges:
            angles_with_context["knee"] = {
                "value": angle_value, "label": lbl, "phase": gait_phase,
                "optimal_min": None, "optimal_max": None,
                "status": "phase_dependent",
                "unscored_reason": "stride position",
                "note": _RUN_KNEE_SWING_NOTE,
            }
            continue
        if sport == "run" and angle_name == "elbow" and contact_metrics \
                and "elbow" not in optimal_ranges:
            angles_with_context["elbow"] = {
                "value": angle_value, "label": lbl,
                "optimal_min": None, "optimal_max": None,
                "status": "phase_dependent",
                "unscored_reason": "arm swing",
                "note": _RUN_ELBOW_UNSCORED_NOTE,
            }
            continue
        if angle_name not in optimal_ranges:
            continue
        opt_min, opt_max = optimal_ranges[angle_name]
        # Bike trunk in an aero position, BELOW the comfort band: the scorer
        # already forgives this side (_AERO_TOLERANCES), so classifying it as
        # "needs_work" was the one place a world-class tuck got filed under
        # "fix this first". Give it its own status instead.
        if (
            sport == "bike"
            and angle_name == "trunk"
            and cycling_position in _AERO_POSITIONS
            and not math.isnan(angle_value)
            and angle_value < opt_min
        ):
            # Amber with the concern, not "aero gain": the same reading on the
            # video, the overlay and the fit plan (AERO_TRUNK_LOW_CONCERN).
            # Below AERO_TRUNK_EXTREME_DEG the sustainability caveat is added.
            extreme = angle_value < AERO_TRUNK_EXTREME_DEG
            note = AERO_TRUNK_LOW_CONCERN.format(lo=opt_min, hi=opt_max)
            if extreme:
                note += " " + _AERO_TRUNK_EXTREME_NOTE
            angles_with_context[angle_name] = {
                "value": angle_value,
                "optimal_min": opt_min,
                "optimal_max": opt_max,
                "status": "acceptable",
                "label": lbl,
                "note": note,
            }
            continue
        # Aero-bar shoulder closed past the band: a choice, not a fault --
        # amber at most, with the reason (see SHOULDER_CLOSED_OK_POSITIONS).
        if (
            sport == "bike"
            and angle_name == "shoulder"
            and cycling_position in SHOULDER_CLOSED_OK_POSITIONS
            and not math.isnan(angle_value)
            and angle_value < opt_min
        ):
            angles_with_context[angle_name] = {
                "value": angle_value,
                "optimal_min": opt_min,
                "optimal_max": opt_max,
                "status": "acceptable",
                "label": lbl,
                "note": _SHOULDER_CLOSED_NOTE,
            }
            continue
        angles_with_context[angle_name] = {
            "value": angle_value,
            "optimal_min": opt_min,
            "optimal_max": opt_max,
            "status": _classify_angle_status(
                angle_value, opt_min, opt_max, metric=angle_name,
            ),
            "label": lbl,
        }

    # Running, behind the flag: the knees overlap, so the stance knee may be
    # drawn on the swing leg. Still scored -- on the study frames it was right
    # seven times in eight -- but the athlete is told where to look.
    if (sport == "run" and contact is not None and "knee" in angles_with_context
            and _run_knees_cross(contact.get("legs"))):
        knee_row = angles_with_context["knee"]
        knee_row["check"] = True
        knee_row["note"] = " ".join(filter(None, (_RUN_KNEE_CHECK_NOTE, knee_row.get("note"))))
        warnings.append(_RUN_KNEE_CHECK_WARNING)

    # 6. Score. On a bike it is computed for every crank position, so the
    # rider's correction of the guessed one changes the number as well as the
    # table -- the server stays the only scorer.
    score_by_phase: dict[str, dict[str, Any]] | None = None
    if sport == "bike":
        withheld = _bike_photo_withheld(angles, cycling_position)
    elif sport == "run" and contact_metrics:
        withheld = _run_photo_withheld(angles, gait_phase, legs)
    else:
        withheld = None
    if withheld:
        score_data = {"overall_score": None, "grade": None, "per_angle": {},
                      "withheld": withheld}
        warnings.append(_WITHHELD_MESSAGES[withheld])
    elif sport == "bike":
        score_by_phase = {
            p: _score_photo_angles(
                angles, _with_helmet_band(
                    _get_cycling_optimal_ranges(cycling_position, p), helmet_scored,
                ), sport,
                _BIKE_PHASE_WEIGHTS[p], cycling_position=cycling_position,
                pedal_phase=p,
            )
            for p in BIKE_PHOTO_PHASES
        }
        # Crank position unknown: the score is the upper body only, and every
        # surface that prints it says so -- the badge, the picture, the phase
        # picker. A bare 100 on such a frame read as a perfect position
        # (Gabriel's photo, 2026-10-13); it was four measures out of six.
        score_by_phase["uncertain"]["partial"] = "upper body only"
        score_data = score_by_phase[pedal_phase]
    elif sport == "run" and contact and contact.get("shin_status"):
        # Footstrike with a measured shin: the shin is scored alongside the
        # angles. Kept out of ``angles`` on purpose -- that dict is the table
        # and the picture's labels; the shin is reported in the contact block.
        shin, (lo, _) = contact["shin_angle"], _RUN_SHIN_BAND
        score_data = _score_photo_angles(
            {**angles, "shin": shin},
            {**optimal_ranges, "shin": _RUN_SHIN_BAND},
            sport,
            {**PHOTO_RUNNING_WEIGHTS, "shin": _RUN_SHIN_WEIGHT},
        )
        if shin < lo:
            # Said in words: a lower number over green joint labels, with the
            # reason nowhere on the picture, reads as the score being wrong.
            warnings.insert(0, (
                f"Shin at touchdown: it leans back {abs(shin):.0f}° from vertical, "
                "so the foot lands out in front of the knee and brakes the "
                f"stride. Within {abs(lo):.0f}° of vertical is the range we "
                "score against. A slightly quicker cadence usually brings the "
                "foot back under you."
            ))
    else:
        score_data = _score_photo_angles(
            angles, optimal_ranges, sport, None, cycling_position=cycling_position,
        )
        if sport == "run" and contact_metrics and "knee" not in optimal_ranges:
            # A swing frame scores trunk, arm and ankle only: the knee and hip
            # carry no verdict at this instant. The bike learned that a bare
            # 100 on such a frame reads as a perfect position (Gabriel's photo,
            # 2026-10-13); a world-class runner's swing frame came back 100 on
            # three measures out of five. Same flag, in the sport's own words
            # -- the bike's say "upper body" and "mark the pedal position".
            score_data["partial"] = "without the knee or hip"
            score_data["partial_badge"] = "partial"
            score_data["partial_hint"] = (
                "a frame at the instant the foot lands scores the knee, the hip "
                "and the shin"
            )

    # 7. Thumbnail
    thumbnail_b64: str | None = None
    if render_thumbnail:
        arc_triplets = _build_arc_triplets(sport, camera_side)
        thumbnail_bytes = _generate_photo_thumbnail(
            cv2, image, nl,
            angles, score_data, camera_side, sport, arc_triplets,
            cycling_position=cycling_position,
            hide_angle_values=hide_angle_values,
            optimal_ranges=optimal_ranges,
            corrected=[int(c["landmark"]) for c in (corrections or [])],
            row_status={k: v.get("status") for k, v in angles_with_context.items()},
            helmet=helmet,
            overstride=(contact if contact and contact.get("measured")
                        and contact.get("overstride_ratio") is not None
                        and not math.isnan(contact["overstride_ratio"]) else None),
            run_pelvis_hip=run_pelvis_hip,
            shin=(contact if contact and contact.get("shin_status") else None),
        )
        thumbnail_b64 = (
            f"data:{_THUMB_MIME};base64,{base64.b64encode(thumbnail_bytes).decode()}"
        )

    processing_time = round(time.time() - start_time, 3)

    # 8. Build response
    response: dict[str, Any] = {
        "sport": sport,
        "camera_side": camera_side,
        "angles": angles,
        "angles_with_context": angles_with_context,
        "score": score_data,
        "thumbnail_base64": thumbnail_b64,
        "processing_time_seconds": processing_time,
        "warnings": warnings,
        # The athlete's own adjustments to the points these numbers were read
        # from, when there are any -- so no reader takes an adjusted
        # measurement for an automatic one.
        "corrections": list(corrections) if corrections else None,
    }
    if helmet is not None:
        # Where the two points were read, so the editor can offer them for
        # dragging: auto_* is the silhouette's own guess, which the client's
        # stored corrections are offsets from.
        response["helmet_points"] = {
            "back": helmet["back_point"], "helmet": helmet["head_point"],
            "auto_back": helmet.get("auto_back_point", helmet["back_point"]),
            "auto_helmet": helmet.get("auto_head_point", helmet["head_point"]),
            "check": bool(helmet.get("check")), "moved": helmet.get("moved", []),
        }

    if sport == "run":
        response["gait_phase"] = gait_phase
        if contact_metrics:
            response["contact"] = contact
        # Tell the athlete how to get the verdict the frame could not give,
        # rather than leaving a silent gap in the table.
        if "hip" not in optimal_ranges:
            warnings.append(
                "Hip angle is measured but not scored: "
                + _RUN_HIP_UNSCORED_NOTES.get(
                    gait_phase, _RUN_HIP_UNSCORED_NOTES["uncertain"],
                )
            )

    if sport == "bike":
        response["cycling_position"] = cycling_position
        response["cycling_position_label"] = get_position_label(cycling_position)
        response["pedal_phase"] = pedal_phase
        # Where the Kona pros sat, beside the bands on aero positions
        # (kona_field.py). Context, never a grade.
        from app.services.video_analysis.biomechanics.kona_field import pro_field
        response["pro_field"] = pro_field(cycling_position)
        if score_by_phase is not None:
            response["score_by_phase"] = score_by_phase
        if pedal_phase == "uncertain" and not withheld:
            warnings.append(
                "We couldn't tell where the pedal was in this photo, so the knee and "
                "hip are measured but not scored, and the score is your upper body. "
                "If the pedal was at the bottom of the stroke, mark it under the "
                "angles -- the knee there is the saddle-height reading."
            )

        # Medical thresholds. The video path has run these since the start; the
        # photo path never did, so a hip closed past the iliac-artery threshold
        # was measured, shown in the table and then passed over in silence.
        # Only at a detected TDC: that is where the hip minimum legitimately
        # occurs, so it is the reading the threshold is defined against.
        if hip_phase == "tdc":
            hip_val = angles.get("hip")
            if hip_val is not None and not math.isnan(hip_val):
                for warn in get_medical_warnings(cycling_position, {
                    "hip_angle_min": hip_val,
                    "trunk_angle_avg": angles.get("trunk", 0.0),
                }):
                    warnings.append(warn["message"])

        # Position archetype classification
        from app.services.video_analysis.biomechanics.cycling_positions import detect_position_archetype
        archetype = detect_position_archetype(
            shoulder_angle=angles.get("shoulder", 0.0),
            elbow_angle=angles.get("elbow", 0.0),
            trunk_angle=angles.get("trunk", 0.0),
            hip_angle=angles.get("hip", 0.0),
            cycling_position=cycling_position,
        )
        if archetype:
            response["position_archetype"] = archetype

    # Where to go from here, whatever the score (see _photo_next_steps).
    response["next_steps"] = _photo_next_steps(
        sport, cycling_position, pedal_phase if sport == "bike" else None,
        bool(withheld), camera_side, helmet,
    )

    logger.info(
        "PHOTO_ANALYSIS_COMPLETE",
        sport=sport,
        camera_side=camera_side,
        num_angles=len(angles),
        overall_score=score_data["overall_score"],
        grade=score_data["grade"],
        processing_time=processing_time,
    )

    return response
