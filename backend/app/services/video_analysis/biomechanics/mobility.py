"""Off-bike mobility screens, and what they are allowed to say about a fit.

The gap this closes
-------------------
Every fit window in ``cycling_positions.py`` is a statement about a bike. None
of them is a statement about the rider. A trunk angle of 22 deg is "good, TT
aero" whether the rider holds it easily or is buying it with a rounded lower
back, and the advice that follows ("you could go lower") is identical either
way. A fitter would never work like that: the first thing they do is put you on
the floor and find out what range you have, because the position you can hold
is bounded by your body before it is bounded by your stem.

What is measured, and what is inferred
--------------------------------------
Two things travel in this module and they are not the same kind of claim.

**Measured.** The angle in each screen, and its tier against published clinical
norms. A supine straight-leg raise and a supine knee-to-chest both happen
entirely in the sagittal plane with the subject still, which is the one
situation where a single side-on photo and a 2D pose model are on solid ground.
The norms (~80 deg ASLR, 120 deg hip flexion) are ordinary clinical reference
values, cited per screen.

**Inferred.** The position ceiling -- "your range supports road drops, not a TT
tuck". This is a rule of thumb from fitting practice, not a calculation, and it
is labelled as one everywhere it surfaces. It is deliberately NOT arithmetic:
the tempting move is to subtract the rider's measured hip flexion from the
flexion a position demands at the top of the stroke, but those two numbers are
not the same quantity. Supine hip flexion is measured with the lumbar spine
flat; the shoulder-hip-knee angle on a bike is whatever the hip, the pelvis and
the lower back produce together. Subtracting one from the other looks precise
and means nothing.

What it never does
------------------
It never rewrites a numeric band. The bands stay as published for the position
the rider actually rode in, because that is the correct reference for what they
actually did. What a limited range changes is which *advice* is allowed: the
plan stops saying "get lower" and starts saying "the limiter here is range, not
the bike". And it never overrules the rider's own experience -- a screen is a
photo on a floor, and somebody who races 180 km in a tuck has better evidence
than we do.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from app.services.video_analysis.biomechanics.angle_calculator import (
    calculate_angle_2d,
    calculate_segment_to_vertical,
)
from app.services.video_analysis.biomechanics.landmarks import PoseLandmark as LM

# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------

TIER_GOOD = "good"
TIER_MODERATE = "moderate"
TIER_LIMITED = "limited"

TIER_ORDER = (TIER_LIMITED, TIER_MODERATE, TIER_GOOD)

TIER_LABELS = {
    TIER_GOOD: "Good range",
    TIER_MODERATE: "Moderate range",
    TIER_LIMITED: "Limited range",
}

# A joint we cannot see well enough is not a joint with no range. Every screen
# refuses rather than guesses when a landmark it needs is occluded.
MIN_SCREEN_VISIBILITY = 0.5

# Both screens are done lying on your back. The torso segment therefore has to
# be roughly horizontal in the photo; if it is not, we are looking at somebody
# sitting up, and a sit-up adds hip flexion that has nothing to do with range.
# Measured as the trunk's angle from vertical, so flat on the floor = 90.
SUPINE_TRUNK_FROM_VERTICAL_MIN = 55.0

# A straight-leg raise stops being a straight-leg raise when the knee bends --
# bending it takes the hamstring out of the movement and inflates the hip angle
# by 30-40 deg. Internal knee angle, 180 = straight.
ASLR_KNEE_STRAIGHT_MIN = 155.0

# Knee-to-chest is the opposite: the knee MUST be bent, or the rider is doing a
# straight-leg raise and we would file a hamstring measurement as a hip one.
KNEE_TO_CHEST_KNEE_BENT_MAX = 120.0


def _sides(landmarks) -> list[tuple[str, int, int, int, int]]:
    """(name, shoulder, hip, knee, ankle) for both sides."""
    return [
        ("left", LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
        ("right", LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ]


def _hip_flexion(landmarks, sh: int, hip: int, knee: int) -> float:
    """Degrees of hip flexion: 0 = leg in line with the torso, 90 = right angle.

    The pose model's own convention is the internal joint angle (180 = the
    segments are in line), which is the opposite direction and reads as
    nonsense on a mobility report. Everything here is flexion, the number a
    clinician would write down.
    """
    internal, _ = calculate_angle_2d(landmarks, sh, hip, knee, MIN_SCREEN_VISIBILITY)
    if math.isnan(internal):
        return float("nan")
    return 180.0 - internal


def _knee_internal(landmarks, hip: int, knee: int, ankle: int) -> float:
    internal, _ = calculate_angle_2d(
        landmarks, hip, knee, ankle, MIN_SCREEN_VISIBILITY,
    )
    return internal


def _trunk_from_vertical(landmarks, sh: int, hip: int) -> float:
    return calculate_segment_to_vertical(
        landmarks, sh, hip, MIN_SCREEN_VISIBILITY,
    )


def _supine_check(landmarks) -> str | None:
    """Reason the subject does not look like they are lying down, or None."""
    angles = [
        _trunk_from_vertical(landmarks, sh, hip)
        for _, sh, hip, _, _ in _sides(landmarks)
    ]
    seen = [a for a in angles if not math.isnan(a)]
    if not seen:
        return (
            "We could not find your torso in the photo. Both shoulders and "
            "hips need to be visible."
        )
    if max(seen) < SUPINE_TRUNK_FROM_VERTICAL_MIN:
        return (
            "You look like you are sitting or standing, not lying down. Both "
            "of these screens are done flat on your back -- sitting up adds "
            "hip angle that is posture, not range."
        )
    return None


# --------------------------------------------------------------------------
# The screens
# --------------------------------------------------------------------------

def _measure_hamstring(landmarks) -> dict[str, Any]:
    """Active straight-leg raise: lie flat, lift one straight leg as high as
    it goes.

    Which leg is being raised is worked out rather than asked: the raised one
    is the one with more hip flexion. Asking would add a question to the form
    and a way to get the answer wrong.
    """
    not_supine = _supine_check(landmarks)
    if not_supine:
        return {"error": not_supine}

    best: tuple[float, str, int, int, int] | None = None
    for name, sh, hip, knee, ankle in _sides(landmarks):
        flexion = _hip_flexion(landmarks, sh, hip, knee)
        if math.isnan(flexion):
            continue
        if best is None or flexion > best[0]:
            best = (flexion, name, hip, knee, ankle)

    if best is None:
        return {
            "error": (
                "We could not measure a hip angle. Shoulder, hip and knee all "
                "need to be visible from the side."
            )
        }

    flexion, side, hip, knee, ankle = best
    knee_angle = _knee_internal(landmarks, hip, knee, ankle)
    if not math.isnan(knee_angle) and knee_angle < ASLR_KNEE_STRAIGHT_MIN:
        return {
            "error": (
                f"The raised knee is bent (about {180 - knee_angle:.0f} deg of "
                "bend). A straight-leg raise only measures hamstring length "
                "while the knee stays straight -- bending it lets the leg come "
                "up much further for no extra range. Retake with the knee locked."
            )
        }

    return {"value": round(flexion, 1), "side": side}


def _measure_hip_flexion(landmarks) -> dict[str, Any]:
    """Knee to chest: lie flat, pull one knee toward your chest.

    Hip flexion with the hamstring taken out of it by the bent knee -- so
    together with the straight-leg raise it separates "tight hamstrings" from
    "the hip itself does not go there", which are different problems with
    different answers.
    """
    not_supine = _supine_check(landmarks)
    if not_supine:
        return {"error": not_supine}

    best: tuple[float, str, int, int, int] | None = None
    for name, sh, hip, knee, ankle in _sides(landmarks):
        flexion = _hip_flexion(landmarks, sh, hip, knee)
        if math.isnan(flexion):
            continue
        if best is None or flexion > best[0]:
            best = (flexion, name, hip, knee, ankle)

    if best is None:
        return {
            "error": (
                "We could not measure a hip angle. Shoulder, hip and knee all "
                "need to be visible from the side."
            )
        }

    flexion, side, hip, knee, ankle = best
    knee_angle = _knee_internal(landmarks, hip, knee, ankle)
    if not math.isnan(knee_angle) and knee_angle > KNEE_TO_CHEST_KNEE_BENT_MAX:
        return {
            "error": (
                "The knee is nearly straight, so this is a straight-leg raise "
                "rather than knee-to-chest. Bend the knee fully and pull it "
                "toward your chest -- that is what takes the hamstring out of "
                "the movement."
            )
        }

    return {"value": round(flexion, 1), "side": side}


# Each screen: how to shoot it, how to measure it, and where the tier cuts sit.
#
# The cut-points are conventional clinical reference values, not thresholds we
# derived. They are named in ``source`` so a reader can go and disagree with
# the reference rather than with us.
MOBILITY_SCREENS: dict[str, dict[str, Any]] = {
    "hamstring": {
        "label": "Straight-leg raise",
        "measures": "Hamstring length",
        "unit": "deg of hip flexion",
        "why": (
            "Hamstring length decides how far your pelvis can rotate forward "
            "before it starts pulling the other way. On the bike that is what "
            "turns a low front end into a rounded lower back."
        ),
        "setup": [
            "Lie flat on your back on the floor, legs straight.",
            "Phone on the floor a couple of metres to your side, level with "
            "your hips, filming you side-on.",
            "Raise one leg as high as it goes with the knee locked straight.",
            "Keep the other leg flat on the floor and take the photo at the top.",
        ],
        "measure": _measure_hamstring,
        # ASLR reference values cluster around 80 deg, with below ~65 widely
        # treated as restricted.
        "tiers": ((80.0, TIER_GOOD), (65.0, TIER_MODERATE)),
        "source": "Clinical active straight-leg-raise norms (~80 deg typical)",
        "reads": {
            TIER_GOOD: (
                "Your hamstrings are not what limits your position. If a low "
                "front end feels wrong, look elsewhere."
            ),
            TIER_MODERATE: (
                "Enough range for most road positions. Going lower is "
                "possible but the last few degrees will come from your lower "
                "back rather than your hips."
            ),
            TIER_LIMITED: (
                "Short hamstrings will pull your pelvis backwards as soon as "
                "you reach for a lower position. Lowering the bars from here "
                "buys aerodynamics and pays for it in the lower back."
            ),
        },
    },
    "hip_flexion": {
        "label": "Knee to chest",
        "measures": "Hip flexion range",
        "unit": "deg of hip flexion",
        "why": (
            "How far the hip joint itself closes. At the top of the pedal "
            "stroke an aero position asks the hip to close a long way, and "
            "range you do not have gets made up by the lower back."
        ),
        "setup": [
            "Lie flat on your back on the floor.",
            "Phone on the floor a couple of metres to your side, level with "
            "your hips, filming you side-on.",
            "Pull one knee toward your chest as far as it comfortably goes.",
            "Keep the other leg flat and your lower back on the floor.",
        ],
        "measure": _measure_hip_flexion,
        # AAOS lists 120 deg as normal hip flexion; below ~105 is restricted.
        "tiers": ((120.0, TIER_GOOD), (105.0, TIER_MODERATE)),
        "source": "AAOS normal hip flexion range of motion (120 deg)",
        "reads": {
            TIER_GOOD: (
                "The hip closes far enough for an aggressive position to come "
                "from the hip rather than the spine."
            ),
            TIER_MODERATE: (
                "Fine for road positions. A deep aero tuck would be asking "
                "close to everything you have at the top of the stroke."
            ),
            TIER_LIMITED: (
                "The hip runs out of travel early. In a low position the top "
                "of the pedal stroke will feel blocked, and the difference "
                "gets taken out of the lower back."
            ),
        },
    },
}


def screen_tier(screen: str, value: float) -> str:
    """Which band a measured angle falls in."""
    cuts = MOBILITY_SCREENS[screen]["tiers"]
    for threshold, tier in cuts:
        if value >= threshold:
            return tier
    return TIER_LIMITED


def measure_screen(screen: str, landmarks) -> dict[str, Any]:
    """Run one screen against one photo's landmarks.

    Returns ``{"screen", "value", "tier", ...}`` or ``{"screen", "error"}``.
    A screen that cannot be measured is an error, never a low score: "we could
    not see your knee" and "your hip does not go there" are opposite findings
    and must never collapse into the same one.
    """
    if screen not in MOBILITY_SCREENS:
        raise ValueError(f"unknown mobility screen: {screen}")

    spec = MOBILITY_SCREENS[screen]
    measure: Callable[[Any], dict[str, Any]] = spec["measure"]
    out = measure(landmarks)
    if "error" in out:
        return {"screen": screen, "label": spec["label"], "error": out["error"]}

    value = out["value"]
    tier = screen_tier(screen, value)
    return {
        "screen": screen,
        "label": spec["label"],
        "measures": spec["measures"],
        "value": value,
        "unit": spec["unit"],
        "side": out.get("side"),
        "tier": tier,
        "tier_label": TIER_LABELS[tier],
        "read": spec["reads"][tier],
        "source": spec["source"],
    }


# --------------------------------------------------------------------------
# From range to position -- the inferred half
# --------------------------------------------------------------------------

# Rungs of how much hip range a position demands, most to least. TT and
# triathlon share a rung on purpose: they are the same demand on the hip, and a
# ceiling that distinguished them would be claiming a resolution these two
# floor measurements do not have.
POSITION_RUNGS: tuple[tuple[str, ...], ...] = (
    ("tt_aero", "triathlon"),
    ("road_drops",),
    ("road_hoods",),
    ("casual",),
)

RUNG_LABELS = (
    "an aero tuck (TT / triathlon)",
    "road bike in the drops",
    "road bike on the hoods",
    "an upright position",
)

# Flat list, kept for callers that just need to know a position is one we rank.
POSITION_AGGRESSION = tuple(p for rung in POSITION_RUNGS for p in rung)


def _rung_of(position: str) -> int:
    for i, rung in enumerate(POSITION_RUNGS):
        if position in rung:
            return i
    raise ValueError(f"unranked cycling position: {position}")

# Said out loud everywhere the ceiling appears. It is a fitter's rule of thumb
# applied to a measurement, and the difference between that and the measurement
# itself is the whole reason this module is trustworthy.
CEILING_CAVEAT = (
    "This is a rule of thumb from fitting practice applied to two floor "
    "measurements -- not a measurement of you on the bike. If you already hold "
    "a lower position comfortably for a full event, your own experience is the "
    "better evidence."
)


def position_ceiling(screens: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """The most demanding position the measured range plausibly supports.

    Hip flexion sets the rung, because the hip angle at the top of the stroke
    is what a low front end actually closes. Short hamstrings then drop it one
    further, because a rider who cannot rotate the pelvis forward reaches a low
    bar by rounding the back instead.

    Returns None when neither screen was measured -- no data is not a verdict.
    """
    hip = (screens.get("hip_flexion") or {}).get("tier")
    ham = (screens.get("hamstring") or {}).get("tier")
    if hip is None and ham is None:
        return None

    reasons: list[str] = []
    rung = 0  # nothing measured against it yet = nothing ruled out

    if hip == TIER_LIMITED:
        rung = 2  # hoods
        reasons.append("hip flexion is in the limited band")
    elif hip == TIER_MODERATE:
        rung = 1  # drops
        reasons.append("hip flexion is moderate")
    elif hip == TIER_GOOD:
        reasons.append("hip flexion is in the good band")

    if ham == TIER_LIMITED:
        rung += 1
        reasons.append("short hamstrings limit how far the pelvis can rotate forward")
    elif ham == TIER_GOOD and hip is None:
        # Hamstrings alone cannot clear a rider for an aero tuck -- that is the
        # hip's call, and the hip was not measured.
        rung = 1
        reasons.append("hamstring length is good, but hip flexion was not measured")

    rung = min(rung, len(POSITION_RUNGS) - 1)
    return {
        "rung": rung,
        "position": POSITION_RUNGS[rung][0],
        "label": RUNG_LABELS[rung],
        "unrestricted": rung == 0,
        "reasons": reasons,
        "caveat": CEILING_CAVEAT,
    }


def build_mobility_profile(
    measurements: dict[str, dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    """Collect the screens that were measured into one profile.

    ``measurements`` maps a screen key to the dict ``measure_screen`` returned.
    Screens that errored are kept, so the profile can say what is missing
    rather than quietly pretending the rider only did one.
    """
    ok = {k: v for k, v in measurements.items() if "value" in v}
    failed = {k: v for k, v in measurements.items() if "error" in v}

    return {
        "screens": ok,
        "unmeasured": failed,
        "screens_done": sorted(ok),
        "screens_available": sorted(MOBILITY_SCREENS),
        "goal": goal if goal in ("comfort", "speed") else None,
        "ceiling": position_ceiling(ok),
    }


def profile_from_stored(
    hamstring_deg: float | None = None,
    hip_flexion_deg: float | None = None,
    goal: str | None = None,
    measured_at: Any = None,
) -> dict[str, Any] | None:
    """Rebuild a profile from the angles saved on the account.

    Only the degrees are stored, never the tiers -- a tier is our reading of
    the number, and the cut-points may move. Re-deriving them here means a
    change to a reference value reaches everybody's next analysis instead of
    only new measurements.

    Returns None when nothing has been measured, so a caller can pass the
    result straight through without having to distinguish "no screens" from
    "an empty profile".
    """
    measured: dict[str, dict[str, Any]] = {}
    for key, value in (("hamstring", hamstring_deg), ("hip_flexion", hip_flexion_deg)):
        if value is None:
            continue
        spec = MOBILITY_SCREENS[key]
        tier = screen_tier(key, float(value))
        measured[key] = {
            "screen": key,
            "label": spec["label"],
            "measures": spec["measures"],
            "value": round(float(value), 1),
            "unit": spec["unit"],
            "tier": tier,
            "tier_label": TIER_LABELS[tier],
            "read": spec["reads"][tier],
            "source": spec["source"],
        }
    if not measured:
        return None

    profile = build_mobility_profile(measured, goal=goal)
    profile["measured_at"] = (
        measured_at.isoformat() if hasattr(measured_at, "isoformat") else measured_at
    )
    return profile


def assess_position(
    profile: dict[str, Any] | None, ridden_position: str | None,
) -> dict[str, Any] | None:
    """Compare the position the rider actually rode against their ceiling.

    This is the one output the analysis acts on. ``within`` False means the
    plan must stop recommending a lower front end and say why.
    """
    if not profile or not ridden_position:
        return None
    ceiling = profile.get("ceiling")
    if not ceiling:
        return None
    if ridden_position not in POSITION_AGGRESSION:
        return None

    within = _rung_of(ridden_position) >= ceiling["rung"]
    reasons = ", and ".join(ceiling.get("reasons") or [])
    if within:
        message = "Your measured range supports the position you rode in."
    else:
        message = (
            "You rode in a position more demanding than your floor "
            f"measurements support -- {reasons}. Going lower is not the next "
            f"move here: what your range supports is {ceiling['label']}, and "
            "the limiter is the range, not the bike."
        )
    return {
        "within": within,
        "ridden": ridden_position,
        "ceiling": ceiling["position"],
        "ceiling_label": ceiling["label"],
        "reasons": ceiling.get("reasons") or [],
        "message": message,
        "caveat": CEILING_CAVEAT,
    }
