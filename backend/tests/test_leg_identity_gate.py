"""Declining to measure a leg beats guessing which leg it is.

The bike path has no working correction for MediaPipe handing a leg's index to
the other leg's foot — every swap-based attempt needed the far leg as evidence,
and the far leg's landmarks are partly invented. So the gate does not correct
anything: it blanks the frames where an ankle left its own track, which the
angle calculators and the Butterworth pass already read as "not measured".

The property that makes this safe is asymmetric: the worst case is losing good
frames. It cannot put the reported series on the wrong leg, which is what
happens today and what every swap attempt risked.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from app.services.video_analysis.biomechanics.landmark_stabilizer import (
    LEG_BREAK_FRAC,
    _gate_leg_identity_breaks,
    stabilize_landmarks,
)

# A trainer clip's geometry: two pedals half a revolution apart on one circle.
BB = (0.35, 0.67)
RADIUS = 0.049           # -> ankle separation 0.098, as measured on IMG_9981
STEP = 2 * math.pi / 40  # ~40 frames a revolution = 90 rpm at 60 fps


def _lm(x: float, y: float, vis: float = 1.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=0.0, visibility=vis)


def _on_circle(angle: float) -> tuple[float, float]:
    return (BB[0] + RADIUS * math.cos(angle), BB[1] + RADIUS * math.sin(angle))


def _frame(left_ankle, right_ankle) -> dict:
    """33 landmarks; only the leg indices carry anything the gate reads."""
    lms = [_lm(0.35, 0.4) for _ in range(33)]
    for idx, (x, y) in ((27, left_ankle), (28, right_ankle)):
        lms[idx] = _lm(x, y)
    # knees/heels/toes ride along so the blanking can be observed on them too
    for idx, (x, y) in ((25, left_ankle), (29, left_ankle), (31, left_ankle)):
        lms[idx] = _lm(x, y - 0.05)
    for idx, (x, y) in ((26, right_ankle), (30, right_ankle), (32, right_ankle)):
        lms[idx] = _lm(x, y - 0.05)
    return {
        "normalized_landmarks": lms,
        "world_landmarks": [_lm(p.x, p.y) for p in lms],
        "timestamp_ms": 0.0,
    }


def _pedalling(n: int = 120) -> list[dict]:
    return [
        _frame(_on_circle(i * STEP), _on_circle(i * STEP + math.pi))
        for i in range(n)
    ]


def _blanked(frame: dict, idx: int) -> bool:
    lm = frame["normalized_landmarks"][idx]
    return math.isnan(lm.x) and math.isnan(lm.y)


def test_a_clean_pedal_stroke_loses_nothing():
    frames = _pedalling()
    out = _gate_leg_identity_breaks(frames)
    assert out["blanked"] == {"left": 0, "right": 0}
    assert out["reseeds"] == {"left": 0, "right": 0}
    assert out["ankle_separation"] == round(2 * RADIUS, 4)


def test_a_foot_that_hops_to_the_other_shoe_is_blanked_not_swapped():
    frames = _pedalling()
    # frame 60: the right index lands on the left foot -- the exact failure
    # seen on IMG_9981 frames 21-22 and 7 times in an 11 s user clip.
    hop = _on_circle(60 * STEP)
    for idx in (26, 28, 30, 32):
        frames[60]["normalized_landmarks"][idx] = _lm(*hop)
        frames[60]["world_landmarks"][idx] = _lm(*hop)

    out = _gate_leg_identity_breaks(frames)

    assert _blanked(frames[60], 28), "the hopped ankle should be unmeasured"
    assert _blanked(frames[60], 26), "the whole leg goes, not just the ankle"
    assert out["blanked"]["right"] >= 1
    assert out["blanked"]["left"] == 0, "the leg that behaved is untouched"


def test_one_bad_frame_does_not_take_its_neighbours_with_it():
    """The cascade a fixed bar plus a decaying velocity produced: one blink made
    the next prediction worse, so it blanked again, all the way to a re-seed."""
    frames = _pedalling()
    hop = _on_circle(60 * STEP)
    for idx in (26, 28, 30, 32):
        frames[60]["normalized_landmarks"][idx] = _lm(*hop)
        frames[60]["world_landmarks"][idx] = _lm(*hop)

    _gate_leg_identity_breaks(frames)

    assert not _blanked(frames[61], 28)
    assert not _blanked(frames[62], 28)


def test_visibility_is_left_alone_so_quality_scores_do_not_silently_drop():
    """Detection-quality metrics read visibility. Blanking is about position."""
    frames = _pedalling()
    hop = _on_circle(60 * STEP)
    for idx in (26, 28, 30, 32):
        frames[60]["normalized_landmarks"][idx] = _lm(*hop)
        frames[60]["world_landmarks"][idx] = _lm(*hop)

    _gate_leg_identity_breaks(frames)

    assert frames[60]["normalized_landmarks"][28].visibility == 1.0


def test_feet_that_never_resolve_as_two_feet_are_left_alone():
    """Below the separation floor there is no scale to judge anything against,
    and blanking on a bad scale would erase the clip rather than clean it."""
    frames = [_frame((0.35, 0.67), (0.3505, 0.6705)) for _ in range(60)]
    out = _gate_leg_identity_breaks(frames)
    assert out["blanked"] == {"left": 0, "right": 0}


def test_a_clip_too_short_to_have_a_scale_is_left_alone():
    out = _gate_leg_identity_breaks(_pedalling(4))
    assert out["blanked"] == {"left": 0, "right": 0}
    assert out["ankle_separation"] is None


def test_a_sustained_wrong_leg_run_re_seeds_and_says_so():
    """The honest limit: the gate catches the transition, but if the model sits
    on the wrong foot for long enough the track has to accept it. That is what
    the re-seed count exists to report."""
    frames = _pedalling()
    for i in range(60, 80):
        hop = _on_circle(i * STEP)
        for idx in (26, 28, 30, 32):
            frames[i]["normalized_landmarks"][idx] = _lm(*hop)
            frames[i]["world_landmarks"][idx] = _lm(*hop)

    out = _gate_leg_identity_breaks(frames)

    assert out["reseeds"]["right"] >= 1
    assert out["blanked"]["right"] >= 1


def test_the_bar_is_a_fraction_of_the_distance_between_the_feet():
    """Scaled to the crank diameter, not to the frame: a clip filmed closer has
    a bigger pedal circle in normalized units and must get a bigger bar."""
    assert 0.5 < LEG_BREAK_FRAC < 1.0


def test_framing_closer_or_further_gates_the_same_frames():
    """The one property that lets a bar calibrated on one clip travel.

    Normalized coordinates mean a tighter framing scales the whole pedal circle
    up, so a bar expressed in pixels-of-the-frame would gate a different set of
    frames on the same riding. Scaling the bar by the measured ankle separation
    makes the decision depend on the geometry rather than on how close the phone
    was, which is what a threshold picked from a single clip has to survive.
    """
    def _hopped(scale: float) -> list[int]:
        frames = []
        for i in range(120):
            la = _on_circle(i * STEP)
            ra = _on_circle(i * STEP + math.pi)
            grow = lambda p: (  # noqa: E731 -- local, one line, reads better inline
                BB[0] + (p[0] - BB[0]) * scale, BB[1] + (p[1] - BB[1]) * scale
            )
            frames.append(_frame(grow(la), grow(ra)))
        hop = frames[60]["normalized_landmarks"][27]
        for idx in (26, 28, 30, 32):
            frames[60]["normalized_landmarks"][idx] = _lm(hop.x, hop.y)
            frames[60]["world_landmarks"][idx] = _lm(hop.x, hop.y)
        _gate_leg_identity_breaks(frames)
        return [i for i, f in enumerate(frames) if _blanked(f, 28)]

    assert _hopped(1.0) == _hopped(2.5) == [60]


def test_running_clips_do_not_get_this_pass():
    """Run has a working anti-swap; this is the bike-only answer to not having
    one. Turning it on for run would blank frames a correction already fixes."""
    frames = _pedalling()
    ctx: dict = {}
    stabilize_landmarks(frames, "run", None, fps=60.0, context=ctx)
    assert ctx["leg_identity_gate"] is None


def test_a_mutual_label_exchange_is_relabelled_not_blanked():
    """The one failure the resolver CAN fix on a bike: both legs' indices
    swap for a frame and swap back. Blanking it loses a good frame; the
    whole-clip resolver hands the points back to their legs and the gate
    then has nothing to catch."""
    frames = _pedalling()
    ra = _on_circle(60 * STEP + math.pi)
    for left_i, right_i in ((25, 26), (27, 28), (29, 30), (31, 32)):
        for key in ("normalized_landmarks", "world_landmarks"):
            lms = frames[60][key]
            lms[left_i], lms[right_i] = lms[right_i], lms[left_i]

    ctx: dict = {}
    stabilize_landmarks(frames, "bike", None, fps=60.0, context=ctx)

    lm = frames[60]["normalized_landmarks"][28]
    assert not math.isnan(lm.x), "the exchanged frame must come back, not blank"
    assert math.dist((lm.x, lm.y), ra) < 0.02, "and on its own leg"
    assert (ctx["leg_identity"] or {}).get("method") == "dp"
    assert ctx["leg_swap_pct"] is None, "bike still reports no swap share"


def test_a_one_sided_hop_is_still_blanked_after_the_resolver():
    """A hop with no mutual exchange carries no far-leg truth to relabel
    from; the gate must keep blanking it. The resolver in front must not
    weaken the blink-never-jump property."""
    frames = _pedalling()
    hop = _on_circle(60 * STEP)
    for idx in (26, 28, 30, 32):
        frames[60]["normalized_landmarks"][idx] = _lm(*hop)
        frames[60]["world_landmarks"][idx] = _lm(*hop)

    ctx: dict = {}
    stabilize_landmarks(frames, "bike", None, fps=60.0, context=ctx)

    assert _blanked(frames[60], 28)
    assert (ctx["leg_identity_gate"] or {}).get("blanked", {}).get("right", 0) >= 1
