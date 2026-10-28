"""What a running photo at initial contact can say — and the gate in front of it.

Three readings exist only at the instant the foot meets the ground: how far
ahead of the hip it lands (overstride), which way the shin leans, and which
part of the foot arrives first. At toe-off or mid-swing they are not
approximate, they are meaningless -- so they are computed only on a frame the
phase detector has placed at contact, and every one of them is NaN or None
rather than a number whenever its landmarks cannot be read.

Alongside them, the "no score" rule the bike path has had since the quality
gate and the run path did not: a frame the detector cannot place in the stride
was still being scored against bands chosen for a phase nobody could name.

All of it sits behind ``contact_metrics``, off by default, because none of it
has been checked on real broadcast frames yet (docs/RUN_PHOTO_PLAN_RU.md).
The last tests here are the ones that matter most: with the flag off, nothing
an athlete sees has changed.

Pure functions and a hand-built pose -- no mediapipe, no opencv.
"""

from __future__ import annotations

import math

import pytest

from app.services.video_analysis.photo_analyzer import (
    _FOOT_STRIKE_FORE_BELOW_DEG,
    _FOOT_STRIKE_HEEL_ABOVE_DEG,
    _WITHHELD_MESSAGES,
    _foot_angle_to_ground,
    _foot_strike_from_ground_angle,
    _run_contact_metrics,
    _run_photo_withheld,
)


class LM:
    def __init__(self, x: float, y: float, visibility: float = 1.0):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, visibility


def _pose(*, hip=(0.0, 1.0), knee=(0.0, 0.5), ankle=(0.0, 0.0),
          hip_vis=1.0, ankle_vis=1.0) -> list[LM]:
    """World-ish landmarks for the LEFT side (hip 23, knee 25, ankle 27).

    y grows upwards here, like MediaPipe's world frame; x is fore-aft with
    forward = +1. Leg length is 1.0 by default so ratios read directly.
    """
    lms = [LM(0.0, 0.0) for _ in range(33)]
    lms[23] = LM(*hip, hip_vis)
    lms[25] = LM(*knee)
    lms[27] = LM(*ankle, ankle_vis)
    lms[29] = LM(ankle[0] - 0.05, ankle[1])   # heel
    lms[31] = LM(ankle[0] + 0.10, ankle[1])   # toe
    return lms


# --------------------------------------------------------------------------
# foot strike, from the foot's angle to the GROUND
# --------------------------------------------------------------------------

@pytest.mark.parametrize("deg,expected", [
    (20.0, "heel"), (8.2, "heel"),
    (8.1, "midfoot"), (0.0, "midfoot"), (-1.6, "midfoot"),
    (-1.7, "forefoot"), (-12.0, "forefoot"),
])
def test_foot_strike_bands(deg, expected):
    """Altman & Davis 2012: > 8.1 rearfoot, -1.6..8.1 midfoot, < -1.6 forefoot."""
    assert _foot_strike_from_ground_angle(deg) == expected


def test_foot_strike_unknown_without_an_angle():
    assert _foot_strike_from_ground_angle(math.nan) is None
    assert _foot_strike_from_ground_angle(None) is None


def test_a_flat_foot_is_midfoot():
    assert _FOOT_STRIKE_FORE_BELOW_DEG < 0.0 < _FOOT_STRIKE_HEEL_ABOVE_DEG


def _foot(heel, toe, vis=1.0):
    """Image-plane landmarks with only the left heel (29) and toe (31) set."""
    lms = [LM(0.5, 0.5) for _ in range(33)]
    lms[29], lms[31] = LM(*heel, vis), LM(*toe, vis)
    return lms


def test_toe_up_is_positive_and_reads_heel():
    """Image y grows downwards, so a toe ABOVE the heel has the smaller y."""
    nl = _foot(heel=(0.50, 0.90), toe=(0.60, 0.88))
    a = _foot_angle_to_ground(nl, 29, 31, 1.0, 1000, 1000)
    assert a > 0
    assert _foot_strike_from_ground_angle(a) == "heel"


def test_toe_down_is_negative_and_reads_forefoot():
    nl = _foot(heel=(0.50, 0.88), toe=(0.60, 0.90))
    assert _foot_strike_from_ground_angle(
        _foot_angle_to_ground(nl, 29, 31, 1.0, 1000, 1000)) == "forefoot"


def test_running_left_reads_the_same_as_running_right():
    right = _foot(heel=(0.50, 0.90), toe=(0.60, 0.88))
    left = _foot(heel=(0.50, 0.90), toe=(0.40, 0.88))
    assert _foot_angle_to_ground(right, 29, 31, 1.0, 1000, 1000) == pytest.approx(
        _foot_angle_to_ground(left, 29, 31, -1.0, 1000, 1000))


def test_the_angle_is_measured_in_pixels_not_normalized_units():
    """On a non-square frame normalized x and y are different units. The same
    landmarks on a 2:1 frame are a flatter foot than on a square one."""
    nl = _foot(heel=(0.50, 0.90), toe=(0.60, 0.88))
    square = _foot_angle_to_ground(nl, 29, 31, 1.0, 1000, 1000)
    wide = _foot_angle_to_ground(nl, 29, 31, 1.0, 2000, 1000)
    assert wide < square


def test_a_foot_read_backwards_is_not_a_reading():
    """Toe behind the heel along the direction of travel is a misread foot."""
    nl = _foot(heel=(0.60, 0.90), toe=(0.50, 0.88))
    assert math.isnan(_foot_angle_to_ground(nl, 29, 31, 1.0, 1000, 1000))


def test_an_invisible_foot_is_not_a_reading():
    nl = _foot(heel=(0.50, 0.90), toe=(0.60, 0.88), vis=0.1)
    assert math.isnan(_foot_angle_to_ground(nl, 29, 31, 1.0, 1000, 1000))


def test_the_shin_does_not_leak_into_the_foot_strike():
    """The bug this replaced. A real touchdown: shin leaning back 16 deg, toe
    9.4 deg above the heel. Read against the shin those add up to ~111 deg and
    came back "forefoot"; read against the ground it is a heel strike, which
    is what the overstride beside it (0.35) says too. Here the shank leans
    back hard and the foot is still toe-up: the answer must be heel."""
    lms = [LM(0.0, 0.0) for _ in range(33)]
    lms[23], lms[25], lms[27] = LM(0.0, 1.0), LM(0.0, 0.5), LM(0.3, 0.0)
    nl = _foot(heel=(0.50, 0.90), toe=(0.60, 0.883))      # ~9.6 deg toe-up
    m = _run_contact_metrics(lms, "left", 1.0, {"ankle": 111.0}, nl, 1000, 1000)
    assert m["shin_angle"] < -10
    assert m["foot_strike"] == "heel"


# --------------------------------------------------------------------------
# the contact readings
# --------------------------------------------------------------------------

def test_a_foot_under_the_hip_is_not_an_overstride():
    m = _run_contact_metrics(_pose(), "left", 1.0, {"ankle": 90.0})
    assert m["overstride_ratio"] == pytest.approx(0.0, abs=1e-6)


def test_overstride_is_the_reach_in_leg_lengths():
    """Ankle 0.2 ahead of the hip on a leg of length 1.0 -> 0.2. Independent of
    body size and camera distance, which is why it is a ratio."""
    m = _run_contact_metrics(
        _pose(ankle=(0.2, 0.0), knee=(0.1, 0.5)), "left", 1.0, {"ankle": 90.0})
    leg = math.dist((0, 1), (0.1, 0.5)) + math.dist((0.1, 0.5), (0.2, 0))
    assert m["overstride_ratio"] == pytest.approx(0.2 / leg, abs=1e-3)


def test_overstride_is_a_magnitude():
    """Behind the hip reads the same as ahead: from one side-view frame the
    distance is the signal either way, as in the video analyzer."""
    ahead = _run_contact_metrics(_pose(ankle=(0.2, 0.0)), "left", 1.0, {"ankle": 90.0})
    behind = _run_contact_metrics(_pose(ankle=(-0.2, 0.0)), "left", 1.0, {"ankle": 90.0})
    assert ahead["overstride_ratio"] == behind["overstride_ratio"]


@pytest.mark.parametrize("kw", [{"hip_vis": 0.2}, {"ankle_vis": 0.2}])
def test_overstride_is_absent_when_hip_or_ankle_is_not_visible(kw):
    """Absent beats a number read off an occluded joint."""
    m = _run_contact_metrics(_pose(**kw), "left", 1.0, {"ankle": 90.0})
    assert math.isnan(m["overstride_ratio"])


def test_a_vertical_shin_reads_zero():
    m = _run_contact_metrics(_pose(), "left", 1.0, {"ankle": 90.0})
    assert m["shin_angle"] == pytest.approx(0.0, abs=0.1)


def test_shin_sign_follows_the_direction_of_travel():
    """Knee ahead of the ankle is positive; ankle ahead of the knee -- the shin
    leaning back, a reaching foot -- is negative. Flip the direction of travel
    and the same geometry flips sign."""
    reaching = _pose(ankle=(0.3, 0.0), knee=(0.0, 0.5))      # ankle out in front
    assert _run_contact_metrics(reaching, "left", 1.0, {"ankle": 90.0})["shin_angle"] < 0
    assert _run_contact_metrics(reaching, "left", -1.0, {"ankle": 90.0})["shin_angle"] > 0


def test_shin_is_absent_without_a_direction_of_travel():
    m = _run_contact_metrics(_pose(), "left", 0.0, {"ankle": 90.0})
    assert math.isnan(m["shin_angle"])


def test_without_image_landmarks_there_is_no_foot_strike():
    """No guess from the ankle angle any more: without the image-plane heel
    and toe the category is absent."""
    m = _run_contact_metrics(_pose(), "left", 1.0, {"ankle": 78.0})
    assert m["foot_strike"] is None
    assert "not a verdict" in m["foot_strike_note"]


def test_the_right_side_reads_its_own_leg():
    """Hip 24, knee 26, ankle 28. A reading taken off the wrong side's indices
    would be the far leg, which on a side view is the one you cannot see."""
    lms = [LM(0.0, 0.0) for _ in range(33)]
    lms[24], lms[26], lms[28] = LM(0.0, 1.0), LM(0.0, 0.5), LM(0.25, 0.0)
    lms[30], lms[32] = LM(0.2, 0.0), LM(0.35, 0.0)
    m = _run_contact_metrics(lms, "right", 1.0, {"ankle": 90.0})
    assert m["overstride_ratio"] > 0.2


# --------------------------------------------------------------------------
# the "no score" rule
# --------------------------------------------------------------------------

def test_no_knee_means_no_score():
    assert _run_photo_withheld({"knee": math.nan}, "initial_contact") == "run_leg_not_measured"
    assert _run_photo_withheld({}, "initial_contact") == "run_leg_not_measured"


def test_a_frame_between_phases_gets_no_score():
    assert _run_photo_withheld({"knee": 160.0}, "uncertain") == "stride_phase_uncertain"


@pytest.mark.parametrize("phase", ["initial_contact", "toe_off", "midstance", "swing"])
def test_a_placed_frame_with_a_knee_is_scored(phase):
    assert _run_photo_withheld({"knee": 160.0}, phase) is None


def test_a_missing_knee_outranks_an_uncertain_phase():
    """Both are true; the one the athlete can act on first is the leg."""
    assert _run_photo_withheld({"knee": math.nan}, "uncertain") == "run_leg_not_measured"


def test_legs_that_cannot_be_named_are_not_called_a_stride_instant():
    """No direction of travel / a foot unseen: the athlete is told that, not
    that the frame fell between two instants (which is what flight means)."""
    blank = {"stance": None, "dx": None, "clear": False}
    flight = {"stance": "under", "dx": 0.0, "clear": False}
    assert _run_photo_withheld({"knee": 160.0}, "uncertain", blank) == "run_legs_unreadable"
    assert _run_photo_withheld({"knee": 160.0}, "uncertain", flight) == "stride_phase_uncertain"
    assert _run_photo_withheld({"knee": math.nan}, "uncertain", blank) == "run_leg_not_measured"


def test_every_refusal_has_words():
    """A withheld key with no message is a KeyError at the moment of refusal,
    which turns an honest "no score" into a 500."""
    for key in ("run_leg_not_measured", "run_legs_unreadable", "stride_phase_uncertain"):
        assert key in _WITHHELD_MESSAGES
        assert "No score" in _WITHHELD_MESSAGES[key]


def test_the_run_messages_are_about_running():
    """The bike's message talks about the fit and the bike. Reusing it for a
    runner would tell them to move a bicycle that is not there."""
    assert "bike" not in _WITHHELD_MESSAGES["run_leg_not_measured"]
    assert "stride" in _WITHHELD_MESSAGES["run_leg_not_measured"]
