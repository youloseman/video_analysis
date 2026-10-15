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
    _FOOT_STRIKE_FORE_ABOVE_DEG,
    _FOOT_STRIKE_HEEL_BELOW_DEG,
    _WITHHELD_MESSAGES,
    _foot_strike_from_ankle,
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
# foot strike, from the ankle angle
# --------------------------------------------------------------------------

@pytest.mark.parametrize("deg,expected", [
    (70.0, "heel"), (84.9, "heel"),
    (85.0, "midfoot"), (90.0, "midfoot"), (95.0, "midfoot"),
    (95.1, "forefoot"), (115.0, "forefoot"),
])
def test_foot_strike_bands(deg, expected):
    assert _foot_strike_from_ankle(deg) == expected


def test_foot_strike_unknown_when_the_ankle_was_not_measured():
    assert _foot_strike_from_ankle(math.nan) is None
    assert _foot_strike_from_ankle(None) is None


def test_the_midfoot_band_straddles_neutral():
    """Neutral shank-foot is 90. The thresholds are provisional, but whatever
    they end up as, 90 has to read as midfoot or the category is nonsense."""
    assert _FOOT_STRIKE_HEEL_BELOW_DEG < 90.0 < _FOOT_STRIKE_FORE_ABOVE_DEG


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


def test_foot_strike_rides_on_the_measured_ankle_angle():
    m = _run_contact_metrics(_pose(), "left", 1.0, {"ankle": 78.0})
    assert m["foot_strike"] == "heel"
    assert m["foot_strike_from_ankle_deg"] == 78.0
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


def test_every_refusal_has_words():
    """A withheld key with no message is a KeyError at the moment of refusal,
    which turns an honest "no score" into a 500."""
    for key in ("run_leg_not_measured", "stride_phase_uncertain"):
        assert key in _WITHHELD_MESSAGES
        assert "No score" in _WITHHELD_MESSAGES[key]


def test_the_run_messages_are_about_running():
    """The bike's message talks about the fit and the bike. Reusing it for a
    runner would tell them to move a bicycle that is not there."""
    assert "bike" not in _WITHHELD_MESSAGES["run_leg_not_measured"]
    assert "stride" in _WITHHELD_MESSAGES["run_leg_not_measured"]
