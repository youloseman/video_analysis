"""The flag in front of the contact readings, end to end.

Everything in test_photo_run_contact.py is behind ``contact_metrics``. This
file is about the flag itself, through the real measuring function: with it
OFF a running photo comes back exactly as it did before this work -- no
``contact`` key, and a frame the detector cannot place still scored -- and
with it ON the readings appear and the "no score" rule applies.

The OFF half is the one that matters while nothing has been checked on real
frames: it is the promise that athletes see no change until somebody decides
they should.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="the photo path decodes and draws with OpenCV")

from app.services.video_analysis import photo_analyzer as PA  # noqa: E402

W, H = 800, 1000


def _run_pose(knee_deg: float, thigh_forward_deg: float = 30.0):
    """A LEFT-side runner whose near thigh is ``thigh_forward_deg`` ahead of
    vertical and whose near knee is ``knee_deg``. Feet point forward (+x) so
    the direction of travel resolves. Everything else is filler that stays
    visible and out of the way.

    Image y grows downwards. The world frame is the image frame recentred and
    NOT stretched: the overstride and the phase are both read off world
    landmarks, and a fixture whose world squashes x against y puts the foot
    somewhere other than where its angles say -- the reach tests below need
    the thigh and knee to mean what they say.
    """
    hip = np.array([0.45, 0.50])
    thigh_len, shank_len = 0.18, 0.18
    a = math.radians(thigh_forward_deg)
    knee = hip + thigh_len * np.array([math.sin(a), math.cos(a)])
    # Knee flexion swings the shank BACK, as a knee does: a straight leg puts
    # the ankle on the thigh's line, and bending it brings the foot back
    # towards (and past) the hip. The internal angle is ``knee_deg`` either
    # way, which is why the first version, bent the other way, still passed
    # every phase test -- and would have put every foot far out in front.
    b = a - math.radians(180.0 - knee_deg)
    ankle = knee + shank_len * np.array([math.sin(b), math.cos(b)])
    shoulder = hip + np.array([0.0, -0.28])

    pts = {11: shoulder, 12: shoulder + [0.02, 0], 13: shoulder + [0.05, 0.12],
           14: shoulder + [0.07, 0.12], 15: shoulder + [0.10, 0.20],
           16: shoulder + [0.12, 0.20],
           23: hip, 24: hip + [0.02, 0], 25: knee, 26: hip + [0.0, 0.18],
           27: ankle, 28: hip + [0.0, 0.36],
           29: ankle + [-0.03, 0.02], 31: ankle + [0.06, 0.02],      # heel, toe
           30: hip + [-0.03, 0.38], 32: hip + [0.06, 0.38]}
    norm, world = [], []
    for j in range(33):
        x, y = pts.get(j, (0.5, 0.1))
        norm.append(SimpleNamespace(x=float(x), y=float(y), z=0.0, visibility=1.0))
        world.append(SimpleNamespace(x=float(x) - 0.5, y=float(y) - 0.5,
                                     z=0.0, visibility=1.0))
    return world, norm


def _measure(knee_deg: float, thigh: float = 30.0, **kw) -> dict:
    world, norm = _run_pose(knee_deg, thigh)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    return PA.analyze_from_pose(image, world, norm, "left", [], "run", None,
                                render_thumbnail=False, **kw)


# The geometry must land where the tests think it does, or every assertion
# below is about the wrong frame.

def test_the_fixture_reaches_initial_contact():
    assert _measure(165.0)["gait_phase"] == "initial_contact"


def test_the_fixture_reaches_an_uncertain_phase():
    """Thigh forward, knee neither straight enough to be landing nor folded
    enough to be swinging -- the gap the detector refuses to guess across."""
    assert _measure(128.0)["gait_phase"] == "uncertain"


# --------------------------------------------------------------------------
# flag OFF: nothing has changed
# --------------------------------------------------------------------------

def test_off_by_default_adds_no_contact_block():
    assert "contact" not in _measure(165.0)


def test_off_still_scores_an_uncertain_frame():
    """Today's behaviour, pinned. Arguably wrong -- which is exactly why the
    change to it is behind the flag and not in this test."""
    r = _measure(128.0)
    assert r["gait_phase"] == "uncertain"
    assert r["score"]["overall_score"] is not None
    assert "withheld" not in r["score"]


# --------------------------------------------------------------------------
# flag ON
# --------------------------------------------------------------------------

def test_on_measures_the_contact_readings_at_touchdown():
    c = _measure(165.0, contact_metrics=True)["contact"]
    assert c["measured"] is True
    assert c["phase"] == "initial_contact"
    for key in ("overstride_ratio", "shin_angle", "foot_strike"):
        assert key in c
    assert not math.isnan(c["overstride_ratio"])
    assert not math.isnan(c["shin_angle"])


def test_on_refuses_a_score_between_phases():
    r = _measure(128.0, contact_metrics=True)
    assert r["score"]["overall_score"] is None
    assert r["score"]["withheld"] == "stride_phase_uncertain"
    assert any("No score" in w for w in r["warnings"])


def test_on_gives_no_readings_away_from_touchdown_and_says_why():
    """Toe-off: a scored frame, but overstride does not exist there."""
    r = _measure(170.0, thigh=-20.0, contact_metrics=True)
    assert r["gait_phase"] == "toe_off"
    assert r["score"]["overall_score"] is not None
    c = r["contact"]
    assert c["measured"] is False
    assert c["overstride_ratio"] is None and c["shin_angle"] is None
    assert "touchdown" in c["note"]


# --------------------------------------------------------------------------
# the knee on a swing frame, and the partial score it leaves
# --------------------------------------------------------------------------

def _swing():
    """Thigh forward, knee folded tight -- the detector's swing."""
    return _measure(60.0, thigh=35.0, contact_metrics=True)


def test_the_fixture_reaches_swing():
    assert _swing()["gait_phase"] == "swing"


def test_off_still_faults_a_tightly_folded_swing_knee():
    """Today's behaviour, pinned: the one wide band's lower edge (80) marks the
    fold down. That is the error the flag exists to fix -- a world-class
    runner's swing frame came back "needs work" at 63 deg."""
    r = _measure(60.0, thigh=35.0)
    assert r["angles_with_context"]["knee"]["status"] != "phase_dependent"
    assert "knee" in r["score"]["per_angle"]


def test_on_shows_the_swing_knee_without_judging_it():
    r = _swing()
    k = r["angles_with_context"]["knee"]
    assert k["status"] == "phase_dependent"
    assert k["value"] == pytest.approx(60.0, abs=2.0)      # the number stays
    assert "folded" in k["note"]
    assert "knee" not in r["score"]["per_angle"]


def test_on_marks_a_swing_score_as_partial_in_the_runners_words():
    """A bare 100 on three measures out of five reads as a perfect stride.
    The bike's words ("upper body", "mark the pedal position") would be wrong
    here, so the run's carry their own."""
    sc = _swing()["score"]
    assert sc["partial"] == "without the knee or hip"
    assert sc["partial_badge"] == "partial"
    assert "foot lands" in sc["partial_hint"]
    assert "pedal" not in sc["partial_hint"]


def test_on_footstrike_is_not_partial():
    assert "partial" not in _measure(165.0, contact_metrics=True)["score"]


# --------------------------------------------------------------------------
# overstride in the score
# --------------------------------------------------------------------------

def _reaching():
    """Footstrike on a near-straight leg with the thigh well forward: the foot
    lands far out in front of the hip."""
    return _measure(170.0, thigh=30.0, contact_metrics=True)


def _under():
    """Footstrike with the thigh just past the contact threshold and the knee
    bent to 2 x thigh short of straight: the shank swings back by exactly the
    thigh's lean, so the ankle lands directly under the hip."""
    return _measure(140.0, thigh=20.0, contact_metrics=True)


def test_the_reach_fixtures_are_both_footstrike():
    assert _reaching()["gait_phase"] == "initial_contact"
    assert _under()["gait_phase"] == "initial_contact"


def test_overstride_enters_the_score_at_footstrike():
    r = _measure(165.0, contact_metrics=True)
    assert "overstride" in r["score"]["per_angle"]


def test_overstride_uses_the_videos_band():
    """A photo and a clip of one runner must not disagree about the same reach."""
    from app.services.video_analysis.biomechanics.sport_configs import RUNNING_REFERENCE
    assert PA._RUN_OVERSTRIDE_BAND == RUNNING_REFERENCE["overstride_ratio"]


def test_a_clear_reach_costs_the_score_and_is_explained():
    """The case that came back 100/100 "Excellent" with a 0.35 reach, because
    the knee and hip both sat inside their bands. The score has to move, and
    the reason has to be in words -- it has no line on the picture."""
    r = _reaching()
    c = r["contact"]
    assert c["overstride_ratio"] > PA._RUN_OVERSTRIDE_BAND[1]
    assert r["score"]["per_angle"]["overstride"]["score"] < 100
    assert r["score"]["overall_score"] < 90
    assert r["warnings"][0].startswith("Overstride:")


def test_a_foot_under_the_hip_is_not_penalised():
    r = _under()
    assert r["contact"]["overstride_ratio"] == pytest.approx(0.0, abs=0.02)
    assert r["score"]["per_angle"]["overstride"]["score"] == 100
    assert not any(w.startswith("Overstride:") for w in r["warnings"])


def test_overstride_stays_out_of_the_angle_table():
    """It is neither an angle nor drawable as one; it lives in the contact
    block. In the table it would get a label and an arc it cannot have."""
    r = _measure(165.0, contact_metrics=True)
    assert "overstride" not in r["angles_with_context"]


def test_on_does_not_touch_the_bike():
    """The flag is a running concern. A bike photo measured with it on must
    be the bike photo it always was."""
    from tests.test_photo_pedal_phase import _bike_pose

    world, norm = _bike_pose(140.0)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    off = PA.analyze_from_pose(image, world, norm, "right", [], "bike", "tt_aero",
                               render_thumbnail=False)
    on = PA.analyze_from_pose(image, world, norm, "right", [], "bike", "tt_aero",
                              render_thumbnail=False, contact_metrics=True)
    assert "contact" not in on
    assert on["score"] == off["score"]
