"""The flag in front of the contact readings, end to end.

Everything in test_photo_run_contact.py is behind ``contact_metrics``. This
file is about the flag itself, through the real measuring function: with it
OFF a running photo comes back exactly as it did before this work -- no
``contact`` key, and a frame the detector cannot place still scored -- and
with it ON the readings appear and the "no score" rule applies.

With the flag ON the photo also stops trusting the model's left/right leg
labels (see ``_name_run_legs``): the measured leg is the STANCE leg, the one
whose foot is lowest, and the stride instant is read off where that foot sits
against the hip. The fixture therefore draws two legs, a stance leg and a
swing leg, and can put either on either label.

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
LEFT_LEG = (25, 27, 29, 31)
RIGHT_LEG = (26, 28, 30, 32)


def _run_pose(knee_deg: float, thigh_forward_deg: float = 30.0, *,
              other_lift: float = 0.15, on_far_label: bool = False,
              knee_vis: float = 1.0):
    """A runner seen from the LEFT, facing +x.

    The described leg has its thigh ``thigh_forward_deg`` ahead of vertical
    and its knee at ``knee_deg``. The other leg trails behind with its foot
    ``other_lift`` (frame heights) above where it would touch -- a swing leg,
    so the described one is the stance leg. ``other_lift`` that puts both feet
    at one height is a frame in flight.

    ``on_far_label``: the described leg carries the RIGHT (far) labels, the
    way MediaPipe hands them out on a single frame about half the time.

    Image y grows downwards. The world frame is the image frame recentred and
    NOT stretched. Both hips sit on one point: in a side view they nearly do,
    and it keeps "ahead of the hip" one number.
    """
    hip = np.array([0.45, 0.50])
    thigh_len, shank_len = 0.18, 0.18
    a = math.radians(thigh_forward_deg)
    knee = hip + thigh_len * np.array([math.sin(a), math.cos(a)])
    # Knee flexion swings the shank BACK, as a knee does.
    b = a - math.radians(180.0 - knee_deg)
    ankle = knee + shank_len * np.array([math.sin(b), math.cos(b)])
    shoulder = hip + np.array([0.0, -0.28])

    # The swing leg: behind, knee bent, foot raised off the stance foot's floor.
    floor = ankle[1]
    o_knee = hip + np.array([-0.03, 0.17])
    o_ankle = np.array([hip[0] - 0.16, floor - other_lift])

    leg = {LEFT_LEG[0]: knee, LEFT_LEG[1]: ankle,
           LEFT_LEG[2]: ankle + [-0.03, 0.02], LEFT_LEG[3]: ankle + [0.06, 0.02]}
    oth = {RIGHT_LEG[0]: o_knee, RIGHT_LEG[1]: o_ankle,
           RIGHT_LEG[2]: o_ankle + [-0.03, 0.02], RIGHT_LEG[3]: o_ankle + [0.06, 0.02]}
    vis = {LEFT_LEG[0]: knee_vis}
    if on_far_label:
        swap = dict(zip(LEFT_LEG, RIGHT_LEG)) | dict(zip(RIGHT_LEG, LEFT_LEG))
        leg = {swap[k]: v for k, v in leg.items()}
        oth = {swap[k]: v for k, v in oth.items()}
        vis = {swap[k]: v for k, v in vis.items()}

    pts = {11: shoulder, 12: shoulder + [0.02, 0], 13: shoulder + [0.05, 0.12],
           14: shoulder + [0.07, 0.12], 15: shoulder + [0.10, 0.20],
           16: shoulder + [0.12, 0.20], 23: hip, 24: hip, **leg, **oth}
    # The normalized x is the design x in frame HEIGHTS divided back into frame
    # widths, so the pose is undistorted in pixels -- the flag-on path reads the
    # picture (x * W/H), the flag-off path the world skeleton, and both must
    # see the same body.
    norm, world = [], []
    for j in range(33):
        x, y = pts.get(j, (0.5, 0.1))
        v = vis.get(j, 1.0)
        norm.append(SimpleNamespace(x=float(x) * H / W, y=float(y), z=0.0, visibility=v))
        world.append(SimpleNamespace(x=float(x) - 0.5, y=float(y) - 0.5, z=0.0, visibility=v))
    return world, norm


def _measure(knee_deg: float, thigh: float = 30.0, *, contact_metrics: bool = False, **pose) -> dict:
    world, norm = _run_pose(knee_deg, thigh, **pose)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    return PA.analyze_from_pose(image, world, norm, "left", [], "run", None,
                                render_thumbnail=False, contact_metrics=contact_metrics)


def _on(knee_deg: float, thigh: float = 30.0, **pose) -> dict:
    return _measure(knee_deg, thigh, contact_metrics=True, **pose)


# --------------------------------------------------------------------------
# flag OFF: nothing has changed
# --------------------------------------------------------------------------

def test_off_by_default_adds_no_contact_block():
    assert "contact" not in _measure(165.0)


def test_off_still_scores_an_uncertain_frame():
    """Today's behaviour, pinned: the thigh-and-knee guess cannot place a
    forward thigh on a half-bent knee, and scores it anyway. Arguably wrong --
    which is exactly why the change to it is behind the flag."""
    r = _measure(128.0)
    assert r["gait_phase"] == "uncertain"
    assert r["score"]["overall_score"] is not None
    assert "withheld" not in r["score"]


def test_off_still_faults_a_tightly_folded_swing_knee():
    """Today's behaviour, pinned: the one wide band's lower edge (80) marks the
    fold down -- a world-class runner's swing frame came back "needs work" at
    63 deg. With the flag on, a swing leg is never the measured leg."""
    r = _measure(60.0, thigh=35.0)
    assert r["gait_phase"] == "swing"
    assert "knee" in r["score"]["per_angle"]


def test_off_still_reads_the_near_label_whatever_it_is_on():
    """Off, the leg carrying the near label is measured even when it is the
    swing leg -- the old behaviour, and the bug the flag fixes."""
    r = _measure(165.0, on_far_label=True)
    assert r["angles_with_context"]["knee"]["value"] != pytest.approx(165.0, abs=3.0)


# --------------------------------------------------------------------------
# flag ON: the stance leg, found by position
# --------------------------------------------------------------------------

def test_the_stance_leg_is_measured_whichever_label_it_carries():
    """The model put the landing leg on the FAR labels. It is still the one
    measured -- and drawn, since it is moved onto the camera-side indices."""
    for far in (False, True):
        r = _on(165.0, on_far_label=far)
        assert r["gait_phase"] == "initial_contact"
        assert r["angles_with_context"]["knee"]["value"] == pytest.approx(165.0, abs=1.5)
        assert r["contact"]["legs"]["swapped"] is far
        assert r["contact"]["legs"]["stance"] == "front"


def test_the_swing_leg_is_reported_not_scored():
    r = _on(165.0)
    legs = r["contact"]["legs"]
    assert legs["swing_knee"] is not None and legs["swing_knee"] < 160
    assert "swing_knee" not in r["score"]["per_angle"]


def test_a_knee_point_the_model_calls_hidden_is_still_read_once_moved():
    """Having labelled a leg "far", the model rates that leg's knee hidden
    wherever it actually is (0.37-0.48 on real treadmill photos). Once the
    legs are named by position, such a point is read; one the model barely
    found at all is not."""
    r = _on(165.0, on_far_label=True, knee_vis=0.4)
    assert r["angles_with_context"]["knee"]["value"] == pytest.approx(165.0, abs=1.5)
    r = _on(165.0, on_far_label=True, knee_vis=0.2)
    assert r["score"]["withheld"] == "run_leg_not_measured"


@pytest.mark.parametrize("knee,thigh,phase", [
    (165.0, 30.0, "initial_contact"),   # foot well ahead of the hip
    (160.0, 11.0, "initial_contact"),   # foot just ahead, leg near straight
    (140.0, 20.0, "midstance"),         # foot under the hip, knee loaded
    (170.0, -20.0, "toe_off"),          # foot well behind
])
def test_the_instant_comes_from_where_the_stance_foot_is(knee, thigh, phase):
    assert _on(knee, thigh)["gait_phase"] == phase


def test_a_frame_in_flight_is_not_placed():
    """Both feet at one height: no foot is on the ground, nothing to read the
    instant from -- no score, and saying so."""
    r = _on(165.0, other_lift=0.0)
    assert r["gait_phase"] == "uncertain"
    assert r["score"]["overall_score"] is None
    assert r["score"]["withheld"] == "stride_phase_uncertain"
    assert any("No score" in w for w in r["warnings"])


def test_phase_thresholds():
    ph = PA._run_phase_by_position
    assert ph({"clear": False, "dx": 0.3}, 170.0) == "uncertain"
    assert ph({"clear": True, "dx": None}, 170.0) == "uncertain"
    assert ph({"clear": True, "dx": 0.11}, 120.0) == "initial_contact"
    assert ph({"clear": True, "dx": 0.02}, 156.0) == "initial_contact"
    assert ph({"clear": True, "dx": 0.02}, 150.0) == "midstance"
    assert ph({"clear": True, "dx": -0.13}, 160.0) == "toe_off"


# --------------------------------------------------------------------------
# flag ON: the contact readings
# --------------------------------------------------------------------------

def test_on_measures_the_contact_readings_at_touchdown():
    c = _on(165.0)["contact"]
    assert c["measured"] is True
    assert c["phase"] == "initial_contact"
    for key in ("overstride_ratio", "shin_angle", "foot_strike"):
        assert key in c
    assert not math.isnan(c["overstride_ratio"])
    assert not math.isnan(c["shin_angle"])


def test_on_gives_no_readings_away_from_touchdown_and_says_why():
    """Toe-off: a scored frame, but overstride does not exist there."""
    r = _on(170.0, thigh=-20.0)
    assert r["gait_phase"] == "toe_off"
    assert r["score"]["overall_score"] is not None
    c = r["contact"]
    assert c["measured"] is False
    assert c["overstride_ratio"] is None and c["shin_angle"] is None
    assert "touchdown" in c["note"]


def test_on_footstrike_is_not_partial():
    assert "partial" not in _on(165.0)["score"]


# --------------------------------------------------------------------------
# the touchdown in the score: the shin, not the reach
# --------------------------------------------------------------------------

def _reaching():
    """Footstrike on a near-straight leg with the thigh well forward: the foot
    lands far out in front of the hip, the shin leaning back 20 deg."""
    return _on(170.0, thigh=30.0)


def _under():
    """Footstrike on a near-straight leg with the foot just ahead of the hip."""
    return _on(160.0, thigh=11.0)


def test_the_reach_fixtures_are_both_footstrike():
    assert _reaching()["gait_phase"] == "initial_contact"
    assert _under()["gait_phase"] == "initial_contact"


def test_the_shin_enters_the_score_at_footstrike_and_the_reach_does_not():
    """On all 13 marked race/landing touchdowns the reach sat past the video's
    band -- Kiptum too: a frame is a touchdown only once the foot is 0.10 ahead,
    and the reach grows with speed a still does not have. It is shown, with
    the reason; the shin is what is scored."""
    r = _on(165.0)
    assert "shin" in r["score"]["per_angle"] and "overstride" not in r["score"]["per_angle"]
    c = r["contact"]
    assert c["overstride_ratio"] > PA._RUN_OVERSTRIDE_BAND[1]
    assert "overstride_status" not in c and "not scored" in c["overstride_note"]
    assert c["shin_band"] == list(PA._RUN_SHIN_BAND)


def test_a_clear_reach_costs_the_score_and_is_explained():
    """The shin leaning well back is the reach that brakes: red, it costs the
    score, and the reason is in words."""
    r = _reaching()
    assert r["contact"]["shin_angle"] < PA._RUN_SHIN_BAND[0] - 6
    assert r["contact"]["shin_status"] == "needs_work"
    assert r["score"]["per_angle"]["shin"]["score"] <= 50
    assert r["score"]["overall_score"] < 80
    assert r["warnings"][0].startswith("Shin at touchdown:")


def test_a_shin_a_little_past_the_band_is_amber():
    """Kiptum, Chicago 2023: -11.7 deg. Worth a look, not a fault."""
    r = _on(165.0)
    assert -16 < r["contact"]["shin_angle"] < -10
    assert r["contact"]["shin_status"] == "acceptable"


def test_a_foot_under_the_hip_is_not_penalised():
    r = _under()
    assert r["contact"]["shin_status"] == "optimal"
    assert r["score"]["per_angle"]["shin"]["score"] == 100
    assert not any(w.startswith("Shin at touchdown:") for w in r["warnings"])


def test_the_elbow_is_shown_not_scored():
    """Elites caught with the arm at the front of its swing read 43 and 50 deg
    (Chicago 2026): where the arm is in its swing is when the shutter fired."""
    r = _on(165.0)
    assert "elbow" not in r["score"]["per_angle"]
    row = r["angles_with_context"]["elbow"]
    assert row["status"] == "phase_dependent" and row["unscored_reason"] == "arm swing"


def test_overstride_stays_out_of_the_angle_table():
    """It is neither an angle nor drawable as one; it lives in the contact
    block. In the table it would get a label and an arc it cannot have."""
    assert "overstride" not in _on(165.0)["angles_with_context"]


# --------------------------------------------------------------------------
# the picture, not the world skeleton
# --------------------------------------------------------------------------

def _biased(world_shift: float):
    """The runner with the WORLD skeleton's knee pushed forward and the picture
    left as it is -- the world-vs-image disagreement the bike had, ~8 deg from
    one side. Flag off reads the world, flag on reads the picture."""
    world, norm = _run_pose(165.0)
    world[25] = SimpleNamespace(x=world[25].x + world_shift, y=world[25].y,
                                z=0.0, visibility=1.0)
    return world, norm


def test_on_reads_the_picture_not_the_world_skeleton():
    world, norm = _biased(0.04)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    on = PA.analyze_from_pose(image, world, norm, "left", [], "run", None,
                              render_thumbnail=False, contact_metrics=True)
    off = PA.analyze_from_pose(image, world, norm, "left", [], "run", None,
                               render_thumbnail=False)
    assert on["angles_with_context"]["knee"]["value"] == pytest.approx(165.0, abs=1.0)
    assert abs(off["angles_with_context"]["knee"]["value"] - 165.0) > 5.0


def test_the_picture_is_read_in_pixels_not_normalized_units():
    """Normalized x and y are different units on an 800 x 1000 frame. Read
    raw, this knee would not be 165."""
    world, norm = _run_pose(165.0)
    image = np.full((H, W, 3), 128, dtype=np.uint8)
    pl = PA._image_plane(norm, image)
    assert PA.calculate_angle_2d(pl, 23, 25, 27)[0] == pytest.approx(165.0, abs=0.5)
    assert abs(PA.calculate_angle_2d(norm, 23, 25, 27)[0] - 165.0) > 1.0


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
