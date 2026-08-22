"""Off-bike mobility screens.

Two kinds of assertion here, matching the two kinds of claim the module makes.

The geometry tests build landmarks by hand at known angles and check the screen
reports that angle back -- if the maths is wrong, every downstream tier is
wrong and nothing else in this file means anything.

The rest are about what the module is allowed to SAY. A screen that cannot see
a knee must not report a limited range; a rule of thumb must not be presented
as a measurement; and no amount of floor mobility data may quietly rewrite a
published fit band.
"""

from __future__ import annotations

import math

import pytest

from app.services.video_analysis.biomechanics import mobility as M
from app.services.video_analysis.biomechanics.landmarks import PoseLandmark as LM


# --------------------------------------------------------------------------
# synthetic landmarks
# --------------------------------------------------------------------------

class _P:
    def __init__(self, x: float, y: float, z: float = 0.0, visibility: float = 1.0):
        self.x, self.y, self.z, self.visibility = x, y, z, visibility


def supine(
    raise_deg: float = 0.0,
    knee_bend_deg: float = 0.0,
    *,
    raised: str = "left",
    visibility: float = 1.0,
) -> list[_P]:
    """A body lying on its back, seen from the side, one leg raised.

    Image coordinates: +x to the right, +y DOWN. The athlete lies along the x
    axis with the head at x=0, so the torso runs head -> hip in +x and the legs
    continue in +x. Raising a leg lifts the knee, which is -y.

    ``raise_deg`` is hip flexion: 0 = leg flat in line with the torso.
    ``knee_bend_deg`` bends the shank back from the thigh by that much.
    """
    pts = [_P(0.0, 0.0, 0.0, visibility) for _ in range(33)]

    head_x, hip_x, floor_y = 0.0, 0.50, 0.60
    thigh, shank = 0.22, 0.22

    for sh, hip, knee, ankle, is_raised in (
        (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE, raised == "left"),
        (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE, raised == "right"),
    ):
        pts[sh] = _P(head_x, floor_y, 0.0, visibility)
        pts[hip] = _P(hip_x, floor_y, 0.0, visibility)

        theta = math.radians(raise_deg if is_raised else 0.0)
        kx = hip_x + thigh * math.cos(theta)
        ky = floor_y - thigh * math.sin(theta)
        pts[knee] = _P(kx, ky, 0.0, visibility)

        # Shank continues the thigh, rotated back by the knee bend.
        phi = theta - math.radians(knee_bend_deg if is_raised else 0.0)
        pts[ankle] = _P(kx + shank * math.cos(phi), ky - shank * math.sin(phi), 0.0, visibility)

    return pts


def standing() -> list[_P]:
    """Upright: torso vertical, so the supine check must reject it."""
    pts = [_P(0.0, 0.0) for _ in range(33)]
    for sh, hip, knee, ankle in (
        (LM.LEFT_SHOULDER, LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE),
        (LM.RIGHT_SHOULDER, LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    ):
        pts[sh] = _P(0.5, 0.20)
        pts[hip] = _P(0.5, 0.50)
        pts[knee] = _P(0.5, 0.72)
        pts[ankle] = _P(0.5, 0.94)
    return pts


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0.0, 45.0, 65.0, 80.0, 95.0])
def test_straight_leg_raise_reads_back_the_angle_it_was_given(deg):
    got = M.measure_screen("hamstring", supine(raise_deg=deg))
    assert "error" not in got, got
    assert got["value"] == pytest.approx(deg, abs=0.5)


def test_the_raised_leg_is_found_whichever_side_it_is():
    for side in ("left", "right"):
        got = M.measure_screen("hamstring", supine(raise_deg=72.0, raised=side))
        assert got["value"] == pytest.approx(72.0, abs=0.5)
        assert got["side"] == side


def test_knee_to_chest_reads_hip_flexion_with_the_knee_folded():
    got = M.measure_screen(
        "hip_flexion", supine(raise_deg=118.0, knee_bend_deg=110.0),
    )
    assert "error" not in got, got
    assert got["value"] == pytest.approx(118.0, abs=0.5)


# --------------------------------------------------------------------------
# refusing to measure the wrong thing
# --------------------------------------------------------------------------

def test_a_bent_knee_invalidates_the_straight_leg_raise():
    """Bending the knee takes the hamstring out of the movement and buys
    30-40 deg of hip angle for free. Reporting that as hamstring length would
    hand the most restricted riders the best scores."""
    got = M.measure_screen("hamstring", supine(raise_deg=110.0, knee_bend_deg=80.0))
    assert "error" in got
    assert "straight" in got["error"].lower()
    assert "value" not in got


def test_a_straight_knee_invalidates_knee_to_chest():
    got = M.measure_screen("hip_flexion", supine(raise_deg=75.0, knee_bend_deg=0.0))
    assert "error" in got
    assert "knee-to-chest" in got["error"]


@pytest.mark.parametrize("screen", ["hamstring", "hip_flexion"])
def test_standing_up_is_refused_not_measured(screen):
    got = M.measure_screen(screen, standing())
    assert "error" in got
    assert "lying down" in got["error"]


@pytest.mark.parametrize("screen", ["hamstring", "hip_flexion"])
def test_an_invisible_joint_is_an_error_not_a_limited_range(screen):
    """"We could not see your knee" and "your hip does not go there" are
    opposite findings. Collapsing them would tell a mobile rider they are
    stiff because the room was dark."""
    got = M.measure_screen(screen, supine(raise_deg=80.0, visibility=0.1))
    assert "error" in got
    assert got.get("tier") is None


def test_unknown_screen_is_a_programming_error():
    with pytest.raises(ValueError):
        M.measure_screen("thoracic_rotation", supine())


# --------------------------------------------------------------------------
# tiers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("deg,tier", [
    (95.0, M.TIER_GOOD), (80.0, M.TIER_GOOD),
    (79.9, M.TIER_MODERATE), (65.0, M.TIER_MODERATE),
    (64.9, M.TIER_LIMITED), (30.0, M.TIER_LIMITED),
])
def test_hamstring_tier_cuts(deg, tier):
    assert M.screen_tier("hamstring", deg) == tier


@pytest.mark.parametrize("deg,tier", [
    (135.0, M.TIER_GOOD), (120.0, M.TIER_GOOD),
    (119.9, M.TIER_MODERATE), (105.0, M.TIER_MODERATE),
    (104.9, M.TIER_LIMITED),
])
def test_hip_flexion_tier_cuts(deg, tier):
    assert M.screen_tier("hip_flexion", deg) == tier


def test_every_tier_has_something_to_say():
    for key, spec in M.MOBILITY_SCREENS.items():
        for tier in M.TIER_ORDER:
            assert spec["reads"].get(tier), f"{key}/{tier} has no read"
        assert spec["source"], f"{key} cites no reference for its cut-points"
        assert spec["setup"], f"{key} does not say how to shoot it"


# --------------------------------------------------------------------------
# the ceiling -- the inferred half
# --------------------------------------------------------------------------

def _screens(hamstring=None, hip=None):
    out = {}
    if hamstring is not None:
        out["hamstring"] = M.measure_screen("hamstring", supine(raise_deg=hamstring))
    if hip is not None:
        out["hip_flexion"] = M.measure_screen(
            "hip_flexion", supine(raise_deg=hip, knee_bend_deg=110.0),
        )
    return out


def test_no_screens_means_no_verdict():
    """Silence is not a diagnosis. A rider who has not done the screens gets
    no ceiling at all, rather than a cautious one."""
    assert M.position_ceiling({}) is None
    assert M.build_mobility_profile({})["ceiling"] is None


def test_good_hip_flexion_rules_nothing_out():
    ceiling = M.position_ceiling(_screens(hip=130.0, hamstring=85.0))
    assert ceiling["unrestricted"] is True
    assert ceiling["rung"] == 0


def test_limited_hip_flexion_stops_at_the_hoods():
    ceiling = M.position_ceiling(_screens(hip=95.0, hamstring=85.0))
    assert ceiling["position"] == "road_hoods"
    assert any("hip flexion" in r for r in ceiling["reasons"])


def test_short_hamstrings_cost_one_further_rung():
    both_ways = M.position_ceiling(_screens(hip=95.0, hamstring=50.0))
    hip_only = M.position_ceiling(_screens(hip=95.0, hamstring=85.0))
    assert both_ways["rung"] == hip_only["rung"] + 1
    assert both_ways["position"] == "casual"


def test_the_ceiling_cannot_fall_off_the_bottom():
    ceiling = M.position_ceiling(_screens(hip=40.0, hamstring=20.0))
    assert ceiling["rung"] == len(M.POSITION_RUNGS) - 1
    assert ceiling["position"] == "casual"


def test_hamstrings_alone_do_not_clear_a_rider_for_aero():
    """The hip is what closes at the top of the stroke. Excellent hamstrings
    with no hip measurement is not evidence the hip goes there."""
    ceiling = M.position_ceiling(_screens(hamstring=95.0))
    assert ceiling["unrestricted"] is False
    assert ceiling["position"] == "road_drops"


def test_the_rule_of_thumb_says_so_every_time():
    """The measurement is a measurement; the position guidance is a fitter's
    heuristic. A reader who cannot tell them apart is being misled by the
    honest half."""
    ceiling = M.position_ceiling(_screens(hip=95.0))
    assert "rule of thumb" in ceiling["caveat"]
    assessed = M.assess_position(
        M.build_mobility_profile(_screens(hip=95.0)), "tt_aero",
    )
    assert "rule of thumb" in assessed["caveat"]


# --------------------------------------------------------------------------
# what the analysis does with it
# --------------------------------------------------------------------------

def test_riding_below_your_ceiling_is_flagged():
    profile = M.build_mobility_profile(_screens(hip=95.0, hamstring=85.0))
    got = M.assess_position(profile, "tt_aero")
    assert got["within"] is False
    assert "not the next move" in got["message"]
    assert got["ceiling"] == "road_hoods"


def test_riding_within_your_range_says_so_plainly():
    profile = M.build_mobility_profile(_screens(hip=130.0, hamstring=85.0))
    got = M.assess_position(profile, "tt_aero")
    assert got["within"] is True


def test_tt_and_triathlon_share_a_rung():
    """They ask the same thing of the hip. Splitting them would claim a
    resolution two floor photos do not have."""
    profile = M.build_mobility_profile(_screens(hip=112.0, hamstring=85.0))
    assert M.assess_position(profile, "tt_aero")["within"] is False
    assert M.assess_position(profile, "triathlon")["within"] is False
    assert M.assess_position(profile, "road_drops")["within"] is True


def test_no_profile_means_no_opinion():
    assert M.assess_position(None, "tt_aero") is None
    assert M.assess_position(M.build_mobility_profile({}), "tt_aero") is None


def test_a_position_we_do_not_rank_gets_no_verdict():
    profile = M.build_mobility_profile(_screens(hip=95.0))
    assert M.assess_position(profile, "recumbent") is None


def test_failed_screens_are_kept_so_the_profile_can_say_what_is_missing():
    measured = {
        "hamstring": M.measure_screen("hamstring", supine(raise_deg=85.0)),
        "hip_flexion": M.measure_screen("hip_flexion", standing()),
    }
    profile = M.build_mobility_profile(measured)
    assert profile["screens_done"] == ["hamstring"]
    assert "hip_flexion" in profile["unmeasured"]
    # and the ceiling is the hamstring-only one, not a hip verdict
    assert profile["ceiling"]["position"] == "road_drops"


@pytest.mark.parametrize("goal,expect", [
    ("comfort", "comfort"), ("speed", "speed"), ("fastest", None), (None, None),
])
def test_the_goal_is_a_stated_preference_not_a_free_text_field(goal, expect):
    assert M.build_mobility_profile({}, goal=goal)["goal"] == expect


def test_every_ranked_position_is_a_real_cycling_position():
    """The ceiling names positions the rest of the app has to recognise."""
    from app.services.video_analysis.biomechanics.cycling_positions import (
        CYCLING_POSITIONS,
    )
    assert set(M.POSITION_AGGRESSION) == set(CYCLING_POSITIONS)
    assert len(M.RUNG_LABELS) == len(M.POSITION_RUNGS)
