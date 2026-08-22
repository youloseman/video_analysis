"""What the mobility profile is allowed to change in a fit.

The screens themselves are tested in ``test_mobility.py``. This file is about
the join: a rider's off-bike range reaching the fit plan and the coach prompt,
and -- just as important -- everything it must leave alone.

The failure this guards against is a report that argues with itself. The
mobility card says the hips do not go there; the fit plan says lower the bars.
The bike is the easy thing to change, so that is the instruction the rider
follows, and the range they do not have comes out of the lower back.
"""

from __future__ import annotations

from app.services.video_analysis.biomechanics import mobility as M
from app.services.video_analysis.biomechanics.action_plan_builder import (
    action_plan_to_json,
    build_action_plan,
)
from app.services.video_analysis.biomechanics.cycling_positions import (
    get_cycling_reference,
)
from app.services.video_analysis.llm_recommendations import _fit_plan_block


def _upright_rider(position: str = "road_hoods") -> dict:
    """Metrics that make the plan want to say "lower the bars".

    Trunk above the band's upper bound is the one diagnostic mobility can
    overrule, so the fixture has to actually trigger it.
    """
    ref = get_cycling_reference(position)
    return {
        "trunk_angle_avg": ref["trunk_angle"][1] + 12,  # too upright
        "knee_at_bdc": ref["knee_at_bdc"][0] - 12,      # saddle too low
    }


def _plan(metrics, position="road_hoods", mobility_fit=None):
    return action_plan_to_json(build_action_plan(
        position=position,
        angle_statistics={},
        sport_specific_metrics=metrics,
        technique_score=70,
        letter_grade="C",
        detected_issues=[],
        mobility_fit=mobility_fit,
    ))


def _actions(plan):
    return [d["action"] for d in plan["diagnostics"]]


def _limited_profile():
    screens = {
        "hip_flexion": {"tier": M.TIER_LIMITED},
        "hamstring": {"tier": M.TIER_LIMITED},
    }
    return {"ceiling": M.position_ceiling(screens), "screens": screens}


def _good_profile():
    screens = {
        "hip_flexion": {"tier": M.TIER_GOOD},
        "hamstring": {"tier": M.TIER_GOOD},
    }
    return {"ceiling": M.position_ceiling(screens), "screens": screens}


# --------------------------------------------------------------------------
# the fixture has to bite, or none of the rest means anything
# --------------------------------------------------------------------------

def test_without_mobility_the_plan_says_lower_the_bars():
    assert "lower_bars" in _actions(_plan(_upright_rider()))


# --------------------------------------------------------------------------
# what a limited range changes
# --------------------------------------------------------------------------

def test_a_limited_range_replaces_lower_the_bars():
    fit = M.assess_position(_limited_profile(), "road_hoods")
    assert fit["within"] is False
    plan = _plan(_upright_rider(), mobility_fit=fit)
    assert "lower_bars" not in _actions(plan)
    assert "improve_mobility" in _actions(plan)


def test_the_finding_is_rewritten_not_deleted():
    """The trunk really is above the band, and that stays in the report. What
    changes is the instruction attached to it -- deleting the row would hide a
    measurement to protect a recommendation."""
    fit = M.assess_position(_limited_profile(), "road_hoods")
    plan = _plan(_upright_rider(), mobility_fit=fit)
    row = next(d for d in plan["diagnostics"] if d["action"] == "improve_mobility")
    assert row["component"] == "bar_position"
    assert row["metric_name"] == "trunk_angle"
    assert row["current_value"] == _plan(_upright_rider())["diagnostics"][
        _actions(_plan(_upright_rider())).index("lower_bars")
    ]["current_value"]
    assert row["status"] == "mobility_limited"


def test_the_rewritten_row_says_it_is_a_rule_of_thumb():
    fit = M.assess_position(_limited_profile(), "road_hoods")
    plan = _plan(_upright_rider(), mobility_fit=fit)
    row = next(d for d in plan["diagnostics"] if d["action"] == "improve_mobility")
    assert "rule of thumb" in row["reason"]
    assert "lower back" in row["reason"]


def test_saddle_advice_is_untouched_by_mobility():
    """Saddle height is about the pedalling leg. How far the hip closes at the
    top of the stroke has nothing to say about it, and a mobility screen that
    started editing saddle heights would be overreaching."""
    fit = M.assess_position(_limited_profile(), "road_hoods")
    plain = _plan(_upright_rider())
    gated = _plan(_upright_rider(), mobility_fit=fit)
    saddle = lambda p: [d for d in p["diagnostics"] if d["component"] == "saddle_height"]
    assert saddle(plain) == saddle(gated)
    assert saddle(gated), "fixture no longer produces a saddle diagnostic"


# --------------------------------------------------------------------------
# what it must NOT change
# --------------------------------------------------------------------------

def test_no_profile_changes_nothing():
    assert _plan(_upright_rider()) == _plan(_upright_rider(), mobility_fit=None)


def test_a_range_that_supports_the_position_changes_nothing():
    fit = M.assess_position(_good_profile(), "road_hoods")
    assert fit["within"] is True
    plain = _plan(_upright_rider())
    gated = _plan(_upright_rider(), mobility_fit=fit)
    assert plain["diagnostics"] == gated["diagnostics"]


def test_mobility_never_moves_a_band():
    """The published window for a position is what the rider was measured
    against, and that stays true whatever their hips do. A band that quietly
    shifted would make two riders' identical numbers score differently."""
    fit = M.assess_position(_limited_profile(), "tt_aero")
    before = dict(get_cycling_reference("tt_aero"))
    plan = _plan(_upright_rider("tt_aero"), position="tt_aero", mobility_fit=fit)
    assert get_cycling_reference("tt_aero") == before
    for d in plan["diagnostics"]:
        assert tuple(d["target_range"]) == tuple(
            before.get(d["metric_name"], tuple(d["target_range"]))
        )


def test_the_score_is_not_touched():
    fit = M.assess_position(_limited_profile(), "road_hoods")
    assert _plan(_upright_rider(), mobility_fit=fit)["technique_score"] == 70


# --------------------------------------------------------------------------
# the verdict travels to the coach
# --------------------------------------------------------------------------

def test_the_coach_is_told_not_to_say_get_lower():
    fit = M.assess_position(_limited_profile(), "road_hoods")
    block = _fit_plan_block(_plan(_upright_rider(), mobility_fit=fit))
    assert "OFF-BIKE MOBILITY" in block
    assert "Do NOT tell them to get lower" in block


def test_the_coach_is_told_not_to_invent_a_flexibility_problem():
    fit = M.assess_position(_good_profile(), "road_hoods")
    block = _fit_plan_block(_plan(_upright_rider(), mobility_fit=fit))
    assert "range is not the limiter" in block


def test_the_coach_hears_nothing_when_no_screens_were_done():
    block = _fit_plan_block(_plan(_upright_rider()))
    assert "MOBILITY" not in block


def test_a_mobility_verdict_alone_still_reaches_the_coach():
    """A rider whose fit is otherwise clean can still be riding below their
    ceiling, and that is exactly the rider the prose needs to warn."""
    fit = M.assess_position(_limited_profile(), "tt_aero")
    block = _fit_plan_block({"diagnostics": [], "mobility_fit": fit})
    assert "OFF-BIKE MOBILITY" in block


# --------------------------------------------------------------------------
# the runner hands it through
# --------------------------------------------------------------------------

def test_run_analysis_accepts_a_mobility_profile():
    import inspect
    from app.services.video_analysis.runner import run_analysis

    sig = inspect.signature(run_analysis)
    assert "mobility_profile" in sig.parameters
    assert sig.parameters["mobility_profile"].default is None
