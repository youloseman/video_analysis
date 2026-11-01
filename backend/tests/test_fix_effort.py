"""Who makes each fit change: at home, it depends, or a bike shop.

Mike (bike fitter, 2026-11-01): a remote rider has no mechanic, and swapping
the new aero bars on a disc-brake bike is a shop job. A plan that lists that
next to "raise the saddle 5 mm" as if they cost the same is the thing this
guards against -- and a note on an item that changes nothing would be noise.
"""
from __future__ import annotations

from app.services.report_pdf import _fit_priorities
from app.services.video_analysis.biomechanics.action_plan_builder import (
    ActionPlan,
    Diagnostic,
    action_plan_to_json,
    fix_effort,
)


def _d(component, action, status="needs_adjustment"):
    return Diagnostic(component=component, status=status, metric_name="m", current_value=1.0,
                      target_range=(0.0, 2.0), action=action, amount="5mm", reason="r", priority=1)


def test_each_part_says_who_makes_the_change():
    assert fix_effort("saddle_height", "raise_saddle", "road_hoods")["level"] == "home"
    assert fix_effort("saddle_fore_aft", "move_saddle_forward", "triathlon")["level"] == "home"
    assert fix_effort("crank_length", "consider_shorter_cranks", "road_hoods")["level"] == "shop"


def test_the_bars_depend_on_the_cockpit_and_the_note_follows_the_position():
    road = fix_effort("bar_position", "lower_bars", "road_hoods")
    aero = fix_effort("bar_position", "lower_bars", "triathlon")
    assert road["level"] == aero["level"] == "depends"
    assert "stem" in road["note"] and "extensions" in aero["note"]


def test_nothing_to_change_carries_no_effort():
    assert fix_effort("saddle_fore_aft", "reassess_after_saddle_height", "road_hoods") is None
    plan = ActionPlan(position="road_hoods", position_label="Road", terminology={},
                      technique_score=80, letter_grade="B",
                      diagnostics=[_d("saddle_height", "raise_saddle"),
                                   _d("saddle_height", "", status="optimal")])
    rows = action_plan_to_json(plan)["diagnostics"]
    assert rows[0]["effort"]["level"] == "home"
    assert rows[1]["effort"] is None


def test_the_pdf_card_carries_the_same_words():
    fit = action_plan_to_json(ActionPlan(position="triathlon", position_label="Tri", terminology={},
                                         technique_score=80, letter_grade="B",
                                         diagnostics=[_d("crank_length", "consider_shorter_cranks")]))
    assert _fit_priorities(fit)[0]["drill"].endswith("· bike shop")
    # an old report, saved before the field existed, renders as it always did
    old = {"diagnostics": [{"component": "saddle_height", "status": "needs_adjustment",
                            "action": "raise_saddle", "amount": "5mm", "priority": 1}]}
    assert _fit_priorities(old)[0]["drill"] == "Raise saddle 5mm"
