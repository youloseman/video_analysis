"""One document, one verdict per number.

Five real reports read side by side (2026-11-06) disagreed with themselves: a 94/A
run led with "Fix first: Foot Strike" over a tile reading 0 against a 0-0 range; a
tile said "In range" in red; "leave these alone" listed a number marked "At the
edge"; "Lower saddle 23 mm" sat beside "Each adjustment: 5mm only"; the cover's
table averaged 72 over a score of 46. A coach finds these in two minutes, and then
doesn't believe the rest.
"""

from __future__ import annotations

import re

from app.services.report_pdf import _score_notes, plan_rows, render_report_html
from app.services.video_analysis.biomechanics.action_plan_builder import (
    FIT_STEP_NOTE,
    build_action_plan,
)
from app.services.video_analysis.biomechanics.running_action_plan_builder import (
    build_running_action_plan,
)


def _run(strong=(), issues=(), prios=(), **metrics) -> dict:
    return {
        "sport_type": "run", "technique_score": 94,
        "training_plan": {"strong_points": list(strong), "issues": list(issues),
                          "top_3_priorities": list(prios)},
        "sport_specific_metrics": metrics,
    }


def _good(metric, value, lo, hi) -> dict:
    return {"metric": metric, "value": value, "range": (lo, hi), "description": f"{metric}: {value}"}


FOOT = {"metric": "foot_strike", "label": "Foot Strike", "value": 0.0, "optimal_range": "0.0-0.0",
        "severity": "minor"}
FOOT_PRIO = {"metric": "foot_strike", "label": "Foot Strike", "drill": "Midfoot Landing Drill",
             "cue": "Land softly on the middle of your foot", "weeks": 6}


# --- foot strike: a word, not a number, and never the first thing to fix

def test_foot_strike_is_a_word_with_no_verdict_and_no_zero_range():
    res = _run(strong=[_good("cadence", 180.0, 170, 190)], issues=[FOOT], prios=[FOOT_PRIO],
               foot_strike="heel", foot_strike_estimated=True)
    by = {r["field"]: r for r in plan_rows(res)}
    assert by["foot_strike"]["text"] == "Heel." and by["foot_strike"]["value"] is None
    assert by["foot_strike"]["status"] == "info"
    html = render_report_html(res, {})
    assert "0–0" not in html and "Estimated from 2D video" in html and "Not graded" in html


def test_a_saved_run_does_not_lead_with_foot_strike():
    html = render_report_html(_run(strong=[_good("cadence", 180.0, 170, 190)], issues=[FOOT],
                                   prios=[FOOT_PRIO], foot_strike="heel"), {})
    assert "Fix first" not in html and "Midfoot Landing Drill" not in html
    assert "Fix these, in this order" not in html


def test_a_new_plan_does_not_lead_with_foot_strike():
    plan = build_running_action_plan({"score": 94, "summary": {
        "cadence_spm": 178.0, "vertical_oscillation_m": 0.08, "trunk_lean_avg": 6.0,
        "knee_max": 168.0, "knee_min": 90.0, "elbow_angle_avg": 90.0, "foot_strike": "heel"}})
    assert "foot_strike" not in [d.metric_name for d in plan.top_priorities]
    # Still known to the plan: only the ordering changed.
    assert "foot_strike" in [d.metric_name for d in plan.diagnostics]


def test_a_real_fault_still_leads_beside_a_heel_strike():
    plan = build_running_action_plan({"score": 60, "summary": {
        "cadence_spm": 150.0, "vertical_oscillation_m": 0.08, "trunk_lean_avg": 6.0,
        "knee_max": 168.0, "knee_min": 90.0, "elbow_angle_avg": 90.0, "foot_strike": "heel"}})
    assert plan.top_priorities[0].metric_name == "cadence"


# --- one verdict per tile

def _tile(html: str, label: str) -> str:
    return html[html.index(f"<span class='tile-l'>{label}</span>"):].split("</div></div>")[0]


def test_past_the_band_on_the_harmless_side_is_not_called_in_range():
    html = render_report_html(_run(strong=[_good("vertical_osc", 3.8, 6, 10)]), {})
    tile = _tile(html, "Vertical oscillation")
    assert "Better than range" in tile and "In range" not in tile
    assert "gr-chip good" in tile and "3.8 cm" in tile


def test_a_small_miss_on_the_side_that_matters_is_at_the_edge_and_not_leave_alone():
    res = _run(strong=[_good("elbow_angle", 83.0, 85, 100), _good("trunk_lean", 6.0, -2, 10)])
    by = {r["field"]: r for r in plan_rows(res)}
    assert by["elbow_angle"]["status"] == "warn"
    html = render_report_html(res, {})
    assert "gr-chip warn" in _tile(html, "Arm swing · elbow")
    assert re.search(r"leave these alone</h3><p>Trunk lean\.</p>", html)


def test_the_chip_and_the_number_never_disagree():
    res = _run(strong=[_good("cadence", 195.0, 170, 190), _good("knee_contact", 168.0, 160, 175)],
               issues=[{"metric": "knee_swing", "value": 134.0, "optimal_range": "80-100", "severity": "critical"}])
    word_colour = {"good": ("In range", "Better than range"), "warn": ("At the edge",), "bad": ("Outside",)}
    tiles = re.findall(r"<span class='chip (\w+)'>([^<]+)</span></div><div class='grbar'><div class='gr-chips'>"
                       r"<span class='gr-chip (\w+)'", render_report_html(res, {}))
    assert len(tiles) == 3
    for chip, word, number in tiles:
        assert chip == number and word in word_colour[chip]


def test_every_run_number_has_its_unit():
    html = render_report_html(_run(strong=[_good("cadence", 180.0, 170, 190), _good("knee_contact", 168.0, 160, 175)]), {})
    assert "180 spm" in html and "170–190 spm" in html and "168°" in html


# --- a bike's "leave these alone" comes from the tiles

def _bike(**over) -> dict:
    res = {
        "sport_type": "bike", "technique_score": 46, "cycling_position": "road_hoods",
        "sport_specific_metrics": {"knee_at_bdc": 150.0, "knee_at_tdc": 62.0, "hip_at_tdc": 65.0,
                                   "shoulder_angle_avg": 88.0, "elbow_angle_avg": 94.0, "forearm_tilt_avg": 17.0},
        "reference_bands": {"knee_at_bdc": {"lo": 135, "hi": 145}, "knee_at_tdc": {"lo": 65, "hi": 75},
                            "hip_at_tdc": {"lo": 55, "hi": 70}, "shoulder_angle_avg": {"lo": 80, "hi": 100},
                            "elbow_angle_avg": {"lo": 145, "hi": 165}, "forearm_tilt_avg": {"lo": 5, "hi": 20}},
        # What the fit plan calls good is wider than the tiles: it lists the knee at
        # the top and leaves out the forearm.
        "fit_plan": {"good_metrics": [{"label": "Hip at top of stroke"}, {"label": "Knee flexion at TDC"}],
                     "diagnostics": [{"component": "saddle_height", "status": "needs_adjustment",
                                      "metric_name": "knee_at_bdc", "action": "lower_saddle", "amount": "5mm",
                                      "priority": 1, "reason": "Knee angle at BDC is 150 deg (optimal 135-145 deg)."}]},
    }
    res.update(over)
    return res


def _alone(html: str) -> str:
    return re.search(r"leave these alone</h3><p>([^<]+)</p>", html).group(1)


def test_at_the_edge_is_not_leave_alone_and_every_good_tile_is():
    alone = _alone(render_report_html(_bike(), {}))
    assert "Knee at the top" not in alone and "TDC" not in alone
    assert "Forearm tilt" in alone and "Hip at the top" in alone and "Shoulder" in alone


def test_the_cover_counts_what_the_tiles_show():
    res = _bike()
    res["sport_specific_metrics"]["helmet_check"] = True  # helmet: a tile that says not measured
    html = render_report_html(res, {})
    assert "<b>6 numbers</b>" in html
    assert "3 measurements where they should be" in html


def test_the_chainring_side_knee_is_not_leave_alone():
    res = _bike()
    res["sport_specific_metrics"].update(knee_at_bdc=140.0, capture_report={"checks": [
        {"id": "drive_side", "status": "warn", "measured": "drive side"}]})
    res["fit_plan"]["diagnostics"] = []
    html = render_report_html(res, {})
    assert "Knee at the bottom" not in _alone(html)


def test_the_kona_field_is_only_claimed_where_it_is_drawn():
    assert "Kona" not in render_report_html(_bike(), {})
    res = _bike(pro_field={"metrics": {"elbow_angle_avg": {"median": 84, "q1": 76, "q3": 88}}})
    assert "Kona" in render_report_html(res, {})


# --- a number outside its range is said out loud

def test_a_red_number_with_no_plan_gets_a_finding():
    html = render_report_html(_bike(), {})
    read = html[html.index("What the measurements flagged"):html.index("The numbers") if "The numbers" in html else None]
    assert "<h3>Elbow</h3>" in read and "94°" in read and "145–165°" in read
    assert "isn't in the plan" in read


def test_a_red_number_on_the_plan_does_not_get_a_second_finding():
    html = render_report_html(_bike(), {})
    assert html.count("isn't in the plan") == 1


def test_a_coachs_report_sends_the_reader_to_that_coach():
    html = render_report_html(_bike(), {"coach": {"name": "Coach Mike"}})
    assert "Worth going through with your coach" in html


# --- one number for one thing

def test_a_finding_prints_the_tiles_number():
    res = _run(issues=[{"metric": "knee_swing", "value": 133.7, "optimal_range": "80-100", "severity": "critical"}])
    res["detected_issues"] = [{"type": "insufficient_knee_drive", "severity": "info", "value": "Min knee = 123 deg",
                               "recommendation": "The knee should fold to 100 deg or less."}]
    html = render_report_html(res, {})
    assert "123" not in html and "Min knee" not in html and "Insufficient knee drive" not in html
    assert "<h3>Knee in swing</h3><span class='chip bad'>134°</span>" in html
    assert "100° or less" in html and " deg" not in html.split("<style>")[1].split("</style>")[1]


def test_the_frame_is_not_said_to_be_where_the_numbers_came_from():
    html = render_report_html(_bike(), {"keyframe": "data:image/png;base64,AAAA"})
    assert "frame your angles were read on" not in html
    assert "across the whole clip" in html


def test_prepared_is_said_once():
    html = render_report_html(_bike(), {"prepared": "Prepared 6 Nov 2026"})
    assert "<dt>Prepared</dt><dd>6 Nov 2026</dd>" in html


def test_no_crank_jargon_in_the_plan():
    html = render_report_html(_bike(), {})
    assert "the bottom of the stroke is 150°" in html and "BDC" not in html


# --- the saddle: the total and the step

SADDLE = {"component": "saddle_height", "status": "needs_adjustment", "metric_name": "knee_at_bdc",
          "action": "lower_saddle", "amount": "≈23 mm (±7)", "priority": 1,
          "reason": "Overextension – lower the saddle about ≈23 mm (±7) to bring the knee to mid-band."}
OLD_NOTE = "Adjust in order: saddle height. Each adjustment: 5mm only, ride 30 minutes before reassessing."


def _saddle(note: str = OLD_NOTE, diagnostics=(SADDLE,)) -> str:
    res = _bike()
    res["fit_plan"] = {"good_metrics": [], "diagnostics": list(diagnostics), "fitting_sequence_note": note}
    return render_report_html(res, {})


def test_a_big_change_is_made_in_steps_and_a_saved_note_no_longer_contradicts_it():
    html = _saddle()
    assert "≈23 mm (±7), in 5 mm steps" in html
    assert "5mm only" not in html and "about ≈" not in html
    assert FIT_STEP_NOTE in html


def test_an_order_of_one_is_not_an_order_and_an_order_of_two_stays():
    assert "Adjust in order" not in _saddle()
    second = {**SADDLE, "component": "bar_position", "metric_name": "trunk_angle", "action": "raise_bars",
              "amount": None, "priority": 2}
    assert "Adjust in order: saddle height -&gt; bar position." in _saddle(
        "Adjust in order: saddle height -> bar position. Each adjustment: 5mm only, ride 30 minutes before reassessing.",
        (SADDLE, second))


def test_a_small_change_gets_no_steps():
    html = _saddle(diagnostics=({**SADDLE, "amount": "5mm"},))
    assert "in 5 mm steps" not in html


def test_a_new_fit_plan_has_the_step_in_its_note():
    plan = build_action_plan("road_hoods", {}, {"knee_at_bdc": 155.0}, 60, "C", [])
    assert "5mm only" not in plan.fitting_sequence_note
    assert FIT_STEP_NOTE in plan.fitting_sequence_note


# --- film again follows the filming note

def test_film_again_follows_the_note():
    res = _bike()
    res["fit_plan"]["fitting_sequence_note"] = ""
    res["sport_specific_metrics"]["capture_report"] = {"checks": [
        {"id": "framing", "status": "warn", "label": "How much of the frame you fill",
         "measured": "320 px tall (25% of the frame height)"}]}
    assert "same side, same speed, but closer" in render_report_html(res, {})
    res["sport_specific_metrics"]["capture_report"] = {"checks": [
        {"id": "drive_side", "status": "warn", "measured": "drive side"}]}
    html = render_report_html(res, {})
    assert "from the other side" in html and "same side" not in html


# --- the cover's table can be done by hand

def _parts(**scores) -> dict:
    return {"sport_type": "bike", "score_breakdown": scores}


def test_the_pull_toward_the_worst_part_is_explained():
    res = _parts(knee_bdc=85.4, trunk_angle=50.3, knee_tdc=95.9, elbow_angle=0.0, shoulder_angle=100.0,
                 pelvic_ratio=100.0, saddle_fit=75.0)
    notes = " ".join(_score_notes(res, 46))
    assert "These average 72" in notes and "Elbow at 0" in notes and "down to 46" in notes
    assert "Helmet above back" in notes and "Forearm" in notes


def test_a_score_that_the_sum_does_not_explain_gets_no_explanation():
    res = _parts(knee_bdc=85.4, trunk_angle=50.3, knee_tdc=95.9, elbow_angle=0.0, shoulder_angle=100.0,
                 pelvic_ratio=100.0, saddle_fit=75.0)
    assert not any("These average" in n for n in _score_notes(res, 20))


def test_the_derived_saddle_part_cannot_be_named_as_the_worst():
    res = _parts(knee_bdc=56.2, trunk_angle=100.0, knee_tdc=100.0, elbow_angle=100.0, shoulder_angle=100.0,
                 pelvic_ratio=100.0, forearm_tilt=100.0, saddle_fit=40.0)
    notes = " ".join(_score_notes(res, 75))
    assert "Knee at bottom at 56" in notes and "Saddle height at" not in notes


def test_the_weights_on_the_cover_add_to_a_hundred():
    res = _parts(knee_bdc=85.4, trunk_angle=50.3, knee_tdc=95.9, elbow_angle=0.0, shoulder_angle=100.0,
                 pelvic_ratio=100.0, saddle_fit=75.0)
    res["technique_score"] = 46
    shares = [int(s) for s in re.findall(r"<span class='bd-w'>(\d+)%</span>", render_report_html(res, {}))]
    assert 98 <= sum(shares) <= 102


def test_a_run_score_is_not_given_the_bikes_explanation():
    res = {"sport_type": "run", "score_breakdown": {"trunk_lean": 100.0, "knee_swing": 20.0}}
    assert not any("These average" in n for n in _score_notes(res, 40))
