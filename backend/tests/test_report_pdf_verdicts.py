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

from app.services.report_pdf import _score_notes, changes_since, plan_rows, render_report_html
from app.services.video_analysis.biomechanics.action_plan_builder import (
    FIT_STEP_NOTE,
    build_action_plan,
)
from app.services.video_analysis.biomechanics.plausibility import looks_aero_on_road
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
    # The elbow fault is an arm carried too OPEN (one-sided since 2026-11-07),
    # so the small miss is above the band.
    res = _run(strong=[_good("elbow_angle", 102.0, 85, 100), _good("trunk_lean", 6.0, -2, 10)])
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


def test_the_chainring_side_knee_is_a_number_like_any_other():
    res = _bike()
    res["sport_specific_metrics"].update(knee_at_bdc=140.0, capture_report={"checks": [
        {"id": "drive_side", "status": "warn", "measured": "a saved clip's old warning"}]})
    res["fit_plan"]["diagnostics"] = []
    html = render_report_html(res, {})
    assert "Knee at the bottom" in _alone(html) and "from the other side" not in html


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
        {"id": "drive_side", "status": "warn", "measured": "a saved clip's old warning"}]}
    html = render_report_html(res, {})
    assert "from the other side" not in html and "same side" in html


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


# --- how far to trust it: a doubtful clip doesn't get a confident plan

def _doubtful(**extra) -> dict:
    res = _run(issues=[{"metric": "knee_swing", "label": "Knee in swing", "value": 134.0}],
               prios=[{"metric": "knee_swing", "label": "Knee in swing", "drill": "High Knee March",
                       "why": "Knee stays open", "cue": "Heel to glute", "weeks": 3}],
               capture_report={"checks": [{"id": "duration", "status": "bad", "label": "How much",
                                           "measured": "9.5 s long, but you were only trackable in 3.7 s of it"}]})
    res.update(extra)
    return res


def test_a_doubtful_clip_says_film_again_before_it_says_fix():
    html = render_report_html(_doubtful(), {})
    assert "Film again first" in html and "Before changing anything" in html
    assert "Fix first" not in html and "Fix these, in this order" not in html
    assert "What this clip suggests" in html and "Low · little usable footage" in html
    assert "keep the whole body in the frame" in html
    # The drill stays, as something to try. The prescription of weeks goes.
    assert "Heel Flicks" in html and "3 weeks" not in html


def test_a_clean_clip_is_unchanged():
    res = _doubtful()
    res["sport_specific_metrics"]["capture_report"]["checks"][0]["status"] = "good"
    html = render_report_html(res, {})
    assert "Film again first" not in html and "Reliability" not in html
    assert "Fix these, in this order" in html and "3 weeks" in html


def test_one_doubtful_number_does_not_make_the_clip_doubtful():
    res = _doubtful()
    res["sport_specific_metrics"]["capture_report"]["checks"] = [
        {"id": "camera_motion", "status": "bad", "measured": "camera moved 80% as much as your hips did"}]
    html = render_report_html(res, {})
    assert "Shaky camera" in html and "Film again first" not in html


def test_a_coachs_own_framing_bar_keeps_the_clip_trusted():
    res = _doubtful()
    res["sport_specific_metrics"]["capture_report"]["checks"] = [
        {"id": "framing", "status": "bad", "label": "How much of the frame you fill", "measured": "237 px tall"}]
    assert "Film again first" in render_report_html(res, {})
    html = render_report_html(res, {"capture_bars": {"small_px": 200}})
    assert "Film again first" not in html and "Small in the frame" not in html


def test_road_clip_that_looks_aero_is_called_out_and_keeps_its_score():
    res = {"sport_type": "bike", "technique_score": 46, "cycling_position": "road_hoods",
           "sport_specific_metrics": {"trunk_angle_avg": 23.1, "elbow_angle_avg": 93.6,
                                      "capture_report": {"checks": []}}}
    html = render_report_html(res, {})
    assert "torso at 23° and elbows at 94° look like aero bars" in html
    assert "Low · looks like aero bars" in html
    assert "run the clip with the position actually ridden" in html and "Film this again first" in html


def test_no_advice_is_given_against_ranges_the_clip_was_not_ridden_in():
    res = {"sport_type": "bike", "technique_score": 46, "cycling_position": "road_hoods",
           "fit_plan": {"priorities": [{"metric": "trunk_angle", "label": "Bar position",
                                        "what": "Raise bars", "why": "Too aggressive"}],
                        "good_metrics": [{"metric": "knee_at_tdc", "label": "Knee at the top"}]},
           "detected_issues": [{"type": "trunk_too_aggressive", "severity": "info", "value": "23°",
                                "recommendation": "Raise the bars"}],
           "sport_specific_metrics": {"trunk_angle_avg": 23.1, "elbow_angle_avg": 93.6,
                                      "capture_report": {"checks": []}}}
    html = render_report_html(res, {})
    assert "Raise" not in html and "Too aggressive" not in html and "Trunk too aggressive" not in html
    assert "Fix first" not in html and "Already working" not in html and "leave these alone" not in html
    assert ">46<" in html


def test_a_real_road_position_is_left_alone():
    for pos, trunk, elbow in (("road_hoods", 42, 155), ("road_drops", 28, 110), ("road_hoods", 28, 150),
                              ("road_hoods", 40, 100), ("triathlon", 23, 94), (None, 23, 94)):
        assert looks_aero_on_road(pos, {"trunk_angle_avg": trunk, "elbow_angle_avg": elbow}) is None, pos


# --- printing: the browser's own header and footer

def test_the_page_leaves_no_margin_for_the_browsers_date_and_address():
    html = render_report_html(_run(), {})
    # The last @page rule wins, so read the last one, and the first-page one too.
    assert re.findall(r"@page\{[^}]*\}", html)[-1] == "@page{size:A4;margin:0}"
    assert re.findall(r"@page:first\{[^}]*\}", html)[-1] == "@page:first{margin:0}"
    assert '<table class="flow"><thead>' in html


# --- structure: one document, not three lists of the same fixes

def _bike_plan(**extra) -> dict:
    return {
        "sport_type": "bike", "technique_score": 75, "cycling_position": "road_hoods", "camera_side": "left",
        "reference_bands": {"knee_at_bdc": {"lo": 135, "hi": 145}, "trunk_angle_avg": {"lo": 40, "hi": 55}},
        "detected_issues": [{"type": "saddle_too_high", "severity": "warning", "value": "150 deg",
                             "recommendation": "Lower the saddle"}],
        "fit_plan": {"diagnostics": [{
            "component": "saddle_height", "status": "needs_adjustment", "metric_name": "knee_at_bdc",
            "action": "lower_saddle", "amount": "≈23 mm (±7)", "priority": 1,
            "reason": "Knee angle at BDC is 165 deg (optimal 135-145 deg). Overextension -- lower the saddle "
                      "about ≈23 mm (±7) to bring the knee to mid-band, ride 30 min before reassessing.",
        }]},
        "sport_specific_metrics": {"knee_at_bdc": 165.0, "trunk_angle_avg": 45.0, "capture_report": {"checks": []}},
        **extra,
    }


def test_a_fit_change_has_a_why_and_a_do_this():
    html = render_report_html(_bike_plan(), {})
    assert "<dt>Why</dt><dd>Knee angle at the bottom of the stroke is 165° (optimal 135-145°). Overextension.</dd>" in html
    assert "<dt>Do this</dt><dd>Lower the saddle about 23 mm (±7)" in html
    assert "Work in 5 mm steps." in html


def test_a_finding_the_plan_covers_is_not_said_twice():
    html = render_report_html(_bike_plan(), {})
    assert "Saddle too high" not in html and "Lower saddle" in html


def test_the_plan_comes_first_when_there_is_nothing_else_to_read():
    html = render_report_html(_bike_plan(), {})
    assert "The coach&#x27;s read" not in html and "The coach's read" not in html
    assert html.index("The plan") < html.index("The numbers")


def test_an_old_run_gets_the_drill_that_matches_its_cue():
    res = _doubtful()
    res["sport_specific_metrics"]["capture_report"]["checks"] = []
    res["training_plan"]["top_3_priorities"][0].update(drill="High Knee March", instruction="March with maximum knee lift")
    html = render_report_html(res, {})
    assert "High Knee March" not in html and "March with" not in html and "Heel Flicks" in html


def test_a_tall_frame_sits_beside_the_cards_not_in_a_wide_box():
    html = render_report_html(_bike_plan(), {"keyframe": "data:image/jpeg;base64,AA"})
    assert '<div class="lead"><img' in html and "cover-img" not in html


# --- the document: whose it is, on every page and in the file name

def test_the_saved_file_is_named_after_the_athlete():
    html = render_report_html(_bike_plan(), {"athlete": "Maya Chen", "filmed": "6 Nov 2026"})
    assert "<title>Maya Chen, Cycling, 6 Nov 2026</title>" in html


def test_an_email_never_lands_in_a_file_name():
    html = render_report_html(_bike_plan(), {"athlete": "maya@example.com", "filmed": "6 Nov 2026"})
    assert "<title>Cycling, 6 Nov 2026</title>" in html


def test_every_page_says_whose_report_it_is():
    html = render_report_html(_bike_plan(), {"athlete": "Maya Chen", "filmed": "6 Nov 2026"})
    tfoot = html[html.index("<tfoot>"):html.index("</tfoot>")]
    assert "Maya Chen · Cycling · Filmed 6 Nov 2026" in tfoot and "getflapp.com" in tfoot


def test_the_coach_font_is_loaded():
    assert "IBM+Plex+Sans" in render_report_html(_bike_plan(), {})


# --- what changed since the last clip

def _since(then_knee, then_score=70, **kw) -> str:
    before = _bike_plan(technique_score=then_score)
    before["sport_specific_metrics"] = {**before["sport_specific_metrics"], "knee_at_bdc": then_knee}
    return render_report_html(_bike_plan(**kw), {"previous": {"filmed": "12 Oct 2026", "result": before}})


def test_what_moved_is_shown_with_both_numbers():
    html = _since(172.0)
    assert "Since 12 Oct 2026" in html and "70 → 75" in html
    assert "172° → 165°" in html and "Closer" in html


def test_a_move_inside_the_methods_error_is_not_a_change():
    html = _since(164.0)
    assert "No real change" in html and "164°" not in html
    # The score is left out: it can move when the rules do.
    assert "70 → 75" not in html


def test_getting_further_out_is_said():
    assert "Further out" in _since(158.0)


def test_coming_back_into_range_is_said():
    moved = changes_since(
        [{"field": "x", "label": "Knee", "value": 140.0, "unit": "°", "band": (135, 145), "status": "good"}],
        [{"field": "x", "label": "Knee", "value": 160.0, "unit": "°", "band": (135, 145), "status": "bad"}])
    assert moved == [{"label": "Knee", "then": "160°", "now": "140°", "word": "Back in range", "tone": "good"}]


def test_a_run_number_with_no_measured_floor_counts_only_when_its_verdict_moves():
    row = {"field": "cadence", "label": "Cadence", "unit": " spm", "band": (170, 190)}
    same = changes_since([{**row, "value": 176.0, "status": "good"}], [{**row, "value": 172.0, "status": "good"}])
    assert same == []
    moved = changes_since([{**row, "value": 172.0, "status": "good"}], [{**row, "value": 160.0, "status": "bad"}])
    assert moved[0]["word"] == "Back in range"


def test_a_doubtful_clip_makes_no_claim_about_what_moved():
    res = _bike_plan()
    res["sport_specific_metrics"]["capture_report"]["checks"] = [
        {"id": "framing", "status": "bad", "label": "How much of the frame you fill", "measured": "237 px tall"}]
    html = render_report_html(res, {"previous": {"filmed": "12 Oct 2026", "result": _bike_plan()}})
    assert "Since 12 Oct" not in html
