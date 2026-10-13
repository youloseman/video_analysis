"""No "Nothing to fix" over a red row, and every out-of-band reading is listed.

The first paid Coach Notes order (2026-10-11): triathlon clip, knee at TDC 85
(band 60-72), pelvic ratio 4.9 (band 2-4). The materiality block named both,
the model repeated both -- and headed the section "Nothing to fix", leaning on
"no rule-based issues were flagged". An athlete who came looking for room to
improve was told there was none.

So: the model gets one retry when it does that, then the section is replaced;
and a checklist of every reading outside or on the edge of its band, with what
to check, is added under the report whatever the model wrote.
"""
from __future__ import annotations

from app.services.video_analysis import concern_hints
from app.services.video_analysis import llm_recommendations as llm

# Order 2's numbers (summary fields).
ORDER_2 = {
    "knee_at_bdc": 146.0, "knee_at_tdc": 85.3, "hip_at_tdc": 55.0,
    "hip_angle_avg": 75.0, "trunk_angle_avg": 15.4, "elbow_angle_avg": 92.6,
    "shoulder_angle_avg": 91.6, "forearm_tilt_avg": 15.7, "pelvic_ratio": 4.9,
}
# Clean means inside every band AND inside the Kona field: a clip inside its
# bands but outside the field gets a "Worth a look" of its own now.
CLEAN = {**ORDER_2, "knee_at_bdc": 141.0, "knee_at_tdc": 66.0,
         "trunk_angle_avg": 25.0, "pelvic_ratio": 3.0, "forearm_tilt_avg": 25.0}

THE_REPORT_IT_WROTE = (
    "**Overall** — Your 15.4° trunk angle is lower than the band.\n\n"
    "**What's working** — Elbow and shoulder are locked in.\n\n"
    "**Nothing to fix** — While your TDC knee angle (85.3°) and pelvic ratio "
    "(4.9x) sit outside their optimal bands, no rule-based issues were flagged.\n\n"
    "**Next session** — Keep your upper body relaxed."
)
A_PROPER_REPORT = THE_REPORT_IT_WROTE.replace(
    "**Nothing to fix** — While", "**Fix first** — Your knee barely closes at the top; while")


def _scripted(monkeypatch, *texts):
    queue = list(texts)
    prompts: list[str] = []

    def fake(system, prompt, max_tokens, tag):
        prompts.append(prompt)
        t = queue.pop(0)
        return None if t is None else {"text": t, "model": "fake"}

    monkeypatch.setattr(llm, "_complete", fake)
    return prompts


def _gen(summary):
    return llm.generate_recommendations(
        "bike", 78, "B", [], {}, summary, cycling_position="triathlon")


def test_order_2_lists_both_red_rows_with_what_to_check():
    concerns = llm._video_concerns("bike", "triathlon", ORDER_2)
    material = {c["field"] for c in concerns if c["material"]}
    assert material == {"knee_at_tdc", "pelvic_ratio"}
    # The flat aero trunk and the borderline saddle are on the list too, as edges.
    edges = {c["field"] for c in concerns if not c["material"]}
    assert {"trunk_angle_avg", "knee_at_bdc"} <= edges
    section = concern_hints.worth_a_look(concerns)
    assert section.startswith("**Worth a look**")
    assert "Knee angle at top of stroke (TDC)**: 85.3° vs 60-72°, outside the band" in section
    assert "short cranks" in section
    # Clearly-out readings lead.
    assert section.index("Knee angle at top") < section.index("Trunk angle")


def test_nothing_to_fix_over_a_red_row_is_retried(monkeypatch):
    prompts = _scripted(monkeypatch, THE_REPORT_IT_WROTE, A_PROPER_REPORT)
    rec = _gen(ORDER_2)
    assert len(prompts) == 2
    assert "YOUR PREVIOUS DRAFT WAS REJECTED" in prompts[1]
    assert "Nothing to fix" not in rec["report"]
    assert "**Fix first**" in rec["report"]


def test_if_the_model_insists_the_section_becomes_the_list(monkeypatch):
    _scripted(monkeypatch, THE_REPORT_IT_WROTE, THE_REPORT_IT_WROTE)
    report = _gen(ORDER_2)["report"]
    assert "Nothing to fix" not in report
    assert "no rule-based issues were flagged" not in report
    fix = report.split("**Fix first**")[1].split("**Worth a look**")[0]
    assert "Knee angle at top of stroke" in fix and "Pelvic ratio" in fix


def test_the_checklist_sits_before_the_closing_cue(monkeypatch):
    _scripted(monkeypatch, A_PROPER_REPORT)
    report = _gen(ORDER_2)["report"]
    assert report.index("**Worth a look**") < report.index("**Next session**")
    assert report.rstrip().endswith("Keep your upper body relaxed.")


def test_a_clean_position_keeps_nothing_to_fix_and_gets_no_list(monkeypatch):
    clean_report = THE_REPORT_IT_WROTE.replace(
        "While your TDC knee angle (85.3°) and pelvic ratio (4.9x) sit outside "
        "their optimal bands, no rule-based issues were flagged.", "Hold it.")
    prompts = _scripted(monkeypatch, clean_report)
    rec = _gen(CLEAN)
    assert len(prompts) == 1
    assert rec["report"] == clean_report


def test_the_prompt_no_longer_says_an_empty_issue_list_means_no_problem():
    """The sentence the model leaned on is gone."""
    prompt = llm._build_prompt("bike", 78, "B", "triathlon", [], {}, ORDER_2)
    assert "no material problem found" not in prompt
    assert "MATERIAL PROBLEMS (" in prompt
    assert "is forbidden" in llm.SYSTEM_PROMPT


def test_photo_rows_get_the_same_hints(monkeypatch):
    res = {"score": {"overall_score": 70, "grade": "C"}, "angles_with_context": {
        "knee": {"value": 84.0, "label": "Knee", "phase": "tdc",
                 "optimal_min": 60, "optimal_max": 72, "status": "needs_work"},
        "elbow": {"value": 95.0, "label": "Elbow", "optimal_min": 75,
                  "optimal_max": 110, "status": "optimal"},
        "helmet_above_back": {"value": 27.0, "label": "Helmet above back",
                              "optimal_min": None, "optimal_max": 23.0,
                              "status": "acceptable"},
    }}
    concerns = llm._photo_concerns("bike", res)
    assert [c["field"] for c in concerns] == ["knee_at_tdc", "helmet_above_back"]
    section = concern_hints.worth_a_look(concerns)
    assert "up to 23%" in section and "short cranks" in section


# Artur's own clip (2026-10-13): 100/100, every band met, forearms at 17 --
# below every one of the 32 Kona pros -- and a report that said nothing.
MY_CLIP = {
    "knee_at_bdc": 143.2, "knee_at_tdc": 63.6, "hip_at_tdc": 49.0, "hip_angle_avg": 70.0,
    "trunk_angle_avg": 21.2, "elbow_angle_avg": 92.2, "shoulder_angle_avg": 87.0,
    "forearm_tilt_avg": 17.1, "pelvic_ratio": 3.2, "helmet_above_back": 9.6, "helmet_check": True,
    "camera_side": "right", "tracking_stability": {"framing": {"verdict": "small"}},
}


def test_inside_every_band_but_outside_the_pro_field_is_still_worth_a_look():
    from app.services.video_analysis.biomechanics.kona_field import pro_gaps
    assert llm._video_concerns("bike", "triathlon", MY_CLIP) == [], "every band met"
    gaps = pro_gaps("triathlon", MY_CLIP)
    assert [g["field"] for g in gaps] == ["forearm_tilt_avg"]
    assert gaps[0]["side"] == "low" and gaps[0]["lo"] == 18.8
    section = concern_hints.worth_a_look([], gaps)
    assert section.startswith("**Worth a look**")
    assert "Forearm tilt**: 17.1°, inside its band but below every one of the 32 Kona pros" in section
    # The flagged helmet (9.6, no edge) is not a gap: it was never read.
    assert "Helmet" not in section
    # A field already on the concern list is not listed twice.
    both = concern_hints.worth_a_look(
        [concern_hints.concern("forearm_tilt_avg", "Forearm tilt", 17.1, 5, 30, material=False)], gaps)
    assert both.count("Forearm tilt") == 1
    # Off the aero bars there is no field to compare with.
    assert pro_gaps("road_hoods", MY_CLIP) == []


def test_a_clean_clip_still_ends_with_where_to_go():
    steps = concern_hints.video_next_steps("bike", "triathlon", MY_CLIP, [{"field": "forearm_tilt_avg",
                                           "label": "Forearm tilt", "value": 17.1, "lo": 18.8, "hi": 32.0, "median": 25.9, "n": 32}])
    assert [s["key"] for s in steps] == ["side", "framing", "helmet", "pros", "refilm"]
    assert "forearm tilt 17.1° against 18.8-32°" in steps[3]["text"]
    # A clean, well-filmed clip on a road bike keeps only the re-film step.
    assert [s["key"] for s in concern_hints.video_next_steps("bike", "road_hoods", {"camera_side": "left"})] == ["refilm"]


def test_the_coach_is_told_about_the_field_and_the_next_step(monkeypatch):
    prompts = _scripted(monkeypatch, "**Overall** — fine.\n\n**Nothing to fix** — Hold it.\n\n**Next session** — Relax.")
    rec = llm.generate_recommendations("bike", 100, "A", [], {}, MY_CLIP, cycling_position="triathlon",
                                       next_steps=[{"key": "side", "title": "Film from the non-drive side", "text": "x"}])
    p = prompts[0]
    assert "MATERIAL PROBLEMS: NONE" in p
    assert "OUTSIDE the whole Kona 2026 pro field" in p and "Forearm tilt 17 (all 32 pros 19-32)" in p
    assert "Film from the non-drive side" in p and "Do not end the report with nothing to do." in p
    # And the checklist under the report carries the gap, before Next session.
    r = rec["report"]
    assert "**Worth a look**" in r and r.index("**Worth a look**") < r.index("**Next session**")
    assert "below every one of the 32 Kona pros" in r


def test_the_clip_report_has_the_block_too():
    from conftest import read_spa
    spa = read_spa()
    assert 'id="nextBlock"' in spa
    assert "renderNextSteps('nextBlock','nextList', r.next_steps||(r.quality&&r.quality.next_steps)||[])" in spa
