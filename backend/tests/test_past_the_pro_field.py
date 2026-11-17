"""A rider lower than every Kona pro is not told to move toward the pros.

A time-trial rider (2026-11-17) with the torso flatter and the helmet lower
than the whole field read: "it is where your next gain most likely is: move
toward it in small steps" -- that is, sit up. The same report said his helmet
at -3 % was "where 28 of the 31 Kona pros rode" (the lowest pro: +7 %) and
compared it "against 6.7-33 % for all 32 of them" (the helmet field is 31).
"""
from __future__ import annotations

from app.services.video_analysis import concern_hints
from app.services.video_analysis import llm_recommendations as llm
from app.services.video_analysis.biomechanics import aero_estimator as AE
from app.services.video_analysis.biomechanics.kona_field import (
    KONA_PRO_N,
    field_n,
    past_the_field,
    pro_gaps,
)

LOW = {"trunk_angle_avg": 8.9, "helmet_above_back": -2.6, "helmet_check": False,
       "forearm_tilt_avg": 22.0, "shoulder_angle_avg": 86.3, "elbow_angle_avg": 78.6}


def test_the_helmet_field_is_thirty_one():
    assert KONA_PRO_N == 32 and field_n("trunk_angle_avg") == 32
    assert field_n("helmet_above_back") == 31
    gaps = {g["field"]: g for g in pro_gaps("tt_aero", LOW)}
    assert set(gaps) == {"trunk_angle_avg", "helmet_above_back"}
    assert gaps["helmet_above_back"]["n"] == 31 and gaps["trunk_angle_avg"]["n"] == 32


def test_which_gaps_are_past_the_field():
    assert past_the_field({"field": "trunk_angle_avg", "side": "low"})
    assert past_the_field({"field": "helmet_above_back", "side": "low"})
    # Short of the field: the gain is toward it.
    assert not past_the_field({"field": "trunk_angle_avg", "side": "high"})
    assert not past_the_field({"field": "helmet_above_back", "side": "high"})
    assert not past_the_field({"field": "forearm_tilt_avg", "side": "low"})
    # A gap built by hand without a side (older callers) is read as before.
    assert not past_the_field({"field": "forearm_tilt_avg"})


def test_the_next_step_does_not_send_a_low_rider_back_up():
    gaps = pro_gaps("tt_aero", LOW)
    step = next(s for s in concern_hints.video_next_steps("bike", "tt_aero", LOW, gaps)
                if s["key"] == "pros")
    text = step["text"]
    assert "move toward it" not in text and "next gain" not in text
    assert "not a gain to chase further" in text and "see the road" in text
    assert "helmet above the back -2.6% against 6.7-33% for all 31 of them" in text
    assert "trunk angle 8.9° against 12.2-27.1° for all 32 of them" in text


def test_a_rider_short_of_the_field_reads_as_before():
    summary = {**LOW, "trunk_angle_avg": 21.0, "helmet_above_back": 15.0, "forearm_tilt_avg": 17.1}
    gaps = pro_gaps("tt_aero", summary)
    assert [g["field"] for g in gaps] == ["forearm_tilt_avg"]
    text = concern_hints.video_next_steps("bike", "tt_aero", summary, gaps)[0]["text"]
    assert "where your next gain most likely is: move toward it in small steps" in text
    assert "chase further" not in text


def test_both_kinds_are_told_apart_in_one_step():
    summary = {**LOW, "forearm_tilt_avg": 17.1}
    text = next(s for s in concern_hints.video_next_steps(
        "bike", "tt_aero", summary, pro_gaps("tt_aero", summary)) if s["key"] == "pros")["text"]
    toward, lower = text.split("You are also lower than every pro: ")
    assert "forearm tilt" in toward and "move toward it" in toward and "trunk angle" not in toward
    assert "trunk angle" in lower and "helmet above the back" in lower and "chase further" in lower


def test_the_checklist_line_matches():
    section = concern_hints.worth_a_look([], pro_gaps("tt_aero", {**LOW, "forearm_tilt_avg": 17.1}))
    trunk, forearm, helmet = section.splitlines()[1:]
    assert "nothing to chase further" in trunk and "next gain" not in trunk
    assert "the field is where the next gain usually is" in forearm
    assert "below every one of the 31 Kona pros" in helmet and "see the road" in helmet
    assert "Re-film to confirm" not in helmet, "a low helmet has its own hint"


def test_the_coach_is_not_told_to_announce_a_gain(monkeypatch):
    prompts: list[str] = []

    def fake(system, prompt, max_tokens, tag):
        prompts.append(prompt)
        return {"text": "**Overall** fine.\n\n**Nothing to fix** Hold it.\n\n**Next session** Relax.",
                "model": "fake"}

    monkeypatch.setattr(llm, "_complete", fake)
    clean = {"knee_at_bdc": 141.0, "knee_at_tdc": 66.0, "hip_at_tdc": 50.0, "pelvic_ratio": 3.0}
    llm.generate_recommendations("bike", 100, "A", [], {}, {**LOW, **clean}, cycling_position="tt_aero")
    low = prompts[-1]
    assert "OUTSIDE the whole Kona 2026 pro field: Trunk angle, Helmet above the back." in low
    assert "names what to check before going any lower." in low
    assert "names where the next gain is" not in low
    # Short of the field, the clause is the old one.
    llm.generate_recommendations(
        "bike", 100, "A", [], {},
        {**LOW, **clean, "trunk_angle_avg": 21.0, "helmet_above_back": 15.0, "forearm_tilt_avg": 17.1},
        cycling_position="tt_aero")
    assert "names where the next gain is." in prompts[-1]


def test_the_head_cue_below_the_field_does_not_claim_the_pros_rode_there():
    cue = AE.helmet_lever(-2.6, False, "tt_aero")["cue"]
    assert "28 of the 31" not in cue
    assert cue.startswith("Helmet 3% of a torso below the line of your back: lower than every Kona pro")
    assert "(+7% to +33%)" in cue
    assert AE.helmet_lever(-2.6, False, "tt_aero")["status"] == "ok"
    # Between the lowest pro and the free band the old sentence is true.
    assert "where 28 of the 31 Kona pros rode" in AE.helmet_lever(12.0, False, "tt_aero")["cue"]
    assert "level with your back" in AE.helmet_lever(0.2, False, "tt_aero")["cue"]
    assert "+4% of a torso above your back: lower than every" in AE.helmet_lever(4.0, False, "tt_aero")["cue"]
