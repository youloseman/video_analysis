"""A high saddle the knee angle does not show: the toe reaching for the pedal.

Pointing the toe down at the bottom of the stroke lengthens the leg, so the
knee can read in range on a saddle that is too high. Two signs together make
a card: the knee in the top half of its band and the ankle past the toe-down
flag (125 deg, mean + 1.8 SD of the reference curve). The side-view hip drop
was measured as the third sign and rejected: it moves with the leg, not the
pelvis (3.7-7.2 % of the thigh on every clip, 2026-11-26).
"""

from pathlib import Path

from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan
from app.services.video_analysis.biomechanics.cycling_positions import get_cycling_reference
from app.services.video_analysis.biomechanics.hip_over_bb import report_says_knee_high

POS = "road_hoods"
LO, HI = get_cycling_reference(POS)["knee_at_bdc"]
TOP_HALF = (LO + HI) / 2 + 1
BOTTOM_HALF = (LO + HI) / 2 - 1


def _plan(**metrics):
    return build_action_plan(
        position=POS, angle_statistics={}, sport_specific_metrics=metrics,
        technique_score=70, letter_grade="B", detected_issues=[],
    )


def _saddle(plan):
    return [d for d in plan.diagnostics if d.component == "saddle_height"]


def test_knee_near_the_top_and_toe_down_lowers_the_saddle():
    plan = _plan(knee_at_bdc=TOP_HALF, ankle_at_bdc=131.0)
    (card,) = _saddle(plan)
    assert card.action == "lower_saddle" and card.amount == "5mm"
    # The toe is the number the card rests on: the knee is in range.
    assert card.metric_name == "ankle_at_bdc"
    assert card.current_value == 131.0 and card.target_range == (93.0, 125.0)
    assert "toe" in card.reason and "film again" in card.reason
    assert "—" not in card.reason
    # In range, but not listed as good next to a card that says lower it.
    assert not any(g["metric"] == "knee_at_bdc" for g in plan.good_metrics)


def test_one_sign_alone_is_not_a_card():
    # Toe down, knee in the bottom half: the toe is not hiding a high saddle.
    assert not _saddle(_plan(knee_at_bdc=BOTTOM_HALF, ankle_at_bdc=131.0))
    # Knee near the top, a flat foot: in range, nothing to change.
    plan = _plan(knee_at_bdc=TOP_HALF, ankle_at_bdc=118.0)
    assert not _saddle(plan)
    assert any(g["metric"] == "knee_at_bdc" for g in plan.good_metrics)
    # No foot reading at all.
    assert not _saddle(_plan(knee_at_bdc=TOP_HALF))
    assert not _saddle(_plan(knee_at_bdc=TOP_HALF, ankle_at_bdc=None))


def test_knee_already_high_says_the_toe_as_a_second_sign():
    (card,) = _saddle(_plan(knee_at_bdc=HI + 8, ankle_at_bdc=132.0))
    assert card.metric_name == "knee_at_bdc" and card.action == "lower_saddle"
    assert "toe also points down" in card.reason
    (plain,) = _saddle(_plan(knee_at_bdc=HI + 8, ankle_at_bdc=115.0))
    assert "toe" not in plain.reason


def test_a_low_knee_still_raises_the_saddle_whatever_the_toe():
    (card,) = _saddle(_plan(knee_at_bdc=LO - 8, ankle_at_bdc=131.0))
    assert card.action == "raise_saddle"


def test_the_saddle_tilt_reads_the_toe_card_as_a_high_saddle():
    from app.services.video_analysis.biomechanics.action_plan_builder import action_plan_to_json

    plan = action_plan_to_json(_plan(knee_at_bdc=TOP_HALF, ankle_at_bdc=131.0))
    assert report_says_knee_high({"fit_plan": plan})


def test_the_app_names_the_foot():
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "ankle_at_bdc:'Foot at the bottom'" in html
    assert "from the toe pointing down at the bottom of the stroke" in html
