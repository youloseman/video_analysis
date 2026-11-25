"""A hip closed at the top of the stroke: cranks on the road, saddle forward on TT.

On the hoods, sliding the saddle forward moves the rider over the bottom bracket
and onto the hands. Shorter cranks open the top of the stroke without moving a
contact point and, across 145-195 mm, cost no power (Martin & Spirduso 2001).
"""

from app.services.video_analysis.biomechanics.action_plan_builder import build_action_plan
from app.services.video_analysis.biomechanics.fit_tradeoffs import rank_adjustments


def _plan(position, crank=None, **metrics):
    sm = {"knee_at_bdc": 145.0, **metrics}
    return build_action_plan(
        position=position, angle_statistics={}, sport_specific_metrics=sm,
        technique_score=70, letter_grade="B", detected_issues=[], crank_length_mm=crank,
    )


def _hip_card(plan):
    return next(d for d in plan.diagnostics if d.metric_name == "hip_at_tdc")


def test_road_closed_hip_says_shorter_cranks_not_saddle_forward():
    card = _hip_card(_plan("road_hoods", hip_at_tdc=40.0))
    assert card.component == "crank_length"
    assert card.action == "consider_shorter_cranks"
    assert "hands" in card.reason
    assert "—" not in card.reason


def test_road_names_a_size_from_the_riders_crank():
    card = _hip_card(_plan("road_drops", crank=172.5, hip_at_tdc=40.0))
    assert card.amount == "172.5 to 167.5 mm"
    assert "172.5 to 167.5 mm" in card.reason


def test_road_crank_read_from_the_saddle_estimate_when_not_passed():
    est = {"crank_length_mm": 170.0, "crank_source": "given"}
    card = _hip_card(_plan("road_hoods", hip_at_tdc=40.0, saddle_estimate=est))
    assert card.amount == "170 to 165 mm"
    assumed = {"crank_length_mm": 172.5, "crank_source": "assumed"}
    card = _hip_card(_plan("road_hoods", hip_at_tdc=40.0, saddle_estimate=assumed))
    assert card.amount == "about 5mm shorter"


def test_road_already_short_cranks_raise_the_bars():
    card = _hip_card(_plan("road_hoods", crank=155.0, hip_at_tdc=40.0))
    assert card.component == "bar_position"
    assert card.action == "raise_bars"


def test_tt_keeps_saddle_forward_and_mentions_cranks_second():
    card = _hip_card(_plan("triathlon", hip_at_tdc=30.0))
    assert card.action == "move_saddle_forward"
    assert "shorter cranks" in card.reason


def test_fitting_order_names_the_crank():
    plan = _plan("road_hoods", hip_at_tdc=40.0)
    assert "crank length" in plan.fitting_sequence_note


def test_road_knee_tight_at_top_below_band_is_no_longer_dropped():
    # 58 against 65-75 used to fall through: neither a card nor a good metric.
    plan = _plan("road_hoods", knee_at_tdc=58.0)
    cards = [d for d in plan.diagnostics if d.metric_name == "knee_at_tdc"]
    assert cards and cards[0].action == "consider_shorter_cranks"


def test_closed_hip_and_tight_knee_give_one_crank_card():
    plan = _plan("road_hoods", hip_at_tdc=40.0, knee_at_tdc=55.0)
    assert sum(d.component == "crank_length" for d in plan.diagnostics) == 1


def test_road_ranking_drops_pelvis_rotation():
    angles = {"hip": {"value": 40, "optimal_min": 55, "optimal_max": 70,
                      "status": "needs_work", "label": "Hip angle"}}
    road = [r["label"] for r in rank_adjustments(angles, "road_hoods")]
    tt = [r["label"] for r in rank_adjustments(angles, "triathlon")]
    assert not any("pelvis" in label for label in road)
    assert any("crank" in label for label in road)
    assert any("pelvis" in label for label in tt)
