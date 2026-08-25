"""The ranking that decides which fit adjustment the coach recommends.

The failure this module exists to prevent: the coach praising a trunk angle and
then, two lines later, prescribing the one adjustment whose whole effect is to
undo it. The ordering is the product here, so it is worth pinning.
"""

from app.services.video_analysis.biomechanics.fit_tradeoffs import (
    build_tradeoff_block,
    rank_adjustments,
)


def angle(label, value, lo, hi, status):
    return {
        "label": label, "value": value,
        "optimal_min": lo, "optimal_max": hi, "status": status,
    }


# The real 2026-07-30 run: closed hip, unrotated pelvis, everything else clean.
CLOSED_HIP = {
    "knee": angle("Knee Angle", 56.4, 48, 80, "optimal"),
    "hip": angle("Hip Angle", 34.7, 45, 62, "needs_work"),
    "elbow": angle("Elbow Angle", 81.9, 75, 110, "optimal"),
    "shoulder": angle("Shoulder Angle", 89.1, 80, 105, "optimal"),
    "trunk": angle("Trunk Angle", 24.3, 20, 30, "optimal"),
    "pelvic_ratio": angle("Pelvic Rotation", 1.43, 2.0, 4.0, "needs_work"),
}


def test_pelvic_rotation_outranks_raising_the_bars():
    """The aero-preserving lever wins: it fixes both problems and costs nothing.

    Raising the stack also opens a hip, which is why the coach reached for it
    unprompted -- but it does so by sitting the rider up.
    """
    ranked = rank_adjustments(CLOSED_HIP, "triathlon")
    labels = [r["label"] for r in ranked]

    assert "rotate the pelvis forward" in labels[0]
    raise_bars = next(i for i, text in enumerate(labels) if "raise the aerobar" in text)
    assert raise_bars > 0, "raising the stack must never be the first suggestion here"


def test_top_option_fixes_both_problems_without_a_cost():
    top = rank_adjustments(CLOSED_HIP, "triathlon")[0]
    assert set(top["fixes"]) == {"Hip Angle", "Pelvic Rotation"}
    assert top["costs"] == []
    assert top["breaks"] == 0


def test_raising_the_bars_carries_its_aero_cost():
    """The cost a band cannot express: 24 deg and 28 deg are both "optimal"."""
    ranked = rank_adjustments(CLOSED_HIP, "triathlon")
    bars = next(r for r in ranked if "raise the aerobar" in r["label"])
    assert bars["aero_cost"], "raising the trunk must be reported as a drag penalty"
    assert "Trunk Angle" in " ".join(bars["costs"])


def test_no_material_problems_yields_no_options():
    """Nothing to fix must not produce a menu of things to change anyway."""
    clean = {k: {**v, "status": "optimal"} for k, v in CLOSED_HIP.items()}
    assert rank_adjustments(clean, "triathlon") == []
    assert build_tradeoff_block(clean, "triathlon") == ""


def test_a_metric_about_to_leave_its_band_is_flagged():
    angles = {
        "elbow": angle("Elbow Angle", 60, 75, 110, "needs_work"),
        "shoulder": angle("Shoulder Angle", 103, 80, 105, "optimal"),
    }
    top = rank_adjustments(angles, "triathlon")[0]
    assert top["breaks"] == 1
    assert "out of its band" in " ".join(top["costs"])


def test_flatter_trunk_is_a_gain_in_aero_and_a_cost_on_the_hoods():
    """Consistent with _classify_angle_status returning "aero_optimized"."""
    angles = {
        "elbow": angle("Elbow Angle", 60, 75, 110, "needs_work"),
        "trunk": angle("Trunk Angle", 21, 20, 30, "optimal"),
    }
    aero = rank_adjustments(angles, "triathlon")[0]
    road = rank_adjustments(angles, "road_hoods")[0]

    assert "aero gain" in " ".join(aero["costs"])
    assert aero["breaks"] == 0
    assert "out of its band" in " ".join(road["costs"])
    assert road["breaks"] == 1


def test_direction_matters():
    """A hip that is too OPEN needs the opposite levers to one that is closed.

    Physics: saddle height moves the hip and knee TOGETHER -- raising extends
    the leg, opening both (the same direction shorter cranks work in). This
    test shipped for months asserting the inverse and thereby PINNED the
    inverted hip sign that had the coach telling a closed-hip rider to lower
    the saddle. If it fails again, check the sign convention before the test.
    """
    too_open = {"hip": angle("Hip Angle", 70, 45, 62, "needs_work")}
    labels = [r["label"] for r in rank_adjustments(too_open, "triathlon")]

    assert any("lower the saddle" in text for text in labels)
    assert not any("raise the saddle" in text for text in labels)


def test_a_closed_hip_never_ranks_lowering_the_saddle():
    """The exact wrong advice the inverted sign produced on real reports."""
    labels = [r["label"] for r in rank_adjustments(CLOSED_HIP, "triathlon")]
    assert not any("lower the saddle" in text for text in labels)
    assert any("raise the saddle" in text for text in labels)


def test_fore_aft_direction_agrees_with_the_action_plan():
    """Two modules advise on the same lever; they must pull the same way.

    fit_tradeoffs encodes the steep-seat-tube doctrine: rotating the pelvis
    forward (saddle nose down / saddle FORWARD) opens the hip. The action
    plan's fore-aft diagnostic for a closed hip shipped pointing BACK -- the
    direction that stretches the reach, drops the torso and closes the hip
    further. One rider, one report, two opposite instructions.
    """
    from app.services.video_analysis.biomechanics.action_plan_builder import (
        build_action_plan,
    )
    from app.services.video_analysis.biomechanics.fit_tradeoffs import ADJUSTMENTS

    assert ADJUSTMENTS["rotate_pelvis_forward"]["moves"]["hip"] == +1
    assert "forward" in ADJUSTMENTS["rotate_pelvis_forward"]["label"]

    plan = build_action_plan(
        position="triathlon",
        angle_statistics={},
        sport_specific_metrics={"knee_at_bdc": 141.0, "hip_angle_avg": 25.0},
        technique_score=70,
        letter_grade="B",
        detected_issues=[],
    )
    fore_aft = next(
        d for d in plan.diagnostics if d.component == "saddle_fore_aft"
    )
    assert fore_aft.action == "move_saddle_forward"
    assert "forward" in fore_aft.reason
    assert "back" not in fore_aft.reason.split("Reassess")[0].lower()


def test_block_names_the_cost_of_every_option():
    block = build_tradeoff_block(CLOSED_HIP, "triathlon")
    options = [line for line in block.splitlines() if line and line[0].isdigit()]
    assert len(options) >= 3
    assert block.count("cost:") == len(options)
