"""The drill has to match the fault, and the fault has to be real.

Our joint angles are INTERNAL: 180 deg is a straight limb, so a knee angle
BELOW its band carries MORE flexion, not less. Both knee drills were wired to
the below case, which inverted their meaning on live reports:

* a runner whose swing knee folded to 56 deg -- heel nearly at the glute, more
  drive than the 80-100 band asks for -- was handed "High Knee March" as
  priority #1;
* a knee locked at 179 deg at contact, the over-extension the drill copy
  actually describes and the one that pairs with overstriding, produced no
  drill at all.

The second half of this file is the other way a plan misleads: grading noise.
A trunk lean of 3.9 deg against a 4-8 band is one tenth of a degree out, well
inside what 2D video resolves, and it produced a two-week drill -- while the
report's own preamble told the reader that a degree or two outside a band is
not a finding.
"""

from __future__ import annotations

from app.services.video_analysis.biomechanics.running_action_plan_builder import (
    build_running_action_plan,
)


def plan_for(**summary):
    base = {
        "cadence_spm": 178.0,
        "overstride_ratio": 0.10,
        "vertical_oscillation_m": 0.08,
        "trunk_lean_avg": 6.0,
        "knee_max": 168.0,
        "elbow_mean": 92.0,
        "knee_min": 90.0,
    }
    base.update(summary)
    return build_running_action_plan({"score": 80, "summary": base})


def diagnosis(plan, metric):
    return next((d for d in plan.diagnostics if d.metric_name == metric), None)


def is_fine(plan, metric):
    """Optimal metrics are filed under good_metrics, not diagnostics."""
    return metric in {m["metric"] for m in plan.good_metrics}


def drills(plan):
    return {d.metric_name: d.drill_key for d in plan.diagnostics if d.drill_key}


# ---------------------------------------------------------------------------
# Knee swing: the band is 80-100 internal
# ---------------------------------------------------------------------------
def test_a_deeply_folded_swing_knee_is_not_a_fault():
    """56 deg internal = ~124 deg of flexion: more knee drive than asked for."""
    plan = plan_for(knee_min=56.1)
    assert is_fine(plan, "knee_swing")
    assert "knee_swing" not in drills(plan)


def test_a_stiff_swing_knee_gets_the_high_knee_drill():
    """115 deg internal = only 65 deg of flexion: weak knee drive."""
    plan = plan_for(knee_min=115.0)
    assert drills(plan).get("knee_swing") == "knee_swing_insufficient"
    assert "Insufficient knee flexion in swing" in (
        diagnosis(plan, "knee_swing").problem_description
    )


# ---------------------------------------------------------------------------
# Knee at contact: the band is 160-175 internal
# ---------------------------------------------------------------------------
def test_a_knee_locked_at_contact_now_produces_a_drill():
    """The 179.3 deg case from a live report, which used to pass silently."""
    plan = plan_for(knee_max=179.3)
    assert drills(plan).get("knee_contact") == "knee_contact_insufficient_flexion"


def test_a_softer_knee_at_contact_is_not_faulted():
    plan = plan_for(knee_max=150.0)
    assert is_fine(plan, "knee_contact")
    assert "knee_contact" not in drills(plan)


# ---------------------------------------------------------------------------
# Deviations too small to mean anything
# ---------------------------------------------------------------------------
def test_a_tenth_of_a_degree_outside_the_band_is_not_a_finding():
    plan = plan_for(trunk_lean_avg=3.9)
    assert is_fine(plan, "trunk_lean")
    assert "trunk_lean" not in drills(plan)


def test_a_real_trunk_deviation_still_gets_its_drill():
    plan = plan_for(trunk_lean_avg=1.5)
    assert drills(plan).get("trunk_lean") == "trunk_lean_insufficient"

    plan = plan_for(trunk_lean_avg=14.0)
    assert drills(plan).get("trunk_lean") == "trunk_lean_excessive"


def test_bouncing_less_than_the_band_is_not_graded_as_a_deviation():
    """Only-too-high metrics have no near-miss on the good side."""
    plan = plan_for(vertical_oscillation_m=0.04)
    assert is_fine(plan, "vertical_osc")
    note = next(m for m in plan.good_metrics if m["metric"] == "vertical_osc")
    assert "better than the target" in note["description"]


def test_a_cadence_above_the_band_is_not_a_fault_either():
    plan = plan_for(cadence_spm=196.0)
    assert is_fine(plan, "cadence")


def test_the_faults_that_are_real_survive_all_of_this():
    """Guard against the guards: the plan must still find genuine problems."""
    plan = plan_for(
        cadence_spm=150.0, overstride_ratio=0.30, vertical_oscillation_m=0.17,
    )
    keys = drills(plan)
    assert keys.get("cadence") == "cadence_low"
    assert keys.get("overstride") == "overstride_high"
    assert keys.get("vertical_osc") == "vertical_osc_high"
