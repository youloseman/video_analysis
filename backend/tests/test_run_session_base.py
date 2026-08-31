"""Which clip a two-sided run session reports the unmergeable half from.

Found by the second biomechanics audit (2026-08-30).

A run pair shares no rigid object, so most of a report cannot be pooled: the
score, the joint-angle table, the findings and the per-leg metrics all come
from ONE of the two clips. That is fine and unavoidable. What was not fine is
that TWO different rules picked that clip, and they disagreed.

``build_run_session`` seeded its merged summary from whichever summary dict had
more KEYS -- so one optional field that happened to be measurable on one side
decided whose knee the session reported. Meanwhile the caller took the score,
angles and findings from ``results["left"]``, unconditionally. A session could
therefore print the left clip's score above the right clip's knee angles, both
labelled "both sides", with nothing anywhere saying so.

Neither half was wrong on its own. The report was wrong because they were
picked separately.
"""

from __future__ import annotations

import pytest

from app.services.video_analysis.run_session import build_run_session


def clip(side: str, *, frames: int = 300, knee_min: float = 70.0, extra=None):
    summary = {
        # whole-body: both clips measure these, so they merge
        "cadence_spm": 170.0, "trunk_lean_avg": 6.0,
        "vertical_oscillation_m": 0.09, "ground_contact_ms": 240.0,
        "flight_time_ms": 120.0, "stance_fraction": 0.35,
        # per-leg: each clip measures its OWN leg
        "knee_min": knee_min, "knee_max": 160.0, "knee_mean": 120.0,
        "tracking_stability": {
            "leg_identity": {"stability": {"unstable": False, "instability": 0.0}},
        },
    }
    if extra:
        summary.update(extra)
    return {
        "camera_side": side, "sport_specific_metrics": summary,
        "frames_analyzed": frames, "technique_score": 88, "letter_grade": "B",
    }


# --------------------------------------------------------------------------
# the bug
# --------------------------------------------------------------------------

def test_an_optional_field_no_longer_decides_whose_leg_is_reported():
    """The exact shape of the bug: two clips identical except that the right
    one measured a foot-strike angle. Under the old rule that one extra key
    made the right leg the session's leg."""
    left = clip("left", frames=600, knee_min=70.0)
    right = clip("right", frames=100, knee_min=95.0,
                 extra={"foot_strike_angle_deg": 4.2})
    out = build_run_session(left, right)
    assert out["session"]["combined"] is True
    # 600 frames beats 100 frames; the extra key is irrelevant.
    assert out["base_side"] == "left"
    assert out["merged_summary"]["knee_min"] == 70.0


def test_the_clip_with_more_frames_wins():
    """More frames is more strides behind every per-leg number. It is a proxy,
    but unlike counting dict keys it is a proxy for something."""
    out = build_run_session(
        clip("left", frames=200, knee_min=70.0),
        clip("right", frames=500, knee_min=95.0),
    )
    assert out["base_side"] == "right"
    assert out["merged_summary"]["knee_min"] == 95.0


def test_ties_are_deterministic():
    """Two runs of the same pair must not disagree about whose leg it is."""
    pair = (clip("left", frames=300), clip("right", frames=300))
    first = build_run_session(*pair)["base_side"]
    assert first == build_run_session(*pair)["base_side"] == "left"


def test_the_summary_says_which_leg_its_unmerged_values_are_from():
    """`camera_side: both` is true of the averaged metrics and false of
    everything else in the same dict. Without this key the report has no way
    to tell the athlete which leg it is showing them."""
    out = build_run_session(
        clip("left", frames=500), clip("right", frames=100),
    )
    merged = out["merged_summary"]
    assert merged["camera_side"] == "both"
    assert merged["per_leg_source"] == "left"


def test_the_session_publishes_the_base_side_for_the_caller():
    """The caller carries the score, angles and findings across. It has to
    take them from the same clip, or the two halves of the report describe
    two different legs."""
    out = build_run_session(clip("left", frames=100), clip("right", frames=900))
    assert out["session"]["base_side"] == out["base_side"] == "right"


# --------------------------------------------------------------------------
# what must not have changed
# --------------------------------------------------------------------------

def test_whole_body_metrics_are_still_averaged():
    """These are the numbers the session exists for: one runner, measured
    twice."""
    left = clip("left", frames=300)
    right = clip("right", frames=300)
    right["sport_specific_metrics"]["cadence_spm"] = 180.0
    merged = build_run_session(left, right)["merged_summary"]
    assert merged["cadence_spm"] == 175.0


def test_a_refusal_still_names_a_base_side():
    """A refused merge still renders a score and an angle table, and they
    still come from one clip. The rule must not go missing just because the
    merge did."""
    left = clip("left", frames=200)
    right = clip("right", frames=800)
    right["sport_specific_metrics"]["tracking_stability"] = {
        "leg_identity": {"stability": {"unstable": True, "instability": 0.3}},
    }
    out = build_run_session(left, right)
    assert out["session"]["combined"] is False
    assert out["session"]["reason"] == "identity_unstable"
    assert out["base_side"] in ("left", "right")
    assert out["session"]["base_side"] == out["base_side"]


def test_two_clips_from_the_same_side_are_still_refused():
    out = build_run_session(clip("right"), clip("right"))
    assert out["session"]["combined"] is False
    assert out["session"]["reason"] == "sides_not_identified"


def test_clips_that_disagree_are_still_refused():
    left = clip("left", frames=300)
    right = clip("right", frames=300)
    right["sport_specific_metrics"]["trunk_lean_avg"] = 20.0   # 70% apart
    out = build_run_session(left, right)
    assert out["session"]["combined"] is False
    assert out["session"]["reason"] == "clips_disagree"


# --------------------------------------------------------------------------
# the caller, and the sentence the athlete reads
# --------------------------------------------------------------------------

def test_the_caller_takes_its_base_from_the_session():
    """A hardcoded results["left"] here is the other half of the bug."""
    import inspect

    from app import main

    src = inspect.getsource(main._process_pair_job)
    assert 'base = results[session["base_side"]]' in src, (
        "the pair job no longer takes its carried-over fields from the "
        "session's chosen clip"
    )


def test_the_report_tells_the_athlete_which_clip_the_score_is_from():
    """Being internally consistent is not the same as being honest: the page
    still shows one leg's score under a 'two-sided session' heading, and has
    to say so."""
    from pathlib import Path

    spa = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html")
    html = spa.read_text(encoding="utf-8")
    block = html[html.index("function renderRunSession("):]
    block = block[:block.index("\n/* Compare keyframe")]
    assert "s.base_side" in block, "the run session never mentions its base clip"
    assert "not on both" in block


@pytest.mark.parametrize("missing", ["frames_analyzed"])
def test_a_clip_with_no_frame_count_does_not_crash_the_choice(missing):
    left, right = clip("left"), clip("right")
    del left[missing]
    out = build_run_session(left, right)
    assert out["base_side"] in ("left", "right")


# --------------------------------------------------------------------------
# Second audit finding: the correction cap held by accident
# --------------------------------------------------------------------------

def test_the_correction_cap_holds_on_the_total_not_the_step():
    """MAX_OFFSET was checked per entry, and entries for one joint SUM.
    Ten legal 0.24 nudges reached 2.4 -- ten times the documented limit.

    check_plausibility caught it downstream by refusing a point pushed
    off-screen, so nothing shipped broken; but that is a different check with
    a different purpose, and the stated invariant was being enforced by
    accident."""
    from app.services.video_analysis.biomechanics.corrections import (
        MAX_OFFSET,
        normalize_corrections,
    )

    step = MAX_OFFSET * 0.9
    with pytest.raises(ValueError, match="in total"):
        normalize_corrections(
            [{"landmark": 26, "dx": step, "dy": 0.0} for _ in range(4)], "right",
        )


def test_nudging_within_the_cap_still_accumulates():
    """The feature this cap protects: "I moved it, then nudged it a bit more"
    must keep working."""
    from app.services.video_analysis.biomechanics.corrections import (
        normalize_corrections,
    )

    got = normalize_corrections(
        [{"landmark": 26, "dx": 0.05, "dy": 0.0},
         {"landmark": 26, "dx": 0.03, "dy": 0.0}], "right",
    )
    assert got == [{"landmark": 26, "dx": 0.08, "dy": 0.0}]


def test_opposite_nudges_do_not_trip_the_cap():
    """Moving a joint out and most of the way back is a small net offset, and
    a cap on the running total must not punish the path taken to it."""
    from app.services.video_analysis.biomechanics.corrections import (
        normalize_corrections,
    )

    got = normalize_corrections(
        [{"landmark": 26, "dx": 0.2, "dy": 0.0},
         {"landmark": 26, "dx": -0.18, "dy": 0.0}], "right",
    )
    assert got[0]["dx"] == pytest.approx(0.02)
