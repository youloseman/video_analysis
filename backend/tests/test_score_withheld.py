"""A clip the gate rejected gets no score at all.

Artur's call, 2026-08-31, after IMG_4262: the pose model confused the legs on
46.8% of frames, the honest exclusions removed the five per-leg components,
and the four that remained all happened to be good -- so it read 100/100. Every
caveat around it was true and none of them was going to win an argument with a
100. A partial analysis used to carry a full score with a banner over it, which
puts a number and its own disclaimer in the same card and lets the reader pick.
They pick the number.

So the number goes. Everything else stays: the angles, the findings, the plan,
the coverage line, and the capture report that says how to refilm.

The score is still COMPUTED -- the fit plan and the drill builder rank their
advice against it and still have work to do on a bad clip. What changes is
whether it is presented, and these tests are about every surface it could leak
out of: the result, the image that travels on its own, the coach's prose, and
the free tier where the score used to be the entire product.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.result_gating import _SAFE_KEYS, gate_free_result
from app.services.video_analysis.biomechanics.quality_gate import (
    evaluate_quality_gate,
)

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


# --------------------------------------------------------------------------
# the gate criterion that started this
# --------------------------------------------------------------------------

def test_unstable_legs_trigger_the_gate():
    """Every other input the gate takes is about whether landmarks were FOUND.
    This is the one failure where they were found and put on the wrong leg,
    which no nan_pct can express."""
    got = evaluate_quality_gate(
        unknown_phase_pct=0.0, angle_statistics={"knee": {"nan_pct": 3.0}},
        valid_frames=560, frames_processed=569, sport="run",
        leg_identity_unstable=True,
    )
    assert got["triggered"] is True
    assert any("swapping which leg" in r for r in got["reasons"])
    assert got["criteria"]["leg_identity_unstable"] is True


def test_stable_legs_do_not_trigger_it():
    got = evaluate_quality_gate(
        unknown_phase_pct=0.0, angle_statistics={"knee": {"nan_pct": 3.0}},
        valid_frames=560, frames_processed=569, sport="run",
        leg_identity_unstable=False,
    )
    assert got["triggered"] is False


def test_a_caller_that_does_not_know_skips_the_criterion():
    """None is "no opinion", not "unstable" -- bike and swim never pass it.

    ``unknown_phase_pct`` is None here because that is what the runner passes
    for bike, which has no per-frame phase channel. Passing a NUMBER with
    sport="bike" raises KeyError inside the gate, because the bike profiles
    genuinely have no such threshold -- a latent sharp edge, unreachable
    today, noted rather than papered over here.
    """
    got = evaluate_quality_gate(
        unknown_phase_pct=None, angle_statistics={"knee": {"nan_pct": 3.0}},
        valid_frames=560, frames_processed=569, sport="bike",
        cycling_position="tt_aero", leg_identity_unstable=None,
    )
    assert got["triggered"] is False


# --------------------------------------------------------------------------
# the free tier, where the score WAS the product
# --------------------------------------------------------------------------

def test_the_reason_survives_the_paywall():
    """A free result is a score and nothing else. Take the score away without
    the reason and the reader gets an empty card that looks like a broken
    page -- and loses the one sentence that tells them how to get a clip we
    can measure."""
    assert "score_withheld" in _SAFE_KEYS


def test_a_free_user_is_told_why_there_is_no_number():
    gated = gate_free_result({
        "status": "completed", "sport_type": "run",
        "technique_score": None, "letter_grade": None,
        "score_withheld": {"reason": "quality_gate", "detail": "The pose model kept swapping which leg is which."},
        "quality_gate_triggered": True,
        "keyframe_base64": "data:image/jpeg;base64,AAAA",
        "angle_statistics": {"knee": {"mean": 120.0}},
    })
    assert gated["technique_score"] is None
    assert gated["score_withheld"]["detail"]
    assert "angle_statistics" not in gated       # still paid


# --------------------------------------------------------------------------
# the image that travels on its own
# --------------------------------------------------------------------------

def test_the_kinogram_badge_handles_a_withheld_score():
    """This picture gets saved, shared and pasted into messages with none of
    the caveats attached, so a grade stamped on it outlives every warning
    that explained why the grade was wrong."""
    import inspect

    from app.services.video_analysis import kinogram

    src = inspect.getsource(kinogram)
    assert "if technique_score is None:" in src
    assert '"not scored"' in src


def test_the_runner_passes_none_to_both_kinograms():
    import inspect

    from app.services.video_analysis import runner

    src = inspect.getsource(runner.analyze_from_frames)
    assert src.count("None if score_withheld else (") == 2, (
        "a kinogram call site is not honouring the withheld score"
    )


# --------------------------------------------------------------------------
# the coach
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sport,needle", [
    ("run", "Sport: running"), ("bike", "Sport: cycling"),
])
def test_the_coach_is_told_the_score_was_withheld(sport, needle):
    """Handed None, the prompt used to read "Technique score: None/100 (None)"
    -- which invites the model to invent one."""
    from app.services.video_analysis.llm_recommendations import build_metrics_block

    block = build_metrics_block(
        sport, None, None, "tt_aero" if sport == "bike" else None, [], {}, {},
    )
    assert needle in block
    assert "WITHHELD" in block
    assert "None/100" not in block
    assert "Do not invent a score" in block


def test_the_coach_still_gets_a_real_score_when_there_is_one():
    from app.services.video_analysis.llm_recommendations import build_metrics_block

    block = build_metrics_block("run", 91, "A", None, [], {}, {})
    assert "Technique score: 91/100 (A)" in block
    assert "WITHHELD" not in block


# --------------------------------------------------------------------------
# the score card
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def spa() -> str:
    return SPA.read_text(encoding="utf-8")


def test_the_card_says_not_scored_rather_than_a_dash(spa: str):
    """An em-dash reads as something that failed to load. This was a
    decision."""
    assert 'id="scoreWithheld"' in spa
    assert "Not scored" in spa


def test_the_card_explains_and_points_at_what_survived(spa: str):
    block = spa[spa.index("const withheld = r.score_withheld"):]
    block = block[:block.index("setGrade(")]
    assert "withheld.detail" in block
    # The reader has to know the analysis still happened.
    assert "still below" in block


def test_the_withheld_message_is_hidden_when_there_is_a_score(spa: str):
    block = spa[spa.index("const withheld = r.score_withheld"):]
    block = block[:block.index("setGrade(")]
    assert "classList.toggle('hidden', !withheld)" in block


def test_the_style_lives_in_the_stylesheet():
    from tests.conftest import read_app_css

    css = read_app_css()
    assert ".score-none{" in css
    assert ".score-none-why{" in css
