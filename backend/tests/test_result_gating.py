"""The paywall. These tests exist because a leak here is silent: the free tier
would simply start giving away the paid analysis, and nothing would look broken.

The module is an ALLOWLIST on purpose, so the load-bearing test is not "is
``ai_recommendations`` removed" but "does an unknown key added tomorrow get
withheld by default".
"""

from __future__ import annotations

import json

import pytest

from app.models.user import (
    TIER_ADMIN,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
    User,
)
from app.services.result_gating import gate_free_result, gate_result_for_tier, is_free


def full_result() -> dict:
    """A result carrying every paid field the video path can produce."""
    return {
        "status": "completed",
        "sport_type": "run",
        "technique_score": 74,
        "letter_grade": "C",
        "keyframe_base64": "data:image/jpeg;base64,AAAA",
        "camera_side": "right",
        "frames_analyzed": 300,
        # --- everything below is paid ---
        "ai_recommendations": {"summary": "lift your cadence"},
        "training_plan": {"top_3_priorities": [{"drill": "strides"}]},
        "angle_statistics": {"knee": {"mean": 120.0}},
        "detected_issues": [{"issue": "overstriding"}],
        "sport_specific_metrics": {"cadence_spm": 165},
        "score_breakdown": {"cadence": 87.5},
        "overlay_video_path": "/jobs/abc/overlay",
    }


PAID_KEYS = [
    "ai_recommendations", "training_plan", "angle_statistics",
    "detected_issues", "sport_specific_metrics", "score_breakdown",
]


def test_free_result_keeps_only_the_teaser():
    gated = gate_free_result(full_result())
    assert gated["technique_score"] == 74
    assert gated["letter_grade"] == "C"
    assert gated["keyframe_base64"].startswith("data:image/jpeg")
    assert gated["locked"]["reason"] == "starter"
    assert "coaching" in gated["locked"]["unlocks"]


@pytest.mark.parametrize("key", PAID_KEYS)
def test_free_result_drops_every_paid_field(key):
    assert key in full_result(), "test fixture no longer covers this field"
    assert key not in gate_free_result(full_result())


def test_free_result_never_serves_an_overlay_video():
    # Present-but-null, so the client renders the upsell rather than a player.
    assert gate_free_result(full_result())["overlay_video_path"] is None


def test_unknown_future_field_is_withheld_by_default():
    """The allowlist's whole point: a paid field added later must not leak
    just because nobody remembered to update a denylist."""
    result = full_result() | {"vo2max_estimate": 58.1, "bike_fit_report": {"x": 1}}
    gated = gate_free_result(result)
    assert "vo2max_estimate" not in gated
    assert "bike_fit_report" not in gated


def test_photo_result_shape_is_gated_too():
    """The photo path returns ``score``/``thumbnail_base64`` instead of the
    video path's keys; the same allowlist has to cover both."""
    photo = {
        "score": {"overall_score": 81, "letter_grade": "B"},
        "thumbnail_base64": "data:image/jpeg;base64,BBBB",
        "angles": {"knee": 140},
        "ai_recommendations": {"summary": "drop the saddle"},
    }
    gated = gate_free_result(photo)
    assert gated["score"]["overall_score"] == 81
    assert gated["thumbnail_base64"]
    assert "angles" not in gated
    assert "ai_recommendations" not in gated


@pytest.mark.parametrize("tier", [TIER_ENTHUSIAST, TIER_FULL, TIER_ADMIN])
def test_paid_tiers_pass_through_untouched(tier):
    result = full_result()
    assert gate_result_for_tier(result, User(email="a@b.c", password_hash="x", tier=tier)) == result


def test_starter_and_anonymous_are_both_free():
    starter = User(email="a@b.c", password_hash="x", tier=TIER_STARTER)
    assert is_free(starter) is True
    assert is_free(None) is True
    assert is_free(User(email="a@b.c", password_hash="x", tier=TIER_FULL)) is False


def test_gate_result_for_tier_gates_anonymous_callers():
    gated = gate_result_for_tier(full_result(), None)
    assert "locked" in gated
    assert "ai_recommendations" not in gated


# --------------------------------------------------------------------------
# The quality caveat
#
# A free report is a score with no angles and no coaching to argue with. If the
# analyzer half-tracked the athlete, or the clip was shot from the wrong angle,
# that score is the ONLY thing they see -- so the caveat travels with it. This is
# the difference between "we couldn't measure this properly" and a confident
# verdict on footage the analyzer already distrusted.
# --------------------------------------------------------------------------
def gated_result() -> dict:
    return full_result() | {
        "quality_gate_triggered": True,
        "sport_specific_metrics": {
            "cadence_spm": 165,                       # paid
            "vertical_oscillation_m": 0.09,           # paid
            "quality_gate": {
                "triggered": True,
                "reasons": ["Only 38% of frames had a usable lower body."],
                "criteria": {"valid_frames_ratio": 0.38},   # paid detail
            },
            "quality_warnings": [
                "Low landmark detection quality. The body may be too far away.",
            ],
            "analysis_confidence": {
                "level": "low",
                "factors": {"landmark_quality_pct": 41},    # paid detail
                "explanation": "long prose",               # paid detail
            },
            "time_base_uncertain": True,
            "sampling_degraded": "This clip is 74.0s, so we analyzed every 4th frame.",
        },
    }


def test_free_callers_are_told_the_analyzer_distrusted_the_clip():
    gated = gate_free_result(gated_result())
    assert gated["quality_gate_triggered"] is True
    q = gated["quality"]
    assert q["triggered"] is True
    assert q["reasons"] == ["Only 38% of frames had a usable lower body."]
    assert q["warnings"] == [
        "Low landmark detection quality. The body may be too far away.",
    ]
    assert q["confidence"] == "low"
    assert q["time_base_uncertain"] is True
    assert q["sampling_degraded"].startswith("This clip is 74.0s")


def test_the_caveat_does_not_smuggle_out_the_paid_metrics():
    """It reads FROM sport_specific_metrics but must never carry it along."""
    gated = gate_free_result(gated_result())
    assert "sport_specific_metrics" not in gated
    blob = json.dumps(gated)
    for paid in ("cadence_spm", "vertical_oscillation_m", "valid_frames_ratio",
                 "landmark_quality_pct", "long prose"):
        assert paid not in blob


def test_confidence_is_reduced_to_its_level():
    """The tier is a trust signal; the per-factor breakdown behind it is paid."""
    q = gate_free_result(gated_result())["quality"]
    assert q["confidence"] == "low"
    assert not isinstance(q["confidence"], dict)


def test_a_clean_result_still_carries_an_explicit_all_clear():
    """Absence of a warning has to be stated, not inferred from a missing key —
    the client renders straight off this block."""
    q = gate_free_result(full_result())["quality"]
    assert q == {"triggered": False, "reasons": [], "warnings": []}


def test_the_caveat_survives_a_result_with_no_metrics_at_all():
    """The photo path has no sport_specific_metrics; it must not blow up."""
    q = gate_free_result({"score": {"overall_score": 70}})["quality"]
    assert q["triggered"] is False
    assert q["warnings"] == []


def test_paid_results_are_untouched_by_the_caveat_logic():
    result = gated_result()
    user = User(email="a@b.c", password_hash="x", tier=TIER_FULL)
    assert gate_result_for_tier(result, user) is result
