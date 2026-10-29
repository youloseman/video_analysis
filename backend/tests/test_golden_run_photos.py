"""Thirteen real treadmill photos of one runner, read with the legs named by
position (``_name_run_legs``).

Why these: on them the model's left/right labels put the "near" leg on the
REAR leg in 12 of 13 frames while the runner's tattooed calf showed the two
physical legs plainly alternating, so the old reading never found a touchdown
and filed early-swing frames as toe-off. Hand-checked on the overlays
(2026-10-19): photo_1, 5 and 9 are a foot landing ahead of the hip, 2, 6, 7, 8
and 13 a foot under the body, 4 and 11 both feet in the air.

The photos are gitignored (upload/), so like the golden clips this is a local
guard: it skips loudly without them, and runs with ``-m golden``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.video_analysis.photo_analyzer import analyze_photo

DIR = Path(__file__).resolve().parents[2] / "upload" / "photo_run"

# photo number -> the instant hand-read off the picture
EXPECTED = {
    1: "initial_contact", 5: "initial_contact", 9: "initial_contact",
    2: "midstance", 6: "midstance", 7: "midstance", 8: "midstance", 13: "midstance",
    4: "uncertain", 11: "uncertain",
    10: "toe_off", 12: "toe_off",
}


def _photo(n: int) -> Path:
    hits = sorted(DIR.glob(f"photo_{n}_*.jpg"))
    if not hits:
        pytest.skip(f"upload/photo_run/photo_{n}_*.jpg is not on this machine (gitignored)")
    return hits[0]


@pytest.mark.golden
@pytest.mark.parametrize("n,phase", sorted(EXPECTED.items()))
def test_the_instant_matches_the_picture(n, phase):
    r = analyze_photo(_photo(n).read_bytes(), "run", None, False, True)
    assert r["gait_phase"] == phase


@pytest.mark.golden
def test_a_touchdown_photo_reads_the_landing_leg():
    """photo_1: the landing leg carries the far label. Measured anyway: a
    near-straight knee, the foot well ahead of the hip, heel first."""
    r = analyze_photo(_photo(1).read_bytes(), "run", None, False, True)
    c = r["contact"]
    assert c["legs"]["stance"] == "front" and c["legs"]["swapped"] is True
    assert r["angles_with_context"]["knee"]["value"] > 155
    assert c["overstride_ratio"] > 0.25
    assert c["foot_strike"] == "heel"
    # The reach is shown, not graded; the shin leaning back is (-15.7 deg:
    # amber, worth a look).
    assert "overstride_status" not in c
    assert c["shin_band"] == [-10.0, 20.0] and c["shin_status"] == "acceptable"
    assert "check" not in r["angles_with_context"]["knee"]


@pytest.mark.golden
def test_crossed_knees_ask_for_a_look_at_the_knee_point():
    """photo_3 (runphoto2026 p07): the knee point sat on the swing leg, 18 deg
    off the hand-placed one. Still scored; the athlete is told to check it."""
    r = analyze_photo(_photo(3).read_bytes(), "run", None, False, True)
    assert r["contact"]["legs"]["knee_gap"] < 0.2
    knee = r["angles_with_context"]["knee"]
    assert knee["check"] is True and "overlap" in knee["note"]
    assert any(w.startswith("Knee: your two knees overlap") for w in r["warnings"])
    assert r["score"]["overall_score"] is not None


@pytest.mark.golden
@pytest.mark.parametrize("n", range(1, 14))
def test_the_pose_handed_out_re_measures_to_the_same_report(n):
    """The pose a run photo hands the joint editor is the one with the legs
    already named; re-measuring from it, nothing moved but a toe by a hair,
    must reproduce the report the athlete saw -- angles, phase, score."""
    from app.services.photo_corrections import recompute_photo

    data = _photo(n).read_bytes()
    r = analyze_photo(data, "run", None, False, True)
    heel = 30 if r["pose"]["camera_side"] == "right" else 29
    out = recompute_photo(data, "run", None, r["pose"],
                          [{"landmark": heel, "dx": 0.0, "dy": 0.0001}])
    b = out["baseline"]
    assert b["angles"] == r["angles"]
    assert b["gait_phase"] == r["gait_phase"]
    assert (b["score"] or {}).get("overall_score") == (r["score"] or {}).get("overall_score")
