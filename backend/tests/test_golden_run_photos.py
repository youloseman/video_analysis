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
    # Graded with the score's own band, so the report's block and the line on
    # the picture carry the same verdict.
    assert c["overstride_band"] == [0.0, 0.15] and c["overstride_status"] == "needs_work"
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
