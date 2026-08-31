"""Real clips, real pixels, compared with what they measured last time.

The gap this closes is the one the biomechanics audit walked into: 1218 tests,
all green, and three HIGH bugs that had been changing what users saw for
months. Every one of those tests checks logic against synthetic input -- the
right way to test logic, and no way at all to notice that the logic is
producing wrong values from real footage.

These tests are the other half. They run the actual pipeline over the actual
clips and compare a reduced record (see services/video_analysis/golden.py)
against a committed baseline.

Running them
------------
The clips are gitignored, so this is a LOCAL guard, not a CI one. Without them
every test here skips with a reason that says so -- loudly, because a
regression test that skips silently is worse than none: it looks like coverage.

They are also slow (~40-55 s per clip), so they are deselected by default:

    python -m pytest                      # normal suite, these skip
    python -m pytest -m golden            # just these
    python -m pytest --golden             # everything, including these

When a change moves a number on purpose, re-record and commit the new baseline
with the change:

    python scripts/golden_baseline.py --update
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.video_analysis.golden import build_record, compare

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
BASELINE_DIR = BACKEND / "tests" / "golden"

# Kept in step with scripts/golden_baseline.py -- a baseline recorded with
# different arguments compares nothing, so the test imports the same table
# rather than restating it.
from scripts.golden_baseline import CLIPS  # noqa: E402


def _run(spec) -> dict:
    from app.services.video_analysis.runner import run_analysis

    return build_record(run_analysis(
        str(REPO / str(spec["path"])), str(spec["sport"]), spec["position"],
        recommendations=False, kinogram=True, overlay_path=None,
    ))


@pytest.mark.golden
@pytest.mark.parametrize("name", sorted(CLIPS))
def test_the_clip_still_measures_what_it_measured(name):
    spec = CLIPS[name]
    clip = REPO / str(spec["path"])
    baseline_path = BASELINE_DIR / f"{name}.json"

    if not clip.exists():
        pytest.skip(
            f"reference clip {spec['path']} is not on this machine (it is "
            "gitignored) -- this guard only runs where the footage lives"
        )
    if not baseline_path.exists():
        pytest.skip(
            f"no baseline for {name}; record one with "
            "`python scripts/golden_baseline.py --update`"
        )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    diffs = compare(baseline, _run(spec))
    assert not diffs, (
        f"{name} measures differently than the committed baseline:\n  "
        + "\n  ".join(diffs)
        + "\n\nIf the change was deliberate, re-record with "
          "`python scripts/golden_baseline.py --update` and commit the new "
          "baseline alongside the code that moved it."
    )


# --------------------------------------------------------------------------
# The baselines themselves are data, and data can rot. These run everywhere,
# clips or not.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CLIPS))
def test_every_clip_has_a_committed_baseline(name):
    """A missing baseline turns the guard above into a permanent skip, which
    reads as passing."""
    assert (BASELINE_DIR / f"{name}.json").exists(), (
        f"{name} has no baseline; record one with "
        "`python scripts/golden_baseline.py --update`"
    )


@pytest.mark.parametrize("name", sorted(CLIPS))
def test_the_baseline_records_conclusions_and_not_only_numbers(name):
    """The three bugs that got through were a flipped ACTION, a component
    that vanished from the rubric, and angles from the wrong landmark space.
    A baseline of nothing but floats would have caught the third and missed
    the first two."""
    path = BASELINE_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip("no baseline recorded yet")
    data = json.loads(path.read_text(encoding="utf-8"))
    exact = data.get("exact") or {}

    # A grade, OR an explicit record that one was withheld. Both are
    # conclusions; a baseline that demanded a grade would fail on exactly the
    # clips where refusing to publish one is the behaviour worth pinning.
    assert exact.get("letter_grade") or exact.get("score_withheld"), (
        "the baseline records neither a grade nor a reason for its absence"
    )
    assert exact.get("camera_side"), "no camera side recorded"
    assert "coverage" in exact, "score coverage is not pinned"
    assert "scored" in exact["coverage"], "the scored-component list is not pinned"
    assert "fit_plan" in exact, "the recommended actions are not pinned"
    assert (data.get("approx") or {}).get("angles"), "no joint angles recorded"


def test_the_bike_baseline_pins_the_recommended_action():
    """The inverted hip sign changed only this: same measurement, same score,
    opposite instruction. Nothing else in the record would have moved."""
    path = BASELINE_DIR / "bike_tt_aero.json"
    if not path.exists():
        pytest.skip("no baseline recorded yet")
    plan = json.loads(path.read_text(encoding="utf-8"))["exact"]["fit_plan"]
    assert plan, "the bike clip records no fit actions to compare against"
    for row in plan:
        assert row.get("action"), "a fit row was pinned without its action"
        assert row.get("component"), "a fit row was pinned without its component"


def test_the_run_baseline_pins_the_guessed_time_base():
    """vid1 reads as 8x slow motion, so its cadence is excluded from the score
    rather than graded. If that exclusion ever silently stops happening, the
    clip goes back to being graded on a guess."""
    path = BASELINE_DIR / "run_slowmo.json"
    if not path.exists():
        pytest.skip("no baseline recorded yet")
    exact = json.loads(path.read_text(encoding="utf-8"))["exact"]
    assert exact.get("summary.slow_motion_factor") == 8
    assert "cadence" in exact["coverage"]["excluded"]
