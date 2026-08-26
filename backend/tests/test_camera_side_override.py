"""The user-chosen camera side replaces the depth vote — bike only, and honestly.

Feature history: the Z-depth vote picks the near side well on clean clips, but
Artur's real uploads showed it can be argued with (mirrors behind the trainer,
full-aero far-leg occlusion). The fix was not a smarter vote — it was letting
the rider state the side they filmed from. These tests pin the contract of
``_apply_camera_side_override``:

* on a bike clip a "left"/"right" override wins, and the lock meta says so
  (``user_set``, single-vote list, ``fallback: False`` so quality gates don't
  discount a side the user asserted);
* on a run clip the override is IGNORED — a run side view sees both legs, and
  claiming a user-set unilateral lock there would be a false certainty;
* junk overrides (None, "", "front") leave the detection untouched.
"""

from app.services.video_analysis.runner import _apply_camera_side_override


def test_bike_override_replaces_detection():
    side, meta = _apply_camera_side_override(
        "right", {"votes": ["right", "right"], "fallback": False}, "left", "bike",
    )
    assert side == "left"
    assert meta["user_set"] is True
    assert meta["votes"] == ["left"]
    assert meta["fallback"] is False


def test_bike_override_agreeing_with_detection_still_marks_user_set():
    # Even when the vote agrees, the result must say "chosen", not "detected" —
    # the difference matters when the user later disputes a number.
    side, meta = _apply_camera_side_override(
        "left", {"votes": ["left"], "fallback": False}, "left", "bike",
    )
    assert side == "left"
    assert meta["user_set"] is True


def test_bike_override_clears_fallback():
    # A fallback lock (vote had no signal) discounts camera_side downstream.
    # A user-asserted side is not a fallback.
    side, meta = _apply_camera_side_override(
        None, {"votes": [], "fallback": True}, "right", "bike",
    )
    assert side == "right"
    assert meta["fallback"] is False


def test_run_ignores_override():
    side, meta = _apply_camera_side_override(
        "left", {"votes": ["left"], "fallback": False}, "right", "run",
    )
    assert side == "left"
    assert "user_set" not in meta


def test_no_override_is_a_no_op():
    original = {"votes": ["right"], "fallback": False}
    side, meta = _apply_camera_side_override("right", original, None, "bike")
    assert side == "right"
    assert meta == original


def test_junk_override_is_ignored():
    for junk in ("", "front", "LEFT", "both"):
        side, meta = _apply_camera_side_override("right", {}, junk, "bike")
        assert side == "right"
        assert "user_set" not in meta


def test_none_lock_meta_survives():
    side, meta = _apply_camera_side_override(None, None, "left", "bike")
    assert side == "left"
    assert meta["user_set"] is True
