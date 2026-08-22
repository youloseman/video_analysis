"""The mobility endpoints, and what they are careful about.

Three things are worth a test here and the rest is plumbing.

A refusal must never be stored. "We could not see your knee" and "your hip does
not go there" are opposite findings, and filing the first as the second would
put a fabricated limitation on somebody's account that quietly steers every
bike analysis they ever run afterwards.

Degrees are stored; tiers are not. A tier is our reading of a number against a
reference value we might revise. Storing the reading rather than the verdict is
what lets a corrected cut-point reach everybody instead of only new users.

And the profile has to actually reach the analysis, or the whole feature is a
card nobody's fit ever hears about.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api import mobility as api
from app.services.video_analysis.biomechanics import mobility as M


@pytest.fixture(autouse=True)
def _no_throttle_between_tests():
    """The burst counter is keyed on user id, and every test gets a fresh DB
    where the first user is id 1. Without this one test's uploads throttle the
    next test's user."""
    api._recent_screens.clear()
    yield
    api._recent_screens.clear()


# --------------------------------------------------------------------------
# storage round-trip
# --------------------------------------------------------------------------

async def test_a_fresh_account_has_no_profile(make_user):
    user = await make_user()
    assert api.stored_profile(user) is None
    got = await api.get_mobility(user)
    assert got["profile"] is None
    assert {s["screen"] for s in got["screens"]} == set(M.MOBILITY_SCREENS)


async def test_stored_degrees_rebuild_a_full_profile(make_user):
    user = await make_user(
        mobility_hamstring_deg=72.0, mobility_hip_flexion_deg=112.0,
    )
    profile = api.stored_profile(user)
    assert profile["screens"]["hamstring"]["value"] == 72.0
    assert profile["screens"]["hamstring"]["tier"] == M.TIER_MODERATE
    assert profile["screens"]["hip_flexion"]["tier"] == M.TIER_MODERATE
    assert profile["ceiling"]["position"] == "road_drops"


async def test_one_screen_alone_is_a_valid_profile(make_user):
    """Somebody who did the hamstring test and stopped still gets an answer,
    just a more cautious one -- a half-done screen is not a reason to say
    nothing."""
    user = await make_user(mobility_hamstring_deg=90.0)
    profile = api.stored_profile(user)
    assert profile["screens_done"] == ["hamstring"]
    assert profile["ceiling"]["position"] == "road_drops"


def test_tiers_are_derived_not_stored():
    """The account holds the angle. If a reference value is ever corrected,
    everybody's next analysis gets the new reading -- not just new users."""
    old = M.MOBILITY_SCREENS["hip_flexion"]["tiers"]
    try:
        M.MOBILITY_SCREENS["hip_flexion"]["tiers"] = ((100.0, M.TIER_GOOD), (90.0, M.TIER_MODERATE))
        profile = M.profile_from_stored(hip_flexion_deg=112.0)
        assert profile["screens"]["hip_flexion"]["tier"] == M.TIER_GOOD
    finally:
        M.MOBILITY_SCREENS["hip_flexion"]["tiers"] = old
    assert M.profile_from_stored(hip_flexion_deg=112.0)["screens"][
        "hip_flexion"]["tier"] == M.TIER_MODERATE


# --------------------------------------------------------------------------
# the goal
# --------------------------------------------------------------------------

async def test_the_goal_round_trips(make_user, db):
    user = await make_user(mobility_hip_flexion_deg=125.0)
    got = await api.set_goal(api.GoalIn(goal="speed"), user, db)
    assert got["profile"]["goal"] == "speed"
    assert user.mobility_goal == "speed"


async def test_an_invented_goal_is_rejected(make_user, db):
    user = await make_user()
    with pytest.raises(HTTPException) as e:
        await api.set_goal(api.GoalIn(goal="podium"), user, db)
    assert e.value.status_code == 400


async def test_the_goal_can_be_cleared(make_user, db):
    user = await make_user(mobility_goal="comfort")
    await api.set_goal(api.GoalIn(goal=None), user, db)
    assert user.mobility_goal is None


async def test_the_goal_does_not_invent_a_profile(make_user, db):
    """A preference is not a measurement. Setting one on an account that has
    measured nothing must not conjure a range out of it."""
    user = await make_user()
    got = await api.set_goal(api.GoalIn(goal="speed"), user, db)
    assert got["profile"] is None


# --------------------------------------------------------------------------
# forgetting
# --------------------------------------------------------------------------

async def test_clearing_forgets_the_measurements(make_user, db):
    user = await make_user(
        mobility_hamstring_deg=72.0,
        mobility_hip_flexion_deg=112.0,
        mobility_goal="comfort",
    )
    got = await api.clear_mobility(user, db)
    assert got["profile"] is None
    assert user.mobility_hamstring_deg is None
    assert user.mobility_hip_flexion_deg is None
    assert user.mobility_measured_at is None
    # The preference is not a measurement and survives -- what they want out of
    # the bike did not change because they deleted a photo's result.
    assert user.mobility_goal == "comfort"


# --------------------------------------------------------------------------
# the catalogue is served, not duplicated in the client
# --------------------------------------------------------------------------

async def test_the_catalogue_carries_setup_and_a_source(make_user):
    user = await make_user()
    for screen in (await api.get_mobility(user))["screens"]:
        assert screen["setup"], f"{screen['screen']} has no capture instructions"
        assert screen["source"], f"{screen['screen']} cites no reference"
        assert screen["bands"]["good"] > screen["bands"]["moderate"]


# --------------------------------------------------------------------------
# validation on the way in
# --------------------------------------------------------------------------

async def test_an_unknown_screen_is_a_400(make_user, db):
    user = await make_user()
    with pytest.raises(HTTPException) as e:
        await api.measure_screen("thoracic_rotation", None, user, db)
    assert e.value.status_code == 400


async def test_a_refusal_writes_nothing(make_user, db, monkeypatch):
    """The important one. A photo we could not measure must leave the account
    exactly as it was -- storing a failed screen as a low reading would tell
    every future fit that this rider is stiff."""
    from tests.test_mobility import standing

    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    monkeypatch.setattr(
        "app.services.video_analysis.photo_analyzer.detect_pose_in_image",
        lambda data: {"world": standing(), "warnings": []},
    )

    user = await make_user()
    photo = _FakePhoto(b"not-really-a-jpeg")
    with pytest.raises(HTTPException) as e:
        await api.measure_screen("hamstring", photo, user, db)
    assert e.value.status_code == 422
    assert "lying down" in e.value.detail
    assert user.mobility_hamstring_deg is None
    assert user.mobility_measured_at is None


async def test_a_good_photo_is_stored_with_a_timestamp(make_user, db, monkeypatch):
    from tests.test_mobility import supine

    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    monkeypatch.setattr(
        "app.services.video_analysis.photo_analyzer.detect_pose_in_image",
        lambda data: {"world": supine(raise_deg=84.0), "warnings": ["dim"]},
    )

    user = await make_user()
    got = await api.measure_screen("hamstring", _FakePhoto(b"x"), user, db)
    assert got["measurement"]["value"] == pytest.approx(84.0, abs=0.5)
    assert got["measurement"]["tier"] == M.TIER_GOOD
    assert got["measurement"]["capture_warnings"] == ["dim"]
    assert user.mobility_hamstring_deg == pytest.approx(84.0, abs=0.5)
    assert user.mobility_measured_at is not None
    assert got["profile"]["screens_done"] == ["hamstring"]


async def test_an_empty_upload_is_rejected_before_any_cpu(make_user, db, monkeypatch):
    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    user = await make_user()
    with pytest.raises(HTTPException) as e:
        await api.measure_screen("hamstring", _FakePhoto(b""), user, db)
    assert e.value.status_code == 400


async def test_a_wrong_file_type_is_rejected(make_user, db, monkeypatch):
    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    user = await make_user()
    with pytest.raises(HTTPException) as e:
        await api.measure_screen("hamstring", _FakePhoto(b"x", name="clip.mov"), user, db)
    assert e.value.status_code == 400


async def test_a_rejected_upload_does_not_burn_the_cooldown(make_user, db, monkeypatch):
    """Picking the wrong file and immediately picking the right one is not two
    requests for CPU. Locking somebody out for their own typo is a strange way
    to run a form."""
    from tests.test_mobility import supine

    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    monkeypatch.setattr(
        "app.services.video_analysis.photo_analyzer.detect_pose_in_image",
        lambda data: {"world": supine(raise_deg=84.0), "warnings": []},
    )
    user = await make_user()

    with pytest.raises(HTTPException):
        await api.measure_screen("hamstring", _FakePhoto(b"x", name="clip.mov"), user, db)
    got = await api.measure_screen("hamstring", _FakePhoto(b"x"), user, db)
    assert got["measurement"]["value"] == pytest.approx(84.0, abs=0.5)


async def test_both_screens_back_to_back_are_fine(make_user, db, monkeypatch):
    """The natural flow is two photos in a row, plus a retake when the first
    gets refused for a bent knee. A brake that fought that would be fighting
    the feature."""
    from tests.test_mobility import supine

    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    monkeypatch.setattr(
        "app.services.video_analysis.photo_analyzer.detect_pose_in_image",
        lambda data: {"world": supine(raise_deg=84.0, knee_bend_deg=0.0), "warnings": []},
    )
    user = await make_user()
    await api.measure_screen("hamstring", _FakePhoto(b"x"), user, db)
    monkeypatch.setattr(
        "app.services.video_analysis.photo_analyzer.detect_pose_in_image",
        lambda data: {"world": supine(raise_deg=124.0, knee_bend_deg=110.0), "warnings": []},
    )
    got = await api.measure_screen("hip_flexion", _FakePhoto(b"x"), user, db)
    assert got["profile"]["screens_done"] == ["hamstring", "hip_flexion"]
    assert got["profile"]["ceiling"]["unrestricted"] is True


async def test_a_loop_is_stopped(make_user, db, monkeypatch):
    from tests.test_mobility import supine

    monkeypatch.setattr(api.settings.model_path.__class__, "exists", lambda _: True)
    monkeypatch.setattr(
        "app.services.video_analysis.photo_analyzer.detect_pose_in_image",
        lambda data: {"world": supine(raise_deg=84.0), "warnings": []},
    )
    user = await make_user()
    for _ in range(api.SCREEN_BURST):
        await api.measure_screen("hamstring", _FakePhoto(b"x"), user, db)
    with pytest.raises(HTTPException) as e:
        await api.measure_screen("hamstring", _FakePhoto(b"x"), user, db)
    assert e.value.status_code == 429


class _FakePhoto:
    def __init__(self, data: bytes, name: str = "screen.jpg"):
        self._data, self.filename = data, name

    async def read(self) -> bytes:
        return self._data
