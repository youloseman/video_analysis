"""Measured CdA on setups, the coach's column and entry, and the research gate.

What matters: a number is the athlete's (or their coach's) measurement with
its source and protocol, never ours; a coach writes onto a linked athlete's
setup only while the link is live, stamped, deletable by the athlete; and
nothing is eligible for research without the owner's own switch.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app.api import athletes as roster
from app.api import auth
from app.api import cda as api
from app.api import coach_link
from app.api import me
from app.models.user import TIER_ADMIN, TIER_COACH, TIER_FULL, TIER_STARTER
from app.services.research import profile_research_eligible, research_eligible

TODAY = date.today().isoformat()


def body(v=0.230, *, source="airo", protocol="repeated_loops", on=TODAY, note=None):
    return api.CdaIn(value_m2=v, measured_on=on, source=source, protocol=protocol, note=note)


async def bike(db, user, name="Speedmax", athlete_id=None):
    return await me.create_profile(
        me.ProfileIn(name=name, sport="bike", kind="tt", athlete_id=athlete_id), user, db,
    )


async def listed(db, user, pid):
    return next(p for p in (await me.list_profiles(user, db))["profiles"] if p["id"] == pid)


# --------------------------------------------------------------------------
# The owner's own setups
# --------------------------------------------------------------------------
async def test_a_measurement_lands_on_the_setup_newest_first(make_user, db):
    u = await make_user(tier=TIER_FULL)
    p = await bike(db, u)
    old = (date.today() - timedelta(days=10)).isoformat()
    await api.add_cda(p["id"], body(0.248, on=old), None, u, db)
    await api.add_cda(p["id"], body(0.236, source="windsock", protocol="single_ride"), None, u, db)
    rows = (await listed(db, u, p["id"]))["cda"]
    assert [r["value_m2"] for r in rows] == [0.236, 0.248]
    assert rows[0]["protocol"] == "single_ride" and rows[0]["entered_by_coach"] is None
    assert all(r["can_delete"] for r in rows)


async def test_a_tunnel_or_cfd_number_is_never_a_ride(make_user, db):
    u = await make_user(tier=TIER_FULL)
    p = await bike(db, u)
    for src in ("airo", "wind_tunnel"):
        out = await api.add_cda(p["id"], body(source=src, protocol="single_ride"), None, u, db)
        assert out["protocol"] == "not_a_ride"
    out = await api.add_cda(p["id"], body(source="notio", protocol="single_ride"), None, u, db)
    assert out["protocol"] == "single_ride"


@pytest.mark.parametrize("kw", [
    {"v": 0.05}, {"v": 2.4}, {"on": "yesterday"},
    {"on": (date.today() + timedelta(days=5)).isoformat()}, {"source": "garmin"},
    {"protocol": "vibes"},
])
def test_what_is_not_a_measurement_is_refused(kw):
    with pytest.raises(ValidationError):
        body(**kw)


async def test_somebody_elses_setup_and_a_run_profile_are_not_found(make_user, db):
    a, b = await make_user(tier=TIER_FULL), await make_user(tier=TIER_FULL)
    pa = await bike(db, a)
    shoes = await me.create_profile(me.ProfileIn(name="Vaporfly", sport="run", kind="shoes"), a, db)
    for pid, who in ((pa["id"], b), (shoes["id"], a), (99999, a)):
        with pytest.raises(HTTPException) as exc:
            await api.add_cda(pid, body(), None, who, db)
        assert exc.value.status_code == 404


async def test_deleting_the_bike_deletes_what_was_measured_on_it(make_user, db):
    u = await make_user(tier=TIER_FULL)
    p = await bike(db, u)
    await api.add_cda(p["id"], body(), None, u, db)
    await me.delete_profile(p["id"], u, db)
    p2 = await bike(db, u, "New")
    assert (await listed(db, u, p2["id"]))["cda"] == []
    from sqlalchemy import func, select

    from app.models.cda import CdaMeasurement
    assert await db.scalar(select(func.count(CdaMeasurement.id))) == 0


# --------------------------------------------------------------------------
# The coach
# --------------------------------------------------------------------------
async def _linked(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    ath_user = await make_user(tier=TIER_FULL)
    a = await roster.create_athlete(roster.AthleteIn(name="Tom"), coach, db)
    await coach_link.create_coach_link(coach_link.LinkIn(code=a["upload_token"], consent=True), ath_user, db)
    return coach, ath_user, a


async def test_a_coach_writes_on_a_linked_athletes_setup_stamped_and_deletable(make_user, db):
    coach, ath, a = await _linked(make_user, db)
    p = await bike(db, ath, "P5")
    await api.add_cda(p["id"], body(0.262, source="windsock", protocol="single_ride"), a["id"], coach, db)
    seen = (await listed(db, ath, p["id"]))["cda"][0]
    assert seen["entered_by_coach"] == coach_link.NO_COACH_NAME and seen["can_delete"]
    # The coach sees it on the athlete's setup and in the Squad column.
    setups = (await api.list_athlete_setups(a["id"], coach, db))["setups"]
    assert setups[0]["whose"] == "athlete" and setups[0]["cda"][0]["value_m2"] == 0.262
    squad = (await roster.list_athletes(coach, db))["athletes"][0]["cda_latest"]
    assert squad["value_m2"] == 0.262 and squad["profile_name"] == "P5"
    # The athlete deletes the coach's row.
    await api.delete_cda(p["id"], seen["id"], None, ath, db)
    assert (await listed(db, ath, p["id"]))["cda"] == []


async def test_a_coach_deletes_only_what_the_coach_typed(make_user, db):
    coach, ath, a = await _linked(make_user, db)
    p = await bike(db, ath, "P5")
    own = await api.add_cda(p["id"], body(0.240), None, ath, db)
    mine = await api.add_cda(p["id"], body(0.250), a["id"], coach, db)
    with pytest.raises(HTTPException) as exc:
        await api.delete_cda(p["id"], own["id"], a["id"], coach, db)
    assert exc.value.status_code == 404
    await api.delete_cda(p["id"], mine["id"], a["id"], coach, db)


async def test_unlinked_the_coach_can_neither_write_nor_see(make_user, db):
    coach, ath, a = await _linked(make_user, db)
    p = await bike(db, ath, "P5")
    await coach_link.revoke_coach_link(ath, db)
    with pytest.raises(HTTPException) as exc:
        await api.add_cda(p["id"], body(), a["id"], coach, db)
    assert exc.value.status_code == 404
    assert (await api.list_athlete_setups(a["id"], coach, db))["setups"] == []


async def test_another_coach_and_a_non_coach_are_not_found(make_user, db):
    coach, ath, a = await _linked(make_user, db)
    p = await bike(db, ath, "P5")
    for who in (await make_user(tier=TIER_COACH), await make_user(tier=TIER_FULL)):
        with pytest.raises(HTTPException) as exc:
            await api.add_cda(p["id"], body(), a["id"], who, db)
        assert exc.value.status_code == 404


async def test_a_roster_athlete_without_an_account_keeps_setups_on_the_coachs(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await roster.create_athlete(roster.AthleteIn(name="Mia"), coach, db)
    p = await bike(db, coach, "Mia's Cervelo", athlete_id=a["id"])
    await api.add_cda(p["id"], body(0.231), a["id"], coach, db)
    setups = (await api.list_athlete_setups(a["id"], coach, db))["setups"]
    assert setups[0]["whose"] == "coach" and setups[0]["cda"][0]["entered_by_coach"] is None
    assert (await roster.list_athletes(coach, db))["athletes"][0]["cda_latest"]["value_m2"] == 0.231


async def test_an_athlete_with_nothing_measured_shows_no_number(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    await roster.create_athlete(roster.AthleteIn(name="Sara"), coach, db)
    assert (await roster.list_athletes(coach, db))["athletes"][0]["cda_latest"] is None


# --------------------------------------------------------------------------
# Research consent: off unless the owner says yes
# --------------------------------------------------------------------------
async def test_research_is_off_until_switched_on_and_back_off(make_user, db):
    u = await make_user(tier=TIER_FULL, email="rider1@gmail.com")
    assert u.research_opt_in is False and not research_eligible(u)
    out = await auth.set_research(auth.ResearchBody(opt_in=True), u, db)
    assert out.research_opt_in and u.research_changed_at is not None and research_eligible(u)
    out = await auth.set_research(auth.ResearchBody(opt_in=False), u, db)
    assert not out.research_opt_in and not research_eligible(u)


async def test_a_roster_athletes_setup_on_the_coachs_account_is_never_eligible(make_user, db):
    coach = await make_user(tier=TIER_COACH, research_opt_in=True, email="coach1@gmail.com")
    a = await roster.create_athlete(roster.AthleteIn(name="Mia"), coach, db)
    p = await bike(db, coach, "Mia's", athlete_id=a["id"])
    from app.models.profile import Profile
    row = await db.get(Profile, p["id"])
    assert not profile_research_eligible(row, coach)   # Mia never had the switch
    own = await db.get(Profile, (await bike(db, coach, "Mine"))["id"])
    assert profile_research_eligible(own, coach)


async def test_staff_and_test_accounts_are_not_research_data(make_user):
    assert not research_eligible(await make_user(tier=TIER_ADMIN, research_opt_in=True, email="a@getflapp.com"))
    assert not research_eligible(await make_user(tier=TIER_STARTER, research_opt_in=True, email="x@example.com"))
    assert research_eligible(await make_user(tier=TIER_STARTER, research_opt_in=True, email="rider@gmail.com"))


def test_the_privacy_policy_says_what_the_switch_does():
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "privacy.html").read_text(encoding="utf-8")
    assert 'id="research-consent"' in html and "measured CdA" in html
