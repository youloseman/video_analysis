"""The roster (``/me/athletes``) and the tier boundary around it.

An athlete is a label the coach owns on their own history. The tests worth
writing are the boundary ones: only the Coach plan keeps a roster; the limit
holds; an analysis names an athlete only if the athlete is the caller's own;
deleting unfiles rather than removes; the history cap grows with the tier.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import athletes as api
from app.api import me
from app.models.analysis import Analysis
from app.models.athlete import Athlete
from app.models.profile import Profile
from app.models.user import (
    TIER_ADMIN,
    TIER_COACH,
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
    athlete_limit,
    has_coach,
    has_full,
    history_cap,
    profile_limit,
    tier_limit,
)


async def add(db, user, name="Andrew", **kw):
    return await api.create_athlete(api.AthleteIn(name=name, **kw), user, db)


async def save(db, user, *, client_id="h1", athlete_id=None, score=None, at=1):
    entry = {"id": client_id, "at": at, "score": score}
    if athlete_id is not None:
        entry["athleteId"] = athlete_id
    await me._upsert(db, user, entry)
    await db.commit()


# --------------------------------------------------------------------------
# The tier
# --------------------------------------------------------------------------
def test_coach_sits_above_full():
    assert has_full(TIER_COACH) and has_coach(TIER_COACH)
    assert has_full(TIER_FULL) and not has_coach(TIER_FULL)
    assert has_coach(TIER_ADMIN)
    assert tier_limit(TIER_COACH)[0] > tier_limit(TIER_FULL)[0]
    assert profile_limit(TIER_COACH) > profile_limit(TIER_FULL)


@pytest.mark.parametrize(
    "tier,expected",
    [(TIER_STARTER, 0), (TIER_ENTHUSIAST, 0), (TIER_FULL, 0), (TIER_COACH, 15)],
)
def test_only_the_coach_plan_keeps_a_roster(tier, expected):
    assert athlete_limit(tier) == expected


def test_the_history_cap_grows_with_the_roster():
    assert history_cap(TIER_FULL) == 100
    assert history_cap(TIER_COACH) == 1000
    assert history_cap("platinum_unicorn") == 100


async def test_full_is_told_which_plan_has_a_roster(make_user, db):
    full = await make_user(tier=TIER_FULL)
    listing = await api.list_athletes(full, db)
    assert listing == {"allowed": False, "athletes": [], "limit": 0, "used": 0}
    with pytest.raises(HTTPException) as e:
        await add(db, full)
    assert e.value.status_code == 402 and "Coach plan" in e.value.detail


# --------------------------------------------------------------------------
# The roster
# --------------------------------------------------------------------------
async def test_an_athlete_round_trips_with_an_intake_token(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach, name="  Andrew   Solomon ", email="a@example.com",
                  sport="bike", note=" Worlds   qualifier ")
    assert a["name"] == "Andrew Solomon" and a["sport"] == "bike"
    assert a["note"] == "Worlds qualifier" and a["email"] == "a@example.com"
    assert len(a["upload_token"]) >= 20 and a["analyses"] == 0
    listing = await api.list_athletes(coach, db)
    assert listing["allowed"] and listing["used"] == 1 and listing["limit"] == 15
    assert listing["athletes"][0]["id"] == a["id"]


def test_a_typo_in_the_email_is_refused_without_a_validator_library():
    with pytest.raises(ValueError):
        api.AthleteIn(name="A", email="not-an-email")
    assert api.AthleteIn(name="A", email="  Andy@Example.com ").email == "andy@example.com"
    assert api.AthleteIn(name="A", email="").email is None


async def test_an_unknown_sport_is_the_triathlete(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach, sport="curling")
    assert a["sport"] == "both"


async def test_the_limit_counts_live_athletes_only(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    rows = [await add(db, coach, name=f"A{i}") for i in range(15)]
    with pytest.raises(HTTPException) as e:
        await add(db, coach, name="One more")
    assert e.value.status_code == 402
    await api.update_athlete(rows[0]["id"], api.AthletePatch(archived=True), coach, db)
    again = await add(db, coach, name="One more")
    assert again["name"] == "One more"
    # Un-archiving past the limit is refused rather than silently allowed.
    with pytest.raises(HTTPException):
        await api.update_athlete(rows[0]["id"], api.AthletePatch(archived=False), coach, db)


async def test_rotating_the_token_kills_the_old_link(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    b = await api.update_athlete(a["id"], api.AthletePatch(rotate_token=True), coach, db)
    assert b["upload_token"] != a["upload_token"]
    c = await api.update_athlete(a["id"], api.AthletePatch(name="Andy", email=None), coach, db)
    assert c["name"] == "Andy" and c["upload_token"] == b["upload_token"]


async def test_somebody_elses_athlete_is_not_found(make_user, db):
    c1 = await make_user(tier=TIER_COACH)
    c2 = await make_user(tier=TIER_COACH)
    a = await add(db, c1)
    with pytest.raises(HTTPException) as e:
        await api.update_athlete(a["id"], api.AthletePatch(name="Mine"), c2, db)
    assert e.value.status_code == 404


# --------------------------------------------------------------------------
# Filing analyses under an athlete
# --------------------------------------------------------------------------
async def test_an_entry_files_under_the_coachs_own_athlete_only(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    other = await make_user(tier=TIER_COACH)
    mine = await add(db, coach)
    theirs = await add(db, other, name="Not mine")
    await save(db, coach, client_id="h1", athlete_id=mine["id"], score=81)
    await save(db, coach, client_id="h2", athlete_id=theirs["id"], score=70)
    rows = {
        r.client_id: r for r in (
            await db.execute(select(Analysis).where(Analysis.user_id == coach.id))
        ).scalars().all()
    }
    assert rows["h1"].athlete_id == mine["id"]
    assert rows["h2"].athlete_id is None


async def test_the_listing_carries_count_last_date_and_score(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athlete_id=a["id"], score=70, at=100)
    await save(db, coach, client_id="h2", athlete_id=a["id"], score=84, at=200)
    row = (await api.list_athletes(coach, db))["athletes"][0]
    assert row["analyses"] == 2 and row["last_at_ms"] == 200 and row["last_score"] == 84


async def test_history_filters_by_athlete_and_the_thin_entry_names_them(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    b = await add(db, coach, name="Matt")
    await save(db, coach, client_id="h1", athlete_id=a["id"])
    await save(db, coach, client_id="h2", athlete_id=b["id"])
    await save(db, coach, client_id="h3")
    only_a = await me.list_analyses(coach, db, athlete_id=a["id"])
    assert [e["id"] for e in only_a] == ["h1"]
    assert only_a[0]["athleteId"] == a["id"]
    everything = await me.list_analyses(coach, db)
    assert {e["id"]: e["athleteId"] for e in everything} == {
        "h1": a["id"], "h2": b["id"], "h3": None,
    }


async def test_a_resave_without_the_athlete_keeps_the_filing(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athlete_id=a["id"])
    await save(db, coach, client_id="h1")   # the thin list carries no athleteId
    row = (await db.execute(select(Analysis).where(Analysis.client_id == "h1"))).scalar_one()
    assert row.athlete_id == a["id"]


async def test_deleting_an_athlete_unfiles_rather_than_removes(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athlete_id=a["id"])
    prof = await me.create_profile(
        me.ProfileIn(name="TT", sport="bike", kind="tt", athlete_id=a["id"]), coach, db,
    )
    assert prof["athlete_id"] == a["id"]
    await api.delete_athlete(a["id"], coach, db)
    assert (await db.execute(select(Athlete))).scalars().all() == []
    row = (await db.execute(select(Analysis).where(Analysis.client_id == "h1"))).scalar_one()
    assert row.athlete_id is None
    p = (await db.execute(select(Profile).where(Profile.id == prof["id"]))).scalar_one()
    assert p.athlete_id is None


async def test_a_profile_for_somebody_elses_athlete_is_the_coachs_own(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    other = await make_user(tier=TIER_COACH)
    theirs = await add(db, other)
    prof = await me.create_profile(
        me.ProfileIn(name="TT", sport="bike", athlete_id=theirs["id"]), coach, db,
    )
    assert prof["athlete_id"] is None
