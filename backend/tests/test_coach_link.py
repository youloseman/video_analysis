"""The coach link (model A) and what it opens.

The consent is the thing: no link without the box; one coach at a time;
the athlete can take it back and the coach's reads stop at once. The
coach reads the athlete's rows at the ATHLETE's access level, writes a note
onto the athlete's entry, and never a bought one. A linked athlete takes no
seat on the roster.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import athletes as roster
from app.api import coach_link as api
from app.api import me
from app.models.analysis import Analysis
from app.models.user import TIER_COACH, TIER_ENTHUSIAST, TIER_STARTER


async def add(db, coach, name="Andy", **kw):
    return await roster.create_athlete(roster.AthleteIn(name=name, **kw), coach, db)


async def save(db, user, *, client_id="h1", score=None, at=1, result=None, **extra):
    entry = {"id": client_id, "at": at, "score": score, **extra}
    await me._upsert(db, user, entry)
    if result is not None:
        row = (await db.execute(select(Analysis).where(
            Analysis.user_id == user.id, Analysis.client_id == client_id))).scalar_one()
        row.result = result
    await db.commit()


async def link(db, athlete_user, code, consent=True):
    return await api.create_coach_link(api.LinkIn(code=code, consent=consent), athlete_user, db)


# --------------------------------------------------------------------------
# Making and breaking the link
# --------------------------------------------------------------------------
async def test_no_box_no_link(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    with pytest.raises(HTTPException) as e:
        await link(db, andy, a["upload_token"], consent=False)
    assert e.value.status_code == 400
    assert (await api.get_coach_link(andy, db))["coach"] is None


async def test_the_code_links_and_the_coach_is_named(make_user, db):
    coach = await make_user(tier=TIER_COACH, email="john@example.com")
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    out = await link(db, andy, a["upload_token"])
    assert out["coach"]["name"] == "john" and out["coach"]["athlete_name"] == "Andy"
    assert "switch this off" in out["consent_text"]
    # Idempotent: the same code again is the same link.
    again = await link(db, andy, a["upload_token"])
    assert again["coach"]["since_ms"] == out["coach"]["since_ms"]
    listing = await roster.list_athletes(coach, db)
    assert listing["athletes"][0]["linked"] is True
    # A linked athlete takes no seat.
    assert listing["used"] == 0


async def test_wrong_code_own_roster_and_second_coach_are_refused(make_user, db):
    c1 = await make_user(tier=TIER_COACH)
    c2 = await make_user(tier=TIER_COACH)
    andy = await make_user()
    a1 = await add(db, c1)
    a2 = await add(db, c2, name="Andy at c2")
    with pytest.raises(HTTPException) as e:
        await link(db, andy, "nope-nope-nope")
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await link(db, c1, a1["upload_token"])
    assert e.value.status_code == 400
    await link(db, andy, a1["upload_token"])
    with pytest.raises(HTTPException) as e:
        await link(db, andy, a2["upload_token"])
    assert e.value.status_code == 409
    # Somebody else cannot take a code that is already in use.
    other = await make_user()
    with pytest.raises(HTTPException) as e:
        await link(db, other, a1["upload_token"])
    assert e.value.status_code == 409


async def test_a_coach_whose_plan_ended_cannot_be_linked(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    coach.tier = TIER_STARTER
    await db.commit()
    andy = await make_user()
    with pytest.raises(HTTPException) as e:
        await link(db, andy, a["upload_token"])
    assert e.value.status_code == 404


async def test_unlinking_stops_the_coachs_reads_at_once(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    await save(db, andy, client_id="a1", score=77)
    await link(db, andy, a["upload_token"])
    rows = await api.athlete_analyses(a["id"], coach, db)
    assert [r["id"] for r in rows] == ["a1"] and rows[0]["linkedAthleteId"] == a["id"]
    await api.revoke_coach_link(andy, db)
    assert (await api.get_coach_link(andy, db))["coach"] is None
    # The roster row is back to a model-B label with no clips of its own.
    assert await api.athlete_analyses(a["id"], coach, db) == []
    assert (await roster.list_athletes(coach, db))["athletes"][0]["linked"] is False
    # And the media readers refuse the athlete's rows now.
    with pytest.raises(HTTPException) as e:
        await me.get_keyframe("a1", athlete_id=a["id"], user=coach, db=db)
    assert e.value.status_code == 404


# --------------------------------------------------------------------------
# What the coach reads
# --------------------------------------------------------------------------
async def test_the_coach_reads_at_the_athletes_access_level(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    free_andy = await make_user(tier=TIER_STARTER)
    a = await add(db, coach)
    await link(db, free_andy, a["upload_token"])
    await save(db, free_andy, client_id="a1", score=70,
               result={"sport_type": "run", "technique_score": 70, "kinogram_base64": "k", "timeline": [1]})
    # A Coach-plan reader would see everything on their own rows; on a free
    # athlete's row they see what the athlete sees -- the teaser.
    with pytest.raises(HTTPException) as e:
        await me.get_kinogram("a1", athlete_id=a["id"], user=coach, db=db)
    assert e.value.status_code == 402
    out = await me.get_analysis_result("a1", athlete_id=a["id"], user=coach, db=db)
    assert out["access"] != "full"
    # The thin list says so too.
    rows = await api.athlete_analyses(a["id"], coach, db)
    assert rows[0]["access"] != "full"


async def test_a_paying_athletes_rows_open_in_full_for_the_coach(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="a1", score=80,
               result={"sport_type": "bike", "technique_score": 80, "kinogram_base64": "kino", "timeline": [1, 2]})
    assert (await me.get_kinogram("a1", athlete_id=a["id"], user=coach, db=db))["kinogram"] == "kino"
    assert (await me.get_timeline("a1", athlete_id=a["id"], user=coach, db=db))["timeline"] == [1, 2]
    assert (await me.get_keyframe("a1", athlete_id=a["id"], user=coach, db=db))["keyframe"] is None


async def test_somebody_elses_athlete_id_is_not_found(make_user, db):
    c1 = await make_user(tier=TIER_COACH)
    c2 = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, c1)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="a1")
    with pytest.raises(HTTPException) as e:
        await me.get_keyframe("a1", athlete_id=a["id"], user=c2, db=db)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await api.athlete_analyses(a["id"], c2, db)
    assert e.value.status_code == 404


async def test_without_an_athlete_id_the_readers_stay_the_callers_own(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    await save(db, coach, client_id="mine", score=90)
    assert (await me.get_keyframe("mine", user=coach, db=db))["keyframe"] is None
    with pytest.raises(HTTPException):
        await me.get_keyframe("mine", athlete_id=999, user=coach, db=db)


async def test_a_model_b_athlete_id_reads_the_coachs_own_rows(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="b1", score=65, athleteId=a["id"])
    rows = await api.athlete_analyses(a["id"], coach, db)
    assert [r["id"] for r in rows] == ["b1"] and "linkedAthleteId" not in rows[0]
    assert (await me.get_keyframe("b1", athlete_id=a["id"], user=coach, db=db))["keyframe"] is None


# --------------------------------------------------------------------------
# The coach's note
# --------------------------------------------------------------------------
async def test_the_note_lands_on_the_athletes_entry_under_the_coachs_name(make_user, db, monkeypatch):
    sent = []
    monkeypatch.setattr(api, "_mail_note", lambda athlete, name, row: sent.append((athlete.email, name)))
    coach = await make_user(tier=TIER_COACH, email="coach.john@example.com")
    andy = await make_user(tier=TIER_ENTHUSIAST, email="andy@example.com")
    a = await add(db, coach)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="a1", score=80)
    out = await api.write_note(a["id"], "a1", api.NoteIn(note="  Drop the heel earlier. "), coach, db)
    assert out == {"coachNote": "Drop the heel earlier.", "coachName": "coach.john", "coachBio": ""}
    row = (await db.execute(select(Analysis).where(Analysis.client_id == "a1"))).scalar_one()
    assert row.data["coachNote"] == "Drop the heel earlier." and row.data["coachName"] == "coach.john"
    assert sent == [("andy@example.com", "coach.john")]
    # Editing an existing note does not mail again.
    await api.write_note(a["id"], "a1", api.NoteIn(note="Drop the heel a touch earlier."), coach, db)
    assert len(sent) == 1
    # The athlete's own listing carries it, read-only.
    mine = await me.list_analyses(andy, db)
    assert mine[0]["coachNote"].startswith("Drop the heel")


async def test_bought_coach_notes_are_never_overwritten(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="b1", athleteId=a["id"], coachNote="Bogdan's", coachName="Bogdan", coachOrderId=7)
    with pytest.raises(HTTPException) as e:
        await api.write_note(a["id"], "b1", api.NoteIn(note="Mine"), coach, db)
    assert e.value.status_code == 409


async def test_a_note_on_a_model_b_entry_needs_that_athletes_filing(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    b = await add(db, coach, name="Matt")
    await save(db, coach, client_id="b1", athleteId=a["id"])
    with pytest.raises(HTTPException) as e:
        await api.write_note(b["id"], "b1", api.NoteIn(note="Wrong athlete"), coach, db)
    assert e.value.status_code == 404
    out = await api.write_note(a["id"], "b1", api.NoteIn(note="Right one"), coach, db)
    assert out["coachNote"] == "Right one"
