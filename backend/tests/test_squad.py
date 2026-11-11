"""Races, clip reviews and groups (``api/squad.py``).

The boundaries are the tests worth having: the Coach plan only; nothing of
another coach's roster is reachable; "unseen" starts on the day the mark
shipped; a note marks its clip; the mark is the coach's and never lands on
the athlete's entry; a group or an athlete leaving takes only its own rows.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import func, select

from app.api import athletes as roster
from app.api import coach_link as link_api
from app.api import me
from app.api import squad as api
from app.models.analysis import Analysis
from app.models.squad import (
    MAX_GROUPS,
    MAX_RACES_PER_ATHLETE,
    AthleteGroupMember,
    AthleteRace,
    ClipReview,
)
from app.models.user import TIER_COACH, TIER_ENTHUSIAST, TIER_FULL

NEW = api.UNSEEN_SINCE_MS + 1000   # a clip filmed after the mark existed
OLD = api.UNSEEN_SINCE_MS - 1000   # one from before it


async def add(db, coach, name="Andy", **kw):
    return await roster.create_athlete(roster.AthleteIn(name=name, **kw), coach, db)


async def save(db, user, *, client_id="h1", at=NEW, **extra):
    await me._upsert(db, user, {"id": client_id, "at": at, **extra})
    await db.commit()


async def link(db, athlete_user, code):
    return await link_api.create_coach_link(
        link_api.LinkIn(code=code, consent=True), athlete_user, db,
    )


async def listed(db, coach, athlete_id):
    return next(a for a in (await roster.list_athletes(coach, db))["athletes"] if a["id"] == athlete_id)


async def count(db, model):
    return int(await db.scalar(select(func.count()).select_from(model)) or 0)


# --------------------------------------------------------------------------
# The plan, and other people's rosters
# --------------------------------------------------------------------------
async def test_only_the_coach_plan_keeps_races_reviews_and_groups(make_user, db):
    full = await make_user(tier=TIER_FULL)
    for call in (
        lambda: api.add_race(1, api.RaceIn(name="Kona", race_on="2026-12-13"), full, db),
        lambda: api.mark_clip_reviewed(1, "h1", full, db),
        lambda: api.mark_all_reviewed(1, full, db),
        lambda: api.list_groups(full, db),
        lambda: api.create_group(api.GroupIn(name="Long course"), full, db),
        lambda: api.set_athlete_groups(1, api.GroupMembersIn(group_ids=[]), full, db),
    ):
        with pytest.raises(HTTPException) as e:
            await call()
        assert e.value.status_code == 402


async def test_another_coachs_athlete_is_not_found(make_user, db):
    john = await make_user(tier=TIER_COACH)
    mike = await make_user(tier=TIER_COACH)
    a = await add(db, john)
    await save(db, john, athleteId=a["id"])
    race = await api.add_race(a["id"], api.RaceIn(name="Kona", race_on="2026-12-13"), john, db)
    group = await api.create_group(api.GroupIn(name="Long course"), john, db)
    for call in (
        lambda: api.add_race(a["id"], api.RaceIn(name="X", race_on="2026-12-13"), mike, db),
        lambda: api.delete_race(a["id"], race["id"], mike, db),
        lambda: api.mark_clip_reviewed(a["id"], "h1", mike, db),
        lambda: api.set_athlete_groups(a["id"], api.GroupMembersIn(group_ids=[]), mike, db),
        lambda: api.rename_group(group["id"], api.GroupIn(name="Mine now"), mike, db),
        lambda: api.delete_group(group["id"], mike, db),
    ):
        with pytest.raises(HTTPException) as e:
            await call()
        assert e.value.status_code == 404
    assert (await api.list_groups(mike, db))["groups"] == []


# --------------------------------------------------------------------------
# Races
# --------------------------------------------------------------------------
async def test_the_next_race_is_the_nearest_one_still_ahead(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await api.add_race(a["id"], api.RaceIn(name="Worlds", race_on="2027-09-12", priority="a"), coach, db)
    await api.add_race(a["id"], api.RaceIn(name="Spring tune-up", race_on="2026-04-05", priority="C"), coach, db)
    await api.add_race(a["id"], api.RaceIn(name="70.3 Florida", race_on="2026-12-13"), coach, db)
    extras = (await api.roster_extras(db, coach, [await roster._owned(db, coach, a["id"])], today="2026-11-11"))[a["id"]]
    assert [r["name"] for r in extras["races"]] == ["Spring tune-up", "70.3 Florida", "Worlds"]
    assert extras["races"][0]["past"] is True and extras["races"][0]["priority"] == "C"
    assert extras["next_race"]["name"] == "70.3 Florida"
    assert extras["next_race"]["days_to_go"] == 32
    # Race day itself is still ahead; the day after it is not.
    on_the_day = await api.roster_extras(db, coach, [await roster._owned(db, coach, a["id"])], today="2026-12-13")
    assert on_the_day[a["id"]]["next_race"]["days_to_go"] == 0
    after = await api.roster_extras(db, coach, [await roster._owned(db, coach, a["id"])], today="2026-12-14")
    assert after[a["id"]]["next_race"]["name"] == "Worlds"


async def test_the_roster_listing_carries_the_races(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    assert (await listed(db, coach, a["id"]))["next_race"] is None
    await api.add_race(a["id"], api.RaceIn(name="Far away", race_on="2099-06-01"), coach, db)
    row = await listed(db, coach, a["id"])
    assert row["next_race"]["name"] == "Far away" and len(row["races"]) == 1
    assert row["group_ids"] == [] and row["unreviewed"] == 0


async def test_a_race_can_be_moved_renamed_and_removed(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    r = await api.add_race(a["id"], api.RaceIn(name="Kona", race_on="2026-10-10"), coach, db)
    out = await api.update_race(a["id"], r["id"], api.RacePatch(race_on="2026-10-17", priority="b"), coach, db)
    assert out["race_on"] == "2026-10-17" and out["priority"] == "B" and out["name"] == "Kona"
    await api.delete_race(a["id"], r["id"], coach, db)
    assert await count(db, AthleteRace) == 0


@pytest.mark.parametrize("bad", ["next spring", "13/12/2026", "2026-13-40", "1850-01-01", ""])
def test_a_race_date_that_is_not_a_date_is_refused(bad):
    with pytest.raises(ValidationError):
        api.RaceIn(name="Kona", race_on=bad)


def test_a_race_needs_a_name_and_an_unknown_priority_falls_back():
    with pytest.raises(ValidationError):
        api.RaceIn(name="   ", race_on="2026-12-13")
    assert api.RaceIn(name="Kona", race_on="2026-12-13", priority="Z").priority == "A"


async def test_an_athlete_keeps_a_dozen_races(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    for i in range(MAX_RACES_PER_ATHLETE):
        await api.add_race(a["id"], api.RaceIn(name=f"Race {i}", race_on="2027-01-01"), coach, db)
    with pytest.raises(HTTPException) as e:
        await api.add_race(a["id"], api.RaceIn(name="One more", race_on="2027-01-01"), coach, db)
    assert e.value.status_code == 409


# --------------------------------------------------------------------------
# Clip reviews
# --------------------------------------------------------------------------
async def test_a_new_clip_is_unseen_until_the_coach_marks_it(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athleteId=a["id"])
    await save(db, coach, client_id="h2", athleteId=a["id"])
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 2
    out = await api.mark_clip_reviewed(a["id"], "h1", coach, db)
    assert out["reviewedAt"] > 0
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 1
    # Marking twice keeps the first stamp and adds no second row.
    again = await api.mark_clip_reviewed(a["id"], "h1", coach, db)
    assert again["reviewedAt"] == out["reviewedAt"] and await count(db, ClipReview) == 1
    await api.unmark_clip_reviewed(a["id"], "h1", coach, db)
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 2


async def test_clips_from_before_the_mark_existed_are_not_an_inbox(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="old", athleteId=a["id"], at=OLD)
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0
    await save(db, coach, client_id="new", athleteId=a["id"], at=NEW)
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 1


async def test_the_coachs_own_clips_are_not_counted(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="mine")           # filed under nobody
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0


async def test_a_linked_athletes_clip_is_marked_without_touching_their_entry(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="ride", sport="bike")
    await save(db, andy, client_id="pool", sport="swim")     # a different product
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 1
    entries = await link_api.athlete_analyses(a["id"], coach, db)
    assert [e["reviewedAt"] for e in entries] == [None]
    await api.mark_clip_reviewed(a["id"], "ride", coach, db)
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0
    entries = await link_api.athlete_analyses(a["id"], coach, db)
    assert entries[0]["reviewedAt"] > 0
    # The mark is the coach's row, not a field on the athlete's stored entry.
    stored = (await db.execute(select(Analysis).where(Analysis.user_id == andy.id, Analysis.client_id == "ride"))).scalar_one()
    assert "reviewedAt" not in stored.data
    assert "reviewedAt" not in (await me.list_analyses(andy, db))[0]


async def test_a_revoked_link_closes_the_athletes_clips(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="ride", sport="bike")
    await link_api.revoke_coach_link(andy, db)
    with pytest.raises(HTTPException) as e:
        await api.mark_clip_reviewed(a["id"], "ride", coach, db)
    assert e.value.status_code == 404
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0


async def test_two_coaches_do_not_share_a_mark(make_user, db):
    john = await make_user(tier=TIER_COACH)
    mike = await make_user(tier=TIER_COACH)
    a = await add(db, john)
    b = await add(db, mike, name="Bea")
    await save(db, john, client_id="h1", athleteId=a["id"])
    await save(db, mike, client_id="h1", athleteId=b["id"])
    await api.mark_clip_reviewed(a["id"], "h1", john, db)
    assert (await listed(db, john, a["id"]))["unreviewed"] == 0
    assert (await listed(db, mike, b["id"]))["unreviewed"] == 1


async def test_a_note_on_a_linked_athletes_report_marks_it(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="ride", sport="bike")
    await link_api.write_note(a["id"], "ride", link_api.NoteIn(note="Drop the pads 5 mm."), coach, db)
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0


async def test_a_note_on_the_coachs_own_clip_marks_it(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athleteId=a["id"])
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 1
    await save(db, coach, client_id="h1", athleteId=a["id"], coachNote="Shorter stride.")
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0
    # ...and saving it again does not stamp a second time.
    await save(db, coach, client_id="h1", athleteId=a["id"], coachNote="Shorter stride.")
    assert await count(db, ClipReview) == 1


async def test_everything_unseen_is_marked_in_one_go(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    andy = await make_user(tier=TIER_ENTHUSIAST)
    a = await add(db, coach)
    await link(db, andy, a["upload_token"])
    await save(db, andy, client_id="theirs", sport="bike")
    await save(db, coach, client_id="mine1", athleteId=a["id"])
    await save(db, coach, client_id="mine2", athleteId=a["id"])
    await api.mark_clip_reviewed(a["id"], "mine1", coach, db)
    assert (await api.mark_all_reviewed(a["id"], coach, db)) == {"marked": 2}
    assert (await listed(db, coach, a["id"]))["unreviewed"] == 0
    assert (await api.mark_all_reviewed(a["id"], coach, db)) == {"marked": 0}


async def test_the_coachs_history_says_which_filed_clips_were_seen(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="filed", athleteId=a["id"], at=NEW + 1)
    await save(db, coach, client_id="own", at=NEW)
    await api.mark_clip_reviewed(a["id"], "filed", coach, db)
    filed, own = await me.list_analyses(coach, db)
    assert filed["id"] == "filed" and filed["reviewedAt"] > 0
    assert "reviewedAt" not in own


async def test_a_mark_sent_back_by_the_client_is_not_stored(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athleteId=a["id"], reviewedAt=123)
    stored = (await db.execute(select(Analysis).where(Analysis.client_id == "h1"))).scalar_one()
    assert "reviewedAt" not in stored.data
    assert (await me.list_analyses(coach, db))[0]["reviewedAt"] is None


# --------------------------------------------------------------------------
# Groups
# --------------------------------------------------------------------------
async def test_an_athlete_can_sit_in_several_groups(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    b = await add(db, coach, name="Bea")
    long = await api.create_group(api.GroupIn(name="  Long   course "), coach, db)
    juniors = await api.create_group(api.GroupIn(name="Juniors"), coach, db)
    assert long["name"] == "Long course"
    await api.set_athlete_groups(a["id"], api.GroupMembersIn(group_ids=[long["id"], juniors["id"], long["id"]]), coach, db)
    await api.set_athlete_groups(b["id"], api.GroupMembersIn(group_ids=[long["id"]]), coach, db)
    groups = {g["name"]: g for g in (await api.list_groups(coach, db))["groups"]}
    assert groups["Long course"]["athlete_ids"] == [a["id"], b["id"]] and groups["Long course"]["athletes"] == 2
    assert groups["Juniors"]["athlete_ids"] == [a["id"]]
    assert (await listed(db, coach, a["id"]))["group_ids"] == sorted([long["id"], juniors["id"]])
    # The set is replaced as a whole.
    await api.set_athlete_groups(a["id"], api.GroupMembersIn(group_ids=[]), coach, db)
    assert (await listed(db, coach, a["id"]))["group_ids"] == []


async def test_two_groups_cannot_share_a_name(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    other = await make_user(tier=TIER_COACH)
    g = await api.create_group(api.GroupIn(name="Long course"), coach, db)
    with pytest.raises(HTTPException) as e:
        await api.create_group(api.GroupIn(name="long COURSE"), coach, db)
    assert e.value.status_code == 409
    h = await api.create_group(api.GroupIn(name="Juniors"), coach, db)
    with pytest.raises(HTTPException) as e:
        await api.rename_group(h["id"], api.GroupIn(name="Long course"), coach, db)
    assert e.value.status_code == 409
    # Renaming to its own name is not a clash; another coach may reuse it.
    assert (await api.rename_group(g["id"], api.GroupIn(name="Long Course"), coach, db))["name"] == "Long Course"
    assert (await api.create_group(api.GroupIn(name="Long course"), other, db))["athletes"] == 0


async def test_a_group_that_is_not_the_coachs_saves_nothing(make_user, db):
    john = await make_user(tier=TIER_COACH)
    mike = await make_user(tier=TIER_COACH)
    a = await add(db, john)
    mine = await api.create_group(api.GroupIn(name="Mine"), john, db)
    theirs = await api.create_group(api.GroupIn(name="Theirs"), mike, db)
    with pytest.raises(HTTPException) as e:
        await api.set_athlete_groups(a["id"], api.GroupMembersIn(group_ids=[mine["id"], theirs["id"]]), john, db)
    assert e.value.status_code == 404
    assert (await listed(db, john, a["id"]))["group_ids"] == []


async def test_removing_a_group_keeps_its_athletes(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    g = await api.create_group(api.GroupIn(name="Long course"), coach, db)
    await api.set_athlete_groups(a["id"], api.GroupMembersIn(group_ids=[g["id"]]), coach, db)
    await api.delete_group(g["id"], coach, db)
    assert (await api.list_groups(coach, db))["groups"] == []
    assert await count(db, AthleteGroupMember) == 0
    assert (await listed(db, coach, a["id"]))["group_ids"] == []


async def test_a_squad_keeps_a_limited_number_of_groups(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    for i in range(MAX_GROUPS):
        await api.create_group(api.GroupIn(name=f"Group {i}"), coach, db)
    with pytest.raises(HTTPException) as e:
        await api.create_group(api.GroupIn(name="One more"), coach, db)
    assert e.value.status_code == 409


async def test_races_and_group_seats_leave_with_the_athlete(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    a = await add(db, coach)
    await save(db, coach, client_id="h1", athleteId=a["id"])
    await api.mark_clip_reviewed(a["id"], "h1", coach, db)
    await api.add_race(a["id"], api.RaceIn(name="Kona", race_on="2026-12-13"), coach, db)
    g = await api.create_group(api.GroupIn(name="Long course"), coach, db)
    await api.set_athlete_groups(a["id"], api.GroupMembersIn(group_ids=[g["id"]]), coach, db)
    await roster.delete_athlete(a["id"], coach, db)
    assert await count(db, AthleteRace) == 0 and await count(db, AthleteGroupMember) == 0
    # The group itself stays, empty; the analysis stays, unfiled, with its mark.
    assert (await api.list_groups(coach, db))["groups"][0]["athletes"] == 0
    assert await count(db, Analysis) == 1 and await count(db, ClipReview) == 1
