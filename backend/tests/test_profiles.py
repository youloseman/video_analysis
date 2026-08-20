"""Equipment profiles, and the tier boundary that runs through them.

A rider on a road bike and the same rider on a TT bike produce two correct sets
of angles; averaged into one history they describe nobody. Filing an analysis
under a setup is what makes a trend line mean something -- and keeping several
setups is what makes a comparison BETWEEN them possible, which is the thing a
single bought report structurally cannot be.

So this is where the Full tier actually earns its price, and the tests worth
writing are the boundary ones: the limit, and the fact that a profile is a
label on somebody's history rather than an owner of it.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import me
from app.models.analysis import Analysis
from app.models.profile import Profile
from app.models.user import (
    TIER_ENTHUSIAST,
    TIER_FULL,
    TIER_STARTER,
    profile_limit,
)


async def make_profile(db, user, *, name="Road", sport="bike", kind="road"):
    return await me.create_profile(
        me.ProfileIn(name=name, sport=sport, kind=kind), user, db,
    )


async def save(db, user, *, client_id="h1", profile_id=None, job_id=None):
    entry = {"id": client_id, "at": 1}
    if profile_id is not None:
        entry["profileId"] = profile_id
    if job_id:
        entry["jobId"] = job_id
    await me._upsert(db, user, entry)
    await db.commit()


# --------------------------------------------------------------------------
# The limit is the product boundary
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "tier,expected", [(TIER_STARTER, 1), (TIER_ENTHUSIAST, 1), (TIER_FULL, 10)],
)
def test_the_limit_is_where_full_earns_its_price(tier, expected):
    assert profile_limit(tier) == expected


def test_an_unknown_tier_gets_the_free_limit():
    assert profile_limit("platinum_unicorn") == profile_limit(TIER_STARTER)


async def test_a_free_account_keeps_one_setup(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await make_profile(db, user, name="Road")
    with pytest.raises(HTTPException) as exc:
        await make_profile(db, user, name="TT")
    # 402, not 403: the client shows a different thing for "you have not paid"
    # than for "you cannot".
    assert exc.value.status_code == 402


async def test_full_keeps_several(db, make_user):
    user = await make_user(tier=TIER_FULL)
    for name in ("Road", "TT", "Trainer"):
        await make_profile(db, user, name=name)
    listing = await me.list_profiles(user, db)
    assert listing["used"] == 3
    assert listing["limit"] == 10


async def test_archiving_frees_a_slot(db, make_user):
    """A retired bike should not hold a paying slot hostage."""
    user = await make_user(tier=TIER_STARTER)
    first = await make_profile(db, user, name="Old bike")
    await me.update_profile(first["id"], me.ProfilePatch(archived=True), user, db)
    second = await make_profile(db, user, name="New bike")
    assert second["id"] != first["id"]


async def test_un_archiving_cannot_smuggle_you_past_the_limit(db, make_user):
    """Retire one, downgrade, then bring it back -- that path has to close, or
    the limit is advisory."""
    user = await make_user(tier=TIER_STARTER)
    old = await make_profile(db, user, name="Old")
    await me.update_profile(old["id"], me.ProfilePatch(archived=True), user, db)
    await make_profile(db, user, name="New")

    with pytest.raises(HTTPException) as exc:
        await me.update_profile(old["id"], me.ProfilePatch(archived=False), user, db)
    assert exc.value.status_code == 402


# --------------------------------------------------------------------------
# Filing analyses
# --------------------------------------------------------------------------
async def test_an_analysis_is_filed_under_its_setup(db, make_user):
    user = await make_user(tier=TIER_FULL)
    road = await make_profile(db, user, name="Road")
    await save(db, user, client_id="h1", profile_id=road["id"])
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.profile_id == road["id"]


async def test_history_can_be_read_one_setup_at_a_time(db, make_user):
    """The whole point: a trend line for the TT bike, not for both averaged."""
    user = await make_user(tier=TIER_FULL)
    road = await make_profile(db, user, name="Road")
    tt = await make_profile(db, user, name="TT")
    await save(db, user, client_id="h1", profile_id=road["id"])
    await save(db, user, client_id="h2", profile_id=tt["id"])

    only_tt = await me.list_analyses(user, db, profile_id=tt["id"])
    assert [e["id"] for e in only_tt] == ["h2"]
    assert len(await me.list_analyses(user, db)) == 2


async def test_a_profile_belonging_to_someone_else_is_ignored(db, make_user):
    """Taken on trust this would file an analysis under a stranger's bike --
    which sounds harmless until you notice the comparison screens read it."""
    mine = await make_user(tier=TIER_FULL)
    theirs = await make_user(tier=TIER_FULL)
    not_mine = await make_profile(db, theirs, name="Their TT")

    await save(db, mine, client_id="h1", profile_id=not_mine["id"])
    row = (
        await db.execute(select(Analysis).where(Analysis.user_id == mine.id))
    ).scalar_one()
    assert row.profile_id is None


async def test_a_re_save_without_a_profile_does_not_unfile_the_entry(db, make_user):
    """The client edits entries it fetched from the thin list, which carries no
    profileId. Clearing the filing on every edit would empty the profiles over
    a week of ordinary use."""
    user = await make_user(tier=TIER_FULL)
    road = await make_profile(db, user, name="Road")
    await save(db, user, client_id="h1", profile_id=road["id"])
    await save(db, user, client_id="h1")           # a plain edit

    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.profile_id == road["id"]


async def test_the_list_reports_the_server_side_filing(db, make_user):
    user = await make_user(tier=TIER_FULL)
    road = await make_profile(db, user, name="Road")
    await save(db, user, client_id="h1", profile_id=road["id"])
    entry = (await me.list_analyses(user, db))[0]
    assert entry["profileId"] == road["id"]


async def test_profiles_report_how_much_history_they_hold(db, make_user):
    user = await make_user(tier=TIER_FULL)
    road = await make_profile(db, user, name="Road")
    await make_profile(db, user, name="TT")
    await save(db, user, client_id="h1", profile_id=road["id"])
    await save(db, user, client_id="h2", profile_id=road["id"])

    by_name = {p["name"]: p for p in (await me.list_profiles(user, db))["profiles"]}
    assert by_name["Road"]["analyses"] == 2
    assert by_name["TT"]["analyses"] == 0


# --------------------------------------------------------------------------
# A profile is a label, not an owner
# --------------------------------------------------------------------------
async def test_deleting_a_setup_keeps_the_history_filmed_on_it(db, make_user):
    """Losing three months of running because a pair of shoes was deleted would
    be a strange trade."""
    user = await make_user(tier=TIER_FULL)
    road = await make_profile(db, user, name="Road")
    await save(db, user, client_id="h1", profile_id=road["id"])

    await me.delete_profile(road["id"], user, db)
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.profile_id is None
    assert (await db.execute(select(Profile))).scalar_one_or_none() is None


async def test_touching_another_accounts_profile_is_a_404(db, make_user):
    """404 rather than 403: ids are sequential, and confirming one exists would
    be a way to count somebody else's bikes."""
    mine = await make_user(tier=TIER_FULL)
    theirs = await make_user(tier=TIER_FULL)
    not_mine = await make_profile(db, theirs, name="Their bike")

    for call in (
        me.update_profile(not_mine["id"], me.ProfilePatch(name="x"), mine, db),
        me.delete_profile(not_mine["id"], mine, db),
    ):
        with pytest.raises(HTTPException) as exc:
            await call
        assert exc.value.status_code == 404


async def test_a_nameless_profile_is_refused():
    with pytest.raises(ValueError):
        me.ProfileIn(name="   ")


async def test_an_unknown_kind_falls_back_rather_than_erroring(db, make_user):
    user = await make_user(tier=TIER_FULL)
    out = await make_profile(db, user, name="Weird", kind="hovercraft")
    assert out["kind"] == "other"
