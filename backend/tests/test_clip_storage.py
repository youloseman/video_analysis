"""The clip survives the analysis, and dies when it is supposed to.

Three things had to become true at once for an Expert Review to be worth what
it charges, and each fails silently on its own:

1. the footage outlives the six-hour job (retention, tested next door);
2. something can *find* it weeks later -- a saved analysis and its upload were
   unrelated objects until ``analyses.job_id`` existed, so even an immortal
   clip would have been unreachable;
3. it still disappears the moment its owner asks, which is a promise that only
   became possible to break once clips started living for weeks.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException

from app.api import billing, me
from app.core import jobs as jobstore
from app.core.config import settings
from app.models.analysis import Analysis
from app.models.order import ORDER_PAID, Order
from app.models.user import TIER_ENTHUSIAST
from sqlalchemy import select


@pytest.fixture(autouse=True)
def uploads(tmp_path, monkeypatch):
    """Point the storage helpers at a throwaway directory.

    ``settings`` is frozen, so the module-level reference is swapped instead --
    ``job_dir_for`` reads it at call time.
    """
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(
        jobstore, "settings", dataclasses.replace(settings, uploads_dir=root),
    )
    jobstore.JOBS.clear()
    yield root
    jobstore.JOBS.clear()


def store_clip(uploads, job_id: str, suffix: str = ".mov"):
    job_dir = uploads / job_id
    job_dir.mkdir()
    clip = job_dir / f"input{suffix}"
    clip.write_bytes(b"\x00" * 64)
    return clip


async def save_entry(db, user, *, client_id="h1", job_id="j1"):
    await me._upsert(db, user, {"id": client_id, "at": 1, "jobId": job_id})
    await db.commit()


# --------------------------------------------------------------------------
# 2. The link
# --------------------------------------------------------------------------
async def test_saving_an_analysis_records_which_upload_it_came_from(db, make_user):
    user = await make_user()
    await save_entry(db, user, job_id="job-abc")
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.job_id == "job-abc"


async def test_a_photo_analysis_records_no_upload(db, make_user):
    """Nothing was written to disk for it, so a job id would be a lie."""
    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "kind": "photo"})
    await db.commit()
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.job_id is None


async def test_re_saving_from_the_thin_list_does_not_lose_the_link(db, make_user):
    """The client edits entries it fetched without their keyframe -- and
    without their jobId. Dropping the link on such a re-save would orphan
    footage that is still inside its retention window."""
    user = await make_user()
    await save_entry(db, user, job_id="job-abc")
    await me._upsert(db, user, {"id": "h1", "at": 2, "score": 71})
    await db.commit()
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.job_id == "job-abc"


# --------------------------------------------------------------------------
# 3. Deleting on request
# --------------------------------------------------------------------------
async def test_deleting_an_analysis_deletes_its_footage(db, make_user, uploads):
    user = await make_user()
    clip = store_clip(uploads, "j1")
    await save_entry(db, user, job_id="j1")

    await me.delete_analysis("h1", user, db)
    assert not clip.exists()


async def test_deleting_the_whole_history_deletes_every_clip(db, make_user, uploads):
    user = await make_user()
    first, second = store_clip(uploads, "j1"), store_clip(uploads, "j2")
    await save_entry(db, user, client_id="h1", job_id="j1")
    await save_entry(db, user, client_id="h2", job_id="j2")

    await me.delete_all_analyses(user, db)
    assert not first.exists() and not second.exists()


async def test_deleting_one_analysis_leaves_the_others_alone(db, make_user, uploads):
    user = await make_user()
    doomed, kept = store_clip(uploads, "j1"), store_clip(uploads, "j2")
    await save_entry(db, user, client_id="h1", job_id="j1")
    await save_entry(db, user, client_id="h2", job_id="j2")

    await me.delete_analysis("h1", user, db)
    assert not doomed.exists()
    assert kept.exists()


async def test_deleting_never_reaches_another_accounts_footage(db, make_user, uploads):
    mine = await make_user()
    theirs = await make_user()
    ours, not_ours = store_clip(uploads, "j1"), store_clip(uploads, "j2")
    await save_entry(db, mine, client_id="h1", job_id="j1")
    await save_entry(db, theirs, client_id="h1", job_id="j2")

    await me.delete_all_analyses(mine, db)
    assert not ours.exists()
    assert not_ours.exists()


async def test_a_missing_file_does_not_block_the_deletion_asked_for(db, make_user):
    """The row must go even if the clip already went."""
    user = await make_user()
    await save_entry(db, user, job_id="gone")
    await me.delete_analysis("h1", user, db)
    assert (await db.execute(select(Analysis))).scalar_one_or_none() is None


# --------------------------------------------------------------------------
# The reviewer's copy
# --------------------------------------------------------------------------
async def make_order(db, user, *, client_id="h1"):
    order = Order(
        user_id=user.id, stripe_session_id="cs_1", plan="expert",
        status=ORDER_PAID, analysis_client_id=client_id,
        created_at_ms=1, updated_at_ms=1,
    )
    db.add(order)
    await db.commit()
    return order


async def test_the_reviewer_can_reach_the_clip(db, make_user, uploads):
    user = await make_user(tier=TIER_ENTHUSIAST)
    clip = store_clip(uploads, "j1")
    await save_entry(db, user, job_id="j1")
    order = await make_order(db, user)

    assert await billing._clip_path(db, order) == clip


async def test_the_clip_is_found_whatever_extension_it_arrived_with(
    db, make_user, uploads,
):
    user = await make_user()
    clip = store_clip(uploads, "j1", suffix=".mp4")
    await save_entry(db, user, job_id="j1")
    order = await make_order(db, user)

    assert await billing._clip_path(db, order) == clip


async def test_an_expired_clip_is_reported_as_absent_not_as_an_error(
    db, make_user, uploads,
):
    """Three different histories -- expired, deleted by its owner, or from
    before analyses recorded their upload -- and one honest answer: there is no
    clip. The queue draws no player rather than a broken one."""
    user = await make_user()
    await save_entry(db, user, job_id="j1")       # nothing on disk
    order = await make_order(db, user)

    assert await billing._clip_path(db, order) is None
    with pytest.raises(HTTPException) as exc:
        await billing.admin_order_clip(order.id, user, db)
    assert exc.value.status_code == 404


async def test_an_order_with_no_analysis_chosen_has_no_clip(db, make_user):
    user = await make_user()
    order = await make_order(db, user, client_id=None)
    assert await billing._clip_path(db, order) is None
