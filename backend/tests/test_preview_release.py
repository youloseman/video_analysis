"""The free preview comes back after a restart that lost the job.

``main.preview_available`` reads "job not in the store" as "spent", which is
right for a job that aged out and wrong for one the deploy killed mid-run. On
startup the store cannot tell the two apart; whether a saved analysis exists
for the claim can.
"""

from __future__ import annotations

from sqlalchemy import select

from app.models.analysis import Analysis
from app.models.user import User
from app.services.preview_release import release_orphaned_previews


async def _claims(db) -> dict[str, str | None]:
    rows = (await db.execute(select(User))).scalars().all()
    return {u.email: u.free_preview_job_id for u in rows}


async def test_a_claim_with_no_saved_analysis_is_released(db, make_user):
    """The deploy killed the run; nobody was shown a report."""
    await make_user(email="lost@example.com", free_preview_job_id="job-killed")
    assert await release_orphaned_previews(db) == 1
    assert (await _claims(db))["lost@example.com"] is None


async def test_a_delivered_preview_stays_spent(db, make_user):
    """The client saved the entry with its jobId seconds after the result
    appeared -- that row is the proof the preview was delivered."""
    user = await make_user(email="seen@example.com", free_preview_job_id="job-seen")
    db.add(Analysis(user_id=user.id, client_id="h1", created_at_ms=1,
                    job_id="job-seen", data={"id": "h1"}))
    await db.commit()
    assert await release_orphaned_previews(db) == 0
    assert (await _claims(db))["seen@example.com"] == "job-seen"


async def test_a_photo_claim_is_never_released(db, make_user):
    """A photo is analysed inline: it either raised before claiming or was
    returned. Its claim names no job, so "no saved row" proves nothing."""
    await make_user(email="photo@example.com", free_preview_job_id="photo:1700000000000")
    assert await release_orphaned_previews(db) == 0
    assert (await _claims(db))["photo@example.com"] == "photo:1700000000000"


async def test_accounts_without_a_claim_are_untouched(db, make_user):
    await make_user(email="fresh@example.com")
    assert await release_orphaned_previews(db) == 0


async def test_someone_elses_saved_row_does_not_count(db, make_user):
    """Only a row pointing at THIS job keeps the claim; a neighbour's saved
    analysis is not evidence that this account was shown anything."""
    other = await make_user(email="other@example.com")
    await make_user(email="lost@example.com", free_preview_job_id="job-killed")
    db.add(Analysis(user_id=other.id, client_id="h9", created_at_ms=1,
                    job_id="job-other", data={"id": "h9"}))
    await db.commit()
    assert await release_orphaned_previews(db) == 1
    claims = await _claims(db)
    assert claims["lost@example.com"] is None
