"""The one free preview cannot be claimed by two clips sent together.

``user`` is loaded before the upload streams in, so two uploads from a free
account both read the preview as free and both wrote their own job id: two
full reports for one preview. The claim is now a compare-and-swap on the
value that was read.
"""

from __future__ import annotations

from sqlalchemy import update
from sqlalchemy.orm.attributes import set_committed_value

from app.main import claim_preview
from app.models.user import User


async def test_a_fresh_account_claims_it(db, make_user):
    user = await make_user()
    assert await claim_preview(db, user, "job-a")
    await db.refresh(user)
    assert user.free_preview_job_id == "job-a"


async def test_the_second_of_two_racing_uploads_loses(db, make_user):
    user = await make_user()
    assert await claim_preview(db, user, "job-a")
    set_committed_value(user, "free_preview_job_id", None)   # what the other request read
    assert not await claim_preview(db, user, "job-b")
    await db.refresh(user)
    assert user.free_preview_job_id == "job-a"


async def test_a_handed_back_claim_can_be_taken_again(db, make_user):
    """A failed run or a legacy photo claim leaves a value that
    preview_available() treats as free; the swap is against that value."""
    user = await make_user()
    await db.execute(update(User).where(User.id == user.id).values(free_preview_job_id="photo:123"))
    await db.commit()
    await db.refresh(user)
    assert await claim_preview(db, user, "job-c")
    await db.refresh(user)
    assert user.free_preview_job_id == "job-c"
