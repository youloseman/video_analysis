"""Hand back the free preview an account lost to a restart.

The one free preview is claimed at upload time by writing the job id onto the
account (``users.free_preview_job_id``), and ``main.preview_available`` hands
it back when that job failed or declined to publish a score -- by looking the
job up in the in-memory store. The store empties on every deploy. A job that
was still running when the process went away is therefore neither "failed" nor
"completed": it is simply gone, and the rule "gone means spent" (which exists
so that a preview does not regenerate every ``job_ttl_hours``) charges the
account for a report that was never delivered.

That account is a brand-new one, on its first clip, and this service ships to
``main`` several times on a busy day. So on every startup, before anything is
served, the claims that never became a saved analysis are released.

"Never became a saved analysis" is the test rather than "was still running",
because the store cannot say what it was doing when it died. A delivered
preview is saved by the client within seconds of appearing (with its
``jobId``), so a claim with no saved row behind it is one whose result nobody
was shown. The photo path's claim (``photo:<ms>``) is left alone: a photo is
analysed inline, so it either raised -- and never claimed -- or was returned.
"""

from __future__ import annotations

import structlog
from typing import Any

from sqlalchemy import exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis
from app.models.user import User

logger = structlog.get_logger()

PHOTO_CLAIM_PREFIX = "photo:"


async def preview_state(db: AsyncSession, user: User, available: bool) -> dict[str, Any]:
    """The account's one free full report, as /auth/me reports it.

    Still to spend: ``{"available": True}``. Spent: when, and on which history
    entry, read off the saved analysis the claim points at -- so the uploader
    can say "your free full report was Oct 12" BEFORE the next clip is
    analysed, instead of the teaser card saying it after. A claim with no
    saved row behind it (a job still running, or a photo claim from the old
    rule) is spent without a date, and the line says so without one.
    """
    if available:
        return {"available": True}
    claimed = user.free_preview_job_id
    row = None
    if claimed and not claimed.startswith(PHOTO_CLAIM_PREFIX):
        row = (
            await db.execute(
                select(Analysis.client_id, Analysis.created_at_ms)
                .where(Analysis.user_id == user.id, Analysis.job_id == claimed)
                .limit(1)
            )
        ).first()
    return {
        "available": False,
        "spent_at_ms": int(row.created_at_ms) if row else None,
        "entry_id": row.client_id if row else None,
    }


async def release_orphaned_previews(db: AsyncSession) -> int:
    """Clear ``free_preview_job_id`` where no saved analysis vouches for the
    job. Returns how many accounts got their preview back."""
    saved = exists(
        select(Analysis.id).where(Analysis.job_id == User.free_preview_job_id)
    )
    result = await db.execute(
        update(User)
        .where(
            User.free_preview_job_id.is_not(None),
            User.free_preview_job_id.not_like(f"{PHOTO_CLAIM_PREFIX}%"),
            ~saved,
        )
        .values(free_preview_job_id=None)
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    released = int(result.rowcount or 0)
    if released:
        logger.info("PREVIEWS_RELEASED", accounts=released)
    return released
