"""Per-user cloud history/progress: save, list, delete, import analyses.

The client keeps the same entry shape it uses in localStorage; here we persist
it per account so history + progress survive device switches. All endpoints
require a valid bearer token.

The list endpoint returns entries *without* their annotated keyframe: a frame is
a ~100-160 KB base64 JPEG, so a full history would otherwise be an 8-16 MB JSON
on every dashboard load -- worst for the most active (most valuable) accounts.
Each entry carries ``has_keyframe`` instead and the client pulls frames one at a
time from ``/analyses/{client_id}/keyframe`` as cards scroll into view.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.user import User
from app.services.result_gating import (
    ACCESS_FULL,
    access_for_stored,
    gate_for_access,
)
from app.services.video_analysis.llm_recommendations import (
    generate_progress_summary,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/me", tags=["me"])

MAX_PER_USER = 100


class AnalysisIn(BaseModel):
    # The full frontend entry; only these are read explicitly, the rest is kept.
    model_config = ConfigDict(extra="allow")
    id: str = Field(min_length=1, max_length=64)
    at: int = 0
    kind: str | None = None
    sport: str | None = None
    score: int | None = None


class ImportIn(BaseModel):
    items: list[AnalysisIn] = Field(default_factory=list)


async def _upsert(db: AsyncSession, user: User, entry: dict[str, Any]) -> None:
    cid = str(entry.get("id") or "")[:64]
    if not cid:
        return
    row = (
        await db.execute(
            select(Analysis).where(
                Analysis.user_id == user.id, Analysis.client_id == cid,
            )
        )
    ).scalar_one_or_none()
    # ``has_keyframe`` is a transport flag from the thin list, never stored.
    entry.pop("has_keyframe", None)
    at = int(entry.get("at") or 0)
    # The upload behind this entry. Kept as a column, not just inside data,
    # because the storage sweeper joins on it -- and because it is what lets an
    # Expert Review reach the actual footage weeks later. Photo analyses have
    # none: nothing was ever written to disk for them.
    job_id = str(entry.get("jobId") or "")[:64] or None
    sport = entry.get("sport")
    kind = entry.get("kind")
    score = entry.get("score")
    if row is not None:
        # The client edits entries it fetched from the thin list (no frame), so
        # a re-save that omits the keyframe must not wipe the stored one.
        if not entry.get("keyframe"):
            stored = (row.data or {}).get("keyframe")
            if stored:
                entry["keyframe"] = stored
        row.data = entry
        row.created_at_ms = at
        # A re-save from the thin list carries no jobId, and losing the link
        # would orphan footage that is still inside its retention window.
        if job_id:
            row.job_id = job_id
            _attach_result(row, job_id)
        row.sport = sport
        row.kind = kind
        row.score = score
    else:
        row = Analysis(
            user_id=user.id, client_id=cid, created_at_ms=at, job_id=job_id,
            sport=sport, kind=kind, score=score, data=entry,
        )
        if job_id:
            _attach_result(row, job_id)
        db.add(row)


async def _enforce_cap(db: AsyncSession, user: User) -> None:
    stale = (
        await db.execute(
            select(Analysis.id)
            .where(Analysis.user_id == user.id)
            .order_by(Analysis.created_at_ms.desc())
            .offset(MAX_PER_USER)
        )
    ).scalars().all()
    if stale:
        await db.execute(delete(Analysis).where(Analysis.id.in_(stale)))


def _attach_result(row: Analysis, job_id: str) -> None:
    """Copy the analyzer's full output onto the row, from the live job store.

    The client cannot supply this: it only ever received what its plan allowed,
    and letting it post the "full" result would make the browser the authority
    on its own entitlements. So the server takes it from the job it ran, which
    is the only place the complete object exists.

    Silent when the job has aged out of the store -- the entry is still saved,
    it just cannot be sold or opened by a later upgrade. The client saves within
    seconds of the result appearing, so that window is theoretical.
    """
    from app.core.jobs import JOBS

    job = JOBS.get(job_id) or {}
    result = job.get("result")
    if not result:
        return
    row.result = result
    # Whether this was the account's free preview is decided by the server at
    # upload time and recorded here, so re-opening the report a year later still
    # shows what it showed on the day.
    if job.get("preview"):
        row.preview = True


async def _delete_stored_clips(
    db: AsyncSession, user_id: int, client_ids: list[str] | None = None,
) -> int:
    """Remove the footage behind these analyses, before the rows go.

    Clips outlive their analysis now, which quietly created a way to break the
    privacy policy: deleting a history entry used to remove the only reference
    to a file that was already gone, and would now leave it sitting on the
    volume for the rest of its retention period. Deleting the row is the moment
    the person asked for the data to go, so the file goes with it.

    Failures here are logged, not raised: a clip that cannot be unlinked must
    not stop the deletion the customer actually asked for -- the sweeper will
    reach it once no row vouches for it.
    """
    from app.core.jobs import delete_job_files
    from app.services.retention import job_ids_for_user

    job_ids = await job_ids_for_user(db, user_id, client_ids)
    removed = sum(1 for jid in job_ids if delete_job_files(jid))
    if job_ids:
        logger.info(
            "CLIPS_DELETED", user_id=user_id, requested=len(job_ids), removed=removed,
        )
    return removed


def _without_keyframe(data: dict[str, Any], row: Analysis | None = None,
                     user: User | None = None) -> dict[str, Any]:
    """Entry minus its base64 frame, flagged so the client knows to fetch it.

    ``access`` rides along because the client stored whatever it was shown at
    the time, which may now be less than it is entitled to -- after subscribing,
    or after paying to unlock this one report. It is the signal to go and fetch
    the real thing from ``/analyses/{id}/result`` instead of re-rendering a
    stale teaser.
    """
    thin = {k: v for k, v in data.items() if k != "keyframe"}
    thin["has_keyframe"] = bool(data.get("keyframe"))
    if row is not None:
        thin["access"] = access_for_stored(user, row)
        # Nothing to reveal: written before results were stored server-side, so
        # an unlock would sell an empty report.
        thin["sellable"] = bool(row.result)
    return thin


@router.get("/analyses")
async def list_analyses(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> list[dict[str, Any]]:
    rows = (
        await db.execute(
            select(Analysis)
            .where(Analysis.user_id == user.id)
            .order_by(Analysis.created_at_ms.desc())
            .limit(MAX_PER_USER)
        )
    ).scalars().all()
    return [_without_keyframe(r.data, r, user) for r in rows]


@router.get("/analyses/{client_id}/keyframe")
async def get_keyframe(
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The annotated keyframe for one of the caller's analyses (data URI)."""
    row = (
        await db.execute(
            select(Analysis).where(
                Analysis.user_id == user.id, Analysis.client_id == client_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.",
        )
    return {"keyframe": (row.data or {}).get("keyframe")}


@router.get("/analyses/{client_id}/result")
async def get_analysis_result(
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    """The stored analysis itself, trimmed to what the caller may now see.

    Distinct from the history entry (``data``), which is the client's own
    rendering of a report and was frozen at whatever it was allowed to see that
    day. This serves the analyzer's output, gated at read time -- so subscribing
    opens every past report, and a one-off unlock opens exactly one.
    """
    row = (
        await db.execute(
            select(Analysis).where(
                Analysis.user_id == user.id, Analysis.client_id == client_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found.",
        )
    if not row.result:
        # Ran before results were stored server-side. Saying so is better than
        # returning a shell that looks like a report we lost.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This analysis was run before full reports were stored.",
        )
    access = access_for_stored(user, row)
    if access == ACCESS_FULL:
        await _ensure_coaching(db, row)
    return {
        "access": access,
        "result": gate_for_access(row.result, access),
    }


async def _ensure_coaching(db: AsyncSession, row: Analysis) -> None:
    """Write the coaching notes if this report was run without them.

    A teaser makes no Gemini call: generating coaching for every free analysis
    would bill us on output nobody sees, and the overwhelming majority of them
    are never bought. So the call is deferred to the moment somebody is
    entitled to read it -- on unlock, or when a subscription opens the history
    retroactively -- and the answer is stored, because they may open it again.

    Failure is not an error here. The rest of the report -- every measurement,
    every finding, the drills -- is what was actually paid for and is already
    on the row; refusing to serve it because a language model was unreachable
    would be a strange way to treat somebody who has just paid.
    """
    result = row.result or {}
    if result.get("ai_recommendations") or result.get("kind") == "photo":
        return
    try:
        from app.services.video_analysis.llm_recommendations import (
            generate_recommendations,
        )

        notes = await run_in_threadpool(
            generate_recommendations,
            result.get("sport_type") or result.get("sport") or "run",
            result.get("technique_score"),
            result.get("letter_grade"),
            result.get("detected_issues") or [],
            result.get("angle_statistics") or {},
            result.get("sport_specific_metrics") or {},
            result.get("cycling_position"),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("LAZY_COACHING_FAILED", analysis=row.client_id, err=str(e))
        return
    if not notes:
        return
    # Reassigned rather than mutated: SQLAlchemy does not track edits made
    # inside a JSON column's dict, so an in-place update would be dropped on
    # commit and the call would be repeated on every read.
    row.result = {**result, "ai_recommendations": notes}
    await db.commit()
    logger.info("LAZY_COACHING_WRITTEN", analysis=row.client_id)


@router.post("/analyses", status_code=status.HTTP_201_CREATED)
async def save_analysis(
    body: AnalysisIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    await _upsert(db, user, body.model_dump())
    await _enforce_cap(db, user)
    await db.commit()
    return {"ok": True, "id": body.id}


@router.post("/analyses/import")
async def import_analyses(
    body: ImportIn,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    for item in body.items[:MAX_PER_USER]:
        await _upsert(db, user, item.model_dump())
    await _enforce_cap(db, user)
    await db.commit()
    return {"imported": len(body.items[:MAX_PER_USER])}


class CompareSide(BaseModel):
    model_config = ConfigDict(extra="allow")
    score: int | None = None
    grade: str | None = None
    trends: dict[str, Any] = Field(default_factory=dict)
    cat: dict[str, Any] = Field(default_factory=dict)
    at: int = 0


class CompareIn(BaseModel):
    sport: str = "run"
    before: CompareSide
    after: CompareSide


@router.post("/compare-summary")
async def compare_summary(
    body: CompareIn,
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """AI progress read comparing two of the caller's analyses (paid feature).

    Sends only the compact numeric trend maps to the LLM (no video, no
    personal data). Degrades to a 503 when the LLM is unavailable so the
    client can fall back to the numeric comparison it already shows.
    """
    if not user.is_paid:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="The AI progress summary is available on paid plans.",
        )
    sport = "bike" if body.sport == "bike" else "run"
    # ``generate_progress_summary`` is a blocking HTTP call to Gemini. Awaiting it
    # inline would stall the event loop for its whole duration -- and this service
    # runs a single worker, so every other caller's job-status poll would stall
    # with it. Hand it to the threadpool like the analyze endpoints do.
    result = await run_in_threadpool(
        generate_progress_summary,
        sport, body.before.model_dump(), body.after.model_dump(),
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not generate a progress summary right now.",
        )
    return result


@router.delete("/analyses", status_code=status.HTTP_204_NO_CONTENT)
async def delete_all_analyses(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    """Wipe the caller's whole history (e.g. after analyses run for other people
    polluted their stats). Distinct route from the per-item delete below."""
    await _delete_stored_clips(db, user.id)
    await db.execute(delete(Analysis).where(Analysis.user_id == user.id))
    await db.commit()


@router.delete("/analyses/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    await _delete_stored_clips(db, user.id, [client_id])
    await db.execute(
        delete(Analysis).where(
            Analysis.user_id == user.id, Analysis.client_id == client_id,
        )
    )
    await db.commit()
