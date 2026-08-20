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
        row.sport = sport
        row.kind = kind
        row.score = score
    else:
        db.add(Analysis(
            user_id=user.id, client_id=cid, created_at_ms=at, job_id=job_id,
            sport=sport, kind=kind, score=score, data=entry,
        ))


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


def _without_keyframe(data: dict[str, Any]) -> dict[str, Any]:
    """Entry minus its base64 frame, flagged so the client knows to fetch it."""
    thin = {k: v for k, v in data.items() if k != "keyframe"}
    thin["has_keyframe"] = bool(data.get("keyframe"))
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
    return [_without_keyframe(r.data) for r in rows]


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
