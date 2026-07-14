"""Per-user cloud history/progress: save, list, delete, import analyses.

The client keeps the same entry shape it uses in localStorage; here we persist
it per account so history + progress survive device switches. All endpoints
require a valid bearer token.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_session
from app.core.security import get_current_user
from app.models.analysis import Analysis
from app.models.user import User

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
    at = int(entry.get("at") or 0)
    sport = entry.get("sport")
    kind = entry.get("kind")
    score = entry.get("score")
    if row is not None:
        row.data = entry
        row.created_at_ms = at
        row.sport = sport
        row.kind = kind
        row.score = score
    else:
        db.add(Analysis(
            user_id=user.id, client_id=cid, created_at_ms=at,
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
    return [r.data for r in rows]


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


@router.delete("/analyses/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    client_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
) -> None:
    await db.execute(
        delete(Analysis).where(
            Analysis.user_id == user.id, Analysis.client_id == client_id,
        )
    )
    await db.commit()
