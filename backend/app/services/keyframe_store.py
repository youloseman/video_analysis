"""Moving stored keyframes out of the history blob and into their own column.

An analysis entry is ~46 KB in the database and ~45 KB of that is one base64
JPEG: the annotated frame. The history list serves the entry and strips the
frame, so the listing read fifty times what it sent -- for up to a hundred rows,
on every load of the app by a signed-in athlete. A JSON blob cannot be
half-fetched, so the frame had to become a column before the read could skip
it (see models/analysis.py).

New rows are written split from the start. This module is for the ones that
were not.

Why in Python and not in the ALTER
----------------------------------
The obvious version is one UPDATE with a JSON operator. There is no portable
spelling of it: ``data - 'keyframe'`` needs jsonb on Postgres, where the column
is ``json`` and the operator does not exist, and SQLite spells the whole thing
differently again. The test suite only ever sees SQLite, so a dialect-split
written here would be tested on the half that is not production -- which is how
a SQLite-only ``DEFAULT 1`` took the service down for four hours. The ORM
already knows how to read and write this column on both, so the move goes
through it.

Why after startup and not during it
-----------------------------------
It is a data rewrite, not a schema change, and it must not stand between a
deploy and the app serving. It is also resumable by construction: each row is
moved independently and a row is only rewritten when it still has a frame in
its blob, so an interrupted pass leaves a mixture that both readers already
handle, and the next boot finishes the job.
"""

from __future__ import annotations

from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import Analysis

logger = structlog.get_logger()

# Rows per transaction. Small enough that a boot is never blocked behind one
# long write, large enough that a few hundred rows is a handful of round trips.
BATCH = 50

# A ceiling per boot, so that a table which is somehow enormous degrades into
# "a bit better on every deploy" rather than into a startup that never finishes.
# The remainder is picked up next time; nothing is lost by stopping early.
# Checked between batches, so the real stopping point is the first batch
# boundary past it -- exactness would buy nothing here and would mean carrying
# a partial page around the loop.
MAX_PER_BOOT = 5000


def split_keyframe(entry: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """``(entry without its frame, the frame)``.

    The single place that decides the frame does not belong in the blob, so the
    save path and the backfill cannot disagree about where it lives.
    """
    if not isinstance(entry, dict) or "keyframe" not in entry:
        return entry, None
    rest = {k: v for k, v in entry.items() if k != "keyframe"}
    frame = entry.get("keyframe")
    return rest, (frame if isinstance(frame, str) and frame else None)


def stored_keyframe(row: Analysis) -> str | None:
    """This row's frame, from the column or from the blob it used to live in.

    Both are read for as long as un-migrated rows can exist, which is until the
    backfill has run everywhere -- and a reader that guessed wrong would show
    an athlete an empty card where their own frame should be.
    """
    if row.keyframe:
        return row.keyframe
    legacy = (row.data or {}).get("keyframe")
    return legacy if isinstance(legacy, str) and legacy else None


async def backfill_keyframes(db: AsyncSession, limit: int = MAX_PER_BOOT) -> int:
    """Move frames out of ``data`` into ``Analysis.keyframe``. Returns the count.

    Selects on the column being empty rather than on the blob holding a frame,
    because only the former is a condition the database can answer without
    reading every blob. That means the pages also contain rows with no frame
    anywhere -- a photo analysis, an entry saved from the thin list -- which
    are left exactly as they are.

    Paged by ascending id rather than by OFFSET, and that is load-bearing: the
    filter is on the column this loop is filling, so a row that is skipped
    stays matching it. Re-running the same ``LIMIT 50`` would hand back the
    same fifty untouched rows forever.
    """
    moved = 0
    after = 0
    while moved < limit:
        rows = (
            await db.execute(
                select(Analysis)
                .where(Analysis.keyframe.is_(None), Analysis.id > after)
                .order_by(Analysis.id)
                .limit(BATCH)
            )
        ).scalars().all()
        if not rows:
            break
        after = rows[-1].id
        touched = 0
        for row in rows:
            rest, frame = split_keyframe(row.data or {})
            if frame is None:
                continue
            row.data = rest
            row.keyframe = frame
            touched += 1
        if touched:
            await db.commit()
            moved += touched
        if len(rows) < BATCH:
            break
    if moved:
        logger.info("KEYFRAMES_BACKFILLED", rows=moved)
    return moved
