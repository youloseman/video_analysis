"""The annotated frame lives in its own column, and old rows are moved to it.

An entry is ~46 KB in the database and ~45 KB of it is the frame -- 98%,
measured across stored rows. The history list serves the entry and strips the
frame out in Python, so it read fifty times what it sent, for up to a hundred
rows, on every load of the app by a signed-in athlete. A JSON blob cannot be
half-fetched, so the frame had to leave the blob before the read could skip it.

Two things have to stay true or an athlete opens their history and finds their
own frames gone: every reader has to look in both places while un-migrated rows
exist, and the backfill has to be safe to interrupt and re-run. Those are what
is tested here, plus the pagination bug this loop is built to avoid -- it
filters on the column it is filling, so a page that moves nothing would be
handed back forever.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

from app.api import me
from app.models.analysis import Analysis
from app.services.keyframe_store import (
    BATCH,
    backfill_keyframes,
    split_keyframe,
    stored_keyframe,
)

FRAME = "data:image/jpeg;base64," + "A" * 4000
OTHER = "data:image/jpeg;base64," + "B" * 4000


# --------------------------------------------------------------------------
# split_keyframe
# --------------------------------------------------------------------------

def test_the_frame_comes_out_and_the_rest_stays():
    rest, frame = split_keyframe({"id": "h1", "score": 74, "keyframe": FRAME})
    assert frame == FRAME
    assert rest == {"id": "h1", "score": 74}


def test_an_entry_with_no_frame_is_returned_unchanged():
    entry = {"id": "h1", "score": 74}
    rest, frame = split_keyframe(entry)
    assert frame is None
    assert rest == entry


@pytest.mark.parametrize("value", ["", None, 0, [], {}])
def test_an_empty_frame_is_no_frame(value):
    """An entry saved from the thin list carries ``keyframe`` absent or empty.
    Storing that as a frame would make ``has_keyframe`` lie and send the client
    to fetch nothing."""
    rest, frame = split_keyframe({"id": "h1", "keyframe": value})
    assert frame is None
    assert "keyframe" not in rest


# --------------------------------------------------------------------------
# stored_keyframe -- both places, for as long as both can hold one
# --------------------------------------------------------------------------

def test_the_column_is_read():
    row = Analysis(data={"id": "h1"}, keyframe=FRAME)
    assert stored_keyframe(row) == FRAME


def test_the_legacy_blob_is_still_read():
    """A row written before the column existed. Reading only the column would
    show an athlete an empty card where their own frame should be."""
    row = Analysis(data={"id": "h1", "keyframe": FRAME}, keyframe=None)
    assert stored_keyframe(row) == FRAME


def test_the_column_wins_over_the_blob():
    """They can only disagree mid-backfill; the column is the one being
    written, so it is the newer of the two."""
    row = Analysis(data={"id": "h1", "keyframe": OTHER}, keyframe=FRAME)
    assert stored_keyframe(row) == FRAME


def test_no_frame_anywhere_is_none():
    assert stored_keyframe(Analysis(data={"id": "h1"}, keyframe=None)) is None


# --------------------------------------------------------------------------
# the backfill
# --------------------------------------------------------------------------

async def _legacy(db, user, client_id: str, frame: str | None = FRAME):
    """A row as it was written before the column existed."""
    entry = {"id": client_id, "score": 74}
    if frame is not None:
        entry["keyframe"] = frame
    row = Analysis(
        user_id=user.id, client_id=client_id, created_at_ms=1,
        data=entry, keyframe=None,
    )
    db.add(row)
    await db.commit()
    return row


async def test_a_legacy_row_is_moved(db, make_user):
    user = await make_user()
    await _legacy(db, user, "h1")

    assert await backfill_keyframes(db) == 1

    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.keyframe == FRAME
    assert "keyframe" not in row.data
    assert row.data == {"id": "h1", "score": 74}


async def test_it_is_safe_to_run_again(db, make_user):
    """A deploy runs it on every boot. The second pass must find nothing to do
    rather than re-writing rows or losing a frame."""
    user = await make_user()
    await _legacy(db, user, "h1")
    assert await backfill_keyframes(db) == 1
    assert await backfill_keyframes(db) == 0
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.keyframe == FRAME


async def test_rows_with_no_frame_are_left_alone(db, make_user):
    """A photo analysis, or an entry saved from the thin list. They match the
    backfill's filter forever -- their column stays empty because there is
    nothing to put in it -- so the loop must not treat them as work."""
    user = await make_user()
    await _legacy(db, user, "h1", frame=None)
    assert await backfill_keyframes(db) == 0
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.keyframe is None
    assert row.data == {"id": "h1", "score": 74}


async def test_a_page_of_untouchable_rows_does_not_loop_forever(db, make_user):
    """The loop filters on the column it is filling, so anything it skips still
    matches on the next query. Paged by OFFSET or re-queried from the top it
    would hand back the same rows until the boot timed out. More than one full
    batch of frameless rows, so the paging is actually exercised."""
    user = await make_user()
    for i in range(BATCH + 5):
        await _legacy(db, user, f"empty{i}", frame=None)
    await _legacy(db, user, "real", frame=FRAME)

    assert await backfill_keyframes(db) == 1

    moved = (
        await db.execute(select(Analysis).where(Analysis.client_id == "real"))
    ).scalar_one()
    assert moved.keyframe == FRAME


async def test_it_moves_more_rows_than_one_batch(db, make_user):
    user = await make_user()
    for i in range(BATCH + 3):
        await _legacy(db, user, f"h{i}")
    assert await backfill_keyframes(db) == BATCH + 3
    rows = (await db.execute(select(Analysis))).scalars().all()
    assert all(r.keyframe == FRAME for r in rows)
    assert all("keyframe" not in r.data for r in rows)


async def test_an_interrupted_pass_leaves_a_readable_mixture(db, make_user):
    """Both readers handle a half-moved table, which is what makes stopping
    early safe. The remainder is picked up on the next boot.

    The limit is checked between batches, so it stops at the first batch
    boundary past it rather than at an exact row -- which is what makes the
    half here exactly one batch.
    """
    user = await make_user()
    for i in range(BATCH + 3):
        await _legacy(db, user, f"h{i:03d}")

    assert await backfill_keyframes(db, limit=1) == BATCH
    rows = (await db.execute(select(Analysis).order_by(Analysis.id))).scalars().all()
    assert sum(1 for r in rows if r.keyframe) == BATCH
    assert sum(1 for r in rows if "keyframe" in r.data) == 3
    # Mixture or not, every frame is still readable.
    assert [stored_keyframe(r) for r in rows] == [FRAME] * (BATCH + 3)

    assert await backfill_keyframes(db) == 3
    rows = (await db.execute(select(Analysis).order_by(Analysis.id))).scalars().all()
    assert all(r.keyframe == FRAME and "keyframe" not in r.data for r in rows)


# --------------------------------------------------------------------------
# through the API
# --------------------------------------------------------------------------

async def test_saving_puts_the_frame_in_the_column(db, make_user):
    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "keyframe": FRAME})
    await db.commit()
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.keyframe == FRAME
    assert "keyframe" not in row.data


async def test_a_resave_from_the_thin_list_keeps_the_frame(db, make_user):
    """The client edits entries it fetched without their frame. Before the
    column existed this was guarded by copying the frame back into the blob;
    the guard has to survive the move or an edit silently erases it."""
    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "keyframe": FRAME})
    await db.commit()
    await me._upsert(db, user, {"id": "h1", "at": 2, "score": 80})
    await db.commit()
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.keyframe == FRAME
    assert row.data["score"] == 80


async def test_the_endpoint_serves_a_migrated_frame(db, make_user):
    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "keyframe": FRAME})
    await db.commit()
    assert (await me.get_keyframe("h1", user, db))["keyframe"] == FRAME


async def test_the_endpoint_serves_a_legacy_frame(db, make_user):
    user = await make_user()
    await _legacy(db, user, "h1")
    assert (await me.get_keyframe("h1", user, db))["keyframe"] == FRAME


async def test_the_listing_flags_a_frame_in_either_place(db, make_user):
    """has_keyframe is what sends the client to fetch one. It is answered from
    the column now, but a row the backfill has not reached still has its frame
    in the blob -- and a false here is a permanently blank card."""
    user = await make_user()
    await _legacy(db, user, "legacy")
    await me._upsert(db, user, {"id": "moved", "at": 2, "keyframe": FRAME})
    await me._upsert(db, user, {"id": "none", "at": 3})
    await db.commit()

    by_id = {e["id"]: e for e in await me.list_analyses(user, db)}
    assert by_id["legacy"]["has_keyframe"] is True
    assert by_id["moved"]["has_keyframe"] is True
    assert by_id["none"]["has_keyframe"] is False
    # And the frame itself is never in the listing, from either place.
    assert all("keyframe" not in e for e in by_id.values())


async def test_the_listing_never_fetches_the_frame(db, make_user):
    """The point of the column. Pinned on the SQL because the regression is
    invisible in the response: re-adding the column changes nothing an
    assertion on the payload could see."""
    from sqlalchemy import event

    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "keyframe": FRAME})
    await db.commit()

    seen: list[str] = []
    bind = db.get_bind()
    engine = getattr(bind, "sync_engine", bind)

    def record(conn, cursor, statement, params, context, executemany):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        await me.list_analyses(user, db)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    selects = [s for s in seen if "FROM analyses" in s]
    assert len(selects) == 1, f"one query expected, got {len(selects)}"
    fetched = selects[0].replace("analyses.keyframe IS NOT NULL", "")
    assert "analyses.keyframe" not in fetched, (
        f"the listing fetches the frame again:\n{selects[0]}"
    )
