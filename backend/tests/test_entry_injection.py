"""An athlete's history entry cannot carry markup onto their coach's screen.

An entry is written by the client (``POST /me/analyses``, ``/import``) and a
linked athlete's rows are served to their coach, whose SPA puts the entry id
into HTML attributes and the frame into ``<img src>``. With the id free text
and the frame any string, an athlete could store
``"><img src onerror=...>`` and run script in the coach's session (the token
lives in localStorage). So: ids are held to characters that need no escaping,
frames to image data URIs -- on the way in, and on the way out for rows that
were written before the check -- and the SPA escapes the id anyway.
"""

from __future__ import annotations

import re

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select

from app.api import me
from app.models.analysis import Analysis
from app.services.keyframe_store import safe_frame, split_keyframe, stored_keyframe

FRAME = "data:image/jpeg;base64," + "A" * 400
EVIL_ID = '"><img src onerror=alert(1)>'


@pytest.mark.parametrize("frame", [
    FRAME,
    "data:image/png;base64,iVBORw0KGgo=",
    "data:image/webp;base64,UklGRg==",
])
def test_image_data_uris_are_frames(frame):
    assert safe_frame(frame) == frame


@pytest.mark.parametrize("frame", [
    'data:image/jpeg;base64,AAAA" onerror="alert(1)',
    "javascript:alert(1)",
    "https://evil.example/x.png",
    "data:image/svg+xml;base64,PHN2Zz4=",
    "data:text/html;base64,PHNjcmlwdD4=",
    "data:image/jpeg;base64,AA AA",
    "",
    None,
    42,
])
def test_anything_else_is_no_frame(frame):
    assert safe_frame(frame) is None
    _rest, out = split_keyframe({"id": "h1", "keyframe": frame})
    assert out is None


def test_a_bad_frame_stored_earlier_is_not_served():
    """Rows written before the check: the reader filters too."""
    assert stored_keyframe(Analysis(data={}, keyframe="javascript:alert(1)")) is None
    assert stored_keyframe(Analysis(data={"keyframe": '"><svg onload=x>'}, keyframe=None)) is None


def test_the_save_model_refuses_an_id_with_markup():
    with pytest.raises(ValidationError):
        me.AnalysisIn(id=EVIL_ID)
    assert me.AnalysisIn(id="h1732300000000").id == "h1732300000000"


@pytest.mark.parametrize("cid", ["h1732300000000", "job_abc-123", "a.b:c"])
def test_real_ids_still_pass(cid):
    assert me.CLIENT_ID_RE.fullmatch(cid)


async def test_upsert_skips_an_id_with_markup(db, make_user):
    user = await make_user()
    await me._upsert(db, user, {"id": EVIL_ID, "at": 1})
    await db.flush()
    rows = (await db.execute(select(Analysis).where(Analysis.user_id == user.id))).scalars().all()
    assert rows == []


async def test_upsert_drops_a_frame_that_is_not_an_image(db, make_user):
    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "keyframe": "javascript:alert(1)"})
    await db.flush()
    row = (await db.execute(select(Analysis).where(Analysis.user_id == user.id))).scalar_one()
    assert row.keyframe is None
    assert "keyframe" not in (row.data or {})


async def test_import_keeps_the_good_entries_and_skips_the_bad_one(db, make_user):
    """One odd entry must not cost the device the rest of its history."""
    user = await make_user()
    body = me.ImportIn(items=[
        {"id": "h1", "at": 1, "keyframe": FRAME},
        {"id": EVIL_ID, "at": 2},
        {"id": "h3", "at": 3},
    ])
    out = await me.import_analyses(body, user=user, db=db)
    assert out == {"imported": 2}
    ids = sorted(
        r.client_id for r in
        (await db.execute(select(Analysis).where(Analysis.user_id == user.id))).scalars().all()
    )
    assert ids == ["h1", "h3"]


async def test_the_single_save_endpoint_rejects_markup_ids_before_the_handler():
    """FastAPI validates the body against AnalysisIn, so the handler never
    sees the id; the model test above is that gate."""
    assert "pattern" in str(me.AnalysisIn.model_json_schema()["properties"]["id"])
    with pytest.raises((ValidationError, HTTPException)):
        me.AnalysisIn.model_validate({"id": "x y"})


def test_the_spa_escapes_entry_ids_and_gates_frames():
    from conftest import read_spa

    spa = read_spa()
    # Every attribute that carries an entry id goes through esc().
    raw = re.findall(r'data-(?:id|del|kf)="\$\{e\.id\}"', spa)
    assert raw == [], raw
    # Frames reach <img src> only through safeFrame.
    assert "function safeFrame(" in spa
    keyframe_of = spa[spa.index("function keyframeOf("):][:300]
    assert "safeFrame(" in keyframe_of


def test_sign_in_keeps_the_device_history_when_the_import_fails():
    from conftest import read_spa

    spa = read_spa()
    body = spa[spa.index("async function migrateAndLoad("):][:900]
    assert "if(r.ok)" in body and "removeItem(HKEY)" in body
    assert body.index("if(r.ok)") < body.index("removeItem(HKEY)")
