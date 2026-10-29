"""The swim upload: stored, never analysed, capped, and in the history at once.

Imports app.main, so it skips without the analysis stack like the other
endpoint tests.
"""
from __future__ import annotations

import dataclasses
import io

import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile
from sqlalchemy import select
from starlette.requests import Request

main = pytest.importorskip("app.main", reason="needs the analysis stack (mediapipe/opencv)")

from app.models.analysis import Analysis  # noqa: E402
from app.services import swim  # noqa: E402


def _req():
    return Request({"type": "http", "headers": [], "client": ("1.2.3.4", 1), "method": "POST", "path": "/"})


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "settings", dataclasses.replace(main.settings, uploads_dir=tmp_path))
    monkeypatch.setattr(main, "disk_refuses_uploads", lambda: False)
    monkeypatch.setattr(main, "storage_alert", lambda: None)
    return tmp_path


async def _upload(db, user, name="lap.mov", data=b"clip-bytes", **form):
    bt = BackgroundTasks()
    out = await main.upload_swim_endpoint(
        bt, _req(), UploadFile(io.BytesIO(data), filename=name),
        form.get("stroke"), form.get("pool"), form.get("view"), form.get("focus"),
        form.get("consent_training", False), user, db,
    )
    return out, bt


async def test_a_clip_is_stored_and_lands_in_the_history(db, make_user, uploads):
    user = await make_user()
    entry, bt = await _upload(db, user, focus="  my   legs sink ", consent_training=True)
    assert entry["sport"] == "swim" and entry["kind"] == "video" and entry["score"] is None
    assert entry["id"].startswith("s") and entry["focus"] == "my legs sink"
    assert entry["swim"] == {"stroke": "freestyle", "pool": "pool_25m", "view": "underwater_side",
                             "consent_training": True, "filename": "lap.mov"}
    assert (uploads / entry["jobId"] / "input.mov").read_bytes() == b"clip-bytes"
    row = (await db.execute(select(Analysis).where(Analysis.client_id == entry["id"]))).scalar_one()
    assert row.sport == "swim" and row.job_id == entry["jobId"] and row.result is None
    # Compressed later, like every stored clip -- never analysed.
    assert [t.func.__name__ for t in bt.tasks] == ["compress_clip"]


async def test_a_wrong_choice_or_file_is_refused(db, make_user, uploads):
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await _upload(db, user, stroke="doggy")
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        await _upload(db, user, name="lap.jpg")
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException) as exc:
        await _upload(db, user, data=b"")
    assert exc.value.status_code == 400
    assert not any(uploads.iterdir())


async def test_clips_without_a_review_are_capped(db, make_user, uploads):
    user = await make_user()
    for _ in range(swim.MAX_UNORDERED_CLIPS):
        await _upload(db, user)
    with pytest.raises(HTTPException) as exc:
        await _upload(db, user)
    assert exc.value.status_code == 429
