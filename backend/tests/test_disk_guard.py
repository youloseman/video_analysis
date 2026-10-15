"""The video volume refuses a clip with a sentence before it fills, and says so to a person.

On 2026-10-15 the volume was 4.8 GB, 65 % full and filling ~0.7 GB a day; a
full volume breaks the upload half-way, which reads as a broken analysis.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

from app.core import jobs as J

Usage = namedtuple("Usage", "total used free")
GB = 2**30


def test_uploads_are_refused_when_almost_no_room_is_left(monkeypatch):
    monkeypatch.setattr(J.shutil, "disk_usage", lambda p: Usage(5 * GB, int(4.85 * GB), int(0.15 * GB)))
    assert J.disk_refuses_uploads() is True
    monkeypatch.setattr(J.shutil, "disk_usage", lambda p: Usage(5 * GB, 3 * GB, 2 * GB))
    assert J.disk_refuses_uploads() is False
    # 5 % of a large volume, not 400 MB, is the floor there.
    monkeypatch.setattr(J.shutil, "disk_usage", lambda p: Usage(50 * GB, int(47.6 * GB), int(2.4 * GB)))
    assert J.disk_refuses_uploads() is True


def test_the_admin_is_emailed_at_85_percent_at_most_twice_a_day(monkeypatch):
    sent = []
    from app.services import notify
    monkeypatch.setattr(notify, "send_email", lambda to, subj, text, html: sent.append(subj) or True)
    monkeypatch.setattr(J, "_last_storage_alert", 0.0)
    monkeypatch.setattr(J.shutil, "disk_usage", lambda p: Usage(5 * GB, int(4.4 * GB), int(0.6 * GB)))
    assert J.storage_alert(now=100_000.0) is True
    assert J.storage_alert(now=100_000.0 + 3600) is False, "not again within 12 hours"
    assert J.storage_alert(now=100_000.0 + 13 * 3600) is True
    assert sent and "88% full" in sent[0]
    monkeypatch.setattr(J.shutil, "disk_usage", lambda p: Usage(5 * GB, 3 * GB, 2 * GB))
    assert J.storage_alert(now=10**9) is False, "below 85 %: nothing to say"


def test_both_upload_paths_check_before_writing():
    src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
    assert src.count("if await run_in_threadpool(disk_refuses_uploads):") == 2
    assert src.count("if e.errno == errno.ENOSPC:") == 2
