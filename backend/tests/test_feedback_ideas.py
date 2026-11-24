"""Suggest a feature: an idea lands in the private inbox, and its author hears
back once when it ships.

What would hurt if it broke: an idea lost (form says thanks, nothing stored),
ideas skewing the result-rating stats, the shipped mail going twice, or going
to an address the rider never gave.
"""

from __future__ import annotations

import pytest
from fastapi import BackgroundTasks, HTTPException
from starlette.requests import Request

from app.api import feedback as F
from app.models.feedback import KIND_IDEA, Feedback
from app.services import ideas_mail


def _req() -> Request:
    return Request({"type": "http", "method": "POST", "path": "/feedback",
                    "headers": [], "client": ("10.0.0.9", 1234),
                    "scheme": "https", "server": ("getflapp.com", 443),
                    "query_string": b"", "root_path": ""})


@pytest.fixture(autouse=True)
def _no_limit(monkeypatch):
    F._hits.clear()
    sent: list = []
    monkeypatch.setattr(ideas_mail, "send_email",
                        lambda to, subject, text, html: sent.append((to, subject, text)) or True)
    return sent


async def _submit(db, user=None, **body):
    bg = BackgroundTasks()
    out = await F.submit_feedback(F.FeedbackIn(kind="idea", **body), _req(), bg, user, db)
    return out, bg


async def test_a_signed_in_idea_is_stored_with_their_email(db, make_user):
    u = await make_user(tier="full", email="rider@example.com")
    out, bg = await _submit(db, u, message="Compare two bikes side by side")
    row = await db.get(Feedback, out["id"])
    assert row.kind == KIND_IDEA and row.rating is None
    assert row.email == "rider@example.com" and row.context["tier"] == "full"
    assert len(bg.tasks) == 1  # the owner hears about it


async def test_a_guest_idea_keeps_an_email_only_if_given(db):
    out, _ = await _submit(db, message="Rear view for running", email="Guest@Example.com")
    assert (await db.get(Feedback, out["id"])).email == "guest@example.com"
    out, _ = await _submit(db, message="Swim side view please")
    assert (await db.get(Feedback, out["id"])).email is None
    with pytest.raises(HTTPException):
        await _submit(db, message="A real idea here", email="not-an-email")


async def test_an_empty_idea_is_refused(db):
    with pytest.raises(HTTPException):
        await _submit(db, message="  hi ")


async def test_ideas_stay_out_of_the_rating_stats(db, make_user):
    admin = await make_user(tier="admin")
    await _submit(db, message="Garmin sync for cadence")
    s = await F.feedback_stats(admin, db)
    assert s["total"] == 0 and s["rated"] == 0
    assert s["ideas"] == 1 and s["ideas_new"] == 1
    lst = await F.list_feedback(kind="idea", _admin=admin, db=db)
    assert [i["kind"] for i in lst["items"]] == ["idea"]


async def test_the_shipped_mail_goes_once_and_marks_it_fixed(db, make_user, _no_limit):
    admin = await make_user(tier="admin")
    u = await make_user(email="asker@example.com")
    out, _ = await _submit(db, u, message="Export my angles to CSV")
    body = F.TriageIn(notify_shipped=True, shipped_note="Account, Export, CSV")
    res = await F.triage_feedback(out["id"], body, admin, db)
    assert res["status"] == "fixed" and res["notified_at"]
    to, subject, text = _no_limit[-1]
    assert to == "asker@example.com" and "Export my angles to CSV" in text
    assert "Account, Export, CSV" in text and "—" not in subject + text
    with pytest.raises(HTTPException) as e:
        await F.triage_feedback(out["id"], body, admin, db)
    assert e.value.status_code == 409


async def test_no_shipped_mail_without_an_address_or_for_a_rating(db, make_user):
    admin = await make_user(tier="admin")
    out, _ = await _submit(db, message="Anonymous idea text")
    with pytest.raises(HTTPException):
        await F.triage_feedback(out["id"], F.TriageIn(notify_shipped=True), admin, db)
    rating = await F.submit_feedback(F.FeedbackIn(rating=1), _req(), BackgroundTasks(), None, db)
    with pytest.raises(HTTPException):
        await F.triage_feedback(rating["id"], F.TriageIn(notify_shipped=True), admin, db)


def test_the_spa_has_the_form_and_the_inbox_button():
    from conftest import read_spa
    spa = read_spa()
    assert 'id="wnIdea"' in spa and "kind:'idea'" in spa
    assert 'data-fbf="ideas"' in spa and "notify_shipped:true" in spa
    assert "mailto:support@getflapp.com?subject=Flapp%20feedback" not in spa


def test_a_release_riders_asked_for_is_tagged(tmp_path, monkeypatch):
    from app.services.changelog import parser as P
    from app.services.changelog.renderer import render_changelog
    (tmp_path / "2026-11-23-csv.md").write_text(
        "---\nasked: yes\n---\n# Export to CSV\n\nYour angles as a file.\n", encoding="utf-8")
    (tmp_path / "2026-11-22-other.md").write_text("# Other\n\nText.\n", encoding="utf-8")
    monkeypatch.setattr(P, "CONTENT_DIR", tmp_path)
    P.invalidate_cache()
    try:
        es = P.get_entries()
        assert [e.asked for e in es] == [True, False]
        html = render_changelog(es, "https://getflapp.com")
        assert html.count("Asked for by riders") == 1
    finally:
        P.invalidate_cache()
