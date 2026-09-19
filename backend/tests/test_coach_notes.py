"""Coach Notes: the rung under the Expert Review, on the same rails.

Pinned: the format is enforced by the server (one line + exactly three
notes, each capped); checkout needs a report and a slot and honours the
pause; delivery writes the notes ONTO the athlete's history entry, where the
report's note block reads them; the Expert Review is untouched.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import billing, me
from app.api.billing import CheckoutIn, create_checkout
from app.core.config import settings
from app.models.analysis import Analysis
from app.models.order import ORDER_DELIVERED, ORDER_PAID, Order
from app.services import expert_review as er


def _use(monkeypatch, **overrides):
    monkeypatch.setattr(billing, "settings", dataclasses.replace(settings, **overrides))


# --------------------------------------------------------------------------
# the format
# --------------------------------------------------------------------------

def test_notes_are_one_line_and_three_notes_whatever_the_sport():
    assert [s["key"] for s in er.sections_for("bike", "notes")] == ["first", "note_1", "note_2", "note_3"]
    assert er.sections_for("run", "notes") == er.sections_for("bike", "notes")
    # And the Expert Review still has its own list, fit table and all.
    assert "fit" in [s["key"] for s in er.sections_for("bike")]


def test_the_word_budget_is_the_servers_not_the_editors():
    long = "word " * 200
    out = er.normalize_report({"first": long, "note_1": long, "verdict": "smuggled"}, "run", "notes")
    assert len(out["first"]) == er.MAX_FIRST_CHARS
    assert len(out["note_1"]) == er.MAX_NOTE_CHARS
    assert "verdict" not in out                       # allowlist over the section list
    assert er.missing_required(out, "run", "notes") == ["Note 2", "Note 3"]


def test_prefill_seeds_nothing_for_notes():
    entry = {"sport": "run", "gated": True, "confidence": "low", "warnings": ["dark clip"]}
    draft = er.prefill(entry, "notes")
    assert all(not draft[k] for k in ("first", "note_1", "note_2", "note_3"))
    assert er.is_empty(draft)
    assert not er.is_empty({"note_2": "x"})


def test_the_delivered_text_is_the_first_line_then_three_numbered_notes():
    txt = er.notes_text({"first": "Shorten the stride.", "note_1": "A", "note_2": "B", "note_3": "C"})
    assert txt == "First thing to change: Shorten the stride.\n\n1. A\n2. B\n3. C"


# --------------------------------------------------------------------------
# checkout
# --------------------------------------------------------------------------

async def test_notes_need_a_report_to_be_written_on(db, make_user, monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_test_x", stripe_price_notes="price_n")
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="notes"), None, user, db)
    assert exc.value.status_code == 400


async def test_the_pause_button_refuses_with_a_reason(db, make_user, monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_test_x", stripe_price_notes="price_n",
         coach_notes_enabled=False)
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="notes", analysis_client_id="h1"), None, user, db)
    assert exc.value.status_code == 503 and "paused" in exc.value.detail


async def test_notes_have_their_own_weekly_cap(db, make_user, monkeypatch):
    _use(monkeypatch, stripe_secret_key="sk_test_x", stripe_price_notes="price_n",
         coach_notes_slots_per_week=1, expert_review_slots_per_week=1)
    user = await make_user()
    now = billing._now_ms()
    db.add(Order(user_id=user.id, stripe_session_id="cs_n1", plan="notes", status=ORDER_PAID,
                 created_at_ms=now, updated_at_ms=now, analysis_client_id="h1"))
    await db.commit()
    assert await billing.notes_slots_left(db) == 0
    # A sold-out notes week leaves the review's own slot untouched.
    assert await billing.review_slots_left(db) == 1
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="notes", analysis_client_id="h1"), None, user, db)
    assert exc.value.status_code == 503 and "booked" in exc.value.detail


async def test_the_slots_endpoint_reports_both_products(db, monkeypatch):
    _use(monkeypatch, coach_notes_enabled=False)
    out = await billing.expert_review_slots(db)
    assert out["open"] is True
    assert out["notes"]["paused"] is True and out["notes"]["open"] is False


# --------------------------------------------------------------------------
# delivery: onto the report
# --------------------------------------------------------------------------

async def _paid_notes_order(db, user, client_id="h1"):
    await me._upsert(db, user, {"id": client_id, "at": 1, "jobId": "j1", "sport": "run"})
    now = billing._now_ms()
    order = Order(user_id=user.id, stripe_session_id="cs_notes", plan="notes", status=ORDER_PAID,
                  created_at_ms=now, updated_at_ms=now, analysis_client_id=client_id)
    db.add(order)
    await db.commit()
    return order


async def test_publishing_refuses_until_all_three_notes_exist(db, make_user):
    user = await make_user()
    order = await _paid_notes_order(db, user)
    await billing.admin_update_order(
        order.id, billing.OrderPatch(report={"first": "Shorten the stride.", "note_1": "A"}), user, db,
    )
    with pytest.raises(HTTPException) as exc:
        await billing.admin_publish_report(order.id, SimpleNamespace(headers={}, url=SimpleNamespace(scheme="https", netloc="x")), user, db)
    assert exc.value.status_code == 400 and "Note 2" in exc.value.detail


async def test_delivered_notes_land_on_the_athletes_entry(db, make_user, monkeypatch):
    monkeypatch.setattr(billing, "_base_url", lambda request: "https://getflapp.com")
    monkeypatch.setattr(billing.notify, "send_email", lambda *a, **k: True)
    user = await make_user()
    order = await _paid_notes_order(db, user)
    await billing.admin_update_order(
        order.id,
        billing.OrderPatch(report={"reviewer": "Artur", "first": "Shorten the stride.",
                                   "note_1": "A", "note_2": "B", "note_3": "C"}),
        user, db,
    )
    out = await billing.admin_publish_report(order.id, None, user, db)
    assert out["status"] == ORDER_DELIVERED and out["emailed"] is True

    row = (await db.execute(select(Analysis).where(Analysis.client_id == "h1"))).scalar_one()
    assert row.data["coachNote"].startswith("First thing to change: Shorten the stride.")
    assert "3. C" in row.data["coachNote"]
    assert row.data["coachName"] == "Artur" and row.data["coachOrderId"] == order.id

    # A re-publish with an edit reaches the report, and the athlete's copy
    # answers with the notes' own section list.
    await billing.admin_update_order(
        order.id, billing.OrderPatch(report={"reviewer": "Artur", "first": "Land softer.",
                                             "note_1": "A", "note_2": "B", "note_3": "C"}), user, db,
    )
    await billing.admin_publish_report(order.id, None, user, db)
    row = (await db.execute(select(Analysis).where(Analysis.client_id == "h1"))).scalar_one()
    assert row.data["coachNote"].startswith("First thing to change: Land softer.")
    mine = await billing.my_report(order.id, user, db)
    assert [s["key"] for s in mine["sections"]] == ["first", "note_1", "note_2", "note_3"]


def test_the_ready_email_carries_the_first_line_only():
    subject, text, html = er.notes_ready_email(
        {"reviewer": "Artur", "first": "Shorten the stride.", "note_1": "secret A"}, "https://x/app",
    )
    assert "Coach Notes" in subject
    assert "Shorten the stride." in text and "secret A" not in text
    assert "secret A" not in html
