"""Swimming as a coach-only review: no numbers, the same products.

Pinned: a swim clip has no $4 report to unlock; its Expert Review drops the
fit table and the internal "disagree" box and asks the coach swim questions;
the re-test and the capture notes are the swim ones; the coach's brief has
the footage and a checklist instead of a table, for Notes and for a review;
the queue and the owner's sale email say SWIM; and the owner can play their
own clip without a paid plan.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.api import billing, me
from app.models.order import ORDER_PAID, ORDER_REFUNDED, Order
from app.services import coach_brief as cb
from app.services import expert_review as er
from app.services import notify, swim

PAID_MS = 1793196120000   # Wed 28 Oct 2026, 07:02 in Vancouver


def _entry(cid="s1", job="swjob1", focus="My legs sink"):
    return swim.build_entry(
        client_id=cid, at_ms=PAID_MS, job_id=job, stroke="freestyle", pool="pool_25m",
        view="underwater_side", question=focus, consent_training=False, filename="IMG_1.MOV",
    )


async def _swim_row(db, user, cid="s1", job="swjob1"):
    from app.models.analysis import Analysis

    entry = _entry(cid, job)
    db.add(Analysis(user_id=user.id, client_id=cid, created_at_ms=PAID_MS, job_id=job,
                    sport="swim", kind="video", score=None, data=entry))
    await db.commit()
    return entry


async def _order(db, user, plan="notes", cid="s1", status=ORDER_PAID):
    now = billing._now_ms()
    order = Order(user_id=user.id, stripe_session_id=f"cs_{plan}_{cid}_{status}", plan=plan,
                  status=status, created_at_ms=now, updated_at_ms=now, analysis_client_id=cid)
    db.add(order)
    await db.commit()
    return order


# --------------------------------------------------------------------------
# the review template
# --------------------------------------------------------------------------

def test_a_swim_review_has_no_fit_table_and_no_disagree_box():
    keys = [s["key"] for s in er.sections_for("swim", er.PLAN_EXPERT, coach=True)]
    assert "fit" not in keys and "disagree" not in keys
    assert {"verdict", "seen", "working", "priorities", "plan", "retest"} <= set(keys)
    # Run and bike keep theirs.
    assert "disagree" in [s["key"] for s in er.sections_for("bike", er.PLAN_EXPERT, coach=True)]
    assert "fit" in [s["key"] for s in er.sections_for("bike", er.PLAN_EXPERT, coach=True)]


def test_the_coach_is_asked_swim_questions():
    seen = next(s for s in er.sections_for("swim", er.PLAN_EXPERT, coach=True) if s["key"] == "seen")
    assert "catch" in seen["hint"] and "kick" in seen["hint"]
    # The shared section list is not mutated for the next run report.
    run_seen = next(s for s in er.SECTIONS if s["key"] == "seen")
    assert "catch" not in run_seen["hint"]


def test_the_draft_starts_with_the_swim_retest_and_no_capture_notes():
    draft = er.prefill({"sport": "swim"}, er.PLAN_EXPERT)
    assert "pool" in draft["retest"] and "underwater" in draft["retest"]
    assert er.capture_notes({"sport": "swim"}) == ""
    # Notes stay three notes, whatever the sport.
    assert er.sections_for("swim", er.PLAN_NOTES) == er.NOTES_SECTIONS


def test_a_swim_review_is_complete_without_a_fit_table():
    report = er.normalize_report({
        "verdict": "v", "seen": "s", "working": "w", "plan": "p", "retest": "r",
        "priorities": [{"title": "Head", "why": "lifts", "change": "breathe sideways", "check": "legs up"}],
        "fit": [{"part": "saddle", "now": "x", "suggest": "y", "why": "z", "risk": "r"}],
        "disagree": "smuggled",
    }, "swim", er.PLAN_EXPERT)
    assert "fit" not in report and "disagree" not in report
    assert er.missing_required(report, "swim", er.PLAN_EXPERT) == []


# --------------------------------------------------------------------------
# the coach's brief
# --------------------------------------------------------------------------

def test_the_swim_brief_for_notes_is_footage_checklist_and_three_notes():
    html = cb.render_swim_brief_html(order_id=21, plan="notes", paid_ms=PAID_MS,
                                     entry=_entry(focus="<b>legs</b> sink"), clip=b"mp4")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "Coach Notes: техника плавания" in html
    assert "data:video/mp4;base64," in html
    assert "Кроль, бассейн 25 м, снято под водой, сбоку." in html
    assert "Захват" in html and "<table" not in html
    assert "&lt;b&gt;legs" in html and "<b>legs" not in html
    assert html.count("<textarea") == 4
    assert "субботы 31 октября" in html                       # 72 h
    assert "flapp-coach-order21-draft" in html and "{{" not in html


def test_the_swim_brief_for_a_review_has_the_review_sections_and_five_days():
    html = cb.render_swim_brief_html(order_id=22, plan="expert", paid_ms=PAID_MS,
                                     entry=_entry(focus=None), clip=None)
    assert "Expert Review: техника плавания" in html
    assert "План на 4 недели" in html and "Приоритет 3" in html
    assert html.count("<textarea") == 9
    assert "понедельника 2 ноября" in html                     # 5 days
    assert "Видео в файл не попало" in html
    assert "Отдельного вопроса атлет не задал" in html


def test_the_run_and_bike_notes_brief_keeps_its_form():
    html = cb.render_brief_html(order_id=14, paid_ms=PAID_MS, result={"sport_type": "bike"},
                                entry={"kind": "photo"}, media={}, clip=None)
    assert html.count("<textarea") == 4
    assert "Coach Notes, заказ №14" in html and "flapp-coach-order14-draft" in html


async def test_the_brief_endpoint_serves_swim_notes_and_swim_reviews(db, make_user):
    user = await make_user()
    await _swim_row(db, user)
    for plan in ("notes", "expert"):
        order = await _order(db, user, plan=plan)
        resp = await billing.admin_order_brief(order.id, user, db)
        assert "техника плавания".encode() in resp.body


async def test_a_run_or_bike_review_still_has_no_brief(db, make_user):
    user = await make_user()
    await me._upsert(db, user, {"id": "h1", "at": 1, "sport": "bike", "kind": "video"})
    await db.commit()
    order = await _order(db, user, plan="expert", cid="h1")
    with pytest.raises(HTTPException) as exc:
        await billing.admin_order_brief(order.id, user, db)
    assert exc.value.status_code == 400


# --------------------------------------------------------------------------
# buying, the queue, the owner
# --------------------------------------------------------------------------

async def test_a_swim_clip_has_nothing_to_unlock(db, make_user):
    user = await make_user()
    await _swim_row(db, user)
    with pytest.raises(HTTPException) as exc:
        await billing._require_unlockable(db, user, "s1")
    assert exc.value.status_code == 409 and "coach" in exc.value.detail


async def test_clips_without_a_live_order_are_counted(db, make_user):
    user = await make_user()
    await _swim_row(db, user, "s1", "j1")
    await _swim_row(db, user, "s2", "j2")
    await _swim_row(db, user, "s3", "j3")
    assert await swim.unordered_clips(db, user.id) == 3
    await _order(db, user, cid="s1")
    assert await swim.unordered_clips(db, user.id) == 2
    # A refunded order frees nothing: the clip is waiting again.
    await _order(db, user, cid="s2", status=ORDER_REFUNDED)
    assert await swim.unordered_clips(db, user.id) == 2


async def test_the_queue_marks_swim_orders(db, make_user):
    user = await make_user()
    await _swim_row(db, user)
    await _order(db, user)
    unlinked = await _order(db, user, plan="expert", cid=None)
    rows = {r["id"]: r for r in await billing.admin_orders(user, db)}
    assert [r["sport"] for r in rows.values() if r["id"] != unlinked.id] == ["swim"]
    assert rows[unlinked.id]["sport"] is None


def test_the_owner_hears_it_is_a_swim_order():
    subject, text, _ = notify.sale_email(
        {"plan": "notes", "amount_total": 1200, "currency": "usd", "email": "a@b.c",
         "paid_ms": PAID_MS, "analysis_client_id": "s1", "sport": "swim"}, "https://x/app")
    assert subject.startswith("ACTION: new Coach Notes · SWIM order")
    run_subject, _, _ = notify.sale_email(
        {"plan": "notes", "amount_total": 1200, "currency": "usd", "email": "a@b.c",
         "paid_ms": PAID_MS, "analysis_client_id": "h1"}, "https://x/app")
    assert run_subject.startswith("ACTION: new Coach Notes order")


async def test_the_owner_can_play_their_swim_clip_without_a_plan(db, make_user, monkeypatch, tmp_path):
    from app.core import jobs

    user = await make_user()                      # starter: no paid access to anything
    await _swim_row(db, user)
    clip = tmp_path / "input.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(jobs, "job_file", lambda job_id, stem: clip if job_id == "swjob1" else None)
    out = await me.get_overlays("s1", user, db)
    assert out["source"].startswith("/jobs/swjob1/source?mt=")
    assert out["overlays"] == {}
