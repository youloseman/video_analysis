"""Selling one report at a time.

The unlock exists because the product is used episodically: somebody changes
their saddle height, films it, and wants that one answer. A subscription is
priced for a habit they do not have, so at $9/month they convert at roughly
zero and are worth nothing. $4 for the report in front of them is the version
of the offer that matches what they came for.

The fence between the two is *time*, not price: an unlock opens one report,
and a report on its own cannot be compared with anything. Trends, before/after
and the overlay video -- everything that needs a second analysis to mean
something -- stay with the subscription. So the tests worth writing are about
what an unlock does NOT open, and about the tier still winning over it.
"""

from __future__ import annotations

import dataclasses

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api import billing, me
from app.api.billing import CheckoutIn, _apply_event, create_checkout
from app.core.config import settings
from app.models.analysis import Analysis
from app.models.order import Order
from app.models.user import TIER_ENTHUSIAST, TIER_STARTER
from app.services.result_gating import (
    ACCESS_FULL,
    ACCESS_PREVIEW,
    ACCESS_TEASER,
    access_for_stored,
)

FULL_RESULT = {
    "status": "completed",
    "sport_type": "run",
    "technique_score": 74,
    "letter_grade": "C",
    "detected_issues": [{"type": "overstriding", "value": "0.21 x leg length"}],
    "angle_statistics": {"knee": {"mean": 142.0}, "hip": {"mean": 31.0}},
    "training_plan": {"top_3_priorities": [{"drill": "strides"}]},
    "ai_recommendations": {"report": "lift your cadence"},
    "sport_specific_metrics": {"cadence_spm": 168},
}


async def store(db, user, *, client_id="h1", preview=False, unlocked=None,
                result=FULL_RESULT):
    row = Analysis(
        user_id=user.id, client_id=client_id, job_id="j1", created_at_ms=1,
        data={"id": client_id}, result=result, preview=preview,
        unlocked_at_ms=unlocked,
    )
    db.add(row)
    await db.commit()
    return row


def unlock_event(user_id, *, client_id="h1", session_id="cs_unlock_1"):
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "client_reference_id": str(user_id),
            "customer": "cus_1",
            "mode": "payment",
            "amount_total": 400,
            "currency": "usd",
            "metadata": {
                "user_id": str(user_id), "plan": "unlock", "tier": "",
                "analysis_client_id": client_id or "",
            },
        }},
    }


# --------------------------------------------------------------------------
# Who may read what
# --------------------------------------------------------------------------
async def test_a_teaser_stays_shut(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user)
    assert access_for_stored(user, row) == ACCESS_TEASER


async def test_an_unlocked_report_opens(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user, unlocked=123)
    assert access_for_stored(user, row) == ACCESS_FULL


async def test_the_free_preview_is_still_the_preview(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user, preview=True)
    assert access_for_stored(user, row) == ACCESS_PREVIEW


async def test_subscribing_opens_everything_that_was_ever_run(db, make_user):
    """Most of the reason the full result is stored. Somebody who paid $4 twice
    and then subscribes should not find their old reports still shut -- and this
    is a better argument for a plan than any wording on the pricing page."""
    user = await make_user(tier=TIER_ENTHUSIAST)
    old_teaser = await store(db, user, client_id="h-old")
    assert access_for_stored(user, old_teaser) == ACCESS_FULL


async def test_an_unlock_does_not_leak_to_the_next_report(db, make_user):
    """One report, not an account-wide switch."""
    user = await make_user(tier=TIER_STARTER)
    bought = await store(db, user, client_id="h1", unlocked=123)
    other = await store(db, user, client_id="h2")
    assert access_for_stored(user, bought) == ACCESS_FULL
    assert access_for_stored(user, other) == ACCESS_TEASER


# --------------------------------------------------------------------------
# Buying one
# --------------------------------------------------------------------------
async def test_an_unlock_must_name_the_report_it_buys(db, make_user, monkeypatch):
    """Unlike an Expert Review there is no human downstream to work out which
    report was meant, so an unnamed unlock buys nothing at all."""
    monkeypatch.setattr(
        billing, "settings", dataclasses.replace(settings, stripe_secret_key="sk_test_x"),
    )
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(CheckoutIn(plan="unlock"), None, user, db)
    assert exc.value.status_code == 400


def _stripe_on(monkeypatch):
    monkeypatch.setattr(
        billing, "settings", dataclasses.replace(settings, stripe_secret_key="sk_test_x"),
    )


async def test_an_unlock_is_refused_for_a_report_that_is_not_theirs(db, make_user, monkeypatch):
    _stripe_on(monkeypatch)
    user = await make_user()
    with pytest.raises(HTTPException) as exc:
        await create_checkout(
            CheckoutIn(plan="unlock", analysis_client_id="nope"), None, user, db,
        )
    assert exc.value.status_code == 404


async def test_a_subscriber_is_not_charged_for_an_unlock(db, make_user, monkeypatch):
    """Their plan already opens it; $4 would buy nothing. The client hides the
    button, but a stale tab from before the upgrade still had it."""
    _stripe_on(monkeypatch)
    user = await make_user(tier=TIER_ENTHUSIAST)
    await store(db, user)
    with pytest.raises(HTTPException) as exc:
        await create_checkout(
            CheckoutIn(plan="unlock", analysis_client_id="h1"), None, user, db,
        )
    assert exc.value.status_code == 409


async def test_an_already_unlocked_report_is_not_sold_twice(db, make_user, monkeypatch):
    _stripe_on(monkeypatch)
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, unlocked=123)
    with pytest.raises(HTTPException) as exc:
        await create_checkout(
            CheckoutIn(plan="unlock", analysis_client_id="h1"), None, user, db,
        )
    assert exc.value.status_code == 409


async def test_a_report_with_nothing_stored_is_not_sold(db, make_user, monkeypatch):
    """Saved before full results were kept: the listing says sellable=false,
    and now so does the server."""
    _stripe_on(monkeypatch)
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, result=None)
    with pytest.raises(HTTPException) as exc:
        await create_checkout(
            CheckoutIn(plan="unlock", analysis_client_id="h1"), None, user, db,
        )
    assert exc.value.status_code == 409
    assert "nothing to unlock" in exc.value.detail


async def test_a_sellable_teaser_reaches_stripe(db, make_user, monkeypatch):
    """The guard lets the legitimate case through to Checkout."""
    _stripe_on(monkeypatch)
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    calls = {}

    def fake_create(**kw):
        calls.update(kw)

        class _S:
            url = "https://checkout.stripe.test/s"

        return _S()

    class _Cust:
        id = "cus_new"

    monkeypatch.setattr(billing.stripe.Customer, "create", lambda **kw: _Cust())
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)
    out = await create_checkout(
        CheckoutIn(plan="unlock", analysis_client_id="h1"), _FakeRequest(), user, db,
    )
    assert out["url"].startswith("https://checkout.stripe.test/")
    assert calls["mode"] == "payment"
    assert calls["metadata"]["analysis_client_id"] == "h1"


class _FakeRequest:
    base_url = "https://getflapp.com/"


async def test_the_free_preview_is_sold_like_a_teaser(db, make_user, monkeypatch):
    """Artur, 2026-10-29: the preview card carries the $4 too. What the preview
    blurred (drills or fit moves, angles, ranges, kinogram) is the stored
    report, so the guard lets it through and paying opens it in full."""
    _stripe_on(monkeypatch)
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user, preview=True)
    calls = {}

    def fake_create(**kw):
        calls.update(kw)

        class _S:
            url = "https://checkout.stripe.test/s"

        return _S()

    class _Cust:
        id = "cus_new"

    monkeypatch.setattr(billing.stripe.Customer, "create", lambda **kw: _Cust())
    monkeypatch.setattr(billing.stripe.checkout.Session, "create", fake_create)
    await create_checkout(
        CheckoutIn(plan="unlock", analysis_client_id="h1"), _FakeRequest(), user, db,
    )
    assert calls["mode"] == "payment" and calls["metadata"]["analysis_client_id"] == "h1"
    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    assert row.preview and row.unlocked_at_ms
    assert access_for_stored(user, row) == ACCESS_FULL


async def test_paying_opens_exactly_that_report(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user)
    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    assert row.unlocked_at_ms
    assert access_for_stored(user, row) == ACCESS_FULL


async def test_paying_records_the_order_too(db, make_user):
    """The entitlement lives on the analysis; the order is the receipt. Both,
    because one answers "may I read this" and the other answers "did I pay"."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    await _apply_event(db, unlock_event(user.id))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.plan == "unlock"
    assert order.amount_total == 400


async def test_a_redelivered_webhook_changes_nothing(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    row = await store(db, user)
    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    first = row.unlocked_at_ms

    await _apply_event(db, unlock_event(user.id))
    await db.refresh(row)
    assert row.unlocked_at_ms == first
    assert len((await db.execute(select(Order))).scalars().all()) == 1


async def test_paying_for_a_deleted_report_does_not_explode(db, make_user):
    """Deleted between paying and the webhook landing. The order stands as the
    record that they paid; there is simply nothing left to open."""
    user = await make_user(tier=TIER_STARTER)
    await _apply_event(db, unlock_event(user.id, client_id="h-gone"))
    order = (await db.execute(select(Order))).scalar_one()
    assert order.plan == "unlock"


async def test_an_unlock_never_reaches_another_account(db, make_user):
    mine = await make_user(tier=TIER_STARTER)
    theirs = await make_user(tier=TIER_STARTER)
    ours = await store(db, mine, client_id="h1")
    not_ours = await store(db, theirs, client_id="h1")

    await _apply_event(db, unlock_event(mine.id, client_id="h1"))
    await db.refresh(ours)
    await db.refresh(not_ours)
    assert ours.unlocked_at_ms
    assert not_ours.unlocked_at_ms is None


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------
async def test_the_result_endpoint_serves_the_whole_report_once_unlocked(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, unlocked=123)
    out = await me.get_analysis_result("h1", user, db)
    assert out["access"] == ACCESS_FULL
    assert out["result"]["angle_statistics"] == FULL_RESULT["angle_statistics"]
    assert out["result"]["training_plan"] == FULL_RESULT["training_plan"]


async def test_the_result_endpoint_still_trims_a_teaser(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    out = await me.get_analysis_result("h1", user, db)
    assert out["access"] == ACCESS_TEASER
    assert "angle_statistics" not in out["result"]
    assert "training_plan" not in out["result"]


async def test_a_report_from_before_full_results_were_stored_says_so(db, make_user):
    """Rather than returning a shell that looks like a report we lost -- and
    rather than selling an unlock that would reveal nothing."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, result=None)
    with pytest.raises(HTTPException) as exc:
        await me.get_analysis_result("h1", user, db)
    assert exc.value.status_code == 409


async def test_the_history_list_says_what_each_entry_may_show(db, make_user):
    """The client stored whatever it was shown that day, so it needs telling
    when it is now entitled to more."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, client_id="h1", unlocked=123)
    await store(db, user, client_id="h2")
    await store(db, user, client_id="h3", result=None)

    by_id = {e["id"]: e for e in await me.list_analyses(user, db)}
    assert by_id["h1"]["access"] == ACCESS_FULL
    assert by_id["h2"]["access"] == ACCESS_TEASER
    assert by_id["h2"]["sellable"] is True
    assert by_id["h3"]["sellable"] is False


async def test_the_history_list_never_fetches_the_stored_results(db, make_user):
    """The listing serves none of ``Analysis.result`` and must not read it.

    It is the analyzer's complete output -- around 490 KB a row against the
    ~140 KB of ``data`` beside it -- and this endpoint runs on every load of
    the app by a signed-in athlete, for up to a hundred rows. A bare
    ``select(Analysis)`` here was pulling tens of megabytes out of the database
    and into Python to build a response of one or two.

    Pinned on the emitted SQL rather than on a timing, because the regression
    is silent: re-adding the column changes nothing an assertion on the
    response could see. ``sellable`` is still expected to be right -- it is
    answered by ``IS NOT NULL``, which reads no value -- so the test next door
    stays the guard on the behaviour while this one guards the cost.
    """
    from sqlalchemy import event

    user = await make_user(tier=TIER_STARTER)
    await store(db, user, client_id="h1")

    seen: list[str] = []
    bind = db.get_bind()
    engine = getattr(bind, "sync_engine", bind)

    def record(conn, cursor, statement, params, context, executemany):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", record)
    try:
        rows = await me.list_analyses(user, db)
    finally:
        event.remove(engine, "before_cursor_execute", record)

    selects = [s for s in seen if "FROM analyses" in s]
    assert selects, "the listing issued no query against analyses"
    for sql in selects:
        # The column may only appear inside the IS NOT NULL that answers
        # "sellable" -- never as a column being fetched.
        fetched = sql.replace("analyses.result IS NOT NULL", "")
        assert "analyses.result" not in fetched, (
            f"the listing fetches the stored result again:\n{sql}"
        )
    # And nothing lazy-loaded it afterwards, which would be worse: one extra
    # query per row instead of one big one.
    assert len(selects) == 1, f"one query expected, got {len(selects)}"
    assert rows[0]["sellable"] is True


async def test_reading_another_accounts_report_is_a_404(db, make_user):
    mine = await make_user()
    theirs = await make_user()
    await store(db, theirs, client_id="h1")
    with pytest.raises(HTTPException) as exc:
        await me.get_analysis_result("h1", mine, db)
    assert exc.value.status_code == 404


async def test_an_unlock_writes_the_coaching_on_the_payment(db, make_user, monkeypatch):
    """The first $4 unlock after the lazy-coaching path shipped came back
    without a coach (2026-10-12): the page showed its own copy and never asked
    the server for the report, so the lazy call never ran. The payment itself
    now writes it."""
    from contextlib import asynccontextmanager

    from app.services.video_analysis import llm_recommendations as llm

    user = await make_user()
    await store(db, user, client_id="u1", result={**FULL_RESULT, "ai_recommendations": None})
    calls = []
    monkeypatch.setattr(llm, "generate_recommendations",
                        lambda *a, **k: calls.append(k) or {"report": "**Overall** — ok.", "model": "fake"})

    @asynccontextmanager
    async def same_session():
        yield db

    import app.core.db as core_db
    monkeypatch.setattr(core_db, "SessionLocal", same_session)
    await billing._coach_after_unlock(user.id, "u1")
    row = (await db.execute(select(Analysis).where(Analysis.client_id == "u1"))).scalar_one()
    assert row.result["ai_recommendations"]["report"] == "**Overall** — ok."
    assert "fit_plan" in calls[0] and "focus" in calls[0]


async def test_an_unlock_is_in_the_queue_as_a_check_not_a_review(db, make_user):
    """Order #3 (2026-10-13), a $4 unlock, opened the full Expert Review form.
    Artur wants every paid report in front of him anyway, to catch what the
    product got wrong -- so an unlock is a CHECK: the stored result and the coach
    text the athlete got, a note, a tick. Never a form, never a send."""
    admin = await make_user()
    buyer = await make_user()
    await store(db, buyer, client_id="h1")
    await _apply_event(db, unlock_event(buyer.id, client_id="h1", session_id="cs_q"))
    queue = await billing.admin_orders(admin, db)
    item = next(o for o in queue if o["plan"] == "unlock")
    assert item["kind"] == "check" and item["checked"] is False and item["has_draft"] is False
    order = (await db.execute(select(Order).where(Order.stripe_session_id == "cs_q"))).scalar_one()
    detail = await billing.admin_order_detail(order.id, admin, db)
    assert detail["kind"] == "check" and "sections" not in detail
    assert detail["result"]["score"] == 74 and detail["result"]["coach_report"] == "lift your cadence"

    # A note and a tick; a report or a status is refused, and publish too.
    out = await billing.admin_update_order(order.id, billing.OrderPatch(admin_note="score too kind", checked=True), admin, db)
    assert out["qa"]["checked_at"] and out["admin_note"] == "score too kind" and out["status"] == "paid"
    with pytest.raises(HTTPException):
        await billing.admin_update_order(order.id, billing.OrderPatch(report={"verdict": "x"}), admin, db)
    with pytest.raises(HTTPException) as exc:
        await billing.admin_publish_report(order.id, None, admin, db)
    assert exc.value.status_code == 400
    assert next(o for o in await billing.admin_orders(admin, db) if o["plan"] == "unlock")["checked"] is True
    # The athlete's own view of the order is untouched: nothing delivered.
    mine = await billing.my_orders(buyer, db)
    assert mine[0]["status"] == "paid" and not mine[0].get("unread")


async def test_a_granted_free_unlock_opens_the_report_without_stripe(db, make_user, monkeypatch):
    """Artur gives a report away by hand -- the first went to a subscriber who
    reported the keyframe drawing his leg wrong (2026-10-14). The unlock button
    spends the credit: the report opens, an order is kept for the review queue
    (as a check, like a paid unlock), and Stripe is never asked."""
    _stripe_on(monkeypatch)

    def no_stripe(**kw):
        raise AssertionError("a granted unlock must not reach Stripe")

    monkeypatch.setattr(billing.stripe.checkout.Session, "create", no_stripe)
    coached = []

    async def fake_coach(user_id, client_id):
        coached.append(client_id)

    monkeypatch.setattr(billing, "_coach_after_unlock", fake_coach)
    user = await make_user(tier=TIER_STARTER)
    user.unlock_credits = 1
    await db.commit()
    row = await store(db, user)
    out = await create_checkout(CheckoutIn(plan="unlock", analysis_client_id="h1"), None, user, db)
    assert out["mode"] == "unlock_credit" and out["credits_left"] == 0
    await db.refresh(row)
    assert access_for_stored(user, row) == ACCESS_FULL
    order = (await db.execute(select(Order))).scalar_one()
    assert order.plan == "unlock_credit" and order.amount_total == 0
    assert order.plan in billing.CHECK_ORDER_PLANS
    import asyncio
    await asyncio.sleep(0)
    assert coached == ["h1"], "the coach text is written for it, like a paid unlock"
    # Spent: the next report goes to the till as usual.
    await store(db, user, client_id="h2")
    monkeypatch.setattr(billing.stripe.checkout.Session, "create",
                        lambda **kw: type("S", (), {"url": "https://checkout.stripe.test/s"})())
    monkeypatch.setattr(billing.stripe.Customer, "create", lambda **kw: type("C", (), {"id": "cus_x"})())
    out = await create_checkout(CheckoutIn(plan="unlock", analysis_client_id="h2"), _FakeRequest(), user, db)
    assert out["url"].startswith("https://checkout.stripe.test/")


def test_the_page_offers_a_granted_unlock_as_free():
    from pathlib import Path
    spa = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "function freeUnlocks()" in spa
    assert "'Unlock this report — free'" in spa
    assert "j.mode==='unlock_credit'" in spa
    assert "unlock_credits:j.unlock_credits||0" in spa
