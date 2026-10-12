"""Every sale is announced to the owner, once; a review order says it owes work.

The first Coach Notes order (2026-10-11, $12, a 72-hour promise) was found two
days in, by accident, in the Stripe dashboard. Nothing in the app told anybody
a review was owed.
"""
from __future__ import annotations

import dataclasses

from app.api.billing import _apply_event
from app.services import notify


def checkout(user_id, *, mode="payment", plan="notes", tier="", session_id="cs_1",
             amount=1200, analysis="a1"):
    return {
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id, "client_reference_id": str(user_id), "customer": "cus_1",
            "mode": mode, "amount_total": amount, "currency": "usd",
            "metadata": {"user_id": str(user_id), "plan": plan, "tier": tier,
                         "analysis_client_id": analysis},
        }},
    }


async def test_a_new_notes_order_is_a_sale_and_a_redelivery_is_not(db, make_user):
    user = await make_user()
    sale = await _apply_event(db, checkout(user.id))
    assert sale["plan"] == "notes" and sale["email"] == user.email
    assert sale["analysis_client_id"] == "a1"
    # Stripe retries until it gets a 2xx: the owner hears about it once.
    assert await _apply_event(db, checkout(user.id)) is None


async def test_a_subscription_is_announced_once(db, make_user):
    user = await make_user()
    ev = checkout(user.id, mode="subscription", plan="enthusiast_monthly",
                  tier="enthusiast", session_id="cs_sub", amount=900, analysis="")
    assert (await _apply_event(db, ev))["tier"] == "enthusiast"
    assert await _apply_event(db, ev) is None


async def test_events_that_are_not_sales_announce_nothing(db):
    assert await _apply_event(db, {"type": "customer.created", "data": {"object": {}}}) is None


def test_the_review_order_email_says_what_is_owed_and_by_when():
    sale = {"plan": "notes", "mode": "payment", "amount_total": 1200, "currency": "usd",
            "email": "buyer@example.com", "analysis_client_id": "a1",
            "paid_ms": 1_760_184_480_000}   # 2025-10-11 12:08 UTC
    subject, text, html = notify.sale_email(sale, "https://getflapp.com/app#reviews")
    assert subject.startswith("ACTION: new Coach Notes order")
    assert "12.00 USD" in subject
    assert "Tue 14 Oct" in text and "72 h" in text
    assert "buyer@example.com" in text
    assert "https://getflapp.com/app#reviews" in text and "review queue" in html


def test_an_unlock_is_news_but_not_work():
    sale = {"plan": "unlock", "mode": "payment", "amount_total": 400, "currency": "usd",
            "email": "b@x.com", "paid_ms": 1_760_184_480_000}
    subject, text, _ = notify.sale_email(sale, "u")
    assert subject == "New sale: Report unlock — 4.00 USD"
    assert "Due" not in text


def test_it_goes_to_the_admin_address(monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "settings",
                        dataclasses.replace(notify.settings, admin_email="owner@x.com"))
    monkeypatch.setattr(notify, "send_email", lambda to, s, t, h: sent.append((to, s)) or True)
    sale = {"plan": "expert", "mode": "payment", "amount_total": 3900, "currency": "usd",
            "email": "b@x.com", "paid_ms": 1}
    assert notify.notify_sale(sale, "u") is True
    assert sent[0][0] == "owner@x.com" and sent[0][1].startswith("ACTION: new Expert Review")


def test_no_admin_address_is_logged_not_raised(monkeypatch):
    monkeypatch.setattr(notify, "settings",
                        dataclasses.replace(notify.settings, admin_email=None))
    assert notify.notify_sale({"plan": "unlock", "paid_ms": 1}, "u") is False
