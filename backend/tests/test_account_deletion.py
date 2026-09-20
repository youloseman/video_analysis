"""Account deletion.

The privacy policy promises erasure ("Until you ask us to delete your account";
"Usage records deleted when your account is deleted"), so what matters is that
the promise is kept in full: every table that references the user, not just the
user row. Orphans left behind by a partial delete are a policy breach that
nothing would surface.

The endpoint function is called directly -- it is a plain coroutine, and going
through HTTP would add nothing but a router.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select

from app.api.auth import DeleteAccountBody, delete_account
from app.core.security import hash_password
from app.models.analysis import Analysis
from app.models.feedback import Feedback
from app.models.order import ORDER_DELIVERED, ORDER_PAID, Order
from app.models.usage import UsageEvent

PASSWORD = "correct-horse-battery"


@pytest.fixture
async def populated(db, make_user):
    """A user with a row in every table that references them, plus a second
    user whose data must survive."""
    user = await make_user(email="me@example.com")
    user.password_hash = hash_password(PASSWORD)
    other = await make_user(email="other@example.com")

    for owner in (user, other):
        db.add_all([
            Analysis(user_id=owner.id, client_id=f"h{owner.id}", created_at_ms=1,
                     sport="run", kind="video", score=70, data={"id": f"h{owner.id}"}),
            UsageEvent(user_id=owner.id, created_at_ms=1, kind="video"),
            Feedback(user_id=owner.id, created_at_ms=1, rating="down"),
            Order(user_id=owner.id, stripe_session_id=f"cs_{owner.id}",
                  plan="expert", status=ORDER_DELIVERED,
                  created_at_ms=1, updated_at_ms=1),
        ])
    await db.commit()
    return user, other


async def count(db, model, user_id) -> int:
    return int(await db.scalar(
        select(func.count()).select_from(model).where(model.user_id == user_id)
    ) or 0)


async def test_the_wrong_password_deletes_nothing(db, populated):
    user, _ = populated
    with pytest.raises(HTTPException) as e:
        await delete_account(DeleteAccountBody(password="not-my-password"), user, db)
    assert e.value.status_code == 401
    assert await db.get(type(user), user.id) is not None
    assert await count(db, Analysis, user.id) == 1


@pytest.mark.parametrize("model", [Analysis, UsageEvent, Feedback, Order])
async def test_every_referencing_row_goes(db, populated, model):
    """Explicitly, per table -- a partial delete leaves data the policy says is
    gone, and nothing else in the system would notice."""
    user, _ = populated
    assert await count(db, model, user.id) == 1
    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert await count(db, model, user.id) == 0


async def test_the_account_itself_goes(db, populated):
    user, _ = populated
    uid, model = user.id, type(user)
    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert await db.get(model, uid) is None


@pytest.mark.parametrize("model", [Analysis, UsageEvent, Feedback, Order])
async def test_nobody_elses_data_is_touched(db, populated, model):
    user, other = populated
    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert await count(db, model, other.id) == 1
    assert await db.get(type(other), other.id) is not None


async def test_deleting_with_an_undelivered_paid_order_still_succeeds(db, make_user):
    """It is their account. The obligation is logged for a refund rather than
    used to hold the account hostage."""
    user = await make_user(email="owed@example.com")
    user.password_hash = hash_password(PASSWORD)
    db.add(Order(user_id=user.id, stripe_session_id="cs_open", plan="expert",
                 status=ORDER_PAID, created_at_ms=1, updated_at_ms=1))
    await db.commit()
    uid, model = user.id, type(user)

    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert await db.get(model, uid) is None


async def test_an_account_with_no_data_deletes_cleanly(db, make_user):
    user = await make_user(email="fresh@example.com")
    user.password_hash = hash_password(PASSWORD)
    await db.commit()
    uid, model = user.id, type(user)
    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert await db.get(model, uid) is None


# --------------------------------------------------------------------------
# A live subscription is cancelled in Stripe, or the deletion does not happen.
#
# Before this, deleting the account removed the row Stripe's webhooks resolve a
# customer to and nothing else: the card kept being charged, the renewal found
# nobody (WEBHOOK_NO_USER), and the customer's first clue was a statement line
# from a service they had left -- a chargeback, not a support ticket.
# --------------------------------------------------------------------------
class _FakeStripe:
    """Just enough of ``stripe.Subscription`` to see what deletion asks of it."""

    def __init__(self, subs, fail=False):
        self.subs = subs
        self.fail = fail
        self.cancelled: list[str] = []
        self.listed_for: list[str] = []

    def list(self, customer, status, limit):
        self.listed_for.append(customer)
        fake = self

        class _Page:
            def auto_paging_iter(self):
                return iter(fake.subs)

        return _Page()

    def cancel(self, sub_id):
        if self.fail:
            import stripe as _stripe

            raise _stripe.StripeError("boom")
        self.cancelled.append(sub_id)


class _Sub:
    def __init__(self, id, status):
        self.id, self.status = id, status


@pytest.fixture
def stripe_on(monkeypatch):
    """Billing configured (the real settings object is frozen) and the
    Subscription API swapped for a recorder."""
    import dataclasses

    from app.api import billing
    from app.core.config import settings

    monkeypatch.setattr(
        billing, "settings", dataclasses.replace(settings, stripe_secret_key="sk_test_x"),
    )

    def _install(subs, fail=False):
        fake = _FakeStripe(subs, fail)
        monkeypatch.setattr(billing.stripe, "Subscription", fake)
        return fake

    return _install


async def _subscriber(make_user, status="active"):
    user = await make_user(
        email="paying@example.com", tier="enthusiast",
        stripe_customer_id="cus_123", subscription_status=status,
    )
    user.password_hash = hash_password(PASSWORD)
    return user


async def test_deleting_a_subscriber_cancels_their_subscription(db, make_user, stripe_on):
    user = await _subscriber(make_user)
    await db.commit()
    fake = stripe_on([_Sub("sub_live", "active"), _Sub("sub_old", "canceled")])
    uid, model = user.id, type(user)

    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)

    assert fake.listed_for == ["cus_123"]
    assert fake.cancelled == ["sub_live"]        # the dead one is left alone
    assert await db.get(model, uid) is None


async def test_a_past_due_subscription_is_cancelled_too(db, make_user, stripe_on):
    """Still billing -- Stripe is retrying the card -- so still stopped."""
    user = await _subscriber(make_user, status="past_due")
    await db.commit()
    fake = stripe_on([_Sub("sub_retrying", "past_due")])
    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert fake.cancelled == ["sub_retrying"]


async def test_if_stripe_refuses_nothing_is_deleted(db, make_user, stripe_on):
    """A deletion that leaves a subscription billing is worse than one that
    has to be retried, so the failure is fatal and the account stays."""
    user = await _subscriber(make_user)
    await db.commit()
    stripe_on([_Sub("sub_live", "active")], fail=True)
    uid, model = user.id, type(user)

    with pytest.raises(HTTPException) as e:
        await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert e.value.status_code == 502
    assert "not deleted" in e.value.detail
    assert await db.get(model, uid) is not None


async def test_the_wrong_password_never_reaches_stripe(db, make_user, stripe_on):
    user = await _subscriber(make_user)
    await db.commit()
    fake = stripe_on([_Sub("sub_live", "active")])
    with pytest.raises(HTTPException):
        await delete_account(DeleteAccountBody(password="nope"), user, db)
    assert fake.cancelled == [] and fake.listed_for == []


async def test_a_free_account_never_touches_stripe(db, make_user, stripe_on):
    """No customer, no subscription: no API call, and no 503 from a deployment
    where billing is not configured at all."""
    user = await make_user(email="free@example.com")
    user.password_hash = hash_password(PASSWORD)
    await db.commit()
    fake = stripe_on([_Sub("sub_x", "active")])
    uid, model = user.id, type(user)
    await delete_account(DeleteAccountBody(password=PASSWORD), user, db)
    assert fake.listed_for == []
    assert await db.get(model, uid) is None
