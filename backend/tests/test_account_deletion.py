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
