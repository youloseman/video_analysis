"""Which stored clips survive a sweep.

This policy replaced a single rule -- delete everything six hours after the
analysis -- that was written when a clip existed only long enough to be
measured. It is why the Expert Review template used to open by apologising that
the footage was gone, and it is the one part of the product where being wrong
in either direction is expensive: delete too eagerly and somebody who paid for
a human to watch their clip gets an apology instead, keep too long and the
privacy policy becomes a false statement.

The pin is the subtle half. It cannot be "keep it once a review is bought",
because a review is bought days after the analysis, from the history screen --
at six hours' retention there was never anything left to pin. So retention is
decided when the file is written, by a rule that does not yet know whether the
clip will ever be reviewed, and the pin only ever *extends* that.
"""

from __future__ import annotations

from app.models.analysis import Analysis
from app.models.order import (
    ORDER_DELIVERED,
    ORDER_IN_REVIEW,
    ORDER_PAID,
    ORDER_REFUNDED,
    Order,
)
from app.models.user import TIER_ADMIN, TIER_COACH, TIER_ENTHUSIAST, TIER_FULL, TIER_STARTER
from app.services.retention import (
    DAY_MS,
    REVIEW_GRACE_DAYS,
    RETENTION_DAYS,
    YEAR_FROM_MS,
    job_ids_for_user,
    job_ids_to_keep,
    retention_days,
)

NOW = 1_760_000_000_000  # fixed clock; nothing here may depend on the real one


def days_ago(n: float) -> int:
    return int(NOW - n * DAY_MS)


async def add_analysis(db, user, *, job_id, client_id="h1", at=None):
    row = Analysis(
        user_id=user.id, client_id=client_id, job_id=job_id,
        created_at_ms=at if at is not None else NOW, data={"id": client_id},
    )
    db.add(row)
    await db.commit()
    return row


async def add_order(db, user, *, client_id="h1", status=ORDER_PAID,
                    delivered_at_ms=None, session="cs_1"):
    row = Order(
        user_id=user.id, stripe_session_id=session, plan="expert",
        status=status, analysis_client_id=client_id,
        created_at_ms=NOW, updated_at_ms=NOW, delivered_at_ms=delivered_at_ms,
    )
    db.add(row)
    await db.commit()
    return row


# --------------------------------------------------------------------------
# Baseline, by tier
# --------------------------------------------------------------------------
def test_an_unknown_tier_gets_the_shortest_period():
    """Erring short is the safe direction: keeping footage longer than promised
    is the failure that matters here."""
    assert retention_days("something-new") == RETENTION_DAYS[TIER_STARTER]
    assert retention_days(None) == RETENTION_DAYS[TIER_STARTER]


async def test_a_fresh_clip_is_kept(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(1))
    assert await job_ids_to_keep(db, NOW) == {"j1"}


async def test_a_free_clip_is_dropped_after_the_free_period(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    limit = RETENTION_DAYS[TIER_STARTER]
    await add_analysis(db, user, job_id="j1", at=days_ago(limit + 1))
    assert await job_ids_to_keep(db, NOW) == set()


async def test_a_paid_clip_outlives_the_free_period(db, make_user):
    """The same age, a different answer -- which is the whole point of tiering
    retention rather than picking one number."""
    free_limit = RETENTION_DAYS[TIER_STARTER]
    paid = await make_user(tier=TIER_ENTHUSIAST)
    await add_analysis(db, paid, job_id="paid", at=days_ago(free_limit + 1))
    assert await job_ids_to_keep(db, NOW) == {"paid"}


async def test_even_a_paid_clip_expires_eventually(db, make_user):
    user = await make_user(tier=TIER_ENTHUSIAST)
    await add_analysis(
        db, user, job_id="j1", at=days_ago(RETENTION_DAYS[TIER_ENTHUSIAST] + 1),
    )
    assert await job_ids_to_keep(db, NOW) == set()


async def test_an_analysis_with_no_stored_upload_contributes_nothing(db, make_user):
    """Photo analyses never touch the disk, so they have no job id and must not
    turn into a None in the keep-set."""
    user = await make_user(tier=TIER_FULL)
    await add_analysis(db, user, job_id=None)
    assert await job_ids_to_keep(db, NOW) == set()


# --------------------------------------------------------------------------
# The pin
# --------------------------------------------------------------------------
async def test_an_unfulfilled_review_pins_a_clip_past_its_expiry(db, make_user):
    """An order that is paid for and still owed keeps its footage however old
    it is. A deadline of ours must not run out while somebody is waiting on
    work from us."""
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(365))
    await add_order(db, user, status=ORDER_PAID)
    assert await job_ids_to_keep(db, NOW) == {"j1"}


async def test_a_review_being_written_pins_too(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(365))
    await add_order(db, user, status=ORDER_IN_REVIEW)
    assert await job_ids_to_keep(db, NOW) == {"j1"}


async def test_a_just_delivered_review_holds_its_clip_for_the_grace_window(db, make_user):
    """So the reviewer can still correct something they wrote about footage
    that would otherwise already be deleted."""
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(365))
    await add_order(
        db, user, status=ORDER_DELIVERED,
        delivered_at_ms=days_ago(REVIEW_GRACE_DAYS - 1),
    )
    assert await job_ids_to_keep(db, NOW) == {"j1"}


async def test_the_pin_lets_go_once_the_grace_window_passes(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(365))
    await add_order(
        db, user, status=ORDER_DELIVERED,
        delivered_at_ms=days_ago(REVIEW_GRACE_DAYS + 1),
    )
    assert await job_ids_to_keep(db, NOW) == set()


async def test_a_refunded_order_pins_nothing(db, make_user):
    """They are not owed a review, so we are not owed their footage."""
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(365))
    await add_order(db, user, status=ORDER_REFUNDED)
    assert await job_ids_to_keep(db, NOW) == set()


async def test_one_athletes_order_cannot_pin_anothers_footage(db, make_user):
    """``client_id`` is minted on the device and is only unique per account, so
    joining an order to an analysis on it alone would cross accounts -- and the
    ids collide in practice, being 'h' plus a millisecond timestamp."""
    buyer = await make_user(tier=TIER_STARTER)
    stranger = await make_user(tier=TIER_STARTER)
    await add_analysis(db, buyer, job_id="mine", client_id="h777", at=days_ago(365))
    await add_analysis(db, stranger, job_id="theirs", client_id="h777", at=days_ago(365))
    await add_order(db, buyer, client_id="h777", status=ORDER_PAID)

    assert await job_ids_to_keep(db, NOW) == {"mine"}


async def test_an_order_pointing_at_nothing_is_harmless(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await add_analysis(db, user, job_id="j1", at=days_ago(365))
    await add_order(db, user, client_id="h-does-not-exist", status=ORDER_PAID)
    assert await job_ids_to_keep(db, NOW) == set()


# --------------------------------------------------------------------------
# Deleting on request
# --------------------------------------------------------------------------
async def test_lookup_for_deletion_finds_a_users_stored_clips(db, make_user):
    user = await make_user(tier=TIER_FULL)
    await add_analysis(db, user, job_id="j1", client_id="h1")
    await add_analysis(db, user, job_id="j2", client_id="h2")
    await add_analysis(db, user, job_id=None, client_id="h3")

    assert set(await job_ids_for_user(db, user.id)) == {"j1", "j2"}
    assert await job_ids_for_user(db, user.id, ["h2"]) == ["j2"]


async def test_lookup_for_deletion_never_reaches_another_account(db, make_user):
    mine = await make_user()
    theirs = await make_user()
    await add_analysis(db, theirs, job_id="theirs", client_id="h1")
    assert await job_ids_for_user(db, mine.id) == []


# --------------------------------------------------------------------------
# A year on a paid plan, from 11 November 2026 (Artur, 2026-11-10)
# --------------------------------------------------------------------------
LATER = YEAR_FROM_MS + 400 * DAY_MS   # a sweep well into the year era


def later_days_ago(n: float) -> int:
    return int(LATER - n * DAY_MS)


async def test_a_paid_clip_is_kept_for_a_year_then_deleted(db, make_user):
    for tier in (TIER_ENTHUSIAST, TIER_FULL, TIER_COACH):
        user = await make_user(tier=tier)
        await add_analysis(db, user, job_id=f"{tier}-old", at=later_days_ago(300))
        await add_analysis(db, user, job_id=f"{tier}-gone", client_id="h2", at=later_days_ago(366))
    keep = await job_ids_to_keep(db, LATER)
    assert keep == {f"{t}-old" for t in (TIER_ENTHUSIAST, TIER_FULL, TIER_COACH)}


async def test_the_free_plan_and_staff_did_not_get_the_year(db, make_user):
    free = await make_user(tier=TIER_STARTER)
    staff = await make_user(tier=TIER_ADMIN)
    await add_analysis(db, free, job_id="free", at=later_days_ago(8))
    await add_analysis(db, staff, job_id="staff", at=later_days_ago(31))
    assert await job_ids_to_keep(db, LATER) == set()


async def test_a_clip_uploaded_under_thirty_days_keeps_thirty_days(db, make_user):
    """Not kept longer than the page said when it was uploaded."""
    user = await make_user(tier=TIER_FULL)
    sweep = YEAR_FROM_MS + 20 * DAY_MS
    await add_analysis(db, user, job_id="before", at=YEAR_FROM_MS - 11 * DAY_MS)
    await add_analysis(db, user, job_id="after", client_id="h2", at=YEAR_FROM_MS + DAY_MS)
    assert await job_ids_to_keep(db, sweep) == {"after"}
    assert retention_days(TIER_COACH, YEAR_FROM_MS - 1) == 60
    assert retention_days(TIER_COACH, YEAR_FROM_MS) == 365


async def test_a_plan_that_ends_puts_the_free_period_on_old_clips(db, make_user):
    user = await make_user(tier=TIER_FULL)
    await add_analysis(db, user, job_id="j1", at=later_days_ago(60))
    assert await job_ids_to_keep(db, LATER) == {"j1"}
    user.tier = TIER_STARTER
    await db.commit()
    assert await job_ids_to_keep(db, LATER) == set()


def test_the_privacy_policy_states_the_year():
    from pathlib import Path
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "privacy.html").read_text(encoding="utf-8")
    assert "<b>12 months</b> on a paid plan" in html and "11 November 2026" in html
    assert "30 days</b> on a paid plan" not in html and "60 days</b> from the analysis" not in html
