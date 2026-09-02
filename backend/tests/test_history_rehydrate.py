"""The client half of "paying opens the report".

`test_unlock.py` proves the server side thoroughly: the webhook writes the
entitlement, `access_for_stored` returns FULL, `/me/analyses/<id>/result`
serves the whole report, and the listing marks each entry with the access the
caller now has. All of it passed. All of it was correct.

Nothing called it.

A saved history entry is the client's own rendering of a report, frozen at
whatever it was allowed to see the day it ran -- and a teaser keeps the score
and the keyframe and strips every measurement. `openHistory` re-rendered that
frozen copy, so the first live $4 unlock charged the card, granted the
entitlement, and showed the athlete the same page they had already seen. The
one screen where the purchase was supposed to become visible was the one that
never asked the server anything.

Which is why these are cross-file tests: an endpoint with no caller is
invisible to a test suite that only ever checks the endpoint. The bug lived in
the gap between two halves that were each individually right.
"""

from __future__ import annotations

import re

import pytest

from tests.conftest import read_spa


@pytest.fixture(scope="module")
def spa() -> str:
    return read_spa()


@pytest.fixture(scope="module")
def me_py() -> str:
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "app" / "api" / "me.py").read_text(
        encoding="utf-8",
    )


def body_of(spa: str, name: str) -> str:
    """Source of one top-level `function name(...)`, to its closing brace."""
    start = spa.index(f"function {name}(")
    depth, i = 0, spa.index("{", start)
    for j in range(i, len(spa)):
        if spa[j] == "{":
            depth += 1
        elif spa[j] == "}":
            depth -= 1
            if depth == 0:
                return spa[start:j + 1]
    raise AssertionError(f"{name} never closes")


# --------------------------------------------------------------------------
# the endpoint has a caller
# --------------------------------------------------------------------------

def test_the_spa_fetches_the_stored_result(spa):
    """The bug in one assertion. This endpoint existed, was gated correctly,
    was tested, and was never requested by anything."""
    assert "/result'" in spa and "/me/analyses/" in spa


def test_the_url_the_client_builds_is_a_route_the_server_declares(spa, me_py):
    """Two files, one string. A rename on either side is a silent 404 that
    looks exactly like "the report just didn't unlock"."""
    assert '/me/analyses/\'+encodeURIComponent(e.id)+\'/result\'' in spa
    assert '@router.get("/analyses/{client_id}/result")' in me_py


# --------------------------------------------------------------------------
# opening a bought report shows the bought report
# --------------------------------------------------------------------------

def test_opening_a_saved_report_asks_whether_it_grew(spa):
    assert "rehydrateEntry(e)" in body_of(spa, "openHistory")


def test_the_refetch_is_not_awaited_before_the_first_paint(spa):
    """Almost every history open has nothing to fetch. Blocking all of them on
    a request to fix one is a tax everybody pays."""
    fn = body_of(spa, "openHistory")
    assert "renderHistoryDetail(e);" in fn
    assert fn.index("renderHistoryDetail(e);") < fn.index("rehydrateEntry(e)")
    assert "await rehydrateEntry" not in fn


def test_a_late_response_cannot_overwrite_a_different_report(spa):
    """The athlete may have navigated on while it was in flight."""
    fn = body_of(spa, "openHistory")
    assert "state.histOpen.id!==fresh.id" in fn


@pytest.mark.parametrize("guard", [
    "e.access!=='full'",     # not entitled to more than they hold
    "e.sellable===false",    # ran before results were stored: 409, not a report
    "entryLooksGated(e)",    # already holds the full thing
    "state.rehydrated[e.id]",  # once per session, not once per open
])
def test_the_refetch_is_skipped_when_it_would_gain_nothing(spa, guard):
    assert guard in body_of(spa, "rehydrateEntry")


def test_a_failed_refetch_leaves_the_athlete_with_what_they_had(spa):
    """A network blip must degrade to the stale entry, not to an error page."""
    fn = body_of(spa, "rehydrateEntry")
    assert "catch(err){ return null; }" in fn
    assert "if(!r.ok) return null;" in fn


# --------------------------------------------------------------------------
# one conversion, used twice
# --------------------------------------------------------------------------

def test_saving_and_rebuilding_share_the_same_conversion(spa):
    """Two functions turning a result into an entry would drift, and the
    rebuilt report would differ from the one a paying user sees on the day."""
    assert "entryFromResult(kind,res," in body_of(spa, "saveToHistory")
    assert "entryFromResult(e.kind, j.result, e)" in body_of(spa, "rehydrateEntry")
    assert len(re.findall(r"function entryFromResult\(", spa)) == 1


@pytest.mark.parametrize("field,expr", [
    ("id", "id:b.id"),                    # not a second entry
    ("at", "at:b.at||Date.now()"),        # not re-dated to today
    ("jobId", "jobId:isV?(b.jobId||null):null"),   # still points at the clip
    ("profileId", "profileId:b.profileId||null"),
    ("planDone", "e.planDone=b.planDone||{}"),     # ticked drills survive
])
def test_a_rebuild_keeps_what_identifies_the_entry(spa, field, expr):
    """`base` is what the entry IS; the result is only what it measured."""
    assert expr in body_of(spa, "entryFromResult"), field


def test_an_entry_records_that_it_was_built_from_a_gated_payload(spa):
    """Precise signal for later: the measurements were withheld, not absent."""
    assert "if(res.locked) e.locked=true;" in body_of(spa, "entryFromResult")


def test_older_entries_are_still_recognised_as_gated(spa):
    """Every report bought before that marker existed -- including the first
    live one -- has no `locked` flag. A teaser has a score and no measurements,
    so that has to be enough to spot one."""
    fn = body_of(spa, "entryLooksGated")
    assert "e.locked" in fn
    assert "angleStats" in fn and "metrics" in fn


# --------------------------------------------------------------------------
# the race the athlete would otherwise lose
# --------------------------------------------------------------------------

def test_the_unlock_return_waits_for_the_webhook(spa):
    """Stripe redirects the browser back immediately; the entitlement is
    written by the webhook a beat later. Opening the report in between shows
    the teaser they just paid to be rid of."""
    fn = body_of(spa, "handleCheckoutReturn") if "function handleCheckoutReturn(" \
        in spa else spa
    block = fn[fn.index("if(plan==='unlock'){"):]
    block = block[:block.index("toast('Payment received")]
    assert "row.access==='full'" in block
    assert "refreshCloud()" in block
    assert "setTimeout" in block


def test_the_wait_is_bounded(spa):
    """A webhook that never lands must not leave a spinner forever: open the
    report anyway, and the next visit picks the entitlement up."""
    block = spa[spa.index("if(plan==='unlock'){"):]
    block = block[:block.index("toast('Payment received")]
    assert re.search(r"for\(let i=0;i<\d+;i\+\+\)", block)
    assert "openHistory(id)" in block
