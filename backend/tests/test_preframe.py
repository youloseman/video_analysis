"""The spent-preview line above the Analyze button (step 4 of the paywall read).

Paywall read of 2026-10-28: the one free full report is spent on the first
clip, and the person who had spent it learned that only after the NEXT clip
was analysed, on the teaser card. Now /auth/me says whether it is spent, when,
and on which history entry (``preview_state``), and the uploader says it
before the button: the date, that this clip shows the score and the skeleton,
and the two ways to a full report. Video only; a photo is the teaser on a free
account regardless and its own banner says so.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.api.auth import PreviewOut, UserOut
from app.models.analysis import Analysis
from app.models.user import TIER_STARTER
from app.services.preview_release import preview_state

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
SPA = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "app.css").read_text(encoding="utf-8")


async def test_a_spent_preview_is_dated_by_the_analysis_the_claim_points_at(db, make_user):
    user = await make_user(TIER_STARTER, free_preview_job_id="job-first-clip")
    db.add(Analysis(user_id=user.id, client_id="h-first", job_id="job-first-clip",
                    created_at_ms=1_761_000_000_000, sport="bike", kind="video",
                    data={"sport": "bike", "kind": "video"}))
    await db.commit()
    assert await preview_state(db, user, available=False) == {
        "available": False, "spent_at_ms": 1_761_000_000_000, "entry_id": "h-first",
    }


async def test_still_to_spend_says_only_that(db, make_user):
    user = await make_user(TIER_STARTER)
    assert await preview_state(db, user, available=True) == {"available": True}


async def test_spent_without_a_saved_row_is_spent_without_a_date(db, make_user):
    """A job still running, or a photo claim from the old rule: no row to
    date it by, and the line says "is spent" rather than inventing a day."""
    running = await make_user(TIER_STARTER, free_preview_job_id="job-running")
    photo = await make_user(TIER_STARTER, free_preview_job_id="photo:1700000000000")
    for user in (running, photo):
        assert await preview_state(db, user, available=False) == {
            "available": False, "spent_at_ms": None, "entry_id": None,
        }


async def test_another_accounts_analysis_never_dates_this_ones_preview(db, make_user):
    other = await make_user(TIER_STARTER)
    me = await make_user(TIER_STARTER, free_preview_job_id="job-shared-id")
    db.add(Analysis(user_id=other.id, client_id="h-other", job_id="job-shared-id",
                    created_at_ms=5, sport="run", kind="video", data={}))
    await db.commit()
    assert (await preview_state(db, me, available=False))["entry_id"] is None


def test_auth_me_carries_the_preview_and_the_sign_in_answers_leave_it_out():
    import inspect

    from app.api import auth

    assert UserOut.model_fields["preview"].default is None
    assert PreviewOut(available=False).model_dump() == {
        "available": False, "spent_at_ms": None, "entry_id": None,
    }
    src = inspect.getsource(auth.me)
    assert "preview_available(user)" in src and "preview_state(db, user" in src
    assert "preview" not in inspect.getsource(auth._user_out)


@pytest.mark.parametrize("needle", [
    'id="preframe"',
    "function syncPreframe(){",
    "state.mode!=='photo'",
    "Your free full report ${when}. This clip shows the <b>score and the skeleton</b>.",
    "track('preframe_shown', {sport:state.sport, has_date:!!pv.spent_at_ms, has_last:!!last});",
    "track('preframe_clicked', {to:a.dataset.pf, sport:state.sport});",
    # The "last report" link opens a locked clip, never a photo and never the
    # preview entry (which is the full report already).
    "e.kind!=='photo' && e.locked && e.lockReason && e.lockReason!=='preview'",
    "e.lockReason=res.locked.reason||'locked';",
    # Spent in place when the preview is shown; filled from /auth/me at boot
    # and after a login.
    "if(auth && res.locked && res.locked.reason==='preview') auth.preview={available:false, spent_at_ms:Date.now(), entry_id:null};",
    "auth.preview=u.preview||null; syncPreframe();",
])
def test_the_uploader_says_it_before_the_button(needle):
    assert needle in SPA


def test_the_line_is_drawn_when_auth_arrives_and_when_the_mode_changes():
    assert SPA.count("syncPreframe();") >= 3
    assert ".preframe{" in CSS and "var(--st-warn-bg)" in CSS
