"""What a $4 unlock hands over — and that it exists to be handed over.

`test_unlock.py` covers who may read what. This covers whether there is
anything to read, which turned out to be a different question.

A free run was rendered as a free run and stored that way: the keyframe drawn
with its angle numbers burned out, no kinogram built, no overlay video. Then
somebody paid, the gate opened, and the report underneath was the teaser --
because the paid version had never been made, and the clip it could have been
made from expires in `job_ttl_hours`. Coaching survived this only because it is
a Gemini call that works just as well later (see me._ensure_coaching); anything
that needs the footage does not get a second chance.

Meanwhile the card sold nine things against one bullet list, for two buttons
with different scopes. One of the nine, `second_phase` -- "Second phase photo
(contact & drive)" -- had no implementation anywhere in the repository. It was
a label, and it was being charged for.

So, three properties: render the sellable artifacts while the footage is still
here, hand the teaser a copy that is not the sellable one, and promise only
what the button being clicked can deliver.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api import me
from app.models.analysis import Analysis
from app.models.user import TIER_ENTHUSIAST, TIER_STARTER
from app.services.result_gating import (
    ACCESS_FULL,
    UNLOCK_SCOPE,
    _UNLOCKS,
    gate_for_access,
    gate_free_result,
    gate_preview_result,
)

BACKEND = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spa() -> str:
    return (BACKEND / "app" / "static" / "index.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def me_py() -> str:
    return (BACKEND / "app" / "api" / "me.py").read_text(encoding="utf-8")


PAID_FRAME = "data:image/jpeg;base64,PAID-with-angles"
FREE_FRAME = "data:image/jpeg;base64,FREE-numbers-hidden"

RESULT = {
    "status": "completed",
    "sport_type": "run",
    "technique_score": 74,
    "letter_grade": "C",
    "keyframe_base64": PAID_FRAME,
    "keyframe_free_base64": FREE_FRAME,
    "kinogram_base64": "data:image/jpeg;base64,KINO",
    "angle_statistics": {"knee": {"mean": 142.0}},
}


async def store(db, user, *, client_id="h1", unlocked=None, result=RESULT):
    row = Analysis(
        user_id=user.id, client_id=client_id, job_id="j1", created_at_ms=1,
        data={"id": client_id}, result=result, unlocked_at_ms=unlocked,
    )
    db.add(row)
    await db.commit()
    return row


# --------------------------------------------------------------------------
# the frame: one is stored, two are served
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gate", [gate_free_result, gate_preview_result])
def test_an_unpaid_reader_gets_the_number_free_copy(gate):
    out = gate(RESULT)
    assert out["keyframe_base64"] == FREE_FRAME


@pytest.mark.parametrize("gate", [gate_free_result, gate_preview_result])
def test_the_stored_frame_never_leaks_under_its_own_name(gate):
    """Serving both would hand the angles over in the second field."""
    out = gate(gate(RESULT))          # idempotent, and still not leaking
    assert PAID_FRAME not in str(out)
    assert "keyframe_free_base64" not in out


def test_a_paid_reader_gets_the_frame_with_its_numbers():
    out = gate_for_access(RESULT, ACCESS_FULL)
    assert out["keyframe_base64"] == PAID_FRAME


def test_a_paid_reader_is_not_shipped_the_teaser_copy_as_well():
    """Two full-size images on a report that already carries two."""
    assert "keyframe_free_base64" not in gate_for_access(RESULT, ACCESS_FULL)


def test_a_result_without_a_free_copy_passes_through_untouched():
    """Paid runs render one frame, not two. Identity, not a rebuilt dict."""
    plain = {k: v for k, v in RESULT.items() if k != "keyframe_free_base64"}
    assert gate_for_access(plain, ACCESS_FULL) is plain


def test_a_teaser_still_has_a_frame_when_only_one_was_rendered():
    """Older stored results, and the paid runs that never needed a copy."""
    plain = {k: v for k, v in RESULT.items() if k != "keyframe_free_base64"}
    assert gate_free_result(plain)["keyframe_base64"] == PAID_FRAME


# --------------------------------------------------------------------------
# the kinogram: made for every run, because it cannot be made later
# --------------------------------------------------------------------------

def test_the_kinogram_is_rendered_for_free_runs_too():
    """`kinogram=not free` is what left a bought report with a hole in it. The
    footage is gone in hours; the decision cannot be deferred to payment the
    way the Gemini call is."""
    src = (BACKEND / "app" / "main.py").read_text(encoding="utf-8")
    assert "kinogram=not free" not in src
    assert "kinogram=True" in src


def test_the_kinogram_is_still_withheld_from_a_free_reader():
    """Rendered to be sold, not to be shown."""
    assert "kinogram_base64" not in gate_free_result(RESULT)
    assert "kinogram_base64" not in gate_preview_result(RESULT)


async def test_the_kinogram_endpoint_serves_a_bought_report(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, unlocked=123)
    assert (await me.get_kinogram("h1", user, db))["kinogram"] == RESULT["kinogram_base64"]


async def test_a_subscriber_gets_it_without_buying_the_report(db, make_user):
    user = await make_user(tier=TIER_ENTHUSIAST)
    await store(db, user)
    assert (await me.get_kinogram("h1", user, db))["kinogram"]


async def test_the_kinogram_endpoint_refuses_an_unpaid_report(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    with pytest.raises(HTTPException) as exc:
        await me.get_kinogram("h1", user, db)
    assert exc.value.status_code == 402


async def test_the_kinogram_endpoint_does_not_reach_another_account(db, make_user):
    mine, theirs = await make_user(), await make_user()
    await store(db, theirs, unlocked=123)
    with pytest.raises(HTTPException) as exc:
        await me.get_kinogram("h1", mine, db)
    assert exc.value.status_code == 404


# --------------------------------------------------------------------------
# the card: promise what the button delivers
# --------------------------------------------------------------------------

def test_the_phantom_feature_is_no_longer_sold():
    """Nothing has ever produced a second phase photo.

    Checked against the lists and the label map rather than the file text: the
    comments explaining why it was removed mention it by name, and they are the
    reason nobody re-adds it.
    """
    from app.services.result_gating import _PREVIEW_UNLOCKS

    for names in (_UNLOCKS, _PREVIEW_UNLOCKS, UNLOCK_SCOPE):
        assert "second_phase" not in names
    spa = (BACKEND / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "second_phase:" not in spa      # the UNLOCK_COPY label


def test_it_is_not_quietly_reintroduced_by_a_gated_result():
    """The label map is keyed off whatever the server sends, so a stray entry
    in either list puts the bullet back on the card."""
    for gate in (gate_free_result, gate_preview_result):
        locked = gate(RESULT)["locked"]
        assert "second_phase" not in locked["unlocks"]
        assert "second_phase" not in locked["unlock_scope"]


def test_the_one_off_unlock_does_not_promise_the_video():
    """The overlay is rendered only for a paying run and lives on the job,
    which expires. Buying a stored report cannot produce one."""
    assert "video" in _UNLOCKS
    assert "video" not in UNLOCK_SCOPE


@pytest.mark.parametrize("key", ["coaching", "angles", "issues", "ranges",
                                 "kinogram", "export", "fit"])
def test_the_unlock_scope_is_everything_it_can_actually_deliver(key):
    assert key in UNLOCK_SCOPE


def test_the_gate_publishes_the_scope_alongside_the_full_list():
    locked = gate_free_result(RESULT)["locked"]
    assert locked["unlocks"] and locked["unlock_scope"]
    assert set(locked["unlock_scope"]) < set(locked["unlocks"])


def test_the_card_reads_the_scope_the_server_sends(spa):
    """A split the client invents is a split that drifts."""
    assert "locked.unlock_scope" in spa
    assert "planOnly" in spa


def test_the_boundary_is_marked_only_when_there_are_two_offers(spa):
    """With no $4 button there is no second offer to distinguish the plan
    from, and the tag would make the plan look like it withholds something."""
    assert "it.planOnly&&opts.unlockId" in spa


# --------------------------------------------------------------------------
# ...and the kinogram is somewhere the buyer can see it
# --------------------------------------------------------------------------

def test_a_saved_report_has_somewhere_to_show_its_kinogram(spa):
    """Rendering it for every run buys nothing if the screen a bought report
    opens on has no kinogram section -- which is what history had, for every
    tier, since it was built."""
    assert 'id="hdKinogramBlock"' in spa
    assert "renderHistoryKinogram(e)" in spa


def test_the_entry_carries_a_flag_and_not_the_picture(spa):
    """Dozens of entries, two full-size images each, almost none opened."""
    assert "if(res.kinogram_base64) e.hasKinogram=true;" in spa
    assert "kinogram:res.kinogram_base64" not in spa


def test_the_picture_is_fetched_from_the_endpoint_that_gates_it(spa, me_py):
    assert "'/me/analyses/'+encodeURIComponent(e.id)+'/kinogram'" in spa
    assert '@router.get("/analyses/{client_id}/kinogram")' in me_py


def test_an_unpaid_entry_is_not_asked_for_one(spa):
    """The endpoint answers 402, and an empty frame reads as a broken page."""
    assert "e.access!=='full'" in spa


# --------------------------------------------------------------------------
# ...and so is the export, which was promised off an expiring job
# --------------------------------------------------------------------------

async def test_the_export_works_from_the_saved_report(db, make_user):
    """`/jobs/<id>/export` answers off the job, which is swept within hours,
    and gates on the tier -- so it could never serve an unlocked report, and it
    stopped serving a subscriber's own report the moment the clip went."""
    user = await make_user(tier=TIER_STARTER)
    await store(db, user, unlocked=123)
    r = await me.export_analysis("h1", "md", user, db)
    assert r.status_code == 200
    assert b"knee" in r.body.lower()
    assert "attachment" in r.headers["content-disposition"]


async def test_the_export_refuses_a_report_that_was_not_bought(db, make_user):
    user = await make_user(tier=TIER_STARTER)
    await store(db, user)
    with pytest.raises(HTTPException) as exc:
        await me.export_analysis("h1", "md", user, db)
    assert exc.value.status_code == 402


async def test_the_export_offers_json_too(db, make_user):
    user = await make_user(tier=TIER_ENTHUSIAST)
    await store(db, user)
    assert (await me.export_analysis("h1", "json", user, db)).status_code == 200


async def test_an_unknown_format_is_a_400_not_a_markdown_file(db, make_user):
    user = await make_user(tier=TIER_ENTHUSIAST)
    await store(db, user)
    with pytest.raises(HTTPException) as exc:
        await me.export_analysis("h1", "pdf", user, db)
    assert exc.value.status_code == 400


async def test_the_export_does_not_reach_another_account(db, make_user):
    mine, theirs = await make_user(), await make_user()
    await store(db, theirs, unlocked=123)
    with pytest.raises(HTTPException) as exc:
        await me.export_analysis("h1", "md", mine, db)
    assert exc.value.status_code == 404


def test_the_cabinet_asks_the_stored_route_not_the_job(spa, me_py):
    assert "'/me/analyses/'+encodeURIComponent(e.id)+'/export?format='+fmt" in spa
    assert '@router.get("/analyses/{client_id}/export")' in me_py
    assert 'id="hdExportBlock"' in spa


def test_the_export_offer_is_hidden_from_a_reader_who_would_get_a_402(spa):
    fn = spa[spa.index("function renderHistoryExport("):]
    fn = fn[:fn.index("\n}")]
    assert "e.access==='full'" in fn


def test_the_preview_scope_never_offers_what_the_preview_already_showed():
    """`coaching` and `issues` are absent from the preview's unlock list; the
    scope is a filter over that list, so it must not reintroduce them."""
    locked = gate_preview_result(RESULT)["locked"]
    assert "coaching" not in locked["unlock_scope"]
    assert set(locked["unlock_scope"]) <= set(locked["unlocks"])
