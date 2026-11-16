"""Which parts fit this frame: the data's rules, and the answers drawn from it.

Two halves. The shipped files are checked against the rules every record must
keep (a source for every part, a status, a reason for every "does not fit").
The questions are tested against a small frame written here, so they do not
move when the real data does.
"""

from __future__ import annotations

import json

import pytest

from app.api import me
from app.models.user import TIER_FULL
from app.services import parts, report_chat


def part(pid, status="manufacturer", *, reach=None, stack=None, angle=None, category="extensions", **kw):
    p = {
        "id": pid, "brand": "Brand", "model": pid.replace("-", " ").title(), "category": category,
        "effect": {"reach_mm": reach, "stack_mm": stack, "angle_deg": angle, "text": ""},
        "status": status,
        "evidence": [{"url": "https://example.com/" + pid, "kind": "manufacturer", "says": "fits"}],
        "url": None, "price": None, "needs": None, "checked": "2026-11-16",
    }
    if status == "incompatible":
        p["reason_incompatible"] = "Clamp is 31.8 mm, the riser takes 22.2 mm."
    p.update(kw)
    return p


FRAME = {
    "id": "testbike-gen1-2020",
    "family": "giant-trinity",
    "generation": "Testbike gen 1",
    "years": [2020, 2024],
    "brakes": "disc",
    "identify": "The one with the red fork.",
    "cockpit": {"type": "integrated_proprietary", "stack_system": "10 mm spacers, 0-80 mm",
                "reach_system": "flip the pads, 0-75 mm", "angle_system": "10/15 deg wedges"},
    "stock_range": [{"size": "M", "pad_stack_mm": [600, 680], "pad_reach_mm": [380, 455],
                     "pad_reference": "back_of_pad", "source": "https://example.com/geo"}],
    "parts": [
        part("plate-forward", reach=[0, 35], category="adapter_plate"),
        part("long-ext", "owners", reach=[0, 60]),
        part("riser-tall", "single_report", stack=[20, 40], category="riser"),
        part("wedge-15", angle=[0, 15], category="wedge"),
        part("std-stem", "incompatible", category="stem"),
    ],
    "gaps": ["no XL size"],
    "checked": "2026-11-16",
}


@pytest.fixture
def one_frame(tmp_path, monkeypatch):
    (tmp_path / f"{FRAME['id']}.json").write_text(json.dumps(FRAME), encoding="utf-8")
    monkeypatch.setattr(parts, "DATA_DIR", tmp_path)
    parts.load_frames.cache_clear()
    yield FRAME
    parts.load_frames.cache_clear()


# --------------------------------------------------------------------------
# The shipped data keeps the rules
# --------------------------------------------------------------------------
def test_every_shipped_frame_file_passes():
    files = sorted(parts.DATA_DIR.glob("*.json"))
    errs = []
    for path in files:
        errs += parts.problems(json.loads(path.read_text(encoding="utf-8")), path.name)
    assert not errs, "\n".join(errs)


def test_shipped_frames_name_a_known_family():
    from app.services.bike_catalog import BIKE_FAMILIES

    parts.load_frames.cache_clear()
    for f in parts.load_frames().values():
        assert f["family"] in BIKE_FAMILIES, f["id"]


@pytest.mark.parametrize("breakage,msg", [
    (lambda p: p.pop("evidence"), "evidence"),
    (lambda p: p.update(status="probably"), "status"),
    (lambda p: p.update(status="incompatible", reason_incompatible=None), "reason_incompatible"),
    (lambda p: p.update(category="gizmo"), "category"),
    (lambda p: p["effect"].update(reach_mm=[30, 0]), "effect.reach_mm"),
    (lambda p: p.update(price={"amount": 99}), "price"),
])
def test_a_record_without_its_proof_is_refused(breakage, msg):
    f = json.loads(json.dumps(FRAME))
    breakage(f["parts"][0])
    errs = parts.problems(f)
    assert any(msg in e for e in errs), errs


def test_a_broken_file_is_skipped_not_half_served(tmp_path, monkeypatch):
    bad = json.loads(json.dumps(FRAME))
    bad["parts"][0].pop("evidence")
    (tmp_path / f"{bad['id']}.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "junk.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(parts, "DATA_DIR", tmp_path)
    parts.load_frames.cache_clear()
    try:
        assert parts.load_frames() == {}
    finally:
        parts.load_frames.cache_clear()


# --------------------------------------------------------------------------
# The answers
# --------------------------------------------------------------------------
def test_more_reach_lists_the_parts_that_move_forward(one_frame):
    out = parts.suggest(one_frame["id"], size="m", reach_mm=50)
    ids = [p["id"] for p in out["helps"]]
    assert ids == ["long-ext", "plate-forward"]          # the one that covers 50 mm first
    assert out["helps"][0]["covers"] == ["reach_mm"]
    assert out["helps"][1]["covers"] == []
    assert out["stock"]["size"]["pad_reach_mm"] == [380, 455]
    assert "riser-tall" in [p["id"] for p in out["other"]]


def test_lower_pads_find_no_riser(one_frame):
    out = parts.suggest(one_frame["id"], stack_mm=-20)
    assert out["helps"] == []
    assert "10 mm spacers" in out["stock"]["stack_system"]


def test_what_does_not_fit_is_always_said(one_frame):
    out = parts.suggest(one_frame["id"], reach_mm=30)
    assert [p["id"] for p in out["incompatible"]] == ["std-stem"]
    assert out["incompatible"][0]["reason_incompatible"]
    assert out["incompatible"][0]["status_label"] == "Does not fit"


def test_an_unknown_frame_is_none(one_frame):
    assert parts.suggest("nope") is None


async def test_the_route_404s_on_an_unknown_frame(one_frame):
    from fastapi import HTTPException

    from app.api import parts as parts_api

    with pytest.raises(HTTPException) as e:
        await parts_api.suggest_parts("nope", None, 0, 0, 0, None)
    assert e.value.status_code == 404


def test_old_records_are_stale(one_frame):
    from datetime import date

    assert not parts.stale(one_frame, date(2026, 12, 1))
    assert parts.stale(one_frame, date(2027, 6, 1))


# --------------------------------------------------------------------------
# The profile and the chat
# --------------------------------------------------------------------------
async def test_a_profile_keeps_a_known_generation_only(db, make_user, one_frame):
    u = await make_user(tier=TIER_FULL)
    p = await me.create_profile(me.ProfileIn(
        name="TT", sport="bike", bike_family="giant-trinity", bike_generation=one_frame["id"],
    ), u, db)
    assert p["bike_generation"] == one_frame["id"]
    assert me.ProfileIn(name="x", bike_generation="made-up-2099").bike_generation is None


def test_the_chat_may_name_only_the_parts_on_record(one_frame):
    md = parts.chat_block(one_frame["id"])
    assert "Name a part ONLY from this list" in md
    assert "Long Ext" in md and "Fitted by several owners" in md
    assert "does NOT fit: Clamp is 31.8 mm" in md
    assert "reach +0..+60 mm" in md
    assert parts.chat_block(None) is None
    assert "Parts on record for this frame" in report_chat.SYSTEM


async def test_a_question_about_a_filed_clip_carries_the_frames_parts(db, make_user, one_frame, monkeypatch):
    from app.api import chat as chat_api
    from app.models.analysis import Analysis

    seen = []
    monkeypatch.setattr(report_chat, "rollout", lambda: "all")
    monkeypatch.setattr(report_chat, "ask", lambda prompt: seen.append(prompt) or {"text": "ok", "model": "fake"})
    u = await make_user(tier=TIER_FULL)
    p = await me.create_profile(me.ProfileIn(
        name="TT", sport="bike", bike_family="giant-trinity", bike_generation=one_frame["id"],
    ), u, db)
    db.add(Analysis(user_id=u.id, client_id="h1", created_at_ms=1, profile_id=p["id"], sport="bike",
                    kind="video", score=84, data={"id": "h1"},
                    result={"sport_type": "bike", "technique_score": 84, "sport_specific_metrics": {}}))
    await db.commit()
    await chat_api.ask_chat("h1", chat_api.QuestionIn(question="Which extensions fit?"), u, db, None)
    assert "Parts on record for this frame (Testbike gen 1" in seen[0]


# --------------------------------------------------------------------------
# What riders tell us back
# --------------------------------------------------------------------------
async def test_a_fit_report_goes_to_the_queue_not_the_data(db, make_user, one_frame):
    from app.api import parts as parts_api
    from app.models.user import TIER_ADMIN

    u = await make_user(tier=TIER_FULL)
    out = await parts_api.report_part(one_frame["id"], parts_api.ReportIn(part_id="long-ext", fits=False), u, db)
    assert out["queued"]
    # The file is untouched: the part keeps the status its sources gave it.
    assert next(p for p in parts.load_frames()[one_frame["id"]]["parts"] if p["id"] == "long-ext")["status"] == "owners"

    admin = await make_user(tier=TIER_ADMIN)
    q = await parts_api.list_reports(True, admin, db)
    assert q["reports"][0]["part_id"] == "long-ext"
    assert q["reports"][0]["fits"] is False
    assert q["reports"][0]["status_on_file"] == "owners"
    await parts_api.mark_reviewed(q["reports"][0]["id"], admin, db)
    assert (await parts_api.list_reports(True, admin, db))["reports"] == []


async def test_a_report_must_name_a_part(db, make_user, one_frame):
    from fastapi import HTTPException

    from app.api import parts as parts_api

    u = await make_user(tier=TIER_FULL)
    with pytest.raises(HTTPException) as e:
        await parts_api.report_part(one_frame["id"], parts_api.ReportIn(part_id="not-listed", fits=True), u, db)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await parts_api.report_part(one_frame["id"], parts_api.ReportIn(fits=True), u, db)
    assert e.value.status_code == 400
    ok = await parts_api.report_part(
        one_frame["id"], parts_api.ReportIn(fits=True, note="Zipp Vuka Alumina fitted fine"), u, db,
    )
    assert ok["queued"]


async def test_the_queue_and_the_demand_are_admin_only(db, make_user, one_frame):
    from fastapi import HTTPException

    from app.api import parts as parts_api

    u = await make_user(tier=TIER_FULL)
    for call in (parts_api.list_reports(True, u, db), parts_api.frame_demand(u, db)):
        with pytest.raises(HTTPException) as e:
            await call
        assert e.value.status_code == 404


async def test_demand_says_which_bikes_lack_data(db, make_user, one_frame):
    from app.api import parts as parts_api
    from app.models.user import TIER_ADMIN

    u = await make_user(tier=TIER_FULL)
    for fam in ("giant-trinity", "giant-trinity", "felt-ia"):
        await me.create_profile(me.ProfileIn(name="b", sport="bike", bike_family=fam), u, db)
    admin = await make_user(tier=TIER_ADMIN)
    fams = {d["family"]: d for d in (await parts_api.frame_demand(admin, db))["families"]}
    assert fams["giant-trinity"]["profiles"] == 2 and fams["giant-trinity"]["data_held"]
    assert fams["felt-ia"]["profiles"] == 1 and not fams["felt-ia"]["data_held"]
