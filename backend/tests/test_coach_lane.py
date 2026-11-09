"""The coach's lane, the intake link, the shared report, and grants over a payment.

Pinned: a NO-GO costs no quota and stays out of History until the coach says run
it; a batch holds ten and one runs at a time; a restart loses nothing; the intake
link dies with the athlete's row or the coach's plan; a shared report ends when it
is switched off; a Full subscriber on a coach pilot goes back to Full.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.api import athletes as roster
from app.api import coach as coach_api
from app.api import coach_reports as reports
from app.api import coach_uploads as api
from app.api import me
from app.api.billing import _set_tier
# Imported up here, before any test swaps core_db.SessionLocal: main binds the name
# when it is first imported, and a lazy first import would bind the test's session.
from app.main import app as _app  # noqa: F401
from app.core import db as core_db
from app.core import jobs as core_jobs
from app.models.analysis import Analysis
from app.models.coach_upload import CoachUpload
from app.models.usage import UsageEvent
from app.models.user import TIER_COACH, TIER_FULL, TIER_STARTER
from app.services import coach_lane, notify
from app.services.intake_page import render_intake_page
from app.services.prescan import capture_check
from app.services.tier_grants import expire_tier_grants, grant_tier

GOOD = {"status": "completed", "result": {
    "status": "completed", "sport_type": "bike", "technique_score": 81, "letter_grade": "B",
    "keyframe_base64": "data:image/jpeg;base64,REAL", "cycling_position": "triathlon",
}}


def req(ip="1.2.3.4"):
    return SimpleNamespace(client=SimpleNamespace(host=ip), headers={})


@pytest.fixture
def lane(monkeypatch, db, tmp_path):
    """The lane on the test database, with the clip store and the analysis faked."""
    monkeypatch.setattr(core_db, "SessionLocal", async_sessionmaker(db.bind, expire_on_commit=False))
    state = SimpleNamespace(verdict="go", runs=[], job=GOOD, queued=[], stored=0)

    async def fake_store(video, suffix):
        state.stored += 1
        d = tmp_path / f"job{state.stored}"
        d.mkdir()
        p = d / f"input{suffix}"
        p.write_bytes(await video.read())
        return f"job{state.stored}", p, p.stat().st_size

    monkeypatch.setattr("app.main._store_clip", fake_store)
    monkeypatch.setattr(core_jobs, "job_file", lambda job_id, stem: tmp_path / job_id / "input.mp4")
    monkeypatch.setattr(coach_lane, "scan_clip", lambda path, sport, bars: {
        "verdict": state.verdict, "reasons": [{"code": "tiny", "level": "no_go", "text": "Too small."}]
        if state.verdict == "no_go" else [],
    })

    def fake_run(job_id, path, params, coach_id):
        state.runs.append(job_id)
        core_jobs.JOBS[job_id] = {"status": "completed", "result": state.job.get("result"), "owner_user_id": coach_id}
        return state.job

    monkeypatch.setattr(coach_lane, "_run_job", fake_run)
    state.shrunk = []
    monkeypatch.setattr(coach_lane, "_shrink", state.shrunk.append)
    monkeypatch.setattr(coach_lane, "enqueue", state.queued.append)
    monkeypatch.setattr(notify, "send_email", lambda *a, **k: True)
    from app.core.config import settings
    monkeypatch.setattr(type(settings), "model_path", property(lambda self: tmp_path), raising=False)
    return state


def clip(name="a.mp4"):
    return UploadFile(io.BytesIO(b"x" * 10), filename=name)


async def coach_with_athlete(make_user, db, **kw):
    coach = await make_user(tier=TIER_COACH, **kw)
    await coach_api.confirm_others(coach, db)
    a = await roster.create_athlete(roster.AthleteIn(name="Andy", email="andy@example.com"), coach, db)
    return coach, a


async def send(db, coach, a=None, batch="batch01", name="a.mp4"):
    return await api.upload_to_batch(
        clip(name), batch, "bike", None, None, a["id"] if a else None, None, coach, db,
    )


# --------------------------------------------------------------------------
# The lane
# --------------------------------------------------------------------------
async def test_a_clip_runs_and_lands_under_its_athlete(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    out = await send(db, coach, a)
    # In line: not in History yet.
    assert await me.list_analyses(coach, db) == []
    await coach_lane.run_upload(out["id"])
    rows = await me.list_analyses(coach, db)
    assert len(rows) == 1 and rows[0]["athleteId"] == a["id"] and rows[0]["serverFiled"] is True
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.keyframe == "data:image/jpeg;base64,REAL" and row.score == 81
    assert (await db.get(CoachUpload, out["id"])).status == "done"
    assert (await roster.list_athletes(coach, db))["athletes"][0]["analyses"] == 1


async def test_a_no_go_costs_no_quota_and_can_be_run_anyway(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    lane.verdict = "no_go"
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    assert lane.runs == [] and len(lane.shrunk) == 1
    assert (await db.execute(select(UsageEvent))).scalars().all() == []
    listed = await api.list_uploads(None, coach, db)
    assert listed["items"][0]["status"] == "no_go" and listed["items"][0]["reasons"][0]["text"] == "Too small."
    assert await me.list_analyses(coach, db) == []
    await api.run_anyway(out["id"], coach, db)
    await coach_lane.run_upload(out["id"])
    assert lane.runs == ["job1"]
    assert len(await me.list_analyses(coach, db)) == 1


async def test_a_finished_clip_carries_what_the_analysis_found(make_user, db, lane, monkeypatch):
    """The quick look's guess gives way to the measurement, and the coach sees it."""
    monkeypatch.setattr(coach_lane, "scan_clip", lambda path, sport, bars: {
        "verdict": "check", "reasons": [{"code": "not_found", "level": "check", "text": "Couldn't find you."}],
    })
    lane.job = {**GOOD, "result": {**GOOD["result"], "sport_specific_metrics": {"capture_report": {"checks": [
        {"id": "duration", "status": "bad", "measured": "9 s long"},
    ]}}}}
    coach, a = await coach_with_athlete(make_user, db)
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    item = (await api.list_uploads(None, coach, db))["items"][0]
    assert item["status"] == "done" and [r["code"] for r in item["reasons"]] == ["duration"]


async def test_a_finished_clip_with_no_capture_report_keeps_the_quick_look(make_user, db, lane, monkeypatch):
    monkeypatch.setattr(coach_lane, "scan_clip", lambda path, sport, bars: {
        "verdict": "check", "reasons": [{"code": "small", "level": "check", "text": "A bit small."}],
    })
    coach, a = await coach_with_athlete(make_user, db)
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    assert (await api.list_uploads(None, coach, db))["items"][0]["reasons"][0]["code"] == "small"


async def test_a_clip_keeps_the_date_it_was_filmed_and_the_clock_for_deleting_it(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    may = 1_780_000_000_000
    out = await api.upload_to_batch(clip("IMG_1.MOV"), "batch01", "bike", None, None, a["id"], None, coach, db, may)
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.data["filmedAt"] == may and row.data["fileName"] == "IMG_1.MOV"
    # Retention counts from the row: a clip filmed months ago must not expire on arrival.
    assert row.created_at_ms > may and row.data["at"] == row.created_at_ms
    await coach_lane.run_upload(out["id"])
    await db.refresh(row)  # the lane wrote through its own session
    assert (await me.list_analyses(coach, db))[0]["filmedAt"] == may


@pytest.mark.parametrize("bad", [0, 1_000_000, 9_999_999_999_999, "soon", None])
async def test_a_date_nobody_could_have_filmed_on_is_not_kept(make_user, db, lane, bad):
    coach, a = await coach_with_athlete(make_user, db)
    await api.upload_to_batch(clip(), "batch01", "bike", None, None, a["id"], None, coach, db, bad)
    assert "filmedAt" not in (await db.execute(select(Analysis))).scalar_one().data


async def test_an_unnamed_coach_never_shows_an_athlete_their_address(make_user, db, lane, monkeypatch):
    from app.api import coach_link

    coach, a = await coach_with_athlete(make_user, db, email="private.person@example.com")
    assert (await coach_link.coach_display(db, coach))["name"] == "Your coach"
    token = (await roster.list_athletes(coach, db))["athletes"][0]["upload_token"]
    page = (await api.intake_page(token, db)).body.decode()
    assert "private.person" not in page
    assert "send your clip to your coach" in page and "I want your coach to see" in page
    sent = []
    monkeypatch.setattr(notify, "send_email", lambda to, subject, text, html: sent.append((subject, text)) or True)
    reports._mail("andy@example.com", "Andy", "Your coach", "bike", "https://x/r/t")
    assert sent[0][0] == "Your coach sent you your ride report"
    assert "sent because your coach uploaded" in sent[0][1]


async def test_a_named_coach_keeps_their_name_in_every_sentence(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    await coach_api.put_coach(coach_api.CoachIn(name="Coach Mike"), coach, db)
    token = (await roster.list_athletes(coach, db))["athletes"][0]["upload_token"]
    page = (await api.intake_page(token, db)).body.decode()
    assert "send your clip to Coach Mike" in page and "I want Coach Mike to see" in page


async def test_a_clip_we_cannot_check_still_runs(make_user, db, lane, monkeypatch):
    coach, a = await coach_with_athlete(make_user, db)
    def boom(*a, **k):
        raise RuntimeError("no codec")
    monkeypatch.setattr(coach_lane, "scan_clip", boom)
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    assert (await db.get(CoachUpload, out["id"])).status == "done"


async def test_a_failed_run_stays_out_of_history(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    lane.job = {"status": "failed", "error": "No pose."}
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    up = await db.get(CoachUpload, out["id"])
    assert up.status == "failed" and "No pose" in up.error
    assert await me.list_analyses(coach, db) == []


async def test_a_batch_holds_ten_and_one_runs_at_a_time(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    for i in range(10):
        await send(db, coach, a, name=f"{i}.mp4")
    with pytest.raises(HTTPException) as e:
        await send(db, coach, a)
    assert e.value.status_code == 400
    # A second batch waits for the first.
    with pytest.raises(HTTPException) as e:
        await send(db, coach, a, batch="batch02")
    assert e.value.status_code == 409


async def test_what_is_in_line_counts_against_the_quota(make_user, db, lane, monkeypatch):
    coach, a = await coach_with_athlete(make_user, db)
    monkeypatch.setitem(__import__("app.models.user", fromlist=["TIER_LIMITS"]).TIER_LIMITS, TIER_COACH, (2, "month"))
    await send(db, coach, a)
    await send(db, coach, a)
    with pytest.raises(HTTPException) as e:
        await send(db, coach, a)
    assert e.value.status_code == 429


async def test_other_peoples_footage_is_confirmed_once(make_user, db, lane):
    coach = await make_user(tier=TIER_COACH)
    a = await roster.create_athlete(roster.AthleteIn(name="Andy"), coach, db)
    with pytest.raises(HTTPException) as e:
        await send(db, coach, a)
    assert e.value.status_code == 409
    # Their own clip needs no such thing.
    assert (await send(db, coach))["athlete_id"] is None
    await coach_api.confirm_others(coach, db)
    assert (await send(db, coach, a))["athlete_id"] == a["id"]


async def test_only_a_coach_has_a_lane_and_only_their_own(make_user, db, lane):
    full = await make_user(tier=TIER_FULL)
    with pytest.raises(HTTPException) as e:
        await send(db, full)
    assert e.value.status_code == 402
    coach, a = await coach_with_athlete(make_user, db)
    other, _ = await coach_with_athlete(make_user, db)
    out = await send(db, coach, a)
    with pytest.raises(HTTPException) as e:
        await api.delete_upload(out["id"], other, db)
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await send(db, other, a)     # somebody else's athlete
    assert e.value.status_code == 404


async def test_a_restart_picks_up_what_was_left(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    one, two = await send(db, coach, a), await send(db, coach, a)
    up = await db.get(CoachUpload, one["id"])
    up.status = "running"          # the process died halfway through this one
    await db.commit()
    lane.queued.clear()
    assert await coach_lane.requeue_open() == 2
    assert lane.queued == [one["id"], two["id"]]
    db.expire_all()
    assert (await db.get(CoachUpload, one["id"])).status == "queued"


async def test_a_coach_whose_plan_ended_does_not_spend_the_lane(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    out = await send(db, coach, a)
    coach.tier = TIER_STARTER
    await db.commit()
    await coach_lane.run_upload(out["id"])
    assert lane.runs == [] and (await db.get(CoachUpload, out["id"])).status == "failed"


async def test_removing_a_stopped_clip_takes_its_row(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    lane.verdict = "no_go"
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    await api.delete_upload(out["id"], coach, db)
    assert (await db.execute(select(Analysis))).scalars().all() == []
    assert (await db.execute(select(CoachUpload))).scalars().all() == []


async def test_the_batch_mail_goes_once_when_the_last_file_is_through(make_user, db, lane, monkeypatch):
    sent = []
    monkeypatch.setattr(notify, "send_email", lambda to, subject, text, html: sent.append(subject) or True)
    coach, a = await coach_with_athlete(make_user, db)
    one, two = await send(db, coach, a), await send(db, coach, a)
    await coach_lane.run_upload(one["id"])
    assert sent == []
    lane.verdict = "no_go"
    await coach_lane.run_upload(two["id"])
    assert sent == ["Your batch is ready: 1 of 2"]


# --------------------------------------------------------------------------
# The intake link
# --------------------------------------------------------------------------
async def test_the_intake_link_files_a_clip_under_the_athlete(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    token = a["upload_token"]
    page = await api.intake_page(token, db)
    assert page.status_code == 200 and b"Andy" in page.body
    with pytest.raises(HTTPException) as e:
        await api.intake_clip(token, req(), clip(), "bike", None, False, db)
    assert e.value.status_code == 400
    out = await api.intake_clip(token, req(), clip(), "bike", None, True, db)
    assert (await api.intake_clip_status(token, out["id"], db))["status"] == "queued"
    up = (await db.execute(select(CoachUpload))).scalar_one()
    await coach_lane.run_upload(up.id)
    rows = await me.list_analyses(coach, db)
    assert rows[0]["athleteId"] == a["id"] and rows[0]["intakeConsentAt"]


async def test_the_verdict_reaches_the_athlete_but_not_the_numbers(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    lane.verdict = "no_go"
    out = await api.intake_clip(a["upload_token"], req("9.9.9.9"), clip(), "bike", None, True, db)
    await coach_lane.run_upload((await db.execute(select(CoachUpload))).scalar_one().id)
    got = await api.intake_clip_status(a["upload_token"], out["id"], db)
    assert got == {"status": "no_go", "reasons": ["Too small."]}


async def test_a_dead_link_says_nothing_about_why(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    token = a["upload_token"]
    assert (await api.intake_page("not-a-token", db)).status_code == 404
    await roster.update_athlete(a["id"], roster.AthletePatch(rotate_token=True), coach, db)
    assert (await api.intake_page(token, db)).status_code == 404
    fresh = (await roster.list_athletes(coach, db))["athletes"][0]["upload_token"]
    assert (await api.intake_page(fresh, db)).status_code == 200
    coach.tier = TIER_STARTER
    await db.commit()
    assert (await api.intake_page(fresh, db)).status_code == 404
    with pytest.raises(HTTPException) as e:
        await api.intake_clip(fresh, req("8.8.8.8"), clip(), "bike", None, True, db)
    assert e.value.status_code == 404


async def test_a_linked_athlete_is_sent_to_the_app(make_user, db, lane):
    from app.api import coach_link

    coach, a = await coach_with_athlete(make_user, db)
    andy = await make_user()
    await coach_link.create_coach_link(coach_link.LinkIn(code=a["upload_token"], consent=True), andy, db)
    page = await api.intake_page(a["upload_token"], db)
    assert b"already linked" in page.body
    with pytest.raises(HTTPException) as e:
        await api.intake_clip(a["upload_token"], req("7.7.7.7"), clip(), "bike", None, True, db)
    assert e.value.status_code == 409


def test_the_page_escapes_what_a_coach_typed():
    html = render_intake_page({"token": "t", "coach": "<script>x</script>", "tagline": "", "logo": "javascript:1",
                               "athlete": "<b>", "sport": "bike", "linked": False})
    assert "<script>x</script>" not in html and "javascript:1" not in html and "&lt;b&gt;" in html


# --------------------------------------------------------------------------
# The shared report
# --------------------------------------------------------------------------
async def filed_report(make_user, db, lane):
    coach, a = await coach_with_athlete(make_user, db)
    await coach_api.put_coach(coach_api.CoachIn(name="Coach Mike", tagline="Fits since 2005"), coach, db)
    out = await send(db, coach, a)
    await coach_lane.run_upload(out["id"])
    row = (await db.execute(select(Analysis))).scalar_one()
    row.result = {**(row.result or {}), **GOOD["result"], "ai_recommendations": {"report": "x"}}
    row.data = {**row.data, "coachNote": "Drop the saddle 5 mm.", "coachName": "Coach Mike"}
    await db.commit()
    return coach, a, row


async def test_the_shared_report_opens_without_an_account_under_the_coachs_mark(make_user, db, lane):
    coach, a, row = await filed_report(make_user, db, lane)
    sent = await reports.send_report(a["id"], row.client_id, coach, db)
    assert sent["mailed"] is True and sent["email"] == "andy@example.com"
    token = sent["url"].rsplit("/", 1)[1]
    page = await reports.shared_report(token, req("5.5.5.5"), db)
    body = page.body.decode()
    assert page.status_code == 200
    assert "Coach Mike" in body and "Measured with Flapp" in body and "Drop the saddle 5 mm." in body
    assert "Andy" in body and coach.email not in body
    # Sending again reuses the link.
    assert (await reports.send_report(a["id"], row.client_id, coach, db))["url"] == sent["url"]


async def test_switching_a_link_off_ends_it_and_a_stranger_cannot(make_user, db, lane):
    coach, a, row = await filed_report(make_user, db, lane)
    sent = await reports.send_report(a["id"], row.client_id, coach, db)
    token = sent["url"].rsplit("/", 1)[1]
    other, other_a = await coach_with_athlete(make_user, db)
    with pytest.raises(HTTPException):
        await reports.stop_sharing(a["id"], row.client_id, other, db)
    await reports.stop_sharing(a["id"], row.client_id, coach, db)
    assert (await reports.shared_report(token, req("5.5.5.6"), db)).status_code == 404
    assert (await reports.get_share(a["id"], row.client_id, coach, db))["url"] is None
    assert (await reports.shared_report("nope", req("5.5.5.7"), db)).status_code == 404


async def test_an_expired_link_has_ended(make_user, db, lane):
    from app.models.coach_upload import ReportShare

    coach, a, row = await filed_report(make_user, db, lane)
    sent = await reports.send_report(a["id"], row.client_id, coach, db)
    share = (await db.execute(select(ReportShare))).scalar_one()
    share.expires_at_ms = 1
    await db.commit()
    token = sent["url"].rsplit("/", 1)[1]
    assert (await reports.shared_report(token, req("5.5.5.8"), db)).status_code == 404


async def test_a_coach_cannot_share_a_linked_athletes_own_report(make_user, db, lane):
    from app.api import coach_link

    coach, a = await coach_with_athlete(make_user, db)
    andy = await make_user(tier=TIER_FULL)
    await coach_link.create_coach_link(coach_link.LinkIn(code=a["upload_token"], consent=True), andy, db)
    await me._upsert(db, andy, {"id": "a1", "at": 1})
    await db.commit()
    with pytest.raises(HTTPException) as e:
        await reports.send_report(a["id"], "a1", coach, db)
    assert e.value.status_code == 404


# --------------------------------------------------------------------------
# The coach's GO / NO-GO grid
# --------------------------------------------------------------------------
FRAME = {"height_frac": 0.5, "facing": -1, "shoulder": 1}


def verdict(px, bars=None):
    frames = [{**FRAME, "height_frac": px / 720}] * 4
    return capture_check(frames, sport="bike", video_w=1280, video_h=720, thresholds=bars)


@pytest.fixture
def every_frame_valid(monkeypatch):
    monkeypatch.setattr("app.services.prescan._valid", lambda f: True)


def test_the_grid_moves_the_bars_and_the_default_stays(every_frame_valid):
    assert verdict(290)["verdict"] == "no_go"
    assert verdict(350)["verdict"] == "check"
    assert verdict(450)["verdict"] == "go"
    assert verdict(290, {"tiny_px": 250, "small_px": 280})["verdict"] == "go"
    assert verdict(450, {"tiny_px": 500, "small_px": 600})["verdict"] == "no_go"


def test_the_chainring_side_is_not_worth_a_word(every_frame_valid):
    frames = [{**FRAME, "height_frac": 0.8, "facing": 1}] * 4
    assert capture_check(frames, sport="bike", video_w=1280, video_h=720)["reasons"] == []


def test_a_coach_is_not_called_you_and_gets_no_pixel_target(every_frame_valid):
    """The clip is usually somebody else's, and "aim for 550+" can contradict their own bars."""
    for px, grid in ((290, {}), (350, {}), (290, {"tiny_px": 300, "small_px": 400})):
        text = verdict(px, grid)["reasons"][0]["text"]
        assert "You're" not in text and "aim for" not in text
    assert "You're" in verdict(290)["reasons"][0]["text"], "an athlete on their own clip stays addressed"


async def test_the_grid_is_bounded_and_survives_a_profile_save(make_user, db):
    coach = await make_user(tier=TIER_COACH)
    for bad in ({"tiny_px": 100, "small_px": 400}, {"tiny_px": 300, "small_px": 250}, {"tiny_px": 900, "small_px": 900}):
        with pytest.raises(HTTPException) as e:
            await coach_api.put_capture(coach_api.CaptureIn(**bad), coach, db)
        assert e.value.status_code == 422
    saved = await coach_api.put_capture(coach_api.CaptureIn(tiny_px=250, small_px=350, drive_side=False), coach, db)
    assert saved["capture"] == {"tiny_px": 250, "small_px": 350, "drive_side": False}
    await coach_api.confirm_others(coach, db)
    again = await coach_api.put_coach(coach_api.CoachIn(name="Mike"), coach, db)
    assert again["capture"]["tiny_px"] == 250 and again["others_ok_at_ms"]
    back = await coach_api.put_capture(coach_api.CaptureIn(reset=True), coach, db)
    assert back["capture"] is None
    full = await make_user(tier=TIER_FULL)
    with pytest.raises(HTTPException):
        await coach_api.put_capture(coach_api.CaptureIn(tiny_px=250, small_px=350), full, db)


# --------------------------------------------------------------------------
# A grant over a payment
# --------------------------------------------------------------------------
async def test_a_full_subscriber_on_a_pilot_goes_back_to_full(make_user, db):
    mike = await make_user(tier=TIER_FULL, subscription_status="active")
    await grant_tier(db, mike.email, TIER_COACH, 30)
    assert mike.tier == TIER_COACH and mike.tier_after_grant == TIER_FULL
    # His monthly renewal lands mid-pilot: the roster stays.
    _set_tier(mike, TIER_FULL)
    await db.commit()
    assert mike.tier == TIER_COACH and mike.tier_until is not None
    from datetime import datetime, timedelta, timezone
    assert await expire_tier_grants(db, datetime.now(timezone.utc) + timedelta(days=31)) == 1
    await db.refresh(mike)
    assert mike.tier == TIER_FULL and mike.tier_until is None and mike.tier_after_grant is None


async def test_a_lapse_during_a_pilot_ends_on_the_free_plan(make_user, db):
    mike = await make_user(tier=TIER_FULL, subscription_status="active")
    await grant_tier(db, mike.email, TIER_COACH, 30)
    _set_tier(mike, TIER_STARTER)
    await db.commit()
    assert mike.tier == TIER_COACH and mike.tier_after_grant is None
    from datetime import datetime, timedelta, timezone
    await expire_tier_grants(db, datetime.now(timezone.utc) + timedelta(days=31))
    await db.refresh(mike)
    assert mike.tier == TIER_STARTER


async def test_buying_coach_during_a_pilot_makes_it_permanent(make_user, db):
    mike = await make_user(tier=TIER_FULL, subscription_status="active")
    await grant_tier(db, mike.email, TIER_COACH, 30)
    _set_tier(mike, TIER_COACH)
    assert mike.tier == TIER_COACH and mike.tier_until is None and mike.tier_after_grant is None


async def test_a_grant_on_nobody_paying_still_ends_on_the_free_plan(make_user, db):
    john = await make_user(tier=TIER_FULL)   # Full by grant, nothing charged
    await grant_tier(db, john.email, TIER_COACH, 30)
    assert john.tier_after_grant is None


# --------------------------------------------------------------------------
# The lane and the app's lifetime
# --------------------------------------------------------------------------
def test_the_app_starts_and_stops_with_the_lane_again_and_again():
    """An app that stops at once must find the lane idle. It once opened a session
    as it started, and stopping inside that hung CI for four hours."""
    from fastapi.testclient import TestClient

    from app.main import app

    for _ in range(5):
        with TestClient(app) as c:
            assert c.get("/health").status_code == 200


def test_the_lane_opens_no_session_until_it_has_a_clip():
    import inspect

    src = inspect.getsource(coach_lane.lane_loop)
    assert "SessionLocal" not in src and "await requeue_open" not in src


def test_a_clip_with_no_lane_running_waits_for_the_next_start(monkeypatch):
    monkeypatch.setattr(coach_lane, "_queue", None)
    coach_lane.enqueue(1)      # no lane: nothing raises
    coach_lane.open_lane()
    coach_lane.enqueue(2)
    assert coach_lane._queue.qsize() == 1
    monkeypatch.setattr(coach_lane, "_queue", None)



# --------------------------------------------------------------------------
# Auto: run or bike, and the bike's family, read from the clip
# --------------------------------------------------------------------------
def _look(monkeypatch, lane, sport, family, verdict="go"):
    seen = []
    monkeypatch.setattr(coach_lane, "scan_clip", lambda path, sp, bars: seen.append(sp) or {
        "verdict": verdict, "reasons": [], "sport": sport, "family": family})
    ran = []

    def fake_run(job_id, path, params, coach_id):
        ran.append(params)
        core_jobs.JOBS[job_id] = {"status": "completed", "result": lane.job.get("result"), "owner_user_id": coach_id}
        return lane.job

    monkeypatch.setattr(coach_lane, "_run_job", fake_run)
    return seen, ran


async def send_auto(db, coach, a, sport="auto", position=None):
    return await api.upload_to_batch(clip(), "batch01", sport, position, None, a["id"], None, coach, db)


async def test_a_row_left_on_auto_runs_as_what_the_clip_shows(make_user, db, lane, monkeypatch):
    seen, ran = _look(monkeypatch, lane, "bike", "aero")
    coach, a = await coach_with_athlete(make_user, db)
    out = await send_auto(db, coach, a)
    assert out["sport"] == "auto" and out["position"] == "auto"
    await coach_lane.run_upload(out["id"])
    # The look is asked without a sport, so its bars are the ones for what it sees.
    assert seen == [None]
    assert ran[0]["sport"] == "bike" and ran[0]["position"] == "triathlon"
    item = (await api.list_uploads(None, coach, db))["items"][0]
    assert item["status"] == "done" and item["sport"] == "bike" and item["position"] == "triathlon"
    row = (await db.execute(select(Analysis))).scalar_one()
    assert row.sport == "bike"


async def test_auto_reads_a_run_and_a_road_bike(make_user, db, lane, monkeypatch):
    _seen, ran = _look(monkeypatch, lane, "run", None)
    coach, a = await coach_with_athlete(make_user, db)
    await coach_lane.run_upload((await send_auto(db, coach, a))["id"])
    assert ran[-1]["sport"] == "run" and ran[-1]["position"] is None
    assert coach_lane.resolve_auto("auto", "auto", {"sport": "bike", "family": "road"})[:2] == ("bike", "road_hoods")
    assert coach_lane.resolve_auto("auto", "auto", {"sport": "bike", "family": "upright"})[:2] == ("bike", "casual")


async def test_what_the_coach_chose_is_never_overruled(make_user, db, lane, monkeypatch):
    _seen, ran = _look(monkeypatch, lane, "bike", "aero")
    coach, a = await coach_with_athlete(make_user, db)
    await coach_lane.run_upload((await send_auto(db, coach, a, "bike", "road_drops"))["id"])
    assert ran[-1]["position"] == "road_drops"
    # A bike row with the position on Auto takes only the position from the look.
    assert coach_lane.resolve_auto("bike", "auto", {"sport": "bike", "family": "aero"})[:2] == ("bike", "triathlon")
    assert coach_lane.resolve_auto("run", "auto", {"sport": "bike", "family": "aero"})[:2] == ("run", None)


async def test_a_clip_nobody_can_call_run_or_bike_is_not_guessed(make_user, db, lane, monkeypatch):
    _seen, ran = _look(monkeypatch, lane, None, None, verdict="check")
    coach, a = await coach_with_athlete(make_user, db)
    out = await send_auto(db, coach, a)
    await coach_lane.run_upload(out["id"])
    assert ran == []
    assert (await db.execute(select(UsageEvent))).scalars().all() == []
    item = (await api.list_uploads(None, coach, db))["items"][0]
    assert item["status"] == "failed" and "Run or Bike" in item["error"]


async def test_a_bike_of_no_clear_family_is_read_as_hoods_and_says_so(make_user, db, lane, monkeypatch):
    _seen, ran = _look(monkeypatch, lane, "bike", None)
    coach, a = await coach_with_athlete(make_user, db)
    out = await send_auto(db, coach, a)
    await coach_lane.run_upload(out["id"])
    assert ran[-1]["position"] == "road_hoods"
    item = (await api.list_uploads(None, coach, db))["items"][0]
    assert item["status"] == "done" and [r["code"] for r in item["reasons"]] == ["position_guessed"]


async def test_run_anyway_still_reads_a_row_left_on_auto(make_user, db, lane, monkeypatch):
    seen, ran = _look(monkeypatch, lane, "bike", "aero", verdict="no_go")
    coach, a = await coach_with_athlete(make_user, db)
    out = await send_auto(db, coach, a)
    await coach_lane.run_upload(out["id"])
    assert ran == [] and (await db.get(CoachUpload, out["id"])).status == "no_go"
    await api.run_anyway(out["id"], coach, db)
    await coach_lane.run_upload(out["id"])
    assert ran[-1]["sport"] == "bike" and ran[-1]["position"] == "triathlon"


async def test_the_athletes_own_link_still_has_to_say_run_or_bike():
    with pytest.raises(HTTPException):
        api._clean_sport("auto")
    assert api._clean_sport("auto", auto=True) == "auto"


def test_a_clip_the_look_cannot_call_is_asked_about_not_guessed():
    # Artur 2026-11-07: the athlete's sport is no stand-in. A triathlete's track
    # run read as their roster "bike" spent a clip on a road-hoods report.
    assert coach_lane.resolve_auto("auto", "auto", {"sport": "run"})[0] == "run"
    assert coach_lane.resolve_auto("auto", "auto", {"sport": None}) == (None, None, None)
    assert coach_lane.resolve_auto("auto", "auto", None) == (None, None, None)
    # What the coach chose still needs no look at all.
    assert coach_lane.resolve_auto("bike", "road_drops", None) == ("bike", "road_drops", None)


async def test_a_bike_athletes_unreadable_clip_is_still_asked_about(make_user, db, lane, monkeypatch):
    _seen, ran = _look(monkeypatch, lane, None, None, verdict="check")
    coach, a = await coach_with_athlete(make_user, db)
    await roster.update_athlete(a["id"], roster.AthletePatch(sport="bike"), coach, db)
    await coach_lane.run_upload((await send_auto(db, coach, a))["id"])
    assert ran == []
    item = (await api.list_uploads(None, coach, db))["items"][0]
    assert item["status"] == "failed" and "Run or Bike" in item["error"]


# --------------------------------------------------------------------------
# The coach's draft: Flapp writes it, the coach edits the words, then approves
# --------------------------------------------------------------------------
FIT = {"diagnostics": [
    {"component": "saddle_height", "status": "needs_adjustment", "metric_name": "knee_at_bdc", "current_value": 149.0,
     "target_range": [138, 145], "action": "lower_saddle", "amount": "5mm", "priority": 1,
     "reason": "Knee angle at BDC is 149 deg (optimal 138-145 deg). Overextension -- lower saddle by 5mm, "
               "ride 30 min before reassessing."},
    {"component": "bar_position", "status": "needs_adjustment", "metric_name": "trunk_angle", "current_value": 30.0,
     "target_range": [20, 25], "action": "lower_bars", "amount": "", "priority": 3,
     "reason": "Trunk angle is 30 deg (optimal 20-25 deg). Position is too upright -- lower the pads stack."}],
    "fitting_sequence_note": "Adjust in order: saddle height -> bar position. One change at a time."}


async def drafted(make_user, db, lane):
    coach, a, row = await filed_report(make_user, db, lane)
    row.result = {**row.result, "fit_plan": FIT, "ai_recommendations": {"report": "**Fix first**\n- Lower it."},
                  "sport_specific_metrics": {"capture_report": {"checks": []}}}
    await db.commit()
    return coach, a, row


async def report_of(db, coach, row) -> str:
    return await me.render_saved_report(db, reader=coach, owner=coach, row=row, athlete_id=row.athlete_id)


async def test_a_coachs_report_is_a_draft_until_it_is_sent(make_user, db, lane):
    coach, a, row = await drafted(make_user, db, lane)
    assert "Draft · only you see it" in await report_of(db, coach, row)
    sent = await reports.send_report(a["id"], row.client_id, coach, db)
    html = await report_of(db, coach, row)
    assert "Draft ·" not in html and "<dt>Reviewed</dt><dd>Coach Mike · " in html
    page = (await reports.shared_report(sent["url"].rsplit("/", 1)[1], req("5.5.5.9"), db)).body.decode()
    assert "Reviewed" in page and "Draft" not in page


async def test_the_coach_rewrites_hides_and_reorders_and_the_link_shows_it(make_user, db, lane):
    coach, a, row = await drafted(make_user, db, lane)
    st = await reports.get_review(a["id"], row.client_id, coach, db)
    keys = [c["key"] for c in st["cards"]]
    assert keys == ["knee_at_bdc|lower_saddle", "trunk_angle|lower_bars"] and "Lower it." in st["coachText"]
    body = reports.ReviewBody(coachText="**Overall** Ride it a week first.", cards=[
        reports.CardEdit(key=keys[1], why="Your back is happy; leave the bars.", do="Nothing yet."),
        reports.CardEdit(key=keys[0], hidden=True)])
    st = await reports.put_review(a["id"], row.client_id, body, coach, db)
    assert st["edited"] and [c["key"] for c in st["cards"]] == [keys[1], keys[0]] and st["cards"][1]["hidden"]
    sent = await reports.send_report(a["id"], row.client_id, coach, db)
    page = (await reports.shared_report(sent["url"].rsplit("/", 1)[1], req("5.5.5.8"), db)).body.decode()
    assert "Ride it a week first." in page and "Lower it." not in page
    assert "Your back is happy; leave the bars." in page and "Lower saddle 5mm" not in page


async def test_back_to_flapps_words_keeps_the_approval(make_user, db, lane):
    coach, a, row = await drafted(make_user, db, lane)
    await reports.put_review(a["id"], row.client_id, reports.ReviewBody(coachText=""), coach, db)
    assert "Lower it." not in await report_of(db, coach, row)
    await reports.send_report(a["id"], row.client_id, coach, db)
    st = await reports.reset_review(a["id"], row.client_id, coach, db)
    assert not st["edited"] and st["reviewedAt"] and "Lower it." in await report_of(db, coach, row)


async def test_another_coach_cannot_read_or_edit_the_draft(make_user, db, lane):
    coach, a, row = await drafted(make_user, db, lane)
    other, _ = await coach_with_athlete(make_user, db)
    for call in (reports.get_review(a["id"], row.client_id, other, db),
                 reports.put_review(a["id"], row.client_id, reports.ReviewBody(), other, db),
                 reports.reset_review(a["id"], row.client_id, other, db)):
        with pytest.raises(HTTPException):
            await call


async def test_a_resave_of_the_entry_keeps_the_coachs_edits(make_user, db, lane):
    from app.core.jobs import JOBS

    coach, a, row = await drafted(make_user, db, lane)
    await reports.put_review(a["id"], row.client_id, reports.ReviewBody(coachText="Mine."), coach, db)
    JOBS["jre"] = {"result": {"sport_type": "bike", "technique_score": 81}}
    try:
        me._attach_result(row, "jre")
    finally:
        JOBS.pop("jre", None)
    assert row.result["coach_edits"]["coachText"] == "Mine."


async def test_the_athletes_link_offers_the_pdf_when_there_is_a_service(make_user, db, lane, monkeypatch):
    import httpx

    from app.services import pdf_render

    coach, a, row = await drafted(make_user, db, lane)
    sent = await reports.send_report(a["id"], row.client_id, coach, db)
    token = sent["url"].rsplit("/", 1)[1]
    monkeypatch.setattr(pdf_render, "service_url", lambda: None)
    page = (await reports.shared_report(token, req("5.5.6.1"), db)).body.decode()
    assert "Download PDF" not in page and "window.print()" in page
    monkeypatch.setattr(pdf_render, "service_url", lambda: "http://pdf.internal:3000")
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"%PDF-1.7 x")), **kw))
    page = (await reports.shared_report(token, req("5.5.6.2"), db)).body.decode()
    assert f'href="/r/{token}/pdf">Download PDF' in page
    pdf = await reports.shared_report_pdf(token, req("5.5.6.3"), db)
    assert pdf.media_type == "application/pdf" and "Andy" in pdf.headers["content-disposition"]
    await reports.stop_sharing(a["id"], row.client_id, coach, db)
    assert (await reports.shared_report_pdf(token, req("5.5.6.4"), db)).status_code == 404
