"""Re-measuring a clip with joint corrections, end to end without a detector.

A finished bike job with stored frames is the starting point; the endpoint
functions are called directly (the pattern the API tests use), and the
background re-measurement is run by hand. No MediaPipe anywhere -- the whole
point of the feature -- but ``app.main`` still needs the analysis stack to
import, hence the skip.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException, Response

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

from app.core.jobs import JOBS  # noqa: E402
from app.services.correction_limits import reset as reset_caps  # noqa: E402
from app.services.video_analysis.landmark_store import save_frames  # noqa: E402
from tests.test_analyze_from_frames import (  # noqa: E402
    FPS,
    N_FRAMES,
    VIDEO_INFO,
    _analyze,
    _frames,
)


def _user(tier="enthusiast", uid=41):
    return SimpleNamespace(id=uid, tier=tier, height_cm=None,
                           mobility_hamstring_deg=None, mobility_hip_flexion_deg=None,
                           mobility_goal=None, mobility_measured_at=None)


@pytest.fixture
def bike_job(tmp_path):
    """A completed bike analysis whose frames are on disk."""
    reset_caps()
    frames = _frames()
    store = tmp_path / "landmarks.npz"
    save_frames(store, frames, meta={
        "sport_type": "bike", "cycling_position": "triathlon",
        "camera_side_override": None, "video_info": VIDEO_INFO,
        "sampling_meta": {}, "stabilizer_ctx": {},
    })
    result = _analyze(_frames())
    job_id = "corrtest01"
    JOBS[job_id] = {
        "status": "completed", "sport": "bike", "cycling_position": "triathlon",
        "result": result, "error": None, "overlay_path": None,
        "created_at": 0.0, "job_dir": str(tmp_path), "token": "tok",
        "owner_user_id": 41, "params": {}, "frames_store": str(store),
        "input_path": str(tmp_path / "input.mp4"), "corrections": [],
        "recompute_rounds": 0, "baseline": None,
    }
    yield job_id
    JOBS.pop(job_id, None)
    reset_caps()


def _run_background(bt: BackgroundTasks) -> None:
    for task in bt.tasks:
        task.func(*task.args, **task.kwargs)


# --- reading the points -----------------------------------------------------

async def test_the_editor_gets_the_near_side_points_and_the_corrections_in_force(bike_job, db):
    out = await main.job_landmarks(bike_job, "tok", _user(), db)
    assert out["camera_side"] == "right"
    assert out["draggable"] == [8, 12, 14, 16, 24, 26, 28, 30, 32]
    assert len(out["frames"]) == N_FRAMES
    assert out["fps"] == FPS and out["frame_width"] == 1080
    assert len(out["frames"][0]["p"]) == 9
    assert out["corrections"] == [] and out["rounds_left"] == 3


async def test_a_free_account_is_refused_before_anything_is_read(bike_job, db):
    with pytest.raises(HTTPException) as e:
        await main.job_landmarks(bike_job, "tok", _user(tier="starter"), db)
    assert e.value.status_code == 402


async def test_somebody_else_cannot_read_the_points(bike_job, db):
    with pytest.raises(HTTPException) as e:
        await main.job_landmarks(bike_job, "wrong-token", _user(uid=99), db)
    assert e.value.status_code == 404


async def test_expired_frames_say_so(bike_job, db):
    Path(JOBS[bike_job]["frames_store"]).unlink()
    with pytest.raises(HTTPException) as e:
        await main.job_landmarks(bike_job, "tok", _user(), db)
    assert e.value.status_code == 410


# --- applying ---------------------------------------------------------------

async def test_a_correction_re_measures_the_clip_and_keeps_the_baseline(bike_job):
    before = JOBS[bike_job]["result"]["sport_specific_metrics"]["knee_at_bdc"]
    bt = BackgroundTasks()
    out = await main.job_corrections(
        bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.0, "dy": 0.05, "frame_idx": 12}]),
        bt, Response(), "tok", _user(),
    )
    assert out["status"] == "processing" and out["rounds_left"] == 2
    assert out["corrections"] == [{"landmark": 24, "dx": 0.0, "dy": 0.05, "frame_idx": 12}]
    assert JOBS[bike_job]["status"] == "processing"

    _run_background(bt)

    job = JOBS[bike_job]
    assert job["status"] == "completed" and job["recompute_error"] is None
    res = job["result"]
    assert res["sport_specific_metrics"]["knee_at_bdc"] < before - 3
    assert res["corrections"] == out["corrections"]
    assert res["baseline"]["metrics"]["knee_at_bdc"] == before
    assert res["baseline"]["technique_score"] is not None
    assert (Path(job["job_dir"]) / "corrections.json").is_file()


async def test_the_status_poll_reports_the_corrections_and_rounds(bike_job):
    bt = BackgroundTasks()
    await main.job_corrections(
        bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.01, "dy": 0.0}]),
        bt, Response(), "tok", _user(),
    )
    _run_background(bt)
    st = main.job_status(bike_job, "tok", _user())
    assert st.status == "completed"
    assert st.corrections == [{"landmark": 24, "dx": 0.01, "dy": 0.0}]
    assert st.rounds_left == 2 and st.recompute_error is None


async def test_rounds_accumulate_and_run_out(bike_job):
    for k in range(3):
        bt = BackgroundTasks()
        await main.job_corrections(
            bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.005, "dy": 0.0}]),
            bt, Response(), "tok", _user(),
        )
        _run_background(bt)
    assert JOBS[bike_job]["corrections"][0]["dx"] == pytest.approx(0.015)
    with pytest.raises(HTTPException) as e:
        await main.job_corrections(
            bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.005, "dy": 0.0}]),
            BackgroundTasks(), Response(), "tok", _user(),
        )
    assert e.value.status_code == 429 and "3 rounds" in e.value.detail


async def test_an_impossible_move_costs_nothing(bike_job):
    with pytest.raises(HTTPException) as e:
        await main.job_corrections(
            bike_job, main.CorrectionsIn(corrections=[{"landmark": 32, "dx": 0.0, "dy": 0.24}]),
            BackgroundTasks(), Response(), "tok", _user(),
        )
    assert e.value.status_code == 422 and "outside the picture" in e.value.detail
    assert JOBS[bike_job]["recompute_rounds"] == 0
    assert JOBS[bike_job]["status"] == "completed"


async def test_the_far_leg_is_refused(bike_job):
    with pytest.raises(HTTPException) as e:
        await main.job_corrections(
            bike_job, main.CorrectionsIn(corrections=[{"landmark": 23, "dx": 0.01, "dy": 0.0}]),
            BackgroundTasks(), Response(), "tok", _user(),
        )
    assert e.value.status_code == 422


async def test_a_free_account_cannot_apply(bike_job):
    with pytest.raises(HTTPException) as e:
        await main.job_corrections(
            bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.01, "dy": 0.0}]),
            BackgroundTasks(), Response(), "tok", _user(tier="starter"),
        )
    assert e.value.status_code == 402


async def test_a_running_job_is_refused(bike_job):
    JOBS[bike_job]["sport"] = "run"
    with pytest.raises(HTTPException) as e:
        await main.job_corrections(
            bike_job, main.CorrectionsIn(corrections=[{"landmark": 23, "dx": 0.01, "dy": 0.0}]),
            BackgroundTasks(), Response(), "tok", _user(),
        )
    assert e.value.status_code == 400


# --- reset ------------------------------------------------------------------

async def test_reset_restores_the_automatic_measurement(bike_job):
    original = JOBS[bike_job]["result"]["sport_specific_metrics"]["knee_at_bdc"]
    bt = BackgroundTasks()
    await main.job_corrections(
        bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.0, "dy": 0.05}]),
        bt, Response(), "tok", _user(),
    )
    _run_background(bt)
    assert JOBS[bike_job]["result"]["sport_specific_metrics"]["knee_at_bdc"] != original

    bt = BackgroundTasks()
    out = await main.job_corrections_reset(bike_job, bt, Response(), "tok", _user())
    assert out["status"] == "processing"
    _run_background(bt)

    job = JOBS[bike_job]
    assert job["corrections"] == [] and job["recompute_rounds"] == 0
    assert job["result"]["corrections"] is None and job["result"]["baseline"] is None
    assert job["result"]["sport_specific_metrics"]["knee_at_bdc"] == original


async def test_reset_with_nothing_to_reset_is_a_no_op(bike_job):
    out = await main.job_corrections_reset(bike_job, BackgroundTasks(), Response(), "tok", _user())
    assert out["status"] == "completed" and out["corrections"] == []


# --- a failed re-measurement leaves the previous result standing ------------

async def test_a_failed_re_measurement_keeps_the_previous_result(bike_job, monkeypatch):
    previous = JOBS[bike_job]["result"]
    bt = BackgroundTasks()
    await main.job_corrections(
        bike_job, main.CorrectionsIn(corrections=[{"landmark": 24, "dx": 0.01, "dy": 0.0}]),
        bt, Response(), "tok", _user(),
    )
    import app.services.video_analysis.runner as runner

    def _boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runner, "analyze_from_frames", _boom)
    _run_background(bt)

    job = JOBS[bike_job]
    assert job["status"] == "completed"
    assert job["result"] is previous
    assert "unchanged" in job["recompute_error"]
    assert job["recompute_rounds"] == 0, "an undelivered round is not spent"
    assert not math.isnan(previous["sport_specific_metrics"]["knee_at_bdc"])


# --- where the rider sits over the crank ------------------------------------

async def test_the_crank_axle_gives_the_hip_over_bb_from_the_clips_median_hip(bike_job, db):
    out = await main.job_bike_points(
        bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}), "tok", _user(tier="full"), db,
    )
    hob = out["hip_over_bb"]
    assert hob["hip_source"] == "video_median" and hob["frames"] > 0
    assert 40 < hob["angle_deg"] < 140 and hob["ref"] == "frame"
    job = JOBS[bike_job]
    assert job["result"]["hip_over_bb"] == hob
    assert job["result"]["sport_specific_metrics"]["hip_over_bb"] == hob
    assert job["bike_points"] == {"C": [0.5, 0.95]}


async def test_a_free_account_cannot_mark_the_crank(bike_job, db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await main.job_bike_points(
            bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}), "tok", _user(tier="starter"), db,
        )
    assert e.value.status_code == 402 and "Full" in e.value.detail


async def test_enthusiast_cannot_mark_the_crank_either(bike_job, db):
    """Full only since 2026-10-26: the reading is what Full sells over Enthusiast."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await main.job_bike_points(
            bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}), "tok", _user(tier="enthusiast"), db,
        )
    assert e.value.status_code == 402 and "Full" in e.value.detail
    with pytest.raises(HTTPException) as e:
        await main.job_bike_auto(bike_job, "tok", _user(tier="enthusiast"), db)
    assert e.value.status_code == 402


async def test_an_enthusiast_who_paid_before_the_move_keeps_the_crank(bike_job, db):
    """crank_until: the end of the period they had paid for when it moved."""
    from datetime import datetime, timedelta, timezone

    from fastapi import HTTPException

    kept = _user(tier="enthusiast")
    kept.crank_until = datetime.now(timezone.utc) + timedelta(days=20)
    out = await main.job_bike_points(
        bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}), "tok", kept, db,
    )
    assert out["hip_over_bb"]["angle_deg"] > 0

    lapsed = _user(tier="enthusiast")
    lapsed.crank_until = datetime.now(timezone.utc) - timedelta(days=1)
    with pytest.raises(HTTPException) as e:
        await main.job_bike_points(
            bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}), "tok", lapsed, db,
        )
    assert e.value.status_code == 402


async def test_a_bad_crank_point_is_refused_with_its_reason(bike_job, db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await main.job_bike_points(
            bike_job, main.BikePointsIn(points={"P": [0.5, 0.95]}), "tok", _user(tier="full"), db,
        )
    assert e.value.status_code == 422 and "crank axle first" in e.value.detail


async def test_the_clip_crank_search_is_paid_and_needs_the_clip(bike_job, db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as e:
        await main.job_bike_auto(bike_job, "tok", _user(tier="starter"))
    assert e.value.status_code == 402
    with pytest.raises(HTTPException) as e:
        await main.job_bike_auto(bike_job, "tok", _user(tier="full"))
    assert e.value.status_code == 410 and "by hand" in e.value.detail


async def test_where_the_crank_came_from_rides_with_the_reading(bike_job, db):
    out = await main.job_bike_points(
        bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}, c_source="auto_chainring"), "tok", _user(tier="full"), db,
    )
    assert out["hip_over_bb"]["c_source"] == "auto_chainring"
    out = await main.job_bike_points(
        bike_job, main.BikePointsIn(points={"C": [0.5, 0.95]}, c_source="<script>"), "tok", _user(tier="full"), db,
    )
    assert out["hip_over_bb"]["c_source"] == "user"


# --- from a saved report: the job store has forgotten the job ----------------

async def _saved_report(db, make_user, tmp_path, monkeypatch, tier="full"):
    """A bike analysis saved to the account, its files on the volume, and the
    in-memory job gone -- what History sees after a deploy."""
    import time

    import dataclasses

    import app.core.jobs as jobstore
    from app.models.analysis import Analysis

    user = await make_user()
    user.tier = tier
    job_id = "histcrank01"
    root = tmp_path / "uploads"
    monkeypatch.setattr(jobstore, "settings", dataclasses.replace(jobstore.settings, uploads_dir=root))
    d = root / job_id
    d.mkdir(parents=True)
    frames = _frames()
    save_frames(d / "landmarks.npz", frames, meta={
        "sport_type": "bike", "cycling_position": "triathlon", "camera_side_override": None,
        "video_info": VIDEO_INFO, "sampling_meta": {}, "stabilizer_ctx": {}})
    (d / "input.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42clip")
    result = _analyze(_frames())
    db.add(Analysis(user_id=user.id, client_id="h1", job_id=job_id, created_at_ms=int(time.time() * 1000),
                    sport="bike", kind="video", data={"id": "h1"}, result=result))
    await db.commit()
    JOBS.pop(job_id, None)
    return user, job_id


async def test_a_saved_report_can_mark_the_crank_after_the_job_is_gone(db, make_user, tmp_path, monkeypatch):
    """Corbin, 2026-10-27: he left the page while the clip was measured and
    came back through History, where the marking did not exist."""
    from sqlalchemy import select

    from app.models.analysis import Analysis

    user, job_id = await _saved_report(db, make_user, tmp_path, monkeypatch)
    pts = await main.job_landmarks(job_id, None, user, db)
    assert len(pts["frames"]) == N_FRAMES
    out = await main.job_bike_points(job_id, main.BikePointsIn(points={"C": [0.5, 0.95]}), None, user, db)
    assert out["hip_over_bb"]["hip_source"] == "video_median"
    row = (await db.execute(select(Analysis).where(Analysis.job_id == job_id))).scalar_one()
    await db.refresh(row)
    assert row.result["hip_over_bb"]["angle_deg"] == out["hip_over_bb"]["angle_deg"], "the reading is kept"


async def test_nobody_else_can_rebuild_someone_elses_job(db, make_user, tmp_path, monkeypatch):
    _, job_id = await _saved_report(db, make_user, tmp_path, monkeypatch)
    stranger = await make_user()
    stranger.tier = "full"
    with pytest.raises(HTTPException) as e:
        await main.job_landmarks(job_id, None, stranger, db)
    assert e.value.status_code == 404


async def test_the_stored_clip_is_served_to_its_owner_only(db, make_user, tmp_path, monkeypatch):
    from app.core.security import create_media_token

    user, job_id = await _saved_report(db, make_user, tmp_path, monkeypatch)
    resp = await main.job_source(job_id, None, user, db)
    assert Path(resp.path).name == "input.mp4"
    resp = await main.job_source(job_id, create_media_token(user.id, job_id), None, db)
    assert Path(resp.path).name == "input.mp4"
    stranger = await make_user()
    with pytest.raises(HTTPException):
        await main.job_source(job_id, None, stranger, db)
    with pytest.raises(HTTPException):
        await main.job_source(job_id, None, None, db)
