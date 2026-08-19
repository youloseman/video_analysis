"""Who may export, and what happens when there is nothing to export.

The document itself is tested in ``test_ai_export.py``; this is the gate around
it. Separate file because importing ``app.main`` pulls in the mediapipe/opencv
stack the rest of the suite deliberately avoids (see conftest) -- keeping them
together would skip the builder tests on any machine without the ML stack.
"""

from __future__ import annotations

import json
import time
import uuid

import pytest

from app.models.user import TIER_ENTHUSIAST, TIER_STARTER, User
from app.services.export import ai_export

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

from fastapi import HTTPException  # noqa: E402

from app.core.jobs import JOBS  # noqa: E402

RESULT = {
    "status": "completed",
    "sport_type": "run",
    "camera_side": "left",
    "frames_analyzed": 148,
    "technique_score": 74,
    "letter_grade": "C+",
    "angle_statistics": {
        "knee": {"min": 62.4, "mean": 121.3, "max": 172.1, "range": 109.7,
                 "valid_frames": 142, "nan_frames": 6},
    },
    "detected_issues": [],
    "sport_specific_metrics": {"cadence_spm": 164.2, "trunk_lean_avg": 6.2},
}


@pytest.fixture
def completed_job():
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "status": "completed",
        "sport": "run",
        "cycling_position": None,
        "result": dict(RESULT),
        "error": None,
        "overlay_path": None,
        "created_at": time.time(),
        "job_dir": None,
        "token": "tok",
        "owner_user_id": None,
    }
    yield job_id
    JOBS.pop(job_id, None)


def _user(tier: str) -> User:
    return User(email="a@b.c", password_hash="x", tier=tier)


def test_markdown_export_is_served_to_a_paid_caller(completed_job):
    r = main.job_export(completed_job, t="tok", user=_user(TIER_ENTHUSIAST))
    assert r.media_type.startswith("text/markdown")
    assert "attachment; filename=" in r.headers["content-disposition"]
    assert b"## Measurements" in r.body


def test_json_format_is_served_too(completed_job):
    r = main.job_export(
        completed_job, format="json", t="tok", user=_user(TIER_ENTHUSIAST),
    )
    assert json.loads(r.body)["flapp_export_schema"] == ai_export.EXPORT_SCHEMA


def test_a_free_caller_is_sent_to_the_paywall_not_handed_a_stub(completed_job):
    """A free result is trimmed to a score before it is ever stored, so its
    export would be one number wrapped in a page of caveats."""
    for user in (None, _user(TIER_STARTER)):
        with pytest.raises(HTTPException) as e:
            main.job_export(completed_job, t="tok", user=user)
        assert e.value.status_code == 402


def test_someone_elses_job_is_not_exportable(completed_job):
    """Same gate as the poll and the overlay: no token, no result. 404 rather
    than 403, so the id is not confirmed to exist."""
    with pytest.raises(HTTPException) as e:
        main.job_export(completed_job, t="wrong-token", user=_user(TIER_ENTHUSIAST))
    assert e.value.status_code == 404


def test_an_unfinished_job_has_nothing_to_export(completed_job):
    JOBS[completed_job]["status"] = "processing"
    JOBS[completed_job]["result"] = None
    with pytest.raises(HTTPException) as e:
        main.job_export(completed_job, t="tok", user=_user(TIER_ENTHUSIAST))
    assert e.value.status_code == 409


def test_an_unknown_format_is_rejected(completed_job):
    with pytest.raises(HTTPException) as e:
        main.job_export(
            completed_job, format="pdf", t="tok", user=_user(TIER_ENTHUSIAST),
        )
    assert e.value.status_code == 400
