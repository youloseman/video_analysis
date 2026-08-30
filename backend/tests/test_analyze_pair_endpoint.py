"""The two-sided upload: who may run it, and what it costs.

A session analyses two clips in one job, so it spends two clips of quota and
is a paid-plan feature. Both of those are checked BEFORE anything is decoded:
discovering the second analysis is unaffordable after the first has run would
burn the compute and still owe the rider an answer.

Guarded by an import skip like the other endpoint tests -- importing app.main
pulls in the mediapipe/opencv stack the rest of the suite avoids.
"""

from __future__ import annotations

import pytest

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

from fastapi.testclient import TestClient  # noqa: E402


def _files():
    return {
        "video_left": ("left.mp4", b"not-a-real-clip", "video/mp4"),
        "video_right": ("right.mp4", b"not-a-real-clip", "video/mp4"),
    }


def test_anonymous_upload_is_rejected():
    client = TestClient(main.app)
    r = client.post("/analyze-pair", files=_files(), data={"position": "road_hoods"})
    assert r.status_code == 401, r.text


def test_a_bogus_session_is_rejected_too():
    client = TestClient(main.app)
    r = client.post(
        "/analyze-pair", files=_files(), data={"position": "road_hoods"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code == 401, r.text


def test_both_clips_are_required():
    """One file is not a session -- and must not quietly become one."""
    client = TestClient(main.app)
    r = client.post(
        "/analyze-pair",
        files={"video_left": ("left.mp4", b"x", "video/mp4")},
        data={"position": "road_hoods"},
    )
    # 422 (missing field) before auth is fine; what must not happen is a 2xx.
    assert r.status_code >= 400, r.text


def test_the_endpoint_takes_a_pair_of_either_sport():
    """It began bike-only, on the reasoning that a run side view already sees
    both legs. Running earned a pair of its own once there was a way to tell a
    trustworthy run clip from one whose legs swapped -- and it merges on
    entirely different rules, because two run clips share no rigid object to
    pool geometry against. See run_session.py."""
    schema = main.app.openapi()
    assert "/analyze-pair" in schema["paths"]
    body = schema["paths"]["/analyze-pair"]["post"]["requestBody"]
    props = next(iter(body["content"].values()))["schema"]
    ref = props.get("$ref", "")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        props = schema["components"]["schemas"][name]
    fields = set((props.get("properties") or {}).keys())
    assert {"video_left", "video_right", "sport"} <= fields
