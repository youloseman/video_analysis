"""The photo re-measurement endpoint: who may call it, and what it refuses.

Guarded by an import skip like the other endpoint tests -- importing app.main
pulls in the mediapipe/opencv stack the rest of the suite avoids.
"""
from __future__ import annotations

import json

import pytest

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

from fastapi.testclient import TestClient  # noqa: E402


def _form(**over):
    data = {
        "sport": "bike", "position": "triathlon",
        "pose": json.dumps({"v": 1}), "corrections": "[]",
    }
    data.update(over)
    return data


def _files():
    return {"photo": ("ride.jpg", b"\xff\xd8not-really-a-jpeg", "image/jpeg")}


def test_anonymous_callers_are_rejected():
    client = TestClient(main.app)
    r = client.post("/analyze-photo/recompute", files=_files(), data=_form())
    assert r.status_code == 401, r.text


def test_the_endpoint_is_documented_with_the_fields_the_client_sends():
    schema = main.app.openapi()
    assert "/analyze-photo/recompute" in schema["paths"]
    body = schema["paths"]["/analyze-photo/recompute"]["post"]["requestBody"]
    props = next(iter(body["content"].values()))["schema"]
    ref = props.get("$ref", "")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        props = schema["components"]["schemas"][name]
    fields = set((props.get("properties") or {}).keys())
    assert {"photo", "sport", "pose", "corrections"} <= fields


def _free_client():
    from types import SimpleNamespace

    from app.core.security import get_current_user

    user = SimpleNamespace(id=987654, tier="starter", email="free@example.com")
    main.app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(main.app)


def _real_photo_and_pose():
    from tests.test_photo_corrections import _photo_bytes, _pose_result

    from app.services import photo_corrections as PC

    photo = _photo_bytes()
    return photo, PC.build_pose_blob(photo, _pose_result(photo))


def test_a_free_account_moves_the_knee_and_keeps_its_score():
    from app.core.security import get_current_user
    from app.services.correction_limits import reset

    reset()
    client = _free_client()
    try:
        photo, blob = _real_photo_and_pose()
        r = client.post("/analyze-photo/recompute",
                        files={"photo": ("ride.jpg", photo, "image/jpeg")},
                        data=_form(pose=json.dumps(blob),
                                   corrections=json.dumps([{"landmark": 26, "dx": 0.0, "dy": 0.02}])))
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["score_kept"] is True and j["locked"]
        assert j["corrections"][0]["landmark"] == 26
        assert "angles" not in j and "baseline" not in j and "ai_recommendations" not in j
        assert j["pose"]["draggable"] == [24, 26]
        assert j["thumbnail_base64"].startswith("data:image/jpeg;base64,")

        r = client.post("/analyze-photo/recompute",
                        files={"photo": ("ride.jpg", photo, "image/jpeg")},
                        data=_form(pose=json.dumps(blob),
                                   corrections=json.dumps([{"landmark": 14, "dx": 0.01, "dy": 0.0}])))
        assert r.status_code == 402 and "hip and the knee" in r.json()["detail"]
    finally:
        main.app.dependency_overrides.pop(get_current_user, None)
        reset()
