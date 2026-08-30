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
