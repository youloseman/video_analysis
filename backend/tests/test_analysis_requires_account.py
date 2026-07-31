"""Analysis is account-only.

The free plan *is* the signed-in Starter tier — there is no anonymous analysis
any more — so both upload endpoints have to turn a session-less caller away
before doing any work. That is the whole sign-up funnel: if this regresses,
the landing page's "create a free account" step becomes optional again and the
per-user quota stops meaning anything.

Guarded by an import skip: importing ``app.main`` pulls in the mediapipe/opencv
stack that the rest of the suite deliberately avoids (see conftest).
"""

from __future__ import annotations

import pytest

main = pytest.importorskip(
    "app.main", reason="needs the analysis stack (mediapipe/opencv)",
)

from fastapi.testclient import TestClient  # noqa: E402


@pytest.mark.parametrize(
    ("path", "field", "filename", "mime"),
    [
        ("/analyze", "file", "clip.mp4", "video/mp4"),
        ("/analyze-photo", "photo", "shot.jpg", "image/jpeg"),
    ],
)
def test_anonymous_upload_is_rejected(path, field, filename, mime):
    client = TestClient(main.app)
    r = client.post(
        path,
        files={field: (filename, b"not-a-real-clip", mime)},
        data={"sport": "run"},
    )
    assert r.status_code == 401, r.text


@pytest.mark.parametrize("path", ["/analyze", "/analyze-photo"])
def test_a_bogus_session_is_rejected_too(path):
    """A stale or forged token must not slip past into the analysis path."""
    client = TestClient(main.app)
    r = client.post(
        path,
        files={"file": ("clip.mp4", b"x", "video/mp4"),
               "photo": ("shot.jpg", b"x", "image/jpeg")},
        data={"sport": "run"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert r.status_code == 401, r.text
