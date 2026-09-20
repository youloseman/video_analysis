"""Security headers ride on every response, including the streamed video.

Tested on a bare Starlette app with the middleware mounted, the way the gzip
middleware is: it is the middleware under test, not the routes behind it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import FileResponse, PlainTextResponse, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.core.headers import SECURITY_HEADERS, SecurityHeadersMiddleware

EXPECTED = {k.decode(): v.decode() for k, v in SECURITY_HEADERS}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    clip = tmp_path / "overlay.mp4"
    clip.write_bytes(b"\x00" * 4096)

    async def text(_):
        return PlainTextResponse("hi")

    async def video(_):
        return FileResponse(clip, media_type="video/mp4")

    async def stream(_):
        async def gen():
            yield b"a" * 1024
            yield b"b" * 1024

        return StreamingResponse(gen(), media_type="application/octet-stream")

    async def already(_):
        # A route that sets its own policy keeps it.
        return PlainTextResponse("framed", headers={"X-Frame-Options": "SAMEORIGIN"})

    app = Starlette(
        routes=[
            Route("/", text), Route("/video", video),
            Route("/stream", stream), Route("/already", already),
        ],
        middleware=[Middleware(SecurityHeadersMiddleware)],
    )
    return TestClient(app)


@pytest.mark.parametrize("path", ["/", "/video", "/stream"])
def test_every_response_carries_the_headers(client, path):
    r = client.get(path)
    assert r.status_code == 200
    for name, value in EXPECTED.items():
        assert r.headers.get(name) == value, name


def test_the_video_still_arrives_whole(client):
    r = client.get("/video")
    assert len(r.content) == 4096
    assert r.headers["content-type"] == "video/mp4"


def test_a_route_that_set_its_own_value_wins(client):
    r = client.get("/already")
    assert r.headers["x-frame-options"] == "SAMEORIGIN"
    assert r.headers["x-content-type-options"] == "nosniff"


def test_hsts_does_not_commit_subdomains():
    """``includeSubDomains`` cannot be taken back for a year in every browser
    that has seen it; it is a separate decision, not a default."""
    assert "includeSubDomains" not in EXPECTED["strict-transport-security"]
    assert "preload" not in EXPECTED["strict-transport-security"]


def test_no_content_security_policy_yet():
    """A CSP on a single-document inline SPA needs nonces and a report-only
    phase; a wrong one takes the app down for everyone. Not sneaked in here."""
    assert "content-security-policy" not in EXPECTED
