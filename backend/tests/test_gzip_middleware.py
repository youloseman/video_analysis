"""Compression. These tests exist because the failure mode is silent.

``SelectiveGZipMiddleware`` works by overriding one method on Starlette's gzip
responder and flipping one of its flags. Both have been renamed once already --
``send_with_gzip``/``content_encoding_set`` in Starlette 0.38 became
``send_with_compression``/``content_type_is_excluded`` in 1.x -- and when that
happens the override stops being called without raising anything. Nothing looks
broken; the overlay video just quietly starts being gzipped again, loses its
``Content-Length``, and stops seeking.

So the load-bearing assertions here are not "is HTML compressed" but "is
``video/mp4`` still left alone, and does a Range request still come back as an
uncompressed 206".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.middleware.gzip import GZipResponder
from starlette.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.core.compression import SelectiveGZipMiddleware

# Comfortably over the middleware's minimum_size, and compressible enough that a
# gzipped body is unmistakably smaller than the original.
BODY = b"the quick brown fox " * 3000
TINY = b"ok"

GZIP = {"Accept-Encoding": "gzip, deflate"}
IDENTITY = {"Accept-Encoding": "identity"}


def _app(tmp_path: Path) -> Starlette:
    """One route per shape of response the real app serves."""
    video = tmp_path / "overlay.mp4"
    video.write_bytes(os.urandom(300_000))

    async def html(_):
        return Response(BODY, media_type="text/html; charset=utf-8")

    async def json_(_):
        return Response(BODY, media_type="application/json")

    async def svg(_):
        return Response(BODY, media_type="image/svg+xml")

    async def png(_):
        return Response(BODY, media_type="image/png")

    async def mp4(_):
        return Response(BODY, media_type="video/mp4")

    async def tiny(_):
        return PlainTextResponse(TINY)

    async def partial(_):
        return Response(
            BODY, status_code=206, media_type="video/mp4",
            headers={"Content-Range": f"bytes 0-{len(BODY) - 1}/999999"},
        )

    async def stream_video(_):
        async def gen():
            yield BODY
            yield BODY
        return StreamingResponse(gen(), media_type="video/mp4")

    async def stream_html(_):
        async def gen():
            yield BODY
            yield BODY
        return StreamingResponse(gen(), media_type="text/html")

    async def overlay(_):
        # The production shape: FileResponse, which since Starlette 1.x answers
        # Range requests itself.
        return FileResponse(video, media_type="video/mp4")

    routes = [
        Route("/html", html), Route("/json", json_), Route("/svg", svg),
        Route("/png", png), Route("/mp4", mp4), Route("/tiny", tiny),
        Route("/partial", partial), Route("/stream-video", stream_video),
        Route("/stream-html", stream_html), Route("/overlay", overlay),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(SelectiveGZipMiddleware, minimum_size=1000)
    return app


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return TestClient(_app(tmp_path))


def test_starlette_hook_contract_is_intact() -> None:
    """The override targets real attributes on Starlette's responder.

    If this fails, Starlette renamed something and app/core/compression.py needs
    updating -- the behavioural tests below are what actually catch the damage,
    but this one names the cause.
    """
    assert hasattr(GZipResponder, "send_with_compression"), (
        "Starlette renamed the gzip responder hook; _SelectiveGZipResponder "
        "overrides a method that no longer exists, so it is being bypassed."
    )
    responder = GZipResponder(app=None, minimum_size=1000)  # type: ignore[arg-type]
    assert hasattr(responder, "content_type_is_excluded"), (
        "Starlette renamed the gzip opt-out flag; the content-type gate in "
        "app/core/compression.py no longer does anything."
    )


@pytest.mark.parametrize("path", ["/html", "/json", "/svg", "/stream-html"])
def test_text_is_compressed(client: TestClient, path: str) -> None:
    r = client.get(path, headers=GZIP)
    assert r.headers.get("content-encoding") == "gzip"
    assert r.content == BODY * (2 if path == "/stream-html" else 1)


@pytest.mark.parametrize("path", ["/mp4", "/png", "/stream-video"])
def test_binary_is_not_compressed(client: TestClient, path: str) -> None:
    """Gzip cannot shrink these, and on the streaming path it would drop
    Content-Length -- which is what <video> needs for duration and seeking."""
    r = client.get(path, headers=GZIP)
    assert "content-encoding" not in r.headers
    assert r.content == BODY * (2 if path == "/stream-video" else 1)


def test_binary_keeps_its_content_length(client: TestClient) -> None:
    r = client.get("/mp4", headers=GZIP)
    assert r.headers["content-length"] == str(len(BODY))


def test_partial_content_is_not_compressed(client: TestClient) -> None:
    """A 206 body is a byte slice described by offsets into the UNCOMPRESSED
    file. Gzipping it makes the response describe one thing and deliver
    another."""
    r = client.get("/partial", headers=GZIP)
    assert r.status_code == 206
    assert "content-encoding" not in r.headers
    assert r.content == BODY


def test_small_responses_are_left_alone(client: TestClient) -> None:
    r = client.get("/tiny", headers=GZIP)
    assert "content-encoding" not in r.headers
    assert r.content == TINY


def test_identity_client_gets_a_plain_body(client: TestClient) -> None:
    r = client.get("/html", headers=IDENTITY)
    assert "content-encoding" not in r.headers
    assert r.content == BODY


def test_compressed_responses_vary_on_accept_encoding(client: TestClient) -> None:
    """Without this a shared cache can hand a gzipped body to a client that
    asked for identity."""
    r = client.get("/html", headers=GZIP)
    assert "accept-encoding" in r.headers.get("vary", "").lower()


def test_overlay_video_still_serves_ranges(client: TestClient) -> None:
    """The end-to-end shape that breaks first if the gate stops working: seeking
    in the annotated overlay."""
    full = client.get("/overlay", headers=GZIP)
    assert full.status_code == 200
    assert "content-encoding" not in full.headers
    assert full.headers.get("accept-ranges") == "bytes"
    assert len(full.content) == 300_000

    part = client.get("/overlay", headers={**GZIP, "Range": "bytes=1000-1999"})
    assert part.status_code == 206
    assert "content-encoding" not in part.headers
    assert part.headers["content-range"] == "bytes 1000-1999/300000"
    assert part.content == full.content[1000:2000]
