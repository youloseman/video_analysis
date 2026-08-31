"""Brotli on the app shell.

Twenty kilobytes off a 163 KB first paint, and the whole value depends on it
costing nothing to produce -- so the tests are about the cache being exact and
the fallbacks being safe, not about the compression itself.

Three ways this could ship broken and still look fine in a browser:
a shared cache handing brotli bytes to a client that only reads gzip (no Vary),
the gzip middleware compressing an already-compressed body (double encoding),
and the cache keyed on something less specific than the bytes (one origin's
document served to another).
"""

from __future__ import annotations

import brotli
import pytest
from fastapi.testclient import TestClient

from app.core import compression as C
from app.main import app


@pytest.fixture(autouse=True)
def _clear_cache():
    C._brotli_cache.clear()
    yield
    C._brotli_cache.clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# accepts_brotli
# --------------------------------------------------------------------------

@pytest.mark.parametrize("header,expected", [
    ("br", True),
    ("gzip, deflate, br", True),
    ("gzip, deflate, br, zstd", True),
    ("br;q=1.0, gzip;q=0.8", True),
    ("gzip, deflate", False),
    ("identity", False),
    ("", False),
    (None, False),
    # The one that matters: a token that merely CONTAINS "br" is not "br".
    ("brotli-ish", False),
    ("gzip, deflateBR", False),
])
def test_accepts_brotli_reads_the_token_not_the_substring(header, expected):
    assert C.accepts_brotli(header) is expected


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def test_the_same_key_returns_the_identical_object():
    """Not just equal -- the same bytes object, which is what proves the second
    call did no work."""
    body = b"<html>" + b"x" * 4000 + b"</html>"
    first = C.brotli_encode(body, "k")
    second = C.brotli_encode(body, "k")
    assert first is second


def test_the_output_decompresses_to_the_input():
    body = ("<html>" + "hello brotli " * 500 + "</html>").encode()
    packed = C.brotli_encode(body, "k")
    assert brotli.decompress(packed) == body
    assert len(packed) < len(body)


def test_incompressible_input_falls_through_to_gzip():
    """When brotli cannot beat the raw body there is nothing to serve; None
    sends the caller back to the gzip path rather than shipping a bigger page."""
    import os
    noise = os.urandom(2048)
    assert C.brotli_encode(noise, "k") is None


def test_the_cache_is_bounded():
    """The key comes from a request-shaped value, so an attacker varying the
    Host header must not be able to grow this without limit."""
    body = b"<html>" + b"y" * 4000 + b"</html>"
    for i in range(C._BROTLI_CACHE_MAX * 3):
        C.brotli_encode(body, f"key-{i}")
    assert len(C._brotli_cache) <= C._BROTLI_CACHE_MAX


def test_missing_brotli_is_not_an_error(monkeypatch):
    """The dependency is optional at runtime; without it the shell must still
    serve, just in gzip."""
    monkeypatch.setattr(C, "_brotli", None)
    assert C.brotli_encode(b"<html>" + b"z" * 4000, "k") is None
    assert C.brotli_available() is False


# --------------------------------------------------------------------------
# the shell over HTTP
# --------------------------------------------------------------------------

def test_a_client_that_asks_for_brotli_gets_it(client):
    r = client.get("/app", headers={"Accept-Encoding": "br"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "br"


def test_a_client_that_cannot_read_brotli_never_gets_it(client):
    r = client.get("/app", headers={"Accept-Encoding": "identity"})
    assert r.headers.get("content-encoding") != "br"


def test_gzip_still_works(client):
    r = client.get("/app", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"


def test_both_encodings_carry_the_same_document(client):
    """httpx decodes both, so this compares the HTML the browser would run.
    A mismatch means one path is serving a different build."""
    br = client.get("/app", headers={"Accept-Encoding": "br"})
    gz = client.get("/app", headers={"Accept-Encoding": "gzip"})
    assert br.text == gz.text
    assert "<title>" in br.text.lower() or "<!doctype" in br.text.lower()


def test_the_response_varies_on_accept_encoding(client):
    """Without this a shared cache serves whichever encoding it saw first to
    everyone -- including brotli to a client that asked for gzip."""
    r = client.get("/app", headers={"Accept-Encoding": "br"})
    assert "accept-encoding" in r.headers.get("vary", "").lower()


def test_the_body_is_encoded_once(client):
    """Starlette's responder passes through anything that already carries a
    Content-Encoding. If that ever stops being true the body would be brotli
    inside gzip, and the header would name only one of them."""
    r = client.get("/app", headers={"Accept-Encoding": "br, gzip"})
    assert r.headers["content-encoding"] == "br"


def test_the_etag_path_still_answers_304(client):
    r = client.get("/app", headers={"Accept-Encoding": "gzip"})
    etag = r.headers["etag"]
    again = client.get(
        "/app", headers={"If-None-Match": etag, "Accept-Encoding": "br"},
    )
    assert again.status_code == 304
    assert "accept-encoding" in again.headers.get("vary", "").lower()


def test_the_landing_page_is_compressed_too(client):
    r = client.get("/", headers={"Accept-Encoding": "br"})
    assert r.headers["content-encoding"] == "br"


def test_two_shells_are_cached_separately(client):
    """Keyed on the ETag, which is a hash of the finished document -- so the
    app and the landing page cannot collide, and neither can two origins."""
    client.get("/app", headers={"Accept-Encoding": "br"})
    client.get("/", headers={"Accept-Encoding": "br"})
    assert len(C._brotli_cache) == 2
    assert len(set(C._brotli_cache.values())) == 2
