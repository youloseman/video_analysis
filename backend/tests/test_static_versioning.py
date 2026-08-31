"""The stylesheet's own URL, and why it carries a hash.

109 KB of CSS used to be inlined in index.html. The document is stamped with a
build id on every deploy, so that CSS was re-downloaded on every push whether
or not a single rule had changed. Out on its own content-hashed URL it is
fetched once and then kept until the CSS itself changes.

The whole saving depends on two things staying true, and both fail silently:
the link in the document must carry the hash (or nothing is cacheable), and
the hash must actually track the file's bytes (or a CSS change never reaches
anyone, which is worse than the bandwidth it saves).
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.main import _static_version, app
from tests.conftest import read_app_css, read_spa

LINK = re.compile(r'<link rel="stylesheet" href="(/app\.css(?:\?v=([0-9a-f]+))?)">')


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

def test_the_css_is_no_longer_inlined():
    """A <style> block creeping back would quietly undo this."""
    assert "<style>" not in read_spa()
    assert len(read_app_css()) > 50_000


def test_the_served_link_carries_a_hash(client):
    doc = client.get("/app").text
    m = LINK.search(doc)
    assert m, "the shell serves no stylesheet link"
    assert m.group(2), "the link has no ?v= hash, so nothing can be cached hard"


def test_the_hash_matches_the_file(client):
    doc = client.get("/app").text
    assert LINK.search(doc).group(2) == _static_version("app.css")


def test_tokens_are_linked_before_the_stylesheet(client):
    """tokens.css defines the variables app.css consumes, and the comment in
    the document says so. Order is load-bearing."""
    doc = client.get("/app").text
    assert doc.index("/tokens.css") < doc.index("/app.css")


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------

def test_a_hashed_request_is_cached_for_a_year(client):
    r = client.get(f"/app.css?v={_static_version('app.css')}")
    assert r.status_code == 200
    assert "immutable" in r.headers["cache-control"]
    assert "max-age=31536000" in r.headers["cache-control"]


def test_an_unhashed_request_revalidates_instead(client):
    """Someone typing the path, or an old document naming it bare. Serving
    that with a year's cache would pin a stale stylesheet in their browser
    with no way to bust it."""
    r = client.get("/app.css")
    assert r.status_code == 200
    assert "no-cache" in r.headers["cache-control"]


def test_a_wrong_hash_revalidates_rather_than_failing(client):
    r = client.get("/app.css?v=deadbeefcafe")
    assert r.status_code == 200
    assert "no-cache" in r.headers["cache-control"]


def test_it_is_served_as_css(client):
    r = client.get("/app.css")
    assert r.headers["content-type"].startswith("text/css")


def test_the_stylesheet_is_compressed(client):
    """25 KB gzipped against 110 KB raw. It is text, and the selective gzip
    middleware has to be letting it through."""
    r = client.get("/app.css", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"


# --------------------------------------------------------------------------
# the hash itself
# --------------------------------------------------------------------------

def test_the_version_tracks_the_bytes(tmp_path, monkeypatch):
    """The failure that matters: a hash that does not change when the CSS does
    means a fixed stylesheet never reaches anyone, held behind a year-long
    cache. Worth more than the bandwidth this whole thing saves."""
    from app import main

    _static_version.cache_clear()
    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    (tmp_path / "x.css").write_bytes(b"a{color:red}")
    first = _static_version("x.css")
    _static_version.cache_clear()
    (tmp_path / "x.css").write_bytes(b"a{color:blue}")
    assert _static_version("x.css") != first
    _static_version.cache_clear()


def test_a_missing_file_does_not_break_the_page(tmp_path, monkeypatch):
    """A shell that raises because a static file is absent is a worse outage
    than one served with an unhelpful cache header."""
    from app import main

    _static_version.cache_clear()
    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    assert _static_version("nothing-here.css") == "0"
    _static_version.cache_clear()


def test_the_version_is_cached_per_process():
    """Hashing 110 KB on every page load to learn what a deploy already
    decided would be a strange way to save bandwidth."""
    _static_version.cache_clear()
    _static_version("app.css")
    hits_before = _static_version.cache_info().hits
    _static_version("app.css")
    assert _static_version.cache_info().hits == hits_before + 1
