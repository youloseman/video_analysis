"""The app shell is rendered once per process, not once per request.

Serving ``/`` or ``/app`` means reading a ~650 KB document off disk, running
ten full-document string replacements over it (analytics, four price tokens,
the stylesheet hash, the build stamp, three per-origin URL rewrites) and
hashing it twice. Measured at about 19 ms of CPU on a developer machine, so
more than that on the container -- and it was being spent on every single
request, including the ones that end in a 304, because the ETag can only be
compared against a document that has already been built.

Every input is fixed for the life of the process except the request's own
origin, which is why the cache is keyed on it. The tests here are about the two
ways that can go wrong: paying the cost anyway (the cache not being consulted),
and one origin being served another's document (the key being too loose).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import _render_shell, app


@pytest.fixture(autouse=True)
def _clear():
    _render_shell.cache_clear()
    yield
    _render_shell.cache_clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_a_second_request_renders_nothing(client):
    """The whole point. A miss here is 19 ms of CPU per page view, on a
    single-worker box whose CPU is already the analysis bottleneck."""
    client.get("/app")
    misses = _render_shell.cache_info().misses
    client.get("/app")
    assert _render_shell.cache_info().misses == misses
    assert _render_shell.cache_info().hits >= 1


def test_a_304_costs_nothing_either(client):
    """The revalidation path was the worst case: full price, no document
    served. It is also the common one -- the shell is ``no-cache``, so every
    repeat visitor takes it."""
    first = client.get("/app")
    misses = _render_shell.cache_info().misses
    again = client.get("/app", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    assert _render_shell.cache_info().misses == misses


def test_the_two_shells_are_not_one_entry(client):
    """The landing page and the app are different documents at different
    canonical paths; sharing a cache entry would serve one as the other."""
    landing = client.get("/").text
    spa = client.get("/app").text
    assert landing != spa
    assert 'property="og:url" content="http://testserver/"' in landing
    assert 'property="og:url" content="http://testserver/app"' in spa


def test_each_origin_keeps_its_own_document(client):
    """og:url, og:image and the canonical link are absolute and built from the
    request's host, so a cache that ignored the origin would advertise one
    domain's URLs on another -- the link-preview bug this rewriting exists to
    prevent, reintroduced by the cache meant to make it cheap."""
    a = client.get("/app", headers={"Host": "getflapp.com"}).text
    b = client.get("/app", headers={"Host": "flapp.up.railway.app"}).text
    assert 'content="http://getflapp.com/app"' in a
    assert 'content="http://getflapp.com/og-image.png"' in a
    assert 'content="http://flapp.up.railway.app/app"' in b
    assert "flapp.up.railway.app" not in a


def test_the_forwarded_protocol_is_part_of_the_key(client):
    """Behind Railway's proxy TLS ends at the edge, so the scheme comes from
    X-Forwarded-Proto. Two requests differing only in that are two documents,
    and the https one must not be served http:// URLs from the cache."""
    plain = client.get("/app", headers={"Host": "getflapp.com"}).text
    secure = client.get(
        "/app",
        headers={"Host": "getflapp.com", "X-Forwarded-Proto": "https"},
    ).text
    assert 'content="http://getflapp.com/app"' in plain
    assert 'content="https://getflapp.com/app"' in secure


def test_the_cache_cannot_be_grown_without_limit(client):
    """The key comes from request headers. An attacker varying Host must not be
    able to push the process's memory up one 650 KB document at a time -- the
    same backstop the brotli cache has, for the same reason."""
    for i in range(40):
        client.get("/app", headers={"Host": f"h{i}.example.com"})
    info = _render_shell.cache_info()
    assert info.currsize <= info.maxsize
    assert info.maxsize is not None and info.maxsize <= 16


def test_the_body_and_the_etag_describe_the_same_document(client):
    """They are cached as one tuple, so they cannot drift -- but a future edit
    could start recomputing one of them outside the cache, and a wrong ETag is
    a browser pinned to a build that no longer exists."""
    import hashlib

    r = client.get("/app", headers={"Accept-Encoding": "identity"})
    expected = '"' + hashlib.sha256(r.content).hexdigest()[:32] + '"'
    assert r.headers["etag"] == expected
