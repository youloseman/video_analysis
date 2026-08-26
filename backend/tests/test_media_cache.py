"""Landing media must be replaceable by a deploy. This is a silent failure.

The HTML shells go out ``no-cache``, so a deploy changes the markup for
everyone at once. ``/media`` used to be mounted as a bare ``StaticFiles``,
with no cache headers at all -- and a response with no ``Cache-Control`` does
not mean "ask again", it means the browser may invent a freshness lifetime
from the file's age. A hero video replaced in August therefore went on playing
the July clip for every returning visitor: fresh markup pointing at stale
film, with nothing in the logs and a green deploy.

The filenames under ``/media`` are stable by design (the landing page links
``hero.mp4``, not ``hero.<hash>.mp4``), so correctness has to come from
revalidation. ``_RevalidatingStatic`` supplies it by overriding one method on
Starlette's ``StaticFiles`` -- the same shape of dependency that already bit
this codebase once in ``test_gzip_middleware``: when Starlette renames the
method, the override simply stops being called and nothing raises.

So the load-bearing assertion is the first one. The other two exist because
the cheap fix for staleness -- refusing to cache at all -- would re-download
four megabytes on every page view and break seeking, and a passing header
test would not notice either.
"""

from __future__ import annotations

from starlette.testclient import TestClient

from app.main import app

HERO = "/media/hero.mp4"


def test_media_is_revalidated_rather_than_heuristically_cached():
    """The header exists at all -- i.e. the StaticFiles override still runs."""
    with TestClient(app) as client:
        r = client.get(HERO)
    assert r.status_code == 200
    cache = r.headers.get("cache-control", "")
    assert "no-cache" in cache, (
        f"/media served with Cache-Control={cache!r}: browsers are free to "
        "invent a freshness lifetime again, and a deployed video swap will "
        "not reach anyone who has already visited"
    )


def test_an_unchanged_file_still_answers_304():
    """Revalidating must stay cheap, or every page view re-sends the video."""
    with TestClient(app) as client:
        first = client.get(HERO)
        etag = first.headers.get("etag")
        assert etag, "no ETag: revalidation has nothing to compare and always re-sends"
        again = client.get(HERO, headers={"If-None-Match": etag})
    assert again.status_code == 304
    assert again.content == b""


def test_range_requests_still_come_back_as_206():
    """What a <video> element issues while seeking; a 200 here means no seeking."""
    with TestClient(app) as client:
        r = client.get(HERO, headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206
    assert r.headers["content-range"].startswith("bytes 0-1023/")
    assert len(r.content) == 1024


def test_a_cache_busting_query_still_resolves_the_file():
    """The landing page appends ?v=N; a mount that 404s on it breaks the hero."""
    with TestClient(app) as client:
        r = client.get(f"{HERO}?v=2")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
