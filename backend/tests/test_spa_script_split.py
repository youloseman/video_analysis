"""The app's code is served as its own file, split out of the document.

index.html carries the whole SPA in one <script> block: 538 KB of code against
~110 KB of markup. It stays that way ON DISK -- sixteen test files read it as a
string, and the mobile shell copies it whole into a local bundle where a second
file would buy nothing. The split happens on the way out instead.

The point is caching. The shell is stamped with a build id and served
no-cache, so every deploy re-sends the whole document to everyone who comes
back. Most deploys do not touch the app's code -- a colour, a heading, a price
-- and those now cost a 25 KB document instead of a 163 KB one, with the code
answered from the browser's cache on a URL that changes only when the code
does.

Three ways this ships broken and still looks fine from Python: the served page
keeps the script inline AND names a src (the app runs twice), the hash stops
tracking the bytes (a code change never reaches anyone, held behind a year-long
cache), or the build stamp stops moving because the token now lives in a file
that nothing templates. The last one actually happened while writing this.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.main import (
    _render_shell,
    _spa_script,
    _spa_script_version,
    app,
)

SRC = re.compile(r'<script src="/app\.js(?:\?v=([0-9a-f]+))?"></script>')


@pytest.fixture(autouse=True)
def _clear():
    _render_shell.cache_clear()
    _spa_script.cache_clear()
    _spa_script_version.cache_clear()
    yield
    _render_shell.cache_clear()
    _spa_script.cache_clear()
    _spa_script_version.cache_clear()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------

def test_the_served_shell_names_a_hashed_script(client):
    m = SRC.search(client.get("/app").text)
    assert m, "the shell serves no /app.js"
    assert m.group(1), "the src has no ?v= hash, so nothing can be cached hard"


def test_the_hash_matches_the_bytes(client):
    doc = client.get("/app").text
    assert SRC.search(doc).group(1) == _spa_script_version("index.html")


def test_the_app_script_is_no_longer_in_the_document(client):
    """Both would mean the app runs twice -- every listener bound twice, every
    timer started twice. It is the one failure mode here that a browser does
    not announce."""
    doc = client.get("/app").text
    body = _spa_script("index.html")
    assert body is not None
    assert body not in doc
    assert len(doc) < 200_000, f"shell is still {len(doc)} chars"


def test_the_head_bootstrap_stays_inline(client):
    """It registers the service worker and sets the build id, and it has to run
    before anything else. Only the BIGGEST script block moves."""
    doc = client.get("/app").text
    assert "serviceWorker" in doc
    assert "__FLAPP_BUILD__" in doc


def test_the_script_src_sits_where_the_inline_block_did(client):
    """Last thing in the body, and not deferred: that is the execution order
    the app was written against."""
    doc = client.get("/app").text
    tail = doc[doc.index('<script src="/app.js'):]
    assert re.fullmatch(r'<script src="/app\.js\?v=[0-9a-f]+"></script>\s*</body>\s*</html>\s*', tail)


def test_the_landing_page_is_not_split(client):
    """Its scripts are small. Trading them for a round trip on the page whose
    whole job is to paint fast would be a loss."""
    assert _spa_script("landing.html") is None
    assert "/app.js" not in client.get("/").text


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------

def test_it_serves_exactly_what_was_inline(client):
    """Byte-identical, relocated. Anything else and the served app is not the
    app the sixteen string-grep tests are reading."""
    r = client.get(f"/app.js?v={_spa_script_version('index.html')}")
    assert r.status_code == 200
    assert r.text == _spa_script("index.html")


def test_a_hashed_request_is_cached_for_a_year(client):
    r = client.get(f"/app.js?v={_spa_script_version('index.html')}")
    assert "immutable" in r.headers["cache-control"]
    assert "max-age=31536000" in r.headers["cache-control"]


@pytest.mark.parametrize("url", ["/app.js", "/app.js?v=deadbeefcafe"])
def test_an_unhashed_or_wrong_hash_revalidates(client, url):
    """Serving those with a year's cache would pin a stale app in somebody's
    browser with no way to bust it."""
    r = client.get(url)
    assert r.status_code == 200
    assert "no-cache" in r.headers["cache-control"]


def test_it_is_served_as_javascript(client):
    r = client.get("/app.js")
    assert r.headers["content-type"].startswith("application/javascript")


def test_it_is_compressed(client):
    """527 KB raw. The selective gzip middleware has to be letting it through
    -- its allow-list is by content type, and this one is new."""
    r = client.get("/app.js", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip"


# --------------------------------------------------------------------------
# the hash, and the build id that rides on it
# --------------------------------------------------------------------------

def test_the_version_tracks_the_bytes(tmp_path, monkeypatch):
    """The failure that matters: a hash that does not move when the code does
    means a fixed app never reaches anyone, held behind a year-long cache."""
    from app import main

    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    big = "x" * 60_000
    (tmp_path / "s.html").write_text(f"<script>{big}</script>", encoding="utf-8")
    first = _spa_script_version("s.html")
    _spa_script.cache_clear()
    _spa_script_version.cache_clear()
    (tmp_path / "s.html").write_text(f"<script>{big}y</script>", encoding="utf-8")
    assert _spa_script_version("s.html") != first


def test_a_missing_file_does_not_break_the_page(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    assert _spa_script("nothing.html") is None
    assert _spa_script_version("nothing.html") == "0"


def test_a_small_script_is_left_alone(tmp_path, monkeypatch):
    """Below the threshold the round trip costs more than the bytes."""
    from app import main

    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    (tmp_path / "s.html").write_text("<script>var a=1;</script>", encoding="utf-8")
    assert _spa_script("s.html") is None


def test_the_biggest_script_is_the_one_that_moves(tmp_path, monkeypatch):
    """A document has several. The head bootstrap must stay put."""
    from app import main

    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    big = "b" * 60_000
    (tmp_path / "s.html").write_text(
        f"<script>var boot=1;</script><script>{big}</script>", encoding="utf-8",
    )
    assert _spa_script("s.html") == big


def test_the_build_stamp_is_not_the_literal_token(client):
    """It is read by the app out of the document, because the app's own file is
    static and nothing templates it. Getting this wrong files every piece of
    feedback against build "dev" -- which is what happened before this test."""
    doc = client.get("/app").text
    m = re.search(r"window\.__FLAPP_BUILD__ = '([^']*)'", doc)
    assert m, "the shell sets no build id"
    assert m.group(1) != "__BUILD__"
    assert re.fullmatch(r"[0-9a-f]{12}", m.group(1)), m.group(1)


def test_a_code_change_moves_the_build_id(tmp_path, monkeypatch):
    """The stamp is a hash of the finished document, and the document carries
    the code's content hash -- so the two move together. If the script src were
    written AFTER the stamp, a JS-only deploy would keep reporting the previous
    build forever."""
    from app import main

    monkeypatch.setattr(main, "STATIC_DIR", tmp_path)
    big = "z" * 60_000
    doc = "<head></head><body><script>%s</script></body>"
    (tmp_path / "s.html").write_text(doc % big, encoding="utf-8")
    first_html, _, first_etag = _render_shell("s.html", "/s", "http://x")

    for fn in (_render_shell, _spa_script, _spa_script_version):
        fn.cache_clear()
    (tmp_path / "s.html").write_text(doc % (big + "q"), encoding="utf-8")
    second_html, _, second_etag = _render_shell("s.html", "/s", "http://x")

    assert first_etag != second_etag
    assert SRC.search(first_html).group(1) != SRC.search(second_html).group(1)
