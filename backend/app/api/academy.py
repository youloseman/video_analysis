"""Academy routes: server-rendered hub + article pages, sitemap, robots.

The whole section is HTML rendered on the server (not a JSON API consumed by
JS) so the article text ships in the initial response and is indexable — the
reason the section exists is organic search.

Routes
    GET /academy            -> hub (article grid)
    GET /academy/{slug}     -> a single article page
    GET /sitemap.xml        -> XML sitemap (home + academy + every article)
    GET /robots.txt         -> allow-all + sitemap pointer

Canonical/OG URLs use ``ACADEMY_BASE_URL`` when set (e.g.
``https://flapp.up.railway.app``); otherwise they are derived from the request
(honouring the ``X-Forwarded-*`` headers Railway sets).
"""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from app.services.academy import get_all_articles, get_article
from app.services.academy.renderer import (
    render_article,
    render_hub,
    render_robots,
    render_sitemap,
)

router = APIRouter(tags=["Academy"])

# Cache-Control for these static-ish HTML pages. 1h browser, 1d CDN — long
# enough to be cheap, short enough that a redeploy's edits show up promptly.
_CACHE = "public, max-age=3600, s-maxage=86400"


def _base_url(request: Request) -> str:
    """Absolute origin (scheme://host) for canonical/OG/sitemap links."""
    env = os.environ.get("ACADEMY_BASE_URL")
    if env:
        return env.rstrip("/")
    # Behind Railway's proxy, request.url.scheme can read as http; trust XFP.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get(
        "host", request.url.netloc
    )
    return f"{proto}://{host}".rstrip("/")


@router.get("/academy", response_class=HTMLResponse, include_in_schema=False)
def academy_hub(request: Request) -> HTMLResponse:
    html_doc = render_hub(get_all_articles(), _base_url(request))
    return HTMLResponse(html_doc, headers={"Cache-Control": _CACHE})


@router.get("/academy/{slug}", response_class=HTMLResponse, include_in_schema=False)
def academy_article(slug: str, request: Request) -> HTMLResponse:
    article = get_article(slug)
    if article is None:
        raise HTTPException(404, "article not found")
    html_doc = render_article(article, _base_url(request))
    return HTMLResponse(html_doc, headers={"Cache-Control": _CACHE})


@router.get("/sitemap.xml", include_in_schema=False)
def sitemap(request: Request) -> PlainTextResponse:
    xml = render_sitemap(get_all_articles(), _base_url(request))
    return PlainTextResponse(
        xml, media_type="application/xml", headers={"Cache-Control": _CACHE}
    )


@router.get("/robots.txt", include_in_schema=False)
def robots(request: Request) -> PlainTextResponse:
    return PlainTextResponse(
        render_robots(_base_url(request)),
        media_type="text/plain",
        headers={"Cache-Control": _CACHE},
    )
