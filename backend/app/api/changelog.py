"""Changelog routes: JSON for the in-app "What's new" view + a public page.

Routes
    GET /changelog.json  -> {latest, entries:[{id, date, date_label, title, html}]}
    GET /changelog       -> server-rendered release-notes page

Both read the same Markdown files (``backend/content/changelog/``). The SPA uses
the JSON to render the view and to decide whether the nav shows an unread dot;
the HTML page exists so a release is linkable and indexable.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse

# Same caching posture as the Academy pages: cheap for repeat views, short
# enough that a redeploy's notes show up promptly.
from app.api.academy import PAGE_CACHE as _CACHE
from app.services.changelog import get_entries, latest_id
from app.services.changelog.renderer import render_changelog

router = APIRouter(tags=["changelog"])


def _base_url(request: Request) -> str:
    """Absolute origin for canonical/OG links (mirrors app/api/academy.py)."""
    env = os.environ.get("ACADEMY_BASE_URL")
    if env:
        return env.rstrip("/")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host") or request.headers.get(
        "host", request.url.netloc
    )
    return f"{proto}://{host}".rstrip("/")


@router.get("/changelog.json")
def changelog_json(response: Response) -> dict[str, Any]:
    # Short cache only: the SPA hits this on load to decide whether to show the
    # unread dot, and a stale hour there means a release goes unannounced.
    response.headers["Cache-Control"] = "public, max-age=300"
    return {
        "latest": latest_id(),
        "entries": [
            {
                "id": e.id,
                "date": e.date,
                "date_label": e.date_label,
                "title": e.title,
                "html": e.body_html,
            }
            for e in get_entries()
        ],
    }


@router.get("/changelog", response_class=HTMLResponse, include_in_schema=False)
def changelog_page(request: Request) -> HTMLResponse:
    html_doc = render_changelog(get_entries(), _base_url(request))
    return HTMLResponse(html_doc, headers={"Cache-Control": _CACHE})
