"""Public sample reports: /examples and /examples/{slug}.

Server-rendered like the Academy, and for the same two reasons: it is a page
search engines have to be able to read, and it is the top of the funnel -- the
thing somebody sees before they have an account. See ``services/examples`` for
what is on it and why the numbers are fixtures.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app.api.academy import PAGE_CACHE as _CACHE
from app.api.academy import _base_url
from app.services import examples

router = APIRouter(tags=["examples"])


@router.get("/examples", include_in_schema=False)
def examples_hub(request: Request) -> HTMLResponse:
    return HTMLResponse(
        examples.render_hub(_base_url(request)),
        headers={"Cache-Control": _CACHE},
    )


@router.get("/examples/{slug}", include_in_schema=False)
def example_page(slug: str, request: Request) -> HTMLResponse:
    sample = examples.get_sample(slug)
    if sample is None:
        # A missing sample is not an error worth a 500 or a bare 404 body: send
        # them to the ones that do exist.
        return HTMLResponse(
            examples.render_hub(_base_url(request)),
            status_code=404,
            headers={"Cache-Control": _CACHE},
        )
    return HTMLResponse(
        examples.render_sample(sample, _base_url(request)),
        headers={"Cache-Control": _CACHE},
    )
