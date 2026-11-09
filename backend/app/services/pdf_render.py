"""The report as a real PDF file, rendered by Gotenberg (a Chromium service).

Until 2026-11-09 the report was an HTML page the reader turned into a PDF with
their own browser's Print: on an iPhone that was US Letter with Safari's own
address, date and page count over our margins, no page numbers anywhere, and no
file the server could keep or attach. Chromium lives in its own Railway service
(GOTENBERG_URL, private network) so a PDF never competes with an analysis for
memory. The same HTML goes in, so the PDF is the page we test.

Unset GOTENBERG_URL (a laptop, a test) or a failed render raises PdfUnavailable,
and the caller falls back to the printable page.
"""

from __future__ import annotations

import asyncio
import re
from html import escape

import structlog

from app.core.config import settings

logger = structlog.get_logger()

# A4, in the inches Gotenberg takes. The bottom margin holds the page footer;
# the rest of the page spacing is the document's own (report_pdf's .flow table).
_PAPER = {"paperWidth": "8.27", "paperHeight": "11.69", "marginTop": "0", "marginBottom": "0.47",
          "marginLeft": "0", "marginRight": "0", "printBackground": "true",
          "emulatedMediaType": "print", "preferCssPageSize": "false"}
TIMEOUT_S = 60.0
# Two at a time from this process: the service queues the rest.
_SLOTS = asyncio.Semaphore(2)


def service_url() -> str | None:
    """Where the PDF service is, or None (read through here so tests can move it)."""
    return (settings.gotenberg_url or "").rstrip("/") or None


class PdfUnavailable(Exception):
    """No PDF service, or it failed: serve the printable page instead."""


def footer_html(left: str) -> str:
    """Chromium's page footer: whose report on the left, the page count on the right.
    It draws in the bottom margin, with its own tiny default font, so it is sized here."""
    return (
        "<html><head><style>html,body{margin:0;padding:0}"
        "div{box-sizing:border-box;width:210mm;padding:0 15mm;display:flex;justify-content:space-between;"
        "font:7pt Helvetica,Arial,sans-serif;color:#6E7787}</style></head><body><div>"
        f"<span>{escape(left)}</span>"
        '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>'
        "</div></body></html>"
    )


def file_name(title: str) -> str:
    """The saved file's name, from the report's title, safe on every filesystem."""
    name = re.sub(r'[\\/:*?"<>|\r\n]+', " ", title).strip() or "Flapp report"
    return f"{name[:120]}.pdf"


async def html_to_pdf(html: str, footer_left: str) -> bytes:
    url = service_url()
    if not url:
        raise PdfUnavailable("no PDF service configured")
    import httpx

    files = [
        ("files", ("index.html", html.encode("utf-8"), "text/html")),
        ("files", ("footer.html", footer_html(footer_left).encode("utf-8"), "text/html")),
    ]
    try:
        async with _SLOTS, httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(f"{url}/forms/chromium/convert/html", data=_PAPER, files=files)
    except Exception as e:  # noqa: BLE001 -- any transport failure means: print instead
        logger.warning("PDF_RENDER_FAILED", err=str(e)[:200])
        raise PdfUnavailable(str(e)) from e
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        logger.warning("PDF_RENDER_FAILED", status=r.status_code, body=r.text[:200])
        raise PdfUnavailable(f"status {r.status_code}")
    return r.content
