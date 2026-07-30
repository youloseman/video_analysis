"""Response compression: gzip for text, hands off everything else.

Why this is not just ``app.add_middleware(GZipMiddleware)``
----------------------------------------------------------
The whole SPA -- CSS, markup and JS -- is inlined in one ~330 KB document, so
compression is worth real money on first paint (it lands around 60 KB). But
Starlette's GZipMiddleware compresses **every** response over ``minimum_size``
without looking at the content type, and for this app that is actively harmful:

* ``/jobs/{id}/overlay`` serves ``video/mp4``. Gzip cannot shrink an already
  compressed container, so the CPU buys nothing -- and the streaming branch
  deletes ``Content-Length``, which is what ``<video>`` uses to know the clip's
  duration and to seek.
* Since Starlette 1.x, ``FileResponse`` answers Range requests with ``206`` and
  a ``Content-Range`` header. Those byte offsets describe the *uncompressed*
  file, so gzipping the body would have the response advertise one thing and
  deliver another. Media scrubbing is exactly the feature that breaks.

So the decision is made per response, once the content type is actually known.

A note on the coupling
----------------------
This reaches into Starlette's responder rather than reimplementing a
compressor, which means it is tied to internals that have moved before: the hook
was called ``send_with_gzip`` in 0.38 and is ``send_with_compression`` in 1.x,
and the opt-out flag was ``content_encoding_set``, now ``content_type_is_excluded``.
When that renaming happened the override simply stopped being called -- no error,
just video quietly being gzipped again. ``tests/test_gzip_middleware.py`` pins
the behaviour so the next such rename fails loudly in the suite instead of
silently in production.
"""

from __future__ import annotations

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipMiddleware, GZipResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Everything this app serves that is worth compressing. An allow-list, not a
# deny-list: a new binary endpoint should default to "left alone", not to
# "gzipped until someone notices the video stopped seeking".
GZIP_CONTENT_TYPES = (
    "text/",
    "application/json",
    "application/javascript",
    "application/xml",
    "application/rss+xml",
    "image/svg+xml",
)


class _SelectiveGZipResponder(GZipResponder):
    """GZipResponder that opts out of binary and partial-content responses.

    Rides the base class's own passthrough: it already forwards a response
    untouched whenever ``content_type_is_excluded`` is set (that is how it skips
    ``text/event-stream``), so raising the flag for anything outside
    :data:`GZIP_CONTENT_TYPES` is the entire change.

    The flag is raised *after* delegating, because the base assigns it from the
    response headers while handling the start message and would otherwise clear
    ours. That ordering is safe: nothing is written to the wire during the start
    message -- it is buffered precisely so the headers can still be edited.
    """

    async def send_with_compression(self, message: Message) -> None:
        await super().send_with_compression(message)
        if message["type"] != "http.response.start" or self.content_type_is_excluded:
            return
        headers = Headers(raw=message["headers"])
        content_type = headers.get("content-type", "").split(";")[0].strip().lower()
        # A 206 body is a slice of the file, described by byte offsets into the
        # uncompressed original -- never compress it.
        partial = message.get("status") == 206 or "content-range" in headers
        if partial or not content_type.startswith(GZIP_CONTENT_TYPES):
            self.content_type_is_excluded = True


class SelectiveGZipMiddleware(GZipMiddleware):
    """GZip, but only for text-ish whole-body responses.

    See :data:`GZIP_CONTENT_TYPES` for what qualifies, and the module docstring
    for why the stock middleware is not usable here as-is.
    """

    def __init__(
        self, app: ASGIApp, minimum_size: int = 1000, compresslevel: int = 6,
    ) -> None:
        # compresslevel 6 rather than Starlette's 9: on a document this size the
        # last three levels buy a couple of percent for several times the CPU.
        super().__init__(app, minimum_size=minimum_size, compresslevel=compresslevel)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and "gzip" in Headers(scope=scope).get("Accept-Encoding", ""):
            responder = _SelectiveGZipResponder(
                self.app, self.minimum_size, compresslevel=self.compresslevel,
            )
            await responder(scope, receive, send)
            return
        # No gzip on offer: let the base class handle it, so clients that ask for
        # identity still get the Vary: Accept-Encoding it adds for caches.
        await super().__call__(scope, receive, send)
