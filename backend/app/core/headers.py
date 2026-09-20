"""Security headers on every response.

Production served none: no HSTS, nothing against framing, nothing against
MIME sniffing, no referrer policy. Each is one line, and each closes a real
door on an app that has a "Delete account" button, a Stripe checkout button
and a media token in the query string of its video URLs.

Pure ASGI rather than ``BaseHTTPMiddleware``: the latter buffers through a
wrapper that has broken streaming responses before (see core/compression.py
for the same choice), and the overlay video is a streamed ``FileResponse``.

What is deliberately NOT here: a Content-Security-Policy. The app shell is a
single document with its script and styles inline, plus PostHog and Google
Fonts; a policy strict enough to mean anything would need nonces threaded
through every ``<script>`` and ``<style>``, and a wrong one takes the whole
app down for everybody at once. It is worth doing, on its own, with a
report-only phase first.
"""

from __future__ import annotations

from collections.abc import Iterable

# (name, value), lower-case names as ASGI wants them.
SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    # A year, host only. ``includeSubDomains`` is left off on purpose: it would
    # commit every subdomain of getflapp.com to TLS, including ones that do not
    # exist yet, and cannot be taken back for the lifetime of the header in
    # every browser that has seen it. Browsers ignore this over plain HTTP, so
    # it is harmless on localhost.
    (b"strict-transport-security", b"max-age=31536000"),
    # Nothing here is meant to be embedded. The mobile shell loads the SPA
    # from its own bundle, not in a frame, and Stripe is a redirect.
    (b"x-frame-options", b"DENY"),
    (b"x-content-type-options", b"nosniff"),
    # Full URL to our own origin, origin only to anyone else. The overlay
    # <video> carries a one-hour media token in its query string, and the
    # report links out to the Academy and to Stripe.
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
)


class SecurityHeadersMiddleware:
    def __init__(self, app, headers: Iterable[tuple[bytes, bytes]] = SECURITY_HEADERS):
        self.app = app
        self.headers = tuple(headers)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in self.headers if k not in present]
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        await self.app(scope, receive, send_with_headers)
