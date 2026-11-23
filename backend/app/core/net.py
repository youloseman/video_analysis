"""Request-level client identity helpers.

Small and shared because two subsystems need the same notion of "who is this
caller" without an account: the analysis rate limiter (``app/main.py``) and the
feedback endpoint (``app/api/feedback.py``).
"""

from __future__ import annotations

import hashlib

from fastapi import Request

from app.core.config import settings


def client_ip(request: Request) -> str:
    """Best-effort client IP. Railway sits behind a proxy, so prefer XFF.

    Leftmost is safe HERE because Railway's edge replaces a caller's
    X-Forwarded-For (and X-Real-IP) rather than appending to it. Measured on
    prod 2026-11-23: ten failed logins, then the same with forged XFF and
    X-Real-IP values -- all still 429, so a forged header does not change the
    key. Re-check this if a CDN or a second proxy is ever put in front.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def ip_hash(ip: str) -> str:
    """Stable, salted digest of an IP for dedupe/limits without storing the IP.

    Salted with ``jwt_secret`` so the digests are not reversible via a rainbow
    table of the IPv4 space. Rotating that secret rotates the digests, which is
    acceptable: they are only used to group a caller's own submissions.
    """
    return hashlib.sha256(
        f"{settings.jwt_secret}:{ip}".encode()
    ).hexdigest()[:32]
