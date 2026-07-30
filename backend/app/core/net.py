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
    """Best-effort client IP. Railway sits behind a proxy, so prefer XFF."""
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
