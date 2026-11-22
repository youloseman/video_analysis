"""GET /bands/evidence.json: why each reference range is what it is.

Public and static (band_evidence.py). Served on its own rather than inside
each result so a photo, an old report and the dashboard all get the same
answer; the client shows an entry only when its band is the one on screen.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from app.services.video_analysis.biomechanics.band_evidence import evidence_catalog

router = APIRouter(tags=["evidence"])


@router.get("/bands/evidence.json")
def bands_evidence(response: Response) -> dict[str, Any]:
    response.headers["Cache-Control"] = "public, max-age=3600"
    return evidence_catalog()
