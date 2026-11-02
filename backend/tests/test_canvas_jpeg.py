"""Canvas JPEGs are encoded with toDataURL, never a bare toBlob.

On a moto g play (Android 16) every toBlob in the Capacitor WebView waited
out a ~4 s idle-task timeout while a <video> was on the page: reading a clip
(six frames) took 31 s and the share card stalled the same way. toDataURL
encodes the same frame in ~11 ms. canvasJpeg() is the one place allowed to
fall back to toBlob.
"""

from __future__ import annotations

import re
from pathlib import Path

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def test_every_canvas_jpeg_goes_through_the_helper() -> None:
    src = SPA.read_text(encoding="utf-8")
    calls = [m.start() for m in re.finditer(r"\.toBlob\(", src)]
    helper = src.index("async function canvasJpeg(")
    helper_end = src.index("\n}\n", helper)
    assert calls, "the helper's fallback should still be there"
    assert all(helper < i < helper_end for i in calls), "a bare toBlob outside canvasJpeg()"
    assert "convertToBlob" not in src.replace("OffscreenCanvas.convertToBlob) as", "")
    assert src.count("canvasJpeg(") >= 4    # the definition + frames, photo, share card
