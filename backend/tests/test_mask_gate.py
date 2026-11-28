"""The person mask (helmet reading, head lever) needs MediaPipe 1.1.

0.10.x aborts on the mask and 1.0.1 dies in the helmet step, so both are
refused; below the floor the skip is logged once instead of the helmet just
going missing (Aero Sim, 2026-11-14 and 2026-11-27).
"""
from __future__ import annotations

import mediapipe
import pytest

from app.services.video_analysis import photo_analyzer as pa


@pytest.mark.parametrize("version, ok", [
    ("1.1.0", True), ("1.2.3", True), ("2.0", True),
    ("1.0.1", False), ("0.10.35", False), ("", False), ("dev", False),
])
def test_the_mask_needs_mediapipe_1_1(monkeypatch, version, ok):
    monkeypatch.setattr(mediapipe, "__version__", version, raising=False)
    monkeypatch.setattr(pa, "_MASK_SKIP_LOGGED", True)
    assert pa._segmentation_mask_supported() is ok


def test_an_old_mediapipe_is_said_once(monkeypatch):
    monkeypatch.setattr(mediapipe, "__version__", "0.10.35", raising=False)
    monkeypatch.setattr(pa, "_MASK_SKIP_LOGGED", False)
    seen = []
    monkeypatch.setattr(pa.logger, "warning", lambda event, **kw: seen.append((event, kw)))
    pa._segmentation_mask_supported()
    pa._segmentation_mask_supported()
    assert seen == [("HELMET_SKIPPED_OLD_MEDIAPIPE", {"mediapipe": "0.10.35", "needs": "1.1"})]
