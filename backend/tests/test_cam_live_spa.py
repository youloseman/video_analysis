"""The live camera's GO / NO-GO bars are the server's.

The in-app camera judges the preview on the phone (index.html CAM_LIVE) and
the prescan judges the picked clip on the server (prescan.capture_check). If
the two drift, the camera shows green and the next screen says re-film -- the
one outcome that makes both checks look broken.
"""
from __future__ import annotations

import re
from pathlib import Path

from app.services.video_analysis.biomechanics.tracking_diagnostics import (
    FRAMING_SMALL_PX,
    FRAMING_TINY_PX,
    RUN_FRAMING_SMALL_PX,
    RUN_FRAMING_TINY_PX,
)
from app.services.video_analysis.capture_report import DETECT_MAX_LONG_EDGE

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def _cam_live() -> dict[str, int]:
    m = re.search(r"const CAM_LIVE=\{([^}]*)\}", SPA.read_text(encoding="utf-8"))
    assert m, "CAM_LIVE not found in index.html"
    return {k: int(v) for k, v in re.findall(r"(\w+):(\d+)", m.group(1))}


def test_the_live_camera_uses_the_prescan_bars():
    live = _cam_live()
    assert live["tinyPx"] == FRAMING_TINY_PX
    assert live["smallPx"] == FRAMING_SMALL_PX
    assert live["runTinyPx"] == RUN_FRAMING_TINY_PX
    assert live["runSmallPx"] == RUN_FRAMING_SMALL_PX
    assert live["detectLongEdge"] == DETECT_MAX_LONG_EDGE


def test_the_live_library_is_pinned_to_an_exact_version():
    src = SPA.read_text(encoding="utf-8")
    assert re.search(r"@mediapipe/tasks-vision@\d+\.\d+\.\d+'", src)
    assert "/float16/1/pose_landmarker_lite.task" in src
