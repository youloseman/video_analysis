"""The report's lead picture is cut to the rider (Gona's PDF, 2026-11-10: over
half of a portrait frame was gym ceiling). The stored frame stays whole; the PDF
cuts it to the rows the analysis kept."""
from __future__ import annotations

import base64
import io
from types import SimpleNamespace

from PIL import Image

from app.services.report_pdf import crop_keyframe, render_report_html
from app.services.video_analysis.video_visualizer import VideoVisualizer


def _viz(ys, vis=0.9):
    v = VideoVisualizer.__new__(VideoVisualizer)
    v.frame_data_list = [{"normalized_landmarks": [SimpleNamespace(x=0.5, y=y, z=0.0, visibility=vis) for y in ys]}]
    return v


def test_a_rider_in_the_lower_half_is_cut_to_with_room_round_them():
    rider = [0.45 + 0.45 * k / 32 for k in range(33)]           # head 0.45, feet 0.90
    y0, y1 = _viz(rider)._keyframe_crop(0)
    assert abs(y0 - (0.45 - 0.25 * 0.45)) < 0.01 and y1 == 1.0, "a quarter above; the ground is kept"


def test_a_rider_filling_the_frame_is_left_whole():
    assert _viz([0.08 + 0.85 * k / 32 for k in range(33)])._keyframe_crop(0) is None


def test_a_cut_near_the_top_keeps_the_badge_whole():
    rider = [0.15 + 0.40 * k / 32 for k in range(33)]           # head 0.15: 0.05 above it
    y0, _ = _viz(rider)._keyframe_crop(0)
    assert y0 == 0.0


def test_too_few_visible_points_cut_nothing():
    assert _viz([0.5] * 33, vis=0.1)._keyframe_crop(0) is None


def _uri(w=90, h=160):
    b = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(b, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _size(uri):
    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).size


def test_the_pdf_cuts_the_picture_and_falls_back_whole():
    assert _size(crop_keyframe(_uri(), [0.4, 1.0])) == (90, 96)
    for bad in (None, [0.9, 0.95], [0.5, 0.2], "x"):
        assert crop_keyframe(_uri(), bad) == _uri()
    html = render_report_html({"sport_type": "bike", "sport_specific_metrics": {"keyframe_crop": [0.4, 1.0]}},
                              {"keyframe": _uri()})
    src = html.split('<div class="lead"><img src="', 1)[1].split('"', 1)[0]
    assert _size(src) == (90, 96)
