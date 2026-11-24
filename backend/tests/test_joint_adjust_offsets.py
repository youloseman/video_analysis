"""Joint adjustments reach analytics as numbers (2026-11-24).

Result stats on stored analyses could not say where the model puts a joint
wrong: photos are never stored and no video correction was kept. So the
``joint_adjust_applied`` event carries the offsets themselves -- which joint,
how far toward the rider's front and how far down, in % of the thigh. Numbers
only, never a picture.
"""
from __future__ import annotations

from pathlib import Path

SPA = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"


def test_photo_and_video_adjustments_carry_their_offsets():
    spa = SPA.read_text(encoding="utf-8")
    assert "function adjOffsets(corrs, hip, knee, aspect, side)" in spa
    assert "offsets:bikeMode?null:adjOffsetsPhoto(state.photoCorrections" in spa
    assert "offsets:kind==='reset'?null:adjOffsetsVideo(j.corrections,j.result)" in spa


def test_the_offsets_are_numbers_not_pixels_or_pictures():
    spa = SPA.read_text(encoding="utf-8")
    body = spa[spa.index("function adjOffsets("):spa.index("async function recomputePhoto(")]
    # normalised by the thigh, capped in count, wrapped so analytics never breaks the edit
    assert "Math.hypot(" in body and ".slice(0,12)" in body
    assert body.count("try{") == 3
    for leak in ("toDataURL", "canvas", "image", "file"):
        assert leak not in body, leak
