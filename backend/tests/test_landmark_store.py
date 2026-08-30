"""The frames an analysis measured from can be written down and read back.

Until the store existed, an analysis was a one-shot: the landmark frames lived
for the length of one call and were gone, so nothing could ever be re-measured
-- not with an athlete's correction, not with a fixed analyzer, not in a test.
The property that matters is fidelity: what comes back is what went in, NaN
gaps and gate flags included, because the re-analysis assumes it is looking at
the same frames.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.video_analysis.landmark_store import (
    FORMAT_VERSION,
    load_frames,
    save_frames,
)


def _lm(x, y, z=0.0, vis=1.0):
    return SimpleNamespace(x=x, y=y, z=z, visibility=vis)


def _frame(i: int, *, gate=None) -> dict:
    rng = np.random.default_rng(i)
    norm = [_lm(*rng.random(2), z=rng.random() - 0.5, vis=rng.random()) for _ in range(33)]
    world = [_lm(*(rng.random(3) - 0.5), vis=rng.random()) for _ in range(33)]
    # a gated point, exactly as the stabilizer leaves one
    norm[27].x = norm[27].y = norm[27].z = math.nan
    world[27].x = world[27].y = world[27].z = math.nan
    f = {
        "normalized_landmarks": norm,
        "world_landmarks": world,
        "timestamp_ms": i * 33.3667,
        "frame_idx": i * 2,
        "frame_width": 1080,
        "frame_height": 1920,
    }
    if gate:
        f["leg_gate_filled"] = set(gate)
    return f


def test_what_comes_back_is_what_went_in(tmp_path):
    frames = [_frame(0), _frame(1, gate={"left"}), _frame(2, gate={"left", "right"})]
    meta = {"sport_type": "bike", "video_info": {"fps": 29.97}, "nested": {"a": [1, 2]}}

    save_frames(tmp_path / "landmarks.npz", frames, meta=meta)
    back, meta_back = load_frames(tmp_path / "landmarks.npz")

    assert meta_back == meta
    assert len(back) == 3
    for a, b in zip(frames, back):
        assert b["timestamp_ms"] == a["timestamp_ms"]
        assert b["frame_idx"] == a["frame_idx"]
        assert (b["frame_width"], b["frame_height"]) == (1080, 1920)
        for key in ("normalized_landmarks", "world_landmarks"):
            for la, lb in zip(a[key], b[key]):
                for attr in ("x", "y", "z", "visibility"):
                    va, vb = getattr(la, attr), getattr(lb, attr)
                    if math.isnan(va):
                        assert math.isnan(vb)
                    else:
                        assert vb == va, f"{key}.{attr} drifted"


def test_the_gate_flag_is_absent_unless_it_was_set(tmp_path):
    """Readers test ``frame.get("leg_gate_filled")`` for truthiness; an empty
    set where there was no key would still be falsy, but keep the shape exact."""
    frames = [_frame(0), _frame(1, gate={"right"})]
    save_frames(tmp_path / "s.npz", frames, meta={})
    back, _ = load_frames(tmp_path / "s.npz")
    assert "leg_gate_filled" not in back[0]
    assert back[1]["leg_gate_filled"] == {"right"}


def test_a_gated_point_stays_gated():
    """Not a store test as such: pins the NaN convention the store relies on."""
    f = _frame(0)
    assert math.isnan(f["normalized_landmarks"][27].x)


def test_nothing_at_the_path_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frames(tmp_path / "missing.npz")


def test_a_store_from_the_future_is_refused(tmp_path):
    save_frames(tmp_path / "s.npz", [_frame(0)], meta={})
    with np.load(tmp_path / "s.npz") as z:
        data = {k: z[k] for k in z.files}
    data["version"] = np.array(FORMAT_VERSION + 1)
    np.savez(tmp_path / "newer.npz", **data)
    with pytest.raises(ValueError, match="newer"):
        load_frames(tmp_path / "newer.npz")


def test_the_parent_directory_is_created(tmp_path):
    save_frames(tmp_path / "deep" / "er" / "s.npz", [_frame(0)], meta={})
    assert (tmp_path / "deep" / "er" / "s.npz").is_file()
