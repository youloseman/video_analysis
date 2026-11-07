"""One clip through the real pipeline, MediaPipe alone or with RTMPose refinement.

    python scripts/pose_benchmark/ab_detector.py <off|on> <clip> <run|bike> [position|auto] [out_dir]

Run from the repository root with PYTHONPATH=backend, under MediaPipe 1.1.0 and
onnxruntime (the model at backend/models/rtmpose-m_body7_halpe26.onnx). Writes
<out_dir>/<clip>.<mode>.json: score, grade, withheld reason, camera side, the
detector stamp, timing and the golden record (golden.build_record), so two
modes compare field by field. ``auto`` reads the bike position the way the
coach lane does (prescan family). One process per clip and mode: three in
parallel ran out of memory silently on a 16 GB machine (2026-11-07).
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time


def main() -> None:
    args = [a.strip() for a in sys.argv[1:]]
    mode, path, sport = args[0], args[1], args[2]
    pos = args[3] if len(args) > 3 and args[3] else None
    out_dir = args[4] if len(args) > 4 else "ab_out"

    from app.core import config

    config.settings = dataclasses.replace(
        config.settings, rtmpose_sports=("bike", "run") if mode == "on" else ())

    from app.services.video_analysis import runner
    from app.services.video_analysis.golden import build_record

    if sport == "bike" and pos == "auto":
        from app.services.coach_lane import resolve_auto, scan_clip

        _s, pos, _n = resolve_auto("bike", "auto", scan_clip(path, "bike", {}))
    if sport == "run":
        pos = None
    t0 = time.time()
    r = runner.run_analysis(path, sport, pos, recommendations=False, kinogram=True, overlay_path=None)
    sm = r.get("sport_specific_metrics") or {}
    out = {
        "clip": os.path.basename(path), "mode": mode, "sport": sport, "position": pos,
        "seconds": round(time.time() - t0, 1),
        "detector": (sm.get("sampling") or {}).get("pose_detector"),
        "status": r.get("status"), "score": r.get("technique_score"), "grade": r.get("letter_grade"),
        "withheld": (r.get("score_withheld") or {}).get("reason"),
        "camera_side": r.get("camera_side"), "error": r.get("error_message"),
        "record": build_record(r),
    }
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    with open(os.path.join(out_dir, f"{name}.{mode}.json"), "w") as f:
        json.dump(out, f, default=str, indent=1)
    print(name, mode, out["seconds"], out["score"], out["grade"], out["withheld"], flush=True)


if __name__ == "__main__":
    main()
