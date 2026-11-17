#!/usr/bin/env python3
"""CLI wrapper around the shared analysis runner.

Thin front-end over ``app.services.video_analysis.runner.run_analysis`` -- the
same code path the FastAPI service uses. Prints the result JSON to stdout
(logs go to stderr).

Usage:
    python analyze_local.py <video_path> <run|bike> [--position road_hoods] [--overlay [PATH]]

Detectors: production refines MediaPipe's points with RTMPose on bike and run
(Railway VA_RTMPOSE_SPORTS=bike,run), and this script does the same unless told
otherwise (``--detector mediapipe``, or VA_RTMPOSE_SPORTS set by hand). It used
to run MediaPipe alone, and a local result was then not the one a user gets:
one time-trial clip read 61/C "lower the saddle 30 mm" here and 89/B "no
change" in production (2026-11-17). The detector that measured is printed last.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

# --- make `app...` importable when run from backend/scripts/ ---------------
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import structlog  # noqa: E402

# Route structlog to stderr so stdout carries ONLY the result JSON (otherwise
# the pretty console logs interleave and break `... > result.json`). MediaPipe's
# native logs already go to stderr.
structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))

from app.core import config  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.services.video_analysis.runner import (  # noqa: E402
    DEFAULT_BIKE_POSITION,
    VALID_POSITIONS,
    _json_safe,
    run_analysis,
)


# What production runs (Railway VA_RTMPOSE_SPORTS), as in golden_baseline.py.
PROD_RTMPOSE_SPORTS = ("bike", "run")
REFINED_SUFFIX = "+rtmpose_m"


def use_detector(choice: str) -> tuple[str, ...]:
    """Point the shared settings at the detectors asked for; returns the sports
    RTMPose will refine. An explicit VA_RTMPOSE_SPORTS in the environment wins
    over the default, never over an explicit --detector."""
    if choice == "mediapipe":
        sports: tuple[str, ...] = ()
    elif choice == "prod":
        sports = PROD_RTMPOSE_SPORTS
    elif os.environ.get("VA_RTMPOSE_SPORTS") is not None:
        sports = settings.rtmpose_sports
    else:
        sports = PROD_RTMPOSE_SPORTS
    config.settings = dataclasses.replace(config.settings, rtmpose_sports=sports)
    return sports


def detector_note(result: dict, sport_type: str, asked: tuple[str, ...]) -> str:
    """One line for stderr: which detector measured, loud when it is not the
    one production uses."""
    used = (result.get("sport_specific_metrics") or {}).get("pose_detector") or "unknown"
    if used.endswith(REFINED_SUFFIX):
        return f"Detector: {used} (as production)."
    if sport_type in asked:
        return (f"WARNING: detector {used}: RTMPose was asked for but did not run "
                "(model file or onnxruntime missing?). This is NOT what production "
                "measures; do not compare these numbers with a user's report.")
    return (f"WARNING: detector {used}: MediaPipe alone. Production refines with "
            "RTMPose; these numbers are NOT what a user gets.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone side-view technique analysis (run/bike).",
    )
    parser.add_argument("video_path", help="Path to a local video file.")
    parser.add_argument("sport", choices=["run", "bike"], help="Sport type.")
    parser.add_argument(
        "--position", default=None,
        help=(
            "Cycling position (bike only): "
            "road_hoods | road_drops | tt_aero | triathlon | casual. "
            f"Default: {DEFAULT_BIKE_POSITION}."
        ),
    )
    parser.add_argument(
        "--overlay", nargs="?", const="__DEFAULT__", default=None, metavar="PATH",
        help=(
            "Also render an annotated overlay video (skeleton + angles + score). "
            "Optionally give an output .mp4 path; default: <video>_overlay.mp4 "
            "next to the input. With ffmpeg on PATH the output is web-safe H.264; "
            "otherwise it is written directly via OpenCV (mp4v)."
        ),
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip the Gemini coaching recommendations (also skipped if no GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--frames-store", nargs="?", const="__DEFAULT__", default=None, metavar="PATH",
        help=(
            "Also write the stabilized landmark frames (.npz) so the analysis "
            "can be re-run without MediaPipe -- see landmark_store. Default "
            "path: <video>_landmarks.npz next to the input."
        ),
    )
    parser.add_argument(
        "--detector", choices=["prod", "mediapipe"], default=None,
        help=(
            "prod: MediaPipe refined by RTMPose on bike and run, as production "
            "(the default unless VA_RTMPOSE_SPORTS is set). mediapipe: MediaPipe alone."
        ),
    )
    args = parser.parse_args()
    asked = use_detector(args.detector or "env")

    video_path = args.video_path
    sport_type = args.sport

    if not Path(video_path).is_file():
        print(f"ERROR: video file not found: {video_path}", file=sys.stderr)
        return 2

    cycling_position: str | None = None
    if sport_type == "bike":
        cycling_position = args.position or DEFAULT_BIKE_POSITION
        if cycling_position not in VALID_POSITIONS:
            print(
                f"ERROR: invalid --position '{cycling_position}'. "
                f"Valid: {', '.join(sorted(VALID_POSITIONS))}",
                file=sys.stderr,
            )
            return 2
    elif args.position:
        print("WARNING: --position is ignored for run.", file=sys.stderr)

    if not settings.model_path.exists():
        print(
            f"WARNING: pose model not found at {settings.model_path}\n"
            f"         Place 'pose_landmarker_heavy.task' in {settings.models_dir} "
            f"(see models/README.md). Attempting anyway...",
            file=sys.stderr,
        )

    overlay_path: str | None = None
    if args.overlay is not None:
        if args.overlay == "__DEFAULT__":
            vp = Path(video_path)
            overlay_path = str(vp.with_name(f"{vp.stem}_overlay.mp4"))
        else:
            overlay_path = args.overlay

    frames_store: str | None = None
    if args.frames_store is not None:
        if args.frames_store == "__DEFAULT__":
            vp = Path(video_path)
            frames_store = str(vp.with_name(f"{vp.stem}_landmarks.npz"))
        else:
            frames_store = args.frames_store

    try:
        result = run_analysis(
            video_path, sport_type, cycling_position, overlay_path=overlay_path,
            recommendations=not args.no_llm, frames_store=frames_store,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    if result.get("overlay_video_path"):
        print(f"\nOverlay video: {result['overlay_video_path']}", file=sys.stderr)
    print(detector_note(result, sport_type, asked), file=sys.stderr)
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
