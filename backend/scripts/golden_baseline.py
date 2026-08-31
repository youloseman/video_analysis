#!/usr/bin/env python
"""Record, or re-record, what the reference clips are supposed to measure.

    python scripts/golden_baseline.py            # check against the baseline
    python scripts/golden_baseline.py --update   # re-record it

The clips are gitignored (they are large and they are Artur's), so this only
runs where they exist. The BASELINE is committed -- it is a few kilobytes of
JSON -- which is what makes a drift visible in a diff rather than only in a
test run.

When a change to the pipeline is deliberate, `--update` and commit the new
baseline WITH the change that caused it. The diff is then a statement about
what the change did to a real measurement, reviewed alongside the code. That
is the whole point: not to freeze the numbers, but to make moving them a
visible act instead of a silent one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.services.video_analysis.golden import build_record, compare  # noqa: E402

BASELINE_DIR = BACKEND / "tests" / "golden"

# The reference clips, and how they must be analysed. Paths are relative to the
# repository root. Keep the arguments here identical to the ones the test uses
# -- a baseline recorded with different settings compares nothing.
CLIPS: dict[str, dict[str, object]] = {
    "bike_tt_aero": {
        "path": "IMG_9981.MOV",
        "sport": "bike",
        "position": "tt_aero",
    },
    "run_slowmo": {
        "path": "upload/vid1.MOV",
        "sport": "run",
        "position": None,
    },
    # Added 2026-08-31, once it turned out nine clips filmed to spec had been
    # sitting in upload/ for days. The two above are pinned as KNOWN inputs
    # rather than good ones -- vid1 is 13% of the frame and reads as 8x slow
    # motion, so half its metrics never exist and the guard cannot see a
    # regression in them. These two can.
    "run_clean": {
        # Every stride metric present and measured: cadence, ground contact,
        # flight, oscillation, overstride, and a five-position kinogram. The
        # first clip in the repo where the running path is fully exercised.
        "path": "upload/IMG_4004.MOV",
        "sport": "run",
        "position": None,
    },
    "run_legs_unstable": {
        # The opposite end, and the reason the withheld-score path exists:
        # the legs traded places on 46.8% of frames, the gate fires, the
        # per-leg components are excluded and no score is published. Pinned so
        # that path cannot quietly start scoring again.
        "path": "upload/IMG_4262.MOV",
        "sport": "run",
        "position": None,
    },
    "bike_aero": {
        # A real aero fit on a trainer, against IMG_9981's indoor TT clip --
        # a second bike geometry so a change that only suits one shows up.
        "path": "upload/IMG_4088.MOV",
        "sport": "bike",
        "position": "tt_aero",
    },
}


def analyse(spec: dict[str, object]) -> dict:
    from app.services.video_analysis.runner import run_analysis

    path = BACKEND.parent / str(spec["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    result = run_analysis(
        str(path), str(spec["sport"]), spec["position"],  # type: ignore[arg-type]
        # Everything that is not deterministic measurement is off: the LLM
        # writes different prose every call, and the overlay is pixels nobody
        # compares. The kinogram stays ON -- whether it publishes at all is a
        # decision the analyzer makes about its own confidence, and that is
        # worth pinning.
        recommendations=False,
        kinogram=True,
        overlay_path=None,
    )
    return build_record(result)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--update", action="store_true", help="re-record the baseline")
    ap.add_argument("--clip", help="only this clip (default: all)")
    args = ap.parse_args()

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    names = [args.clip] if args.clip else sorted(CLIPS)
    failures = 0

    for name in names:
        spec = CLIPS.get(name)
        if spec is None:
            print(f"unknown clip: {name}; have {sorted(CLIPS)}")
            return 2
        target = BASELINE_DIR / f"{name}.json"
        print(f"\n=== {name} ({spec['path']})")
        try:
            record = analyse(spec)
        except FileNotFoundError as e:
            print(f"  SKIP -- clip not on this machine: {e}")
            continue

        if args.update or not target.exists():
            target.write_text(
                json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8",
            )
            print(f"  recorded -> {target.relative_to(BACKEND)}")
            continue

        baseline = json.loads(target.read_text(encoding="utf-8"))
        diffs = compare(baseline, record)
        if not diffs:
            print("  unchanged")
        else:
            failures += 1
            print(f"  {len(diffs)} DIFFERENCE(S):")
            for line in diffs:
                print(f"    {line}")

    if failures:
        print(
            f"\n{failures} clip(s) measure differently than the committed baseline.\n"
            "If that is what the change was for, re-run with --update and commit\n"
            "the new baseline alongside it."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
