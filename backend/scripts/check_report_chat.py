"""Ask the report chat a fixed set of questions against the LIVE models.

The unit tests fake the model; this is the only check of what it actually
says. Run it before widening the chat and after any change to its rules
(services/report_chat.py SYSTEM) or to the model in GEMINI_MODEL:

    railway run --service video-analysis python scripts/check_report_chat.py
    # or locally, with GEMINI_API_KEY (and optionally OPENAI_API_KEY) set:
    python scripts/check_report_chat.py [--only helmet]

Uses the aero example result (content/examples/bike-aero.json) with one red
finding added, and a made-up athlete with a logged CdA. Prints each answer
and flags the rule breaks a string can catch:

* a CdA number that is not the athlete's own logged one;
* a named helmet model, or advice to change helmets;
* "nothing to fix" while a red finding stands;
* a named part (stem, extension, riser brand);
* a reply in English to a Russian question.

A flag is a reason to read the answer, not proof of a fault.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services import report_chat  # noqa: E402
from app.services.export import ai_export  # noqa: E402

QUESTIONS = {
    "saddle": "Why does the report say my saddle is too high, and how much should I lower it?",
    "first": "What should I change first, and what does it cost me?",
    "minutes": "How much time could my position be worth over a 90 km bike leg?",
    "cda": "What is my CdA in this position?",
    "cda_logged": "My Notio says 0.231. How does that compare with what you see in the video?",
    "helmet": "Which aero helmet would suit my head position best?",
    "parts": "Which extensions should I buy to get 30 mm more reach on my Trinity?",
    "nothing": "So is there nothing to fix?",
    "pain": "My lower back hurts after an hour. What is wrong?",
    "ru": "Почему мне рекомендуют опустить седло?",
    "offtopic": "Write me a poem about Kona.",
}

HELMETS = r"\b(giro aerohead|aerohead|kask|bambino|met drone|drone|poc tempor|tempor|rudy|wing|aerotop|lazer|vanquish|s-works tt)\b"
PARTS = r"\b(zipp|vuka|profile design|wattshop|tririg|ski[- ]?bend|enve|vision|3t|fasttt)\b"


def sample_result() -> dict:
    ex = json.loads((ROOT / "content/examples/bike-aero.json").read_text(encoding="utf-8"))
    return {
        "sport_type": "bike",
        "cycling_position": ex.get("cycling_position") or "triathlon",
        "camera_side": ex.get("camera_side") or "left",
        "frames_analyzed": 300,
        # The example scored 100; a red row caps a bike video at 89/B.
        "technique_score": 84,
        "letter_grade": "B",
        "angle_statistics": {},
        "sport_specific_metrics": dict(ex.get("sport_specific_metrics") or {}),
        "detected_issues": [{
            "type": "saddle_too_high", "severity": "moderate", "value": "156°",
            "description": "Knee nearly straight at the bottom of the stroke.",
        }],
    }


def flags(key: str, answer: str) -> list[str]:
    a = answer.lower()
    out = []
    for m in re.findall(r"0\.\d{2,3}", a):
        if m != "0.231":
            out.append(f"CdA-like number {m}")
    if re.search(HELMETS, a):
        out.append("names a helmet")
    if key == "helmet" and re.search(r"\b(try|buy|get|switch to|choose)\b[^.]{0,40}helmet", a):
        out.append("helmet advice")
    if "nothing to fix" in a and "not" not in a[:40]:
        out.append("'nothing to fix' over a red finding")
    if re.search(PARTS, a):
        out.append("names a part brand")
    if key == "ru" and not re.search(r"[а-я]", a):
        out.append("answered a Russian question in English")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated question keys")
    args = ap.parse_args()
    keys = args.only.split(",") if args.only else list(QUESTIONS)

    result = sample_result()
    report_md = ai_export.build_markdown(result, job_id="check")
    athlete_md = report_chat.athlete_block(
        sport="bike", height_cm=181,
        mobility={"hamstring_deg": 72, "hip_flexion_deg": 118, "measured_on": "2026-10-01"},
        profile={"name": "Race bike", "kind": "tt", "crank_length_mm": 170,
                 "bike_family": "giant-trinity", "bike_model": "Advanced Pro",
                 "frame_size": "M", "bike_year": 2019},
        cda=[{"measured_on": "2026-11-01", "value_m2": 0.231, "source": "notio",
              "protocol": "repeated_loops", "note": "base position"}],
        previous=[{"at": 1_760_000_000_000, "score": 78}],
    )
    flagged = 0
    for key in keys:
        q = QUESTIONS[key]
        prompt = report_chat.build_prompt(
            report_md=report_md, athlete_md=athlete_md, withheld_md=None,
            history=[], question=q,
        )
        ans = report_chat.ask(prompt)
        print("=" * 78)
        print(f"[{key}] {q}")
        if not ans:
            print("  !! NO ANSWER (chain exhausted)")
            flagged += 1
            continue
        print(f"  ({ans.get('model')}, {len(prompt)} prompt chars)")
        print(ans["text"])
        f = flags(key, ans["text"])
        if f:
            flagged += 1
            print("  >> FLAGS: " + "; ".join(f))
    print("=" * 78)
    print(f"{len(keys)} questions, {flagged} flagged")
    return 1 if flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
