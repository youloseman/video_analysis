"""Which parts fit this frame: the parts-compatibility data and its questions.

The data is files, not tables (app/data/parts/README.md): one JSON per frame
generation, every part with its evidence and how sure we are. Files because
the data is edited by people reading fitting guides and forum threads, reviewed
in a diff, and shipped with the code that reads it -- not typed into a form.

What the code decides, and what it leaves to the data:

* **The card is drawn from rows.** ``suggest`` returns parts exactly as the file
  states them. Nothing is inferred: a part is never "probably compatible"
  because a similar part is. The report chat may name a part only when it is in
  the rows it was handed (services/report_chat.py).
* **Direction, not fit.** A rider asks for "+30 mm reach, -20 mm stack". A part
  helps when its stated effect moves that way on that axis; ``covers`` says
  whether its range reaches the amount asked. Whether the rider's body wants
  that change is the report's job, not this module's.
* **Incompatible parts are answers too.** They are returned in their own list,
  with the reason, because "that one will not mount" is what saves the money.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "parts" / "frames"

STATUSES = ("manufacturer", "owners", "single_report", "incompatible")
STATUS_LABELS = {
    "manufacturer": "Confirmed by the maker",
    "owners": "Fitted by several owners",
    "single_report": "One owner's report",
    "incompatible": "Does not fit",
}
_RANK = {s: i for i, s in enumerate(STATUSES)}
CATEGORIES = (
    "adapter_plate", "riser", "spacer_kit", "extensions", "armrests", "wedge",
    "stem", "basebar", "full_cockpit", "other",
)
AXES = ("reach_mm", "stack_mm", "angle_deg")
PAD_REFERENCES = ("back_of_pad", "pad_center", "elbow_cup_center", "unspecified")
# A record older than this is flagged by scripts/check_parts_db.py.
STALE_DAYS = 183

_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,79}$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Loading and checking
# ---------------------------------------------------------------------------

def _range_ok(v: Any) -> bool:
    return v is None or (
        isinstance(v, list) and len(v) == 2
        and all(isinstance(x, (int, float)) for x in v) and v[0] <= v[1]
    )


def problems(frame: dict[str, Any], name: str = "") -> list[str]:
    """What is wrong with one frame file. Empty = shippable."""
    out: list[str] = []
    where = name or str(frame.get("id"))

    def bad(msg: str) -> None:
        out.append(f"{where}: {msg}")

    for key in ("id", "family", "generation", "years", "brakes", "cockpit", "parts", "gaps", "checked"):
        if key not in frame:
            bad(f"missing {key}")
    if out:
        return out
    if not _ID.match(str(frame["id"])):
        bad("id must be lowercase-hyphenated")
    if name and Path(name).stem != frame["id"]:
        bad(f"file name must be {frame['id']}.json")
    if not _DAY.match(str(frame["checked"])):
        bad("checked must be YYYY-MM-DD")
    for s in frame.get("stock_range") or []:
        if not _range_ok(s.get("pad_stack_mm")) or not _range_ok(s.get("pad_reach_mm")):
            bad(f"stock_range {s.get('size')}: ranges must be [low, high]")
        if s.get("pad_reference") not in PAD_REFERENCES:
            bad(f"stock_range {s.get('size')}: pad_reference must be one of {PAD_REFERENCES}")
    seen: set[str] = set()
    for p in frame["parts"]:
        pid = str(p.get("id"))
        if pid in seen:
            bad(f"part {pid} listed twice")
        seen.add(pid)
        if not _ID.match(pid):
            bad(f"part id {pid!r} must be lowercase-hyphenated")
        if not p.get("brand") or not p.get("model"):
            bad(f"part {pid}: brand and model are required")
        if p.get("status") not in STATUSES:
            bad(f"part {pid}: status must be one of {STATUSES}")
        if p.get("status") == "incompatible" and not p.get("reason_incompatible"):
            bad(f"part {pid}: incompatible needs reason_incompatible")
        if p.get("category") not in CATEGORIES:
            bad(f"part {pid}: category must be one of {CATEGORIES}")
        ev = p.get("evidence") or []
        if not ev or not all(str(e.get("url", "")).startswith("http") for e in ev):
            bad(f"part {pid}: every part needs evidence with a URL")
        eff = p.get("effect") or {}
        for ax in AXES:
            if not _range_ok(eff.get(ax)):
                bad(f"part {pid}: effect.{ax} must be [low, high] or null")
        price = p.get("price")
        if price is not None and not (
            isinstance(price.get("amount"), (int, float)) and price.get("currency")
            and _DAY.match(str(price.get("checked", "")))
        ):
            bad(f"part {pid}: price needs amount, currency and checked")
        if p.get("checked") and not _DAY.match(str(p["checked"])):
            bad(f"part {pid}: checked must be YYYY-MM-DD")
    return out


@lru_cache(maxsize=1)
def load_frames() -> dict[str, dict[str, Any]]:
    """Every frame file that passes ``problems``. A broken file is skipped
    (and fails the test suite), never half-served."""
    frames: dict[str, dict[str, Any]] = {}
    if not DATA_DIR.is_dir():
        return frames
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            frame = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not problems(frame, path.name):
            frames[frame["id"]] = frame
    return frames


def stale(frame: dict[str, Any], today: date | None = None) -> bool:
    today = today or date.today()
    try:
        return (today - date.fromisoformat(frame["checked"])).days > STALE_DAYS
    except (KeyError, ValueError):
        return True


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

def frame_summary(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f["id"], "family": f["family"], "generation": f["generation"],
        "years": f.get("years"), "brakes": f.get("brakes"), "identify": f.get("identify"),
        "sizes": [s.get("size") for s in f.get("stock_range") or [] if s.get("size")],
        "checked": f["checked"],
    }


def frames_for(family: str | None = None) -> list[dict[str, Any]]:
    return [
        frame_summary(f) for f in load_frames().values()
        if family is None or f["family"] == family
    ]


def _part_out(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": p["id"], "brand": p["brand"], "model": p["model"], "category": p["category"],
        "effect": p.get("effect") or {}, "status": p["status"],
        "status_label": STATUS_LABELS[p["status"]],
        "reason_incompatible": p.get("reason_incompatible"),
        "evidence": p.get("evidence") or [], "url": p.get("url"), "price": p.get("price"),
        "needs": p.get("needs"), "checked": p.get("checked"),
    }


def _moves(rng: list[float] | None, want: float) -> bool:
    """Does a part with this effect range move the axis the way asked?"""
    if not rng or not want:
        return False
    lo, hi = rng
    return hi > 0 if want > 0 else lo < 0


def _covers(rng: list[float] | None, want: float) -> bool:
    if not rng or not want:
        return False
    lo, hi = rng
    return hi >= want if want > 0 else lo <= want


def suggest(
    frame_id: str, *, size: str | None = None,
    reach_mm: float = 0.0, stack_mm: float = 0.0, angle_deg: float = 0.0,
) -> dict[str, Any] | None:
    """Stock adjustment, the parts that move the asked way, and what does not fit.

    None for an unknown frame. With nothing asked, every compatible part comes
    back under ``other``.
    """
    f = load_frames().get(frame_id)
    if f is None:
        return None
    want = {"reach_mm": reach_mm or 0.0, "stack_mm": stack_mm or 0.0, "angle_deg": angle_deg or 0.0}
    asked = {k: v for k, v in want.items() if v}
    cockpit = f.get("cockpit") or {}
    stock_size = next(
        (s for s in f.get("stock_range") or [] if size and str(s.get("size")).lower() == str(size).lower()),
        None,
    )

    helps, other, incompatible = [], [], []
    for p in f["parts"]:
        out = _part_out(p)
        if p["status"] == "incompatible":
            incompatible.append(out)
            continue
        eff = p.get("effect") or {}
        moves = [ax for ax, v in asked.items() if _moves(eff.get(ax), v)]
        if moves:
            out["moves"] = moves
            out["covers"] = [ax for ax in moves if _covers(eff.get(ax), asked[ax])]
            helps.append(out)
        else:
            other.append(out)
    key = lambda p: (-len(p.get("covers") or []), _RANK[p["status"]], p["brand"], p["model"])  # noqa: E731
    helps.sort(key=key)
    other.sort(key=lambda p: (_RANK[p["status"]], p["brand"], p["model"]))
    return {
        "frame": frame_summary(f),
        "asked": asked,
        "stock": {
            "stack_system": cockpit.get("stack_system"),
            "reach_system": cockpit.get("reach_system"),
            "angle_system": cockpit.get("angle_system"),
            "notes": cockpit.get("notes"),
            "size": stock_size,
        },
        "helps": helps,
        "other": other,
        "incompatible": incompatible,
        "gaps": f.get("gaps") or [],
        "stale": stale(f),
    }


def chat_block(frame_id: str | None) -> str | None:
    """The frame's parts, for the report chat: the only parts it may name."""
    if not frame_id:
        return None
    f = load_frames().get(frame_id)
    if f is None:
        return None
    lines = [
        f"### Parts on record for this frame ({f['generation']}, checked {f['checked']})",
        "",
        "Name a part ONLY from this list, with its status. Anything not listed is "
        "unknown for this frame: say so and suggest checking with the maker.",
        "",
    ]
    c = f.get("cockpit") or {}
    for label, key in (("Stock pad height", "stack_system"), ("Stock pad reach", "reach_system"),
                       ("Stock tilt", "angle_system")):
        if c.get(key):
            lines.append(f"- {label}: {c[key]}")
    for p in sorted(f["parts"], key=lambda p: _RANK[p["status"]]):
        eff = p.get("effect") or {}
        moves = ", ".join(
            f"{ax.split('_')[0]} {eff[ax][0]:+g}..{eff[ax][1]:+g} {'deg' if ax == 'angle_deg' else 'mm'}"
            for ax in AXES if eff.get(ax)
        )
        tail = f"; does NOT fit: {p['reason_incompatible']}" if p["status"] == "incompatible" else ""
        lines.append(
            f"- {p['brand']} {p['model']} ({p['category']}; {STATUS_LABELS[p['status']]})"
            f"{': ' + moves if moves else ''}{tail}"
        )
    return "\n".join(lines)
