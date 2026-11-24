"""Parse changelog Markdown files into cached :class:`Entry` objects.

File format:

    backend/content/changelog/YYYY-MM-DD-some-slug.md

    ---
    audience: bike, run
    summary: One line the What's new list shows under the title.
    ---
    # What the athlete gets out of this release

    Body markdown: a short paragraph, then bullets if useful.

* **Date comes from the filename**, which also makes the directory sort
  chronologically in any file browser.
* **Title is the first ``# `` heading**, body is everything after it.
* **The frontmatter lines are optional.** ``audience`` is who it is for
  (``everyone``, ``bike``, ``run``, ``swim``, ``coaches``; default everyone),
  and it drives the list's filter chips. ``summary`` is the one line under
  the title in the app; without it the first sentence of the body stands in.
  ``asked: yes`` marks a release that riders asked for ("Suggest a feature"):
  the note then wears an "Asked for by riders" tag. Nothing else is
  configurable.

Markdown conversion is borrowed from the Academy parser rather than
reimplemented, so both surfaces support exactly the same subset.

One gotcha inherited from that converter: **a list item must sit on one physical
line.** It has no lazy continuation, so a soft-wrapped bullet is parsed as a new
paragraph and visibly breaks the list. Wrap prose freely; never wrap a bullet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.services.academy.parser import _split_frontmatter, markdown_to_html

# backend/content/changelog/  (this file: backend/app/services/changelog/parser.py)
CONTENT_DIR = Path(__file__).resolve().parents[3] / "content" / "changelog"

_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.M)

AUDIENCES = ("everyone", "bike", "run", "swim", "coaches")
_SENTENCE_RE = re.compile(r"^(.+?[.!?])(?:\s|$)", re.S)
_MD_MARK_RE = re.compile(r"[*_`]|\[([^\]]*)\]\([^)]*\)")

_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


@dataclass(frozen=True)
class Entry:
    id: str            # filename stem, e.g. "2026-07-30-result-feedback"
    date: str          # ISO date from the filename, e.g. "2026-07-30"
    title: str
    body_html: str
    summary: str = ""
    audience: tuple[str, ...] = ("everyone",)
    asked: bool = False

    @property
    def date_label(self) -> str:
        """'30 July 2026' -- matches the date style used in Privacy/Terms."""
        try:
            y, m, d = (int(p) for p in self.date.split("-"))
            return f"{d} {_MONTHS[m - 1]} {y}"
        except (ValueError, IndexError):
            return self.date


def _parse_file(path: Path) -> Entry | None:
    m = _FILENAME_RE.match(path.stem)
    if m is None:
        return None
    meta, text = _split_frontmatter(path.read_text(encoding="utf-8"))
    h1 = _H1_RE.search(text)
    if h1 is None:
        return None
    title = h1.group(1).strip()
    body = text[h1.end():].lstrip("\n")
    aud = tuple(a.strip().lower() for a in str(meta.get("audience") or "everyone").split(",") if a.strip())
    aud = tuple(a for a in aud if a in AUDIENCES) or ("everyone",)
    return Entry(
        id=path.stem, date=m.group(1), title=title,
        body_html=markdown_to_html(body),
        summary=str(meta.get("summary") or "").strip() or _first_sentence(body),
        audience=aud,
        asked=str(meta.get("asked") or "").strip().lower() in ("yes", "true", "1"),
    )


def _first_sentence(body: str) -> str:
    """The stand-in summary: the body's first sentence, Markdown marks off."""
    para = next((b for b in body.split("\n\n") if b.strip() and not b.lstrip().startswith(("-", "#", "|"))), "")
    plain = " ".join(_MD_MARK_RE.sub(lambda m: m.group(1) or "", para).split())
    m = _SENTENCE_RE.match(plain)
    return (m.group(1) if m else plain)[:160]


_cache: list[Entry] | None = None


def get_entries() -> list[Entry]:
    """All entries, newest first. Parsed once, then served from memory (a
    redeploy or :func:`invalidate_cache` picks up edits)."""
    global _cache
    if _cache is None:
        entries = []
        if CONTENT_DIR.is_dir():
            for path in sorted(CONTENT_DIR.glob("*.md")):
                entry = _parse_file(path)
                if entry is not None:
                    entries.append(entry)
        # Filename dates sort lexicographically, so reversing is chronological.
        _cache = list(reversed(entries))
    return _cache


def latest_id() -> str | None:
    """Id of the newest entry -- what the app compares against to decide
    whether to show the unread dot."""
    entries = get_entries()
    return entries[0].id if entries else None


def invalidate_cache() -> None:
    global _cache
    _cache = None
