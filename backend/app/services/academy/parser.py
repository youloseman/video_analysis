"""Parse Academy Markdown files into cached :class:`Article` objects.

Design goals
------------
* **File = article.** ``backend/content/academy/<slug>.md`` — YAML-ish
  frontmatter between ``---`` fences, then a Markdown body.
* **No external deps.** The project deliberately keeps ``requirements.txt``
  minimal (see ``core/config.py``), so this module ships its own tiny
  frontmatter reader and Markdown->HTML converter instead of pulling in
  ``python-frontmatter`` / ``markdown``. The converter supports exactly the
  subset the writing guide uses: ATX headings, paragraphs, ordered/unordered
  lists, GitHub-style tables, blockquotes, ``**bold**`` / ``*italic*`` /
  ``` `code` ``` and links.
* **In-memory cache.** First access parses every file; subsequent calls are
  served from memory. Call :func:`invalidate_cache` (or restart the process /
  redeploy) to pick up edits.

Frontmatter fields (see ``content/academy/running-cadence.md`` for a full
example)::

    slug         str   required  URL id: /academy/<slug> (defaults to filename)
    title        str   required  <h1> + <title>
    description  str   required  meta description / OpenGraph / card summary
    category     str   required  one of _CATEGORIES
    sport        str   run|bike  accent colour + schema (default: run)
    read_time    int   opt       "X min read" (default: estimated from words)
    published    str   opt       ISO date (YYYY-MM-DD) for schema/sitemap
    featured     bool  opt       render large on the hub
    sources      list  opt       [{author, year, title, finding}] bibliography
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# backend/content/academy/  (this file: backend/app/services/academy/parser.py)
CONTENT_DIR = Path(__file__).resolve().parents[3] / "content" / "academy"

# category key -> (human label, matching sport for the accent colour).
# The sport in a category is only a *default* tint; a file's own ``sport``
# frontmatter always wins.
_CATEGORIES: dict[str, dict[str, str]] = {
    "running-form": {"label": "Running Form", "sport": "run"},
    "running-economy": {"label": "Running Economy", "sport": "run"},
    "bike-position": {"label": "Bike Position", "sport": "bike"},
    "bike-aero": {"label": "Bike Aerodynamics", "sport": "bike"},
}

_WORDS_PER_MINUTE = 200  # for the read-time estimate fallback


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArticleMeta:
    """Lightweight metadata for hub cards / listings (no body HTML)."""

    slug: str
    title: str
    description: str
    category: str
    category_label: str
    sport: str  # run | bike
    read_time: int
    published: str | None
    featured: bool


@dataclass(frozen=True)
class Article:
    """A fully parsed article: metadata + rendered HTML body + sources."""

    meta: ArticleMeta
    body_html: str
    sources: list[dict[str, Any]] = field(default_factory=list)


# In-memory cache. None => not loaded yet.
_cache: dict[str, Article] | None = None


# ---------------------------------------------------------------------------
# Frontmatter
# ---------------------------------------------------------------------------
def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a ``---``-fenced frontmatter block from the Markdown body.

    Returns ``(metadata, body)``. If there is no frontmatter, metadata is empty
    and the whole text is the body.
    """
    text = text.lstrip("﻿")  # strip BOM if present
    if not text.startswith("---"):
        return {}, text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.DOTALL)
    if not m:
        return {}, text
    return _parse_yaml_ish(m.group(1)), m.group(2)


def _parse_yaml_ish(block: str) -> dict[str, Any]:
    """Parse the small YAML subset we allow in frontmatter.

    Supports ``key: value`` scalars, simple ``- item`` string lists, and
    lists of one-line ``key: value`` maps (used by ``sources``). Deliberately
    tiny — not a real YAML parser.
    """
    data: dict[str, Any] = {}
    lines = block.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        # top-level "key:" or "key: value"
        m = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", raw)
        if not m:
            i += 1
            continue
        key, val = m.group(1), m.group(2).strip()
        if val:  # inline scalar
            data[key] = _coerce_scalar(val)
            i += 1
            continue
        # block value: gather indented "- ..." lines
        items: list[Any] = []
        i += 1
        while i < n and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
            item_line = lines[i]
            i += 1
            stripped = item_line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                items.append(_start_list_item(stripped[2:], lines, i, n))
                # _start_list_item may consume continuation lines; recompute i
                i = _list_item_end
        data[key] = items
    return data


# _start_list_item communicates how many continuation lines it consumed via a
# module-level cursor to keep the caller loop simple.
_list_item_end = 0


def _start_list_item(first: str, lines: list[str], idx: int, n: int) -> Any:
    """Parse one ``- ...`` list item. Either a scalar string, or a map whose
    first ``key: value`` is inline after the dash and whose remaining
    ``key: value`` pairs are more-indented lines that follow.
    """
    global _list_item_end
    m = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", first)
    if not m:  # plain string item
        _list_item_end = idx
        return _coerce_scalar(first)
    obj: dict[str, Any] = {m.group(1): _coerce_scalar(m.group(2).strip())}
    j = idx
    # Continuation lines for this map are indented deeper than the "- ".
    while j < n:
        cont = lines[j]
        if not cont.strip():
            j += 1
            continue
        # A new "- " item or a dedent ends this map.
        if cont.lstrip().startswith("- ") or not cont.startswith(("    ", "\t\t", "\t ", "  ")):
            break
        cm = re.match(r"^\s+([A-Za-z0-9_]+):\s*(.*)$", cont)
        if not cm:
            break
        obj[cm.group(1)] = _coerce_scalar(cm.group(2).strip())
        j += 1
    _list_item_end = j
    return obj


def _coerce_scalar(val: str) -> Any:
    """Strip quotes and coerce obvious bools/ints; otherwise return the str."""
    if (val.startswith('"') and val.endswith('"')) or (
        val.startswith("'") and val.endswith("'")
    ):
        return val[1:-1]
    low = val.lower()
    if low in ("true", "false"):
        return low == "true"
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    return val


# ---------------------------------------------------------------------------
# Markdown -> HTML (small, dependency-free)
# ---------------------------------------------------------------------------
def markdown_to_html(md: str) -> str:
    """Convert the supported Markdown subset to HTML.

    Block grammar handled line-by-line: ATX headings (``##``/``###``), GitHub
    tables, ``>`` blockquotes, ``-``/``*``/``1.`` lists (one level), fenced
    ``` code blocks, and paragraphs. Inline formatting runs on every non-code
    text run via :func:`_inline`.
    """
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:  # blank line -> paragraph break
            i += 1
            continue

        # fenced code block ``` ... ```
        if stripped.startswith("```"):
            i += 1
            code: list[str] = []
            while i < n and not lines[i].strip().startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            out.append(
                "<pre><code>" + html.escape("\n".join(code)) + "</code></pre>"
            )
            continue

        # ATX heading
        h = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if h:
            level = len(h.group(1))
            out.append(f"<h{level}>{_inline(h.group(2).strip())}</h{level}>")
            i += 1
            continue

        # table: a header row followed by a |---|---| separator
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
            block, i = _consume_table(lines, i, n)
            out.append(block)
            continue

        # blockquote
        if stripped.startswith(">"):
            quote: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append("<blockquote>" + _inline(" ".join(quote)) + "</blockquote>")
            continue

        # unordered list
        if re.match(r"^[-*]\s+", stripped):
            items, i = _consume_list(lines, i, n, ordered=False)
            out.append("<ul>" + items + "</ul>")
            continue

        # ordered list
        if re.match(r"^\d+\.\s+", stripped):
            items, i = _consume_list(lines, i, n, ordered=True)
            out.append("<ol>" + items + "</ol>")
            continue

        # paragraph: gather consecutive non-blank, non-block lines
        para: list[str] = []
        while i < n and lines[i].strip() and not _is_block_start(lines, i, n):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + _inline(" ".join(para)) + "</p>")

    return "\n".join(out)


def _is_block_start(lines: list[str], i: int, n: int) -> bool:
    """True if line ``i`` begins a non-paragraph block (so a paragraph stops)."""
    s = lines[i].strip()
    if s.startswith(("#", ">", "```")):
        return True
    if re.match(r"^[-*]\s+", s) or re.match(r"^\d+\.\s+", s):
        return True
    if "|" in lines[i] and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]):
        return True
    return False


def _consume_list(lines: list[str], i: int, n: int, ordered: bool) -> tuple[str, int]:
    pat = r"^\d+\.\s+(.*)$" if ordered else r"^[-*]\s+(.*)$"
    items: list[str] = []
    while i < n:
        m = re.match(pat, lines[i].strip())
        if not m:
            break
        items.append("<li>" + _inline(m.group(1).strip()) + "</li>")
        i += 1
    return "".join(items), i


def _consume_table(lines: list[str], i: int, n: int) -> tuple[str, int]:
    def cells(row: str) -> list[str]:
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [c.strip() for c in row.split("|")]

    header = cells(lines[i])
    i += 2  # skip header + separator
    body_rows: list[str] = []
    while i < n and "|" in lines[i] and lines[i].strip():
        row_cells = cells(lines[i])
        tds = "".join(f"<td>{_inline(c)}</td>" for c in row_cells)
        body_rows.append(f"<tr>{tds}</tr>")
        i += 1
    ths = "".join(f"<th>{_inline(c)}</th>" for c in header)
    return (
        f"<table><thead><tr>{ths}</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>",
        i,
    )


def _inline(text: str) -> str:
    """Escape HTML, then apply inline Markdown: code, bold, italic, links."""
    # Protect inline `code` spans first so their content isn't escaped twice
    # or mangled by the bold/italic passes.
    code_spans: list[str] = []

    def _stash_code(m: re.Match) -> str:
        code_spans.append(html.escape(m.group(1)))
        return f"\x00{len(code_spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash_code, text)
    text = html.escape(text)

    # links [text](url) — url is attribute-escaped by html.escape already
    text = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    # bold then italic (bold first so ** isn't eaten by the * rule)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

    # restore code spans
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{code_spans[int(m.group(1))]}</code>", text)
    return text


# ---------------------------------------------------------------------------
# Loading / cache
# ---------------------------------------------------------------------------
def _estimate_read_time(md_body: str) -> int:
    words = len(re.findall(r"\w+", md_body))
    return max(1, round(words / _WORDS_PER_MINUTE))


def _parse_file(path: Path) -> Article | None:
    """Parse one ``.md`` file into an :class:`Article`; None if unusable.

    Drafts are skipped so unfinished articles never reach the public site:
    a ``*.DRAFT.md`` filename, or frontmatter ``draft: true`` /
    ``published: false``, is treated as not-yet-published.
    """
    if path.name.lower().endswith(".draft.md"):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    fm, body = _split_frontmatter(text)
    if fm.get("draft") is True or fm.get("published") is False:
        return None
    slug = str(fm.get("slug") or path.stem).strip()
    title = str(fm.get("title") or "").strip()
    if not slug or not title:
        return None  # required fields missing -> skip

    category = str(fm.get("category") or "running-form").strip()
    cat_meta = _CATEGORIES.get(category, {"label": category.replace("-", " ").title(), "sport": "run"})
    sport = str(fm.get("sport") or cat_meta["sport"]).strip()
    if sport not in ("run", "bike"):
        sport = "run"

    read_time = fm.get("read_time")
    if not isinstance(read_time, int) or read_time <= 0:
        read_time = _estimate_read_time(body)

    meta = ArticleMeta(
        slug=slug,
        title=title,
        description=str(fm.get("description") or "").strip(),
        category=category,
        category_label=cat_meta["label"],
        sport=sport,
        read_time=read_time,
        published=str(fm["published"]).strip() if fm.get("published") else None,
        featured=bool(fm.get("featured", False)),
    )
    sources = fm.get("sources") if isinstance(fm.get("sources"), list) else []
    return Article(meta=meta, body_html=markdown_to_html(body), sources=sources)


def _load() -> dict[str, Article]:
    articles: dict[str, Article] = {}
    if not CONTENT_DIR.exists():
        return articles
    for path in sorted(CONTENT_DIR.glob("*.md")):
        article = _parse_file(path)
        if article is not None:
            articles[article.meta.slug] = article
    return articles


def _ensure_loaded() -> dict[str, Article]:
    global _cache
    if _cache is None:
        _cache = _load()
    return _cache


def get_all_articles() -> list[ArticleMeta]:
    """All article metadata, featured first then newest ``published`` first."""
    metas = [a.meta for a in _ensure_loaded().values()]
    metas.sort(key=lambda m: (not m.featured, m.published or "", m.title), reverse=False)
    # published desc within the non-featured group: re-sort by a composite key
    metas.sort(key=lambda m: (0 if m.featured else 1, _neg_date(m.published), m.title))
    return metas


def _neg_date(d: str | None) -> str:
    """Sort key that puts newer ISO dates first (undated go last)."""
    if not d:
        return "0000-00-00"
    # invert each char so lexical ascending == date descending
    return "".join(chr(255 - ord(c)) for c in d)


def get_article(slug: str) -> Article | None:
    return _ensure_loaded().get(slug)


def get_categories() -> dict[str, dict[str, Any]]:
    """Category keys -> {label, sport, count} for the hub filter chips."""
    counts: dict[str, int] = {}
    for meta in (a.meta for a in _ensure_loaded().values()):
        counts[meta.category] = counts.get(meta.category, 0) + 1
    result: dict[str, dict[str, Any]] = {}
    for key, info in _CATEGORIES.items():
        if counts.get(key):
            result[key] = {**info, "count": counts[key]}
    return result


def invalidate_cache() -> None:
    """Drop the in-memory cache (tests / hot-reload)."""
    global _cache
    _cache = None
