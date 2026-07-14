"""Academy — file-based, server-rendered educational articles.

An article is a single Markdown file with YAML-ish frontmatter living under
``backend/content/academy/``. The backend parses it into an :class:`Article`
(metadata + rendered HTML body), caches the result in memory, and renders each
one into a full, SEO-friendly HTML page (own URL, ``<title>``/meta/OpenGraph +
``Article`` JSON-LD). No database, no CMS: add a file -> the article appears.

See :mod:`app.services.academy.parser` for the file format and
:mod:`app.services.academy.renderer` for the HTML output.
"""

from app.services.academy.parser import (
    Article,
    ArticleMeta,
    get_all_articles,
    get_article,
    invalidate_cache,
)

__all__ = [
    "Article",
    "ArticleMeta",
    "get_all_articles",
    "get_article",
    "invalidate_cache",
]
