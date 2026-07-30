"""Changelog — file-based release notes, shown in-app and on a public page.

One Markdown file per release under ``backend/content/changelog/``. Add a file
-> the entry appears in the app's "What's new" view, in ``/changelog.json`` and
on the server-rendered ``/changelog`` page. No database, no frontmatter.

The point is to close the feedback loop: athletes send feedback (see
``app/api/feedback.py``) and this is where they see that it went somewhere. A
changelog nobody notices does nothing, so the app puts an unread dot on the nav
until the newest entry has been read.

See :mod:`app.services.changelog.parser` for the file format.
"""

from app.services.changelog.parser import (
    Entry,
    get_entries,
    invalidate_cache,
    latest_id,
)

__all__ = ["Entry", "get_entries", "invalidate_cache", "latest_id"]
