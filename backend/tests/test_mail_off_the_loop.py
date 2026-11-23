"""No async handler sends mail on the event loop.

``notify.send_email`` is a blocking HTTPS (or SMTP) round trip with a timeout
of several seconds. Called straight from an ``async def`` it stalls every
request on the single worker for that long -- a coach writing a note, or the
plan-lapse pass mailing one user after another, froze the app for everyone.
Inside async code it is handed to the threadpool (passed, not called), so a
direct call there is the bug.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"
# Sync helpers whose whole job is to call send_email.
SYNC_MAILERS = {"send_email", "_mail_note", "_mail", "notify_sale", "storage_alert"}


def _name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def test_async_code_never_calls_a_blocking_mailer():
    bad = []
    for path in APP.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and _name(node.func) in SYNC_MAILERS:
                    bad.append(f"{path.relative_to(APP)}:{node.lineno} {fn.name}")
    assert bad == [], bad
