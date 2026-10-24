"""Analysis failures that outlive the process that had them.

Two gaps made crashes uninvestigable. Railway keeps a container's log only
until the next deploy, and several sessions push to main a day -- a crash from
yesterday was simply gone by the time anyone looked (the TypeError three
athletes hit on 23.10 left no trace by the 24th). And the handlers logged
``str(e)`` alone, which names an error but not where it happened.

This records both the message and the frames that raised it to PostHog, which
keeps events for 12 months and is already read by
``scripts/analytics_report.py`` -- and it writes the same frames to the log for
whoever is watching live.

The records are deliberately ANONYMOUS: no account id, no person profile. A
crash report needs *where* it happened, not *who* it happened to, and a
server-side event cannot see the per-browser analytics opt-out the privacy
policy promises. The person stays on the browser's own ``analysis_failed``
event, which does honour it. Nothing here carries media, a filename, or text
the athlete typed -- the policy's line for product events.

It also owns the sentence the athlete sees, so an exception's raw text never
reaches them again: "'>' not supported between instances of 'NoneType' and
'int'" was the whole explanation three people got for a failed analysis.
"""

from __future__ import annotations

import io
import re
import threading
import traceback
from typing import Any

import structlog

from app.services import analytics

logger = structlog.get_logger()

# A fixed non-person id: with person processing off, PostHog files the events
# without creating or updating anybody.
DIAGNOSTICS_ID = "flapp-server-diagnostics"

GENERIC = (
    "Something went wrong on our side while analysing this clip. Please try "
    "again — if it happens twice, try a different clip."
)
UNREADABLE = (
    "We couldn't read this video file. Try a different clip, or export it as "
    "MP4."
)

# Absolute paths in an exception message name the server's filesystem layout
# and job directories; none of it helps diagnosis and none of it belongs in a
# third-party analytics store.
_PATH = re.compile(r"(?:[A-Za-z]:)?(?:[/\\][\w.\-]+){2,}")


def user_message(exc: BaseException) -> str:
    """The sentence an athlete reads when an analysis dies unexpectedly."""
    if isinstance(exc, RuntimeError) and str(exc).startswith("Could not open video"):
        return UNREADABLE
    return GENERIC


def _clean(text: str, limit: int = 300) -> str:
    return _PATH.sub("<path>", text)[:limit]


def _frames(exc: BaseException, limit: int = 8) -> list[str]:
    """The innermost frames, as ``app/...py:line in func``.

    The innermost end is kept because that is where the comparison with None
    happened; the outer frames are the same handful of request plumbing every
    time.
    """
    out = []
    for f in traceback.extract_tb(exc.__traceback__)[-limit:]:
        path = f.filename.replace("\\", "/")
        cut = path.rfind("/app/")
        short = path[cut + 1:] if cut >= 0 else path.rsplit("/", 1)[-1]
        out.append(f"{short}:{f.lineno} in {f.name}")
    return out


def image_meta(data: bytes) -> dict[str, Any]:
    """Shape of an uploaded photo -- never its pixels.

    Enough to tell a 400-pixel screenshot from a 48-megapixel phone shot, a
    sideways EXIF orientation from an upright one, or HEIC from JPEG, which is
    what "no pose detected" has to be told apart by.
    """
    meta: dict[str, Any] = {"img_mb": round(len(data) / 1048576, 2)}
    try:
        from PIL import Image
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except Exception:  # noqa: BLE001 -- optional; JPEG/PNG still read
            pass
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            meta.update({
                "img_w": w,
                "img_h": h,
                "img_mp": round(w * h / 1e6, 1),
                "img_format": im.format,
                # 1 = upright; 6/8 = the sensor was turned, and only the EXIF
                # tag says so. A pose model handed the raw pixels sees a body
                # lying on its side.
                "img_orientation": (im.getexif() or {}).get(274),
            })
    except Exception:  # noqa: BLE001 -- an unreadable header is itself a finding
        meta["img_unreadable"] = True
    return meta


def _send(event: str, props: dict[str, Any]) -> None:
    props = {k: v for k, v in props.items() if v is not None}
    props["$process_person_profile"] = False
    # A daemon thread: these fire from analysis worker threads, and a slow
    # analytics endpoint must not hold an analysis slot open.
    threading.Thread(
        target=analytics.capture, args=(DIAGNOSTICS_ID, event, props), daemon=True,
    ).start()


def report_crash(where: str, exc: BaseException, **context: Any) -> None:
    """An exception nobody planned for: log the frames, keep a copy."""
    frames = _frames(exc)
    err = _clean(str(exc))
    logger.error(
        "ANALYSIS_CRASH", where=where, err_type=type(exc).__name__, err=err,
        frames=frames, **{k: v for k, v in context.items() if v is not None},
    )
    _send("analysis_error", {
        "where": where, "err_type": type(exc).__name__, "err": err,
        "frames": frames, **context,
    })


def report_rejected(where: str, reason: str, **context: Any) -> None:
    """A refusal we meant to give (no pose found): keep what it was refused for."""
    _send("analysis_rejected", {"where": where, "reason": _clean(reason, 200), **context})
