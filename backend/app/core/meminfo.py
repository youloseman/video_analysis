"""How much memory this container has, and how much of it is gone.

``max_concurrent_analyses`` is the valve that stops Starlette's threadpool
starting one MediaPipe run per thread and OOMing the box. It is set to 2, and
until now that was a guess nobody could check: the cost of an analysis was
never written down and the container's ceiling was never visible from inside.

Measured on a real clip (720p, 11 s, pose model "heavy"):

    process after import, no analysis yet     42 MB
    resident once the model is loaded        287 MB
    peak during one analysis                 602 MB

So one analysis is ~315 MB of working set on top of a ~287 MB floor, and a
worker running N of them wants roughly ``287 + N x 315`` MB. At the current
cap of 2 that is around 900 MB before anything else the process is doing --
which is why the answer to "can we raise it" is "look at this endpoint on the
real container first". An OOM here is not a slow request: it takes the single
worker down and every in-flight analysis with it.

Everything is best-effort and Linux-shaped. On a developer's Windows machine
every reader returns None and /health simply says nothing about memory, which
is the truth -- there is no container limit to report.
"""

from __future__ import annotations

from pathlib import Path

# cgroup v2 (what Railway, Fly and modern Docker use), then v1 as a fallback.
# "max" in v2 means unlimited, and v1 spells the same thing as a number so
# large it is obviously not a real allocation.
_V2_LIMIT = Path("/sys/fs/cgroup/memory.max")
_V2_USAGE = Path("/sys/fs/cgroup/memory.current")
_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
_V1_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")

# Past this, a reported "limit" is the host's whole address space showing
# through an unlimited cgroup, not a budget anybody set.
_UNLIMITED_ABOVE = 1 << 50  # 1 PiB


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if 0 < value < _UNLIMITED_ABOVE else None


def container_limit_bytes() -> int | None:
    """The memory ceiling this process is actually held to, if there is one."""
    return _read_int(_V2_LIMIT) or _read_int(_V1_LIMIT)


def container_usage_bytes() -> int | None:
    """How much of that ceiling is currently in use, cgroup-wide."""
    return _read_int(_V2_USAGE) or _read_int(_V1_USAGE)


def process_rss_bytes() -> int | None:
    """This process's resident set, from /proc. No psutil dependency: the one
    number we want is one line of a file the kernel already writes."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                # "VmRSS:    123456 kB"
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        return None
    return None


def memory_health() -> dict[str, object]:
    """What /health reports. Keys are absent rather than null when unknowable,
    so a developer's machine does not print a row of nulls that look like a
    broken container."""
    mb = 1024 * 1024
    out: dict[str, object] = {}
    limit = container_limit_bytes()
    usage = container_usage_bytes()
    rss = process_rss_bytes()
    if limit:
        out["limit_mb"] = round(limit / mb)
    if usage:
        out["used_mb"] = round(usage / mb)
    if rss:
        out["rss_mb"] = round(rss / mb)
    if limit and usage:
        out["pct"] = round(usage / limit * 100, 1)
        # What one more concurrent analysis would need. The figure is measured
        # (see the module docstring) and deliberately stated rather than
        # derived from anything at runtime: it is a property of the pose model,
        # not of this request.
        out["headroom_analyses"] = max(0, int((limit - usage) / (315 * mb)))
    return out
