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
cap of 2 that is around 900 MB.

And then this endpoint answered the question it was built to ask, with a result
that went the other way: production reports 7629 MB with 1.2% of it used.
Memory is nowhere near the constraint -- there is room for something like
twenty more concurrent analyses. What binds is CPU, which is why ``cpu_quota``
is here too: an analysis is CPU-bound from end to end (~95 s of one, measured),
so the cap is really a question about cores, and ``os.cpu_count()`` answers
about the HOST rather than about the slice this container may use.

The cap is still 2, deliberately. Raising it is an env var away
(``VA_MAX_CONCURRENT_ANALYSES``) and wants watching under real concurrent load,
which there has not been any of yet -- and an OOM or a stalled worker here is
not a slow request: it takes the single worker down and every in-flight
analysis with it.

Everything is best-effort and Linux-shaped. On a developer's Windows machine
every reader returns None and /health simply says nothing about memory, which
is the truth -- there is no container limit to report.
"""

from __future__ import annotations

import os
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


_V2_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
_V1_CPU_QUOTA = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
_V1_CPU_PERIOD = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")


def cpu_quota() -> float | None:
    """How many cores this container may actually use, if it is capped.

    The other half of the concurrency question, and the half that binds here.
    Memory turned out not to be the constraint at all -- the box reports 7.6 GB
    with 1.2% used -- while an analysis is CPU-bound from end to end, so what
    decides how many can run at once is how many cores there are to run them
    on. ``os.cpu_count()`` answers about the HOST, which on a shared runtime is
    a much larger and entirely fictional number; the cgroup quota is the real
    one.
    """
    try:
        raw = _V2_CPU_MAX.read_text().split()
        if len(raw) == 2 and raw[0] != "max":
            return int(raw[0]) / int(raw[1])
    except (OSError, ValueError, ZeroDivisionError):
        pass
    try:
        quota = int(_V1_CPU_QUOTA.read_text().strip())
        period = int(_V1_CPU_PERIOD.read_text().strip())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError, ZeroDivisionError):
        pass
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
    cores = cpu_quota()
    if cores:
        out["cpu_cores"] = round(cores, 2)
    # The host's count, for contrast: on a shared runtime it is much larger
    # than the quota, and reading it as available parallelism is how a
    # CPU-bound cap gets set to a number the container cannot honour.
    host = os.cpu_count()
    if host:
        out["host_cpus"] = host
    return out
