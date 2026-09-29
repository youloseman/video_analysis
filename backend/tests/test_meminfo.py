"""The container's memory budget, reported so the concurrency cap can be set
on evidence.

One analysis peaks at ~600 MB and leaves ~287 MB of loaded pose model resident.
``max_concurrent_analyses`` is therefore a memory budget, and it was set to 2
without anybody being able to see the budget from inside. An OOM on a
single-worker deploy does not fail one request -- it takes the worker down and
every clip being analysed with it.

The readers are deliberately incapable of raising: /health must not start
failing because a kernel spelled a cgroup file differently, so every one of
them answers None rather than throwing, and the endpoint simply says nothing
about memory when there is nothing to say.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core import meminfo

MB = 1024 * 1024


@pytest.fixture
def cg(tmp_path, monkeypatch):
    """A fake cgroup v2 directory."""
    def write(limit: str | None = None, usage: str | None = None):
        if limit is not None:
            (tmp_path / "memory.max").write_text(limit)
        if usage is not None:
            (tmp_path / "memory.current").write_text(usage)
        monkeypatch.setattr(meminfo, "_V2_LIMIT", tmp_path / "memory.max")
        monkeypatch.setattr(meminfo, "_V2_USAGE", tmp_path / "memory.current")
        monkeypatch.setattr(meminfo, "_V1_LIMIT", tmp_path / "nope.v1")
        monkeypatch.setattr(meminfo, "_V1_USAGE", tmp_path / "nope.v1u")
    return write


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def test_a_real_limit_is_read(cg):
    cg(limit=str(2048 * MB), usage=str(900 * MB))
    assert meminfo.container_limit_bytes() == 2048 * MB
    assert meminfo.container_usage_bytes() == 900 * MB


def test_unlimited_reads_as_no_limit(cg):
    """cgroup v2 writes the word. A limit of "max" is not a number and must not
    become one -- reporting 0% used of "max" would be worse than silence."""
    cg(limit="max", usage=str(900 * MB))
    assert meminfo.container_limit_bytes() is None


def test_a_v1_style_enormous_limit_is_not_a_budget(cg, tmp_path, monkeypatch):
    """cgroup v1 spells "unlimited" as a number near 2^63, which is the host's
    address space showing through rather than anything anybody allocated."""
    cg(limit=str(1 << 62), usage=str(900 * MB))
    assert meminfo.container_limit_bytes() is None


def test_a_missing_cgroup_is_not_an_error(cg, tmp_path, monkeypatch):
    """A developer's machine. /health has to keep answering."""
    monkeypatch.setattr(meminfo, "_V2_LIMIT", tmp_path / "absent")
    monkeypatch.setattr(meminfo, "_V2_USAGE", tmp_path / "absent")
    monkeypatch.setattr(meminfo, "_V1_LIMIT", tmp_path / "absent")
    monkeypatch.setattr(meminfo, "_V1_USAGE", tmp_path / "absent")
    assert meminfo.container_limit_bytes() is None
    assert meminfo.container_usage_bytes() is None


def test_garbage_is_not_an_error(cg):
    cg(limit="not-a-number", usage="")
    assert meminfo.container_limit_bytes() is None
    assert meminfo.container_usage_bytes() is None


def test_rss_is_read_from_proc_or_absent(monkeypatch, tmp_path):
    status = tmp_path / "status"
    status.write_text("Name:\tpython\nVmRSS:\t  602400 kB\nThreads:\t9\n")
    monkeypatch.setattr(meminfo, "Path", lambda p: status if "status" in str(p) else Path(p))
    assert meminfo.process_rss_bytes() == 602400 * 1024


# --------------------------------------------------------------------------
# what /health shows
# --------------------------------------------------------------------------

def test_the_report_carries_the_budget_and_the_headroom(cg):
    cg(limit=str(2048 * MB), usage=str(900 * MB))
    out = meminfo.memory_health()
    assert out["limit_mb"] == 2048
    assert out["used_mb"] == 900
    assert out["pct"] == pytest.approx(43.9, abs=0.2)
    # (2048 - 900) / 315 = 3 more analyses would fit.
    assert out["headroom_analyses"] == 3


def test_a_nearly_full_container_reports_no_headroom(cg):
    """The number that should stop somebody raising the cap."""
    cg(limit=str(1024 * MB), usage=str(950 * MB))
    out = meminfo.memory_health()
    assert out["headroom_analyses"] == 0
    assert out["pct"] > 90


def test_nothing_knowable_is_an_empty_report(cg, tmp_path, monkeypatch):
    """Keys absent, not null: a row of nulls on a laptop reads like a broken
    container rather than like a machine with no container at all."""
    for name in ("_V2_LIMIT", "_V2_USAGE", "_V1_LIMIT", "_V1_USAGE"):
        monkeypatch.setattr(meminfo, name, tmp_path / "absent")
    monkeypatch.setattr(meminfo, "process_rss_bytes", lambda: None)
    assert meminfo.memory_health() == {}


def test_health_still_answers_without_a_cgroup():
    """The endpoint is what the deploy is gated on (railway.json). It must not
    acquire a way to fail."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert isinstance(r.json()["memory"], dict)
