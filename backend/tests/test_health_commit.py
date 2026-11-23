"""/health names the commit it was built from, and the deploy waits for it.

Railway keeps the old container serving when a new one fails its health check,
so "/health returned 200" after a deploy proved nothing about the new code.
CI writes app/BUILD_SHA before ``railway up``; the smoke test waits until
/health reports that commit.
"""

from __future__ import annotations

from pathlib import Path

from app import main

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "deploy.yml"


def test_the_file_written_by_ci_is_the_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    (tmp_path / "BUILD_SHA").write_text("4e27187c9ad0\n", encoding="ascii")
    assert main.build_commit(tmp_path) == "4e27187c9ad0"


def test_without_the_file_it_is_dev_or_railways_own(tmp_path, monkeypatch):
    monkeypatch.delenv("RAILWAY_GIT_COMMIT_SHA", raising=False)
    assert main.build_commit(tmp_path) == "dev"
    monkeypatch.setenv("RAILWAY_GIT_COMMIT_SHA", "abc123")
    assert main.build_commit(tmp_path) == "abc123"


def test_health_reports_it():
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        assert client.get("/health").json()["commit"] == main.BUILD_COMMIT


def test_the_deploy_writes_the_file_and_waits_for_that_commit():
    wf = WORKFLOW.read_text(encoding="utf-8")
    write = wf.index('echo "$GITHUB_SHA" > backend/app/BUILD_SHA')
    assert write < wf.index("railway up --ci")
    smoke = wf[wf.index("- name: Smoke-test /health"):]
    assert '"commit"' in smoke and '"$GITHUB_SHA"' in smoke
