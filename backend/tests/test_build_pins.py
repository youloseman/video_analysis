"""The image is built from pinned inputs: the pose model and every package.

The model was fetched from ".../latest/..." and only the direct dependencies
were pinned, so a rebuild with no commit could change every number the app
reports (a new model) or break serving (a new transitive, as starlette did on
2026-10-13). Both are now pinned, and CI installs the same set the image does.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (ROOT / "Dockerfile").read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
CONSTRAINTS = (ROOT / "backend" / "constraints.txt").read_text(encoding="utf-8")


def _pins(text: str) -> dict[str, str]:
    return {
        m.group(1).lower().replace("_", "-"): m.group(2)
        for m in re.finditer(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s#]+)", text, re.M)
    }


def test_the_pose_model_is_a_fixed_version_checked_by_hash():
    assert "/latest/" not in DOCKERFILE
    assert re.search(r"sha256sum -c", DOCKERFILE)
    assert re.search(r"[0-9a-f]{64}  models/pose_landmarker_heavy\.task", DOCKERFILE)


def test_the_image_and_ci_install_with_the_constraints():
    assert "-c constraints.txt" in DOCKERFILE
    assert "COPY backend/requirements.txt backend/constraints.txt" in DOCKERFILE
    assert "-c constraints.txt" in WORKFLOW


def test_every_direct_pin_agrees_with_the_constraints():
    """A bump in requirements.txt without regenerating constraints.txt would
    make pip refuse to resolve: catch it here, not in the Railway build."""
    direct = _pins((ROOT / "backend" / "requirements.txt").read_text(encoding="utf-8"))
    locked = _pins(CONSTRAINTS)
    missing = sorted(n for n in direct if n not in locked)
    differ = sorted(f"{n}: {direct[n]} vs {locked[n]}" for n in direct if n in locked and direct[n] != locked[n])
    assert missing == [] and differ == [], (missing, differ)
