"""Production refuses to start fail-open.

``JWT_SECRET`` and ``DATABASE_URL`` both have defaults so a laptop runs with no
setup. On Railway either default is silent and dangerous: a session signed with
the public string in config.py is a session anyone can mint (admin included),
and a SQLite file inside the container loses accounts and paid orders on the
next deploy while /health stays green. lifespan raises instead; Railway keeps
the previous container, so the cost is a blocked deploy, not an outage.
"""

from __future__ import annotations

import dataclasses

import pytest

from app.core.config import Settings, production_config_errors

GOOD = dataclasses.replace(
    Settings(), jwt_secret="a-real-secret", database_url="postgresql+asyncpg://u:p@h/db",
)
RAILWAY = {"RAILWAY_ENVIRONMENT": "production"}


def test_a_configured_production_starts():
    assert production_config_errors(RAILWAY, GOOD) == []


def test_the_default_jwt_secret_is_refused_on_railway():
    s = dataclasses.replace(GOOD, jwt_secret=Settings.jwt_secret)
    errors = production_config_errors(RAILWAY, s)
    assert len(errors) == 1 and "JWT_SECRET" in errors[0]


def test_a_sqlite_database_is_refused_on_railway():
    s = dataclasses.replace(GOOD, database_url=Settings.database_url)
    errors = production_config_errors(RAILWAY, s)
    assert len(errors) == 1 and "DATABASE_URL" in errors[0]


def test_both_are_reported_together():
    errors = production_config_errors(RAILWAY, Settings())
    assert len(errors) == 2


def test_a_laptop_needs_neither():
    assert production_config_errors({}, Settings()) == []


async def test_lifespan_raises_before_touching_the_database(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "production_config_errors", lambda: ["JWT_SECRET is not set"])

    async def boom():  # pragma: no cover - must not be reached
        raise AssertionError("init_db ran on an unsafe config")

    monkeypatch.setattr(main, "init_db", boom)
    with pytest.raises(RuntimeError, match="Refusing to start"):
        async with main.lifespan(main.app):
            pass
