"""The hand-written ALTER TABLE migrations in ``core/db.py``.

Flapp has no Alembic. Schema evolution is raw SQL strings run from the FastAPI
lifespan hook at startup, and until this file existed the test suite never
executed a single one of them: tests build a fresh SQLite schema with
``create_all``, so every migrator hits its ``if not cols: return`` guard and
returns before the first ALTER. Production is Postgres with an existing table
and takes the other branch -- the branch nothing covered.

That gap has already cost a production outage. ``ALTER TABLE users ADD COLUMN
notify_on_ready BOOLEAN NOT NULL DEFAULT 1`` shipped with a green suite: SQLite
accepts ``DEFAULT 1`` for a boolean, Postgres raises DatatypeMismatchError, the
app died in the lifespan hook and every deploy afterwards failed its health
check.

Two kinds of check here, because neither is sufficient alone.

The EXECUTION tests run the migrators against a table built the way production
has one -- older, missing the newer columns -- so the ALTER branch actually
runs. That catches syntax errors and typos, on SQLite.

The DIALECT tests never touch a database. They compare the type each ALTER
declares against the type SQLAlchemy would emit for that column on the
POSTGRES dialect. That is the check that catches the class of bug SQLite hides,
and it is the one that would have caught ``TIMESTAMP`` where the model says
``DateTime(timezone=True)`` -- a bare TIMESTAMP is *without* time zone in
Postgres, so create_all on a fresh database and the ALTER on an existing one
would build two different columns, and the existing one would reject every
write.

A real Postgres-backed run is still the only complete answer. This is what can
be checked without one.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import postgresql

from app.core import db as dbmod
from app.models.analysis import Analysis
from app.models.order import Order
from app.models.user import User

DB_PY = Path(dbmod.__file__)


# --------------------------------------------------------------------------
# harvesting the DDL
# --------------------------------------------------------------------------

_ALTER = re.compile(
    r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)\s+([A-Z][A-Z0-9_ ()]*?)"
    r"(?:\s+NOT\s+NULL)?(?:\s+DEFAULT\s+\S+?)?\s*[\"']",
    re.I,
)

# Long statements are written as adjacent Python string literals, so the SQL a
# scanner needs to read is interrupted by `" "` in the middle. Joining them
# first is not cosmetic: without it a pattern anchored on BOOLEAN stops at the
# quote and never reaches the DEFAULT on the next line -- which is exactly
# where the outage lived.
_CONCAT = re.compile(r"[\"']\s*[\"']")


def _sql_source() -> str:
    return _CONCAT.sub("", DB_PY.read_text(encoding="utf-8"))


def _alters() -> list[tuple[str, str, str]]:
    """(table, column, declared type) for every ALTER ... ADD COLUMN in db.py."""
    found = [(t, c, ty.strip()) for t, c, ty in _ALTER.findall(_sql_source())]
    assert found, "no ALTER statements found -- the scanner regex has gone stale"
    return found


MODELS = {"users": User, "orders": Order, "analyses": Analysis}


def test_the_scanner_actually_sees_the_migrations():
    """This file is worthless if the regex quietly stops matching. Anchor it on
    a column that exists and must keep existing."""
    cols = {(t, c) for t, c, _ in _alters()}
    assert ("users", "notify_on_ready") in cols
    assert ("users", "mobility_measured_at") in cols
    assert len(cols) >= 10


# --------------------------------------------------------------------------
# dialect: what Postgres will actually build
# --------------------------------------------------------------------------

@pytest.mark.parametrize("table,column,declared", _alters())
def test_each_alter_matches_what_postgres_would_create(table, column, declared):
    """A column added by ALTER must be the same column create_all would build.

    When they differ, a fresh deploy and an upgraded one get different schemas,
    and only one of them is the one the ORM thinks it is talking to.
    """
    model = MODELS.get(table)
    if model is None or column not in model.__table__.c:
        pytest.skip(f"{table}.{column} is not a current model column")
    want = model.__table__.c[column].type.compile(postgresql.dialect())
    # JSON is spelled per-dialect in db.py on purpose (JSONB on Postgres,
    # JSON on SQLite) -- that one is deliberately not a literal match.
    if "JSON" in want.upper() or "JSON" in declared.upper():
        pytest.skip("JSON column: dialect-specific by design")
    assert declared.upper() == want.upper(), (
        f"{table}.{column}: the migration says {declared!r} but Postgres "
        f"create_all would build {want!r}"
    )


@pytest.mark.parametrize("table,column,declared", _alters())
def test_no_alter_declares_a_type_postgres_does_not_know(table, column, declared):
    """SQLite accepts any type name at all -- it applies affinity rules and
    moves on. So a typo in a type is invisible until Postgres sees it."""
    known = {
        "INTEGER", "BIGINT", "SMALLINT", "FLOAT", "DOUBLE PRECISION", "REAL",
        "BOOLEAN", "TEXT", "DATE", "TIMESTAMP", "TIMESTAMP WITH TIME ZONE",
        "TIMESTAMP WITHOUT TIME ZONE", "NUMERIC", "JSON", "JSONB",
    }
    base = re.sub(r"\(\d+(,\s*\d+)?\)", "", declared).strip().upper()
    assert base in known or base.startswith("VARCHAR"), (
        f"{table}.{column}: {declared!r} is not a type Postgres is known to "
        "accept -- check it before this reaches a deploy"
    )


def test_no_sqlite_only_boolean_default():
    """The literal outage. SQLite takes DEFAULT 1 for a boolean; Postgres
    raises DatatypeMismatchError, the lifespan hook dies, and every deploy
    afterwards fails its health check."""
    hits = list(re.finditer(r"BOOLEAN\s[^;]*?DEFAULT\s+(\S+?)[\s\"']", _sql_source(), re.I))
    assert hits, "no boolean DEFAULT found -- this guard has stopped guarding"
    for m in hits:
        val = m.group(1).strip("\"' ")
        assert val.lower() in ("true", "false"), (
            f"boolean DEFAULT {val!r} -- Postgres needs true/false, not 0/1"
        )


# --------------------------------------------------------------------------
# execution: run them against a table that predates the columns
# --------------------------------------------------------------------------

def _legacy_users_table(path: Path) -> None:
    """The users table as it existed before any of the migrations ran.

    Written by hand rather than derived from the model: the point is to be a
    table the model has since outgrown, and one generated from the model can
    never be that.
    """
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE users ("
        " id INTEGER PRIMARY KEY,"
        " email VARCHAR(320) NOT NULL,"
        " password_hash VARCHAR(200) NOT NULL,"
        " is_pro BOOLEAN NOT NULL DEFAULT 0,"
        " created_at TIMESTAMP"
        ")"
    )
    con.execute(
        "INSERT INTO users (email, password_hash, is_pro) "
        "VALUES ('legacy@example.com', 'x', 1)"
    )
    con.commit()
    con.close()


def test_the_user_migration_runs_on_a_table_that_predates_it(tmp_path):
    """The branch production takes and the suite never did."""
    path = tmp_path / "legacy.db"
    _legacy_users_table(path)

    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        dbmod._migrate_users(conn)

    with engine.connect() as conn:
        cols = {c["name"] for c in inspect(conn).get_columns("users")}
    engine.dispose()

    for col in (
        "tier", "stripe_customer_id", "subscription_status", "expert_credits",
        "free_preview_job_id", "expert_credit_grant_ref", "height_cm",
        "notify_on_ready", "mobility_hamstring_deg", "mobility_hip_flexion_deg",
        "mobility_measured_at", "mobility_goal",
    ):
        assert col in cols, f"{col} was never added by _migrate_users"


def test_running_the_migration_twice_is_a_no_op(tmp_path):
    """It runs on every boot. A second pass must not raise 'duplicate column'."""
    path = tmp_path / "legacy.db"
    _legacy_users_table(path)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        dbmod._migrate_users(conn)
    with engine.begin() as conn:
        dbmod._migrate_users(conn)      # must not raise
    engine.dispose()


def test_a_legacy_pro_account_is_carried_onto_a_tier(tmp_path):
    """The one migration that moves data rather than adding a column. A pro
    account that came out the other side on the free tier would be a paying
    customer silently downgraded."""
    path = tmp_path / "legacy.db"
    _legacy_users_table(path)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        dbmod._migrate_users(conn)
    with engine.connect() as conn:
        tier = conn.execute(text("SELECT tier FROM users")).scalar()
    engine.dispose()
    assert tier == "enthusiast"


def test_mobility_columns_start_empty(tmp_path):
    """No backfill, on purpose: NULL means "has not done the screens", which
    the analysis has to handle anyway. Inventing a range for existing accounts
    would put a fabricated limitation on every one of them."""
    path = tmp_path / "legacy.db"
    _legacy_users_table(path)
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:
        dbmod._migrate_users(conn)
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT mobility_hamstring_deg, mobility_hip_flexion_deg, "
            "mobility_measured_at, mobility_goal FROM users"
        )).first()
    engine.dispose()
    assert row == (None, None, None, None)
