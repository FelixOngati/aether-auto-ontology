# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for transaction-local OBO/RLS session injection across SQL connectors."""

from __future__ import annotations

import contextlib
import json
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from auto_ontology.connectors.mysql import MySQLDatabase
from auto_ontology.connectors.postgres import PostgresDatabase, _reset_pooled_connection
from auto_ontology.semantic.visit_enter import execute_capped
from auto_ontology.connectors.session_context import (
    MissingCallerContextError,
    UnmappedRbacRoleError,
    mysql_session_variables,
    postgres_rbac_role,
    postgres_rls_settings,
    require_postgres_tenant_guc,
    rls_guc_prefix,
    snowflake_query_tag,
    validate_rbac_configuration,
)
from auto_ontology.connectors.snowflake import SnowflakeDatabase
from auto_ontology.retrieval.text_to_sql.react.run_sql import (
    check_sql,
    enforce_default_row_limit,
)
from auto_ontology.server.identity import CallerOboContext, bind_obo_context


def test_session_context_helpers_format_and_validate_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obo = CallerOboContext(
        user_id="analyst@example.com",
        tenant_id="tenant-eu",
        region="EMEA",
        roles=("finance_analyst", "viewer"),
    )
    assert postgres_rls_settings(obo) == [
        ("aether.tenant_id", "tenant-eu"),
        ("aether.principal", "analyst@example.com"),
        ("aether.region", "EMEA"),
        ("aether.roles", "finance_analyst,viewer"),
    ]
    assert json.loads(snowflake_query_tag(obo) or "{}") == {
        "principal": "analyst@example.com",
        "region": "EMEA",
        "roles": ["finance_analyst", "viewer"],
        "tenant_id": "tenant-eu",
    }
    assert mysql_session_variables(obo) == [
        ("aether_tenant_id", "tenant-eu"),
        ("aether_principal", "analyst@example.com"),
        ("aether_region", "EMEA"),
    ]

    monkeypatch.setenv("AUTO_ONTOLOGY_RLS_GUC_PREFIX", "bad-prefix; DROP TABLE")
    with pytest.raises(ValueError, match="AUTO_ONTOLOGY_RLS_GUC_PREFIX"):
        rls_guc_prefix()


def test_postgres_execute_injects_transaction_local_set_config_with_read_only() -> None:
    executed: list[tuple[str, Any]] = []

    class _FakeCursor:
        description = [("revenue",)]

        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            executed.append((sql, params))

        def fetchall(self) -> list[dict[str, Any]]:
            return [{"revenue": 4200}]

    class _FakeConn:
        def execute(self, sql: str, params: Any = None) -> None:
            executed.append((sql, params))

        def cursor(self, **kwargs: Any) -> _FakeCursor:
            return _FakeCursor()

        @contextlib.contextmanager
        def transaction(self):
            executed.append(("BEGIN", None))
            yield
            executed.append(("COMMIT", None))

    class _FakePool:
        closed = False

        @contextlib.contextmanager
        def connection(self):
            yield _FakeConn()

        def close(self) -> None:
            self.closed = True

    db = object.__new__(PostgresDatabase)
    db._pool = _FakePool()  # type: ignore[assignment]
    db._database_name = "analytics"
    db._finalizer = None  # type: ignore[assignment]

    ctx = CallerOboContext(
        user_id="alice@example.com", tenant_id="tenant-a", region="EMEA"
    )
    with bind_obo_context(ctx):
        df = db.execute("SELECT revenue FROM accounts", read_only=True)

    assert isinstance(df, pd.DataFrame)
    assert df.to_dict(orient="records") == [{"revenue": 4200}]
    assert executed[0] == ("BEGIN", None)
    set_sql, set_params = executed[1]
    assert "set_config('transaction_read_only', %s, true)" in set_sql
    assert "set_config('aether.tenant_id', %s, true)" in set_sql
    assert "set_config('aether.principal', %s, true)" in set_sql
    assert "set_config('aether.region', %s, true)" in set_sql
    assert set_params == ("on", "tenant-a", "alice@example.com", "EMEA")
    assert executed[2][0] == "SELECT revenue FROM accounts"
    assert executed[3] == ("COMMIT", None)


def test_postgres_execute_refuses_without_a_tenant_guc() -> None:
    executed: list[tuple[str, Any]] = []

    class _FakePool:
        @contextlib.contextmanager
        def connection(self):
            executed.append(("connection", None))
            yield object()

    db = object.__new__(PostgresDatabase)
    db._pool = _FakePool()  # type: ignore[assignment]
    db._database_name = "analytics"
    db._finalizer = None  # type: ignore[assignment]

    with pytest.raises(MissingCallerContextError, match="tenant GUC"):
        db.execute("SELECT revenue FROM accounts")
    assert executed == []

    with bind_obo_context(
        CallerOboContext(user_id="alice@example.com", tenant_id="", region="EMEA")
    ):
        with pytest.raises(MissingCallerContextError, match="tenant GUC"):
            db.execute("SELECT revenue FROM accounts")
    assert executed == []
    with pytest.raises(MissingCallerContextError):
        require_postgres_tenant_guc(None)


def test_catalog_sampling_opts_postgres_out_of_caller_rls() -> None:
    db = object.__new__(PostgresDatabase)
    seen: list[dict[str, Any]] = []

    def execute(sql: str, parameters: Any = None, **kwargs: Any) -> pd.DataFrame:
        seen.append(kwargs)
        return pd.DataFrame()

    db.execute = execute  # type: ignore[method-assign]
    execute_capped(db, "SELECT 1")
    assert seen == [{"timeout_s": 120, "enforce_caller_rls": False}]


def test_snowflake_query_tag_and_mysql_variables_are_not_row_filters() -> None:
    snowflake_doc = " ".join((snowflake_query_tag.__doc__ or "").split())
    mysql_doc = " ".join((mysql_session_variables.__doc__ or "").split())
    assert "does not filter rows" in snowflake_doc
    assert "row access policy" in snowflake_doc
    assert "do not filter rows" in mysql_doc


def test_snowflake_and_mysql_execute_propagate_obo_session_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_sf_kwargs: dict[str, Any] = {}

    sf = object.__new__(SnowflakeDatabase)
    sf._connect_kwargs = {"user": "svc", "account": "acct"}
    sf._warehouse = "WH"
    sf._database_name = "SF_DB"
    sf._schema_filter = None

    def fake_sf_execute(
        connect_kwargs: dict[str, Any], sql: str, parameters: Any
    ) -> pd.DataFrame:
        captured_sf_kwargs.update(connect_kwargs)
        return pd.DataFrame([{"n": 1}])

    monkeypatch.setattr(sf, "_execute", fake_sf_execute)
    ctx = CallerOboContext(user_id="bob@example.com", tenant_id="tenant-b", region="US")
    sf.execute("SELECT 1 AS n", obo_context=ctx)
    tag = json.loads(captured_sf_kwargs["session_parameters"]["QUERY_TAG"])
    assert tag["tenant_id"] == "tenant-b"
    assert tag["region"] == "US"

    mysql_calls: list[tuple[str, Any]] = []
    fake_cursor = MagicMock()
    fake_cursor.description = [("n",)]
    fake_cursor.fetchall.return_value = [{"n": 1}]
    fake_cursor.execute.side_effect = lambda q, p=None: mysql_calls.append((q, p))
    fake_conn = MagicMock()
    fake_conn.cursor.return_value = fake_cursor
    monkeypatch.setattr("mysql.connector.connect", lambda **kw: fake_conn)

    my = object.__new__(MySQLDatabase)
    my._connect_kwargs = {"database": "mydb"}
    my._database_name = "mydb"
    my._timeout_variable = None
    my.execute("SELECT 1 AS n", read_only=True, obo_context=ctx)

    assert mysql_calls[0] == ("SET SESSION TRANSACTION READ ONLY", None)
    assert "@aether_tenant_id = %s" in mysql_calls[1][0]
    assert mysql_calls[1][1] == ("tenant-b", "bob@example.com", "US")


def test_patch5_rbac_set_local_role_discard_all_and_sqlglot_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[tuple[str, Any]] = []

    class _FakeCursor:
        description = [("revenue",)]

        def __enter__(self) -> "_FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            executed.append((sql, params))

        def fetchall(self) -> list[dict[str, Any]]:
            return [{"revenue": 100}]

    class _FakeConn:
        def execute(self, sql: str, params: Any = None) -> None:
            executed.append((sql, params))

        def rollback(self) -> None:
            executed.append(("ROLLBACK", None))

        def cursor(self, **kwargs: Any) -> _FakeCursor:
            return _FakeCursor()

        @contextlib.contextmanager
        def transaction(self):
            executed.append(("BEGIN", None))
            yield
            executed.append(("COMMIT", None))

    class _FakePool:
        closed = False

        @contextlib.contextmanager
        def connection(self):
            yield _FakeConn()

        def close(self) -> None:
            self.closed = True

    # 1. Pool reset rolls back an open transaction, then DISCARD ALL.
    reset_conn = _FakeConn()
    _reset_pooled_connection(reset_conn)  # type: ignore[arg-type]
    assert executed == [("ROLLBACK", None), ("DISCARD ALL", None)]
    executed.clear()

    # 2. SET LOCAL ROLE from verified OBO role map + fail-closed identifier validation
    monkeypatch.setenv(
        "AUTO_ONTOLOGY_RBAC_ROLE_MAP",
        json.dumps({"finance_analyst": "aether_finance_ro"}),
    )
    ctx = CallerOboContext(
        user_id="alice@example.com",
        tenant_id="tenant-a",
        region="EMEA",
        roles=("finance_analyst",),
    )
    assert postgres_rbac_role(ctx) == "aether_finance_ro"

    db = object.__new__(PostgresDatabase)
    db._pool = _FakePool()  # type: ignore[assignment]
    db._database_name = "analytics"
    db._finalizer = None  # type: ignore[assignment]
    db.execute("SELECT revenue FROM accounts", read_only=True, obo_context=ctx)
    assert executed[0] == ("BEGIN", None)
    assert executed[1] == ("SET LOCAL ROLE aether_finance_ro", None)
    assert "set_config('aether.tenant_id', %s, true)" in executed[2][0]

    monkeypatch.setenv(
        "AUTO_ONTOLOGY_RBAC_ROLE_MAP",
        json.dumps({"finance_analyst": "bad_role; DROP ROLE postgres"}),
    )
    with pytest.raises(ValueError, match="Invalid PostgreSQL RBAC role identifier"):
        postgres_rbac_role(ctx)

    # 3. sqlglot blocks COPY, GRANT, pg_catalog, information_schema, and injects default LIMIT
    class _StubCatalog:
        def columns(self, table: str, schema: str | None) -> list[str] | None:
            return ["id", "amount"] if table == "invoices" else None

        def schemas_with(self, table: str) -> list[str]:
            return ["public"] if table == "invoices" else []

    cat = _StubCatalog()
    assert check_sql("COPY invoices TO '/tmp/out.csv'", "postgres", cat).problems
    assert check_sql("GRANT SELECT ON invoices TO public", "postgres", cat).problems
    assert check_sql("SELECT * FROM pg_catalog.pg_tables", "postgres", cat).problems
    assert check_sql(
        "SELECT * FROM information_schema.columns", "postgres", cat
    ).problems
    assert check_sql("SELECT * FROM pg_shadow", "postgres", cat).problems
    assert not check_sql("SELECT id, amount FROM invoices", "postgres", cat).problems

    monkeypatch.setenv("AUTO_ONTOLOGY_MAX_QUERY_ROWS", "250")
    bounded = enforce_default_row_limit(
        "SELECT id, amount FROM invoices", dialect="postgres"
    )
    assert "LIMIT 250" in bounded.upper()
    already_bounded = enforce_default_row_limit(
        "SELECT id FROM invoices LIMIT 10", dialect="postgres"
    )
    assert "LIMIT 10" in already_bounded.upper()
    assert "250" not in already_bounded
    clamped = enforce_default_row_limit(
        "SELECT id FROM invoices LIMIT 100000000", dialect="postgres"
    )
    assert "LIMIT 250" in clamped.upper()
    assert "100000000" not in clamped
    monkeypatch.setenv("AUTO_ONTOLOGY_MAX_QUERY_ROWS", "0")
    with pytest.raises(ValueError, match="positive row cap"):
        enforce_default_row_limit("SELECT id FROM invoices", dialect="postgres")


def test_rbac_role_is_required_when_obo_roles_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTO_ONTOLOGY_RBAC_ROLE_MAP", raising=False)
    monkeypatch.delenv("AUTO_ONTOLOGY_ENABLE_RBAC_SET_ROLE", raising=False)
    ctx = CallerOboContext(
        user_id="alice@example.com",
        tenant_id="tenant-a",
        roles=("finance_analyst",),
    )
    assert postgres_rbac_role(ctx) == "aether_role_finance_analyst"

    ambiguous = CallerOboContext(
        user_id="alice@example.com",
        tenant_id="tenant-a",
        roles=("finance_analyst", "viewer"),
    )
    with pytest.raises(UnmappedRbacRoleError, match="more than one"):
        postgres_rbac_role(ambiguous)

    monkeypatch.setenv(
        "AUTO_ONTOLOGY_RBAC_ROLE_MAP",
        json.dumps(
            {"finance_analyst": "aether_finance_ro", "viewer": "aether_finance_ro"}
        ),
    )
    assert postgres_rbac_role(ambiguous) == "aether_finance_ro"
    missing = CallerOboContext(
        user_id="alice@example.com",
        tenant_id="tenant-a",
        roles=("finance_analyst", "auditor"),
    )
    with pytest.raises(UnmappedRbacRoleError, match="auditor"):
        postgres_rbac_role(missing)
    split = CallerOboContext(
        user_id="alice@example.com",
        tenant_id="tenant-a",
        roles=("finance_analyst", "auditor"),
    )
    monkeypatch.setenv(
        "AUTO_ONTOLOGY_RBAC_ROLE_MAP",
        json.dumps(
            {"finance_analyst": "aether_finance_ro", "auditor": "aether_audit_ro"}
        ),
    )
    with pytest.raises(UnmappedRbacRoleError, match="more than one"):
        postgres_rbac_role(split)

    monkeypatch.setenv("AUTO_ONTOLOGY_RBAC_ROLE_MAP", "{")
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_rbac_configuration()
