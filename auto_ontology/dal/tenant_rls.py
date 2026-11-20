# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PostgreSQL Row-Level Security (RLS) helpers for multi-tenant catalog tables.

In pooled multi-tenant deployments where multiple tenants share a single
Auto Ontology metadata store, catalog tables carrying a ``tenant_id`` column can
be isolated with PostgreSQL RLS policies bound to the session GUC
``<prefix>.tenant_id`` (matching :mod:`auto_ontology.connectors.session_context`).
"""

from __future__ import annotations

import re
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.engine import Connection

from auto_ontology.connectors.session_context import rls_guc_prefix

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

DEFAULT_TENANT_SCOPED_TABLES: tuple[str, ...] = (
    "connections",
    "catalog_database",
    "catalog_schema",
    "catalog_table",
    "catalog_column",
    "ontology_term",
    "ontology_sql_attribute",
)


def _validate_identifier(name: str, kind: str) -> str:
    if not _IDENT_RE.match(name):
        raise ValueError(f"Unsafe SQL {kind} identifier: {name!r}")
    return name


def build_tenant_rls_statements(
    table_name: str,
    *,
    schema_name: str = "public",
    tenant_column: str = "tenant_id",
    guc_prefix: str | None = None,
    allow_unset_admin_bypass: bool = False,
) -> list[str]:
    """Return idempotent PostgreSQL DDL statements enabling tenant RLS on *table_name*.

    When *allow_unset_admin_bypass* is ``False`` (the zero-trust default), rows
    are invisible whenever ``current_setting('<prefix>.tenant_id', true)`` is
    empty or unset — failing closed if a caller forgets to bind tenant context.
    """
    tbl = _validate_identifier(table_name, "table")
    sch = _validate_identifier(schema_name, "schema")
    col = _validate_identifier(tenant_column, "column")
    prefix = _validate_identifier(guc_prefix or rls_guc_prefix(), "GUC prefix")
    policy_name = _validate_identifier(f"{tbl}_tenant_isolation", "policy")
    guc_name = f"{prefix}.tenant_id"

    if allow_unset_admin_bypass:
        predicate = (
            f"(NULLIF(current_setting('{guc_name}', true), '') IS NULL "
            f"OR {col} = current_setting('{guc_name}', true))"
        )
    else:
        predicate = (
            f"(NULLIF(current_setting('{guc_name}', true), '') IS NOT NULL "
            f"AND {col} = current_setting('{guc_name}', true))"
        )

    qualified = f'"{sch}"."{tbl}"'
    return [
        f"ALTER TABLE {qualified} ADD COLUMN IF NOT EXISTS {col} TEXT",
        f"ALTER TABLE {qualified} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {qualified} FORCE ROW LEVEL SECURITY",
        f'DROP POLICY IF EXISTS "{policy_name}" ON {qualified}',
        (
            f'CREATE POLICY "{policy_name}" ON {qualified} '
            f"USING {predicate} WITH CHECK {predicate}"
        ),
    ]


def apply_catalog_tenant_rls(
    conn: Connection,
    tables: Sequence[str] = DEFAULT_TENANT_SCOPED_TABLES,
    *,
    schema_name: str = "public",
    tenant_column: str = "tenant_id",
    guc_prefix: str | None = None,
    allow_unset_admin_bypass: bool = False,
) -> int:
    """Execute tenant RLS DDL for each table in *tables*; return policy count."""
    applied = 0
    for table_name in tables:
        for stmt in build_tenant_rls_statements(
            table_name,
            schema_name=schema_name,
            tenant_column=tenant_column,
            guc_prefix=guc_prefix,
            allow_unset_admin_bypass=allow_unset_admin_bypass,
        ):
            conn.execute(text(stmt))
        applied += 1
    return applied
