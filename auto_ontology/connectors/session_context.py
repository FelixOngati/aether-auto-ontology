# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller On-Behalf-Of (OBO) session attributes for SQL connectors.

PostgreSQL is the only engine here that binds a transaction-local tenant GUC
(``set_config``) for row-level security policies. A caller-facing Postgres
query refuses to run when that tenant GUC cannot be set. An empty settings
list is not permission to run as the pool role.

Snowflake ``QUERY_TAG`` is an audit label. A row access policy does not read
it, and this connector does not set a session variable those policies
evaluate, so the tag does not filter rows.

MySQL ``@aether_*`` user variables are caller attributes for that connection.
This connector does not install a view or policy that reads them, so they do
not filter rows either.
"""

from __future__ import annotations

import json
import os
import re

from auto_ontology.server.identity import CallerOboContext, current_obo_context

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


class MissingCallerContextError(PermissionError):
    """Caller-facing SQL was asked to run without a tenant RLS binding."""


def resolve_effective_obo(
    obo_context: CallerOboContext | None = None,
) -> CallerOboContext | None:
    """Return explicit *obo_context* or the currently bound request OBO context."""
    return obo_context if obo_context is not None else current_obo_context()


def rls_guc_prefix() -> str:
    """Validated PostgreSQL GUC namespace prefix (default ``aether``)."""
    prefix = (os.environ.get("AUTO_ONTOLOGY_RLS_GUC_PREFIX") or "aether").strip()
    if not _IDENT_RE.match(prefix):
        raise ValueError(
            f"Invalid AUTO_ONTOLOGY_RLS_GUC_PREFIX {prefix!r}: must match [a-z_][a-z0-9_]*"
        )
    return prefix


def postgres_rls_settings(
    obo_context: CallerOboContext | None = None,
) -> list[tuple[str, str]]:
    """Build transaction-local ``(guc_name, value)`` pairs for PostgreSQL ``set_config``.

    Returns an empty list when no caller is bound. Caller-facing execution
    must use :func:`require_postgres_tenant_guc` instead of treating that
    empty list as a successful binding.
    """
    obo = resolve_effective_obo(obo_context)
    if obo is None:
        return []
    prefix = rls_guc_prefix()
    pairs: list[tuple[str, str]] = []
    if obo.tenant_id:
        pairs.append((f"{prefix}.tenant_id", obo.tenant_id))
    if obo.user_id:
        pairs.append((f"{prefix}.principal", obo.user_id))
    if obo.region:
        pairs.append((f"{prefix}.region", obo.region))
    if obo.roles:
        pairs.append((f"{prefix}.roles", ",".join(obo.roles)))
    return pairs


def require_postgres_tenant_guc(
    obo_context: CallerOboContext | None = None,
) -> list[tuple[str, str]]:
    """Return Postgres RLS settings, refusing when the tenant GUC is absent.

    The pool login is often the table owner. PostgreSQL skips row-level
    security for the owner unless the table uses ``FORCE ROW LEVEL SECURITY``,
    so a query with no ``tenant_id`` GUC must not run at all.
    """
    settings = postgres_rls_settings(obo_context)
    tenant_guc = f"{rls_guc_prefix()}.tenant_id"
    if not any(name == tenant_guc and value for name, value in settings):
        raise MissingCallerContextError(
            "Refusing to run SQL without a transaction-local tenant GUC. "
            "The pool role is not a substitute for caller row-level security."
        )
    return settings


def snowflake_query_tag(obo_context: CallerOboContext | None = None) -> str | None:
    """Serialize caller identity as a Snowflake ``QUERY_TAG`` audit label.

    The tag is not an input to a row access policy. Setting it does not
    filter rows.
    """
    obo = resolve_effective_obo(obo_context)
    if obo is None:
        return None
    payload = {
        "tenant_id": obo.tenant_id,
        "principal": obo.user_id,
        "region": obo.region,
        "roles": list(obo.roles),
    }
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def mysql_session_variables(
    obo_context: CallerOboContext | None = None,
) -> list[tuple[str, str]]:
    """Build per-connection MySQL user-variable assignments.

    These are caller attributes for the connection. This connector does not
    install a view or policy that reads them, so they do not filter rows.
    """
    obo = resolve_effective_obo(obo_context)
    if obo is None:
        return []
    prefix = rls_guc_prefix()
    variables: list[tuple[str, str]] = []
    if obo.tenant_id:
        variables.append((f"{prefix}_tenant_id", obo.tenant_id))
    if obo.user_id:
        variables.append((f"{prefix}_principal", obo.user_id))
    if obo.region:
        variables.append((f"{prefix}_region", obo.region))
    return variables
