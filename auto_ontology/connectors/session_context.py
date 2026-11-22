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


class UnmappedRbacRoleError(PermissionError):
    """OBO roles were present and did not resolve to one PostgreSQL role."""


_role_map_cache: tuple[str, dict[str, str] | None] | None = None


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


def _rbac_role_prefix() -> str:
    prefix = (
        os.environ.get("AUTO_ONTOLOGY_RBAC_ROLE_PREFIX") or "aether_role_"
    ).strip()
    if not _IDENT_RE.match(prefix):
        raise ValueError(
            f"Invalid AUTO_ONTOLOGY_RBAC_ROLE_PREFIX {prefix!r}: "
            "must match [a-z_][a-z0-9_]*"
        )
    return prefix


def _parse_rbac_role_map(raw: str) -> dict[str, str] | None:
    """Parse ``AUTO_ONTOLOGY_RBAC_ROLE_MAP``. An empty value means prefix mapping."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "AUTO_ONTOLOGY_RBAC_ROLE_MAP is not valid JSON. "
            "Fix it before serving queries; the pool login is not a fallback role."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("AUTO_ONTOLOGY_RBAC_ROLE_MAP must be a JSON object")
    cleaned: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError(
                "AUTO_ONTOLOGY_RBAC_ROLE_MAP values must be non-empty PostgreSQL role names"
            )
        role = value.strip()
        if not _IDENT_RE.match(role):
            raise ValueError(
                f"Invalid PostgreSQL RBAC role identifier {role!r}: "
                "must match [a-z_][a-z0-9_]*"
            )
        cleaned[key] = role
    return cleaned


def rbac_role_map() -> dict[str, str] | None:
    """Return the validated role map, parsing it once per distinct env value."""
    global _role_map_cache
    raw = (os.environ.get("AUTO_ONTOLOGY_RBAC_ROLE_MAP") or "").strip()
    if _role_map_cache is not None and _role_map_cache[0] == raw:
        return _role_map_cache[1]
    parsed = _parse_rbac_role_map(raw)
    _role_map_cache = (raw, parsed)
    return parsed


def validate_rbac_configuration() -> None:
    """Fail process startup when the RBAC map or role prefix cannot be used."""
    rbac_role_map()
    _rbac_role_prefix()


def postgres_rbac_role(obo_context: CallerOboContext | None = None) -> str | None:
    """Resolve the transaction-local PostgreSQL role for a caller query.

    When the caller has OBO roles, this always returns one PostgreSQL role.
    ``AUTO_ONTOLOGY_RBAC_ROLE_MAP`` names that role explicitly. Every OBO role
    must appear in the map, and they must all name the same PostgreSQL role,
    so claim order cannot pick a privilege. With no map, each OBO role is
    prefixed with ``AUTO_ONTOLOGY_RBAC_ROLE_PREFIX`` (default ``aether_role_``).
    ``AUTO_ONTOLOGY_ENABLE_RBAC_SET_ROLE`` does not turn this off.

    A missing or ambiguous mapping raises :class:`UnmappedRbacRoleError`
    instead of leaving the statement on the pool login. Role strings are
    checked against ``^[a-z_][a-z0-9_]*$`` before they are interpolated into
    ``SET LOCAL ROLE``.
    """
    obo = resolve_effective_obo(obo_context)
    if obo is None or not obo.roles:
        return None

    mapping = rbac_role_map()
    if mapping is not None:
        missing = [role for role in obo.roles if role not in mapping]
        if missing:
            raise UnmappedRbacRoleError(
                "Refusing to run SQL as the pool login. "
                f"OBO roles {list(missing)} are not in AUTO_ONTOLOGY_RBAC_ROLE_MAP."
            )
        resolved = [mapping[role] for role in obo.roles]
    else:
        prefix = _rbac_role_prefix()
        resolved = []
        for role in obo.roles:
            candidate = f"{prefix}{role.strip().lower()}"
            if not _IDENT_RE.match(candidate):
                raise ValueError(
                    f"Invalid PostgreSQL RBAC role identifier {candidate!r}: "
                    "must match [a-z_][a-z0-9_]*"
                )
            resolved.append(candidate)

    unique = list(dict.fromkeys(resolved))
    if len(unique) != 1:
        raise UnmappedRbacRoleError(
            "Refusing to run SQL as the pool login. "
            "OBO roles map to more than one PostgreSQL role."
        )
    return unique[0]


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
