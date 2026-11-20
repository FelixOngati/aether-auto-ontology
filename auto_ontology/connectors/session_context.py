# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Caller On-Behalf-Of (OBO) session attribute formatting for SQL connectors.

Translates :class:`~auto_ontology.server.identity.CallerOboContext` into
engine-specific transaction-local or per-connection session parameters so
source databases can enforce Row-Level Security (RLS) policies without leaking
caller identity across pooled connections.
"""

from __future__ import annotations

import json
import os
import re

from auto_ontology.server.identity import CallerOboContext, current_obo_context

_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")


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
    """Build transaction-local ``(guc_name, value)`` pairs for PostgreSQL ``set_config``."""
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


def snowflake_query_tag(obo_context: CallerOboContext | None = None) -> str | None:
    """Serialize caller OBO identity as a deterministic JSON ``QUERY_TAG``."""
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
    """Build per-connection MySQL user session variable assignments."""
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
