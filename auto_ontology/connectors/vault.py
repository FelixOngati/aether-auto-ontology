# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HashiCorp Vault access for UI-managed database connection secrets.

Connection credentials are stored in Vault keyed by ``{tenant_id}/{database_name}``
(when a tenant context is present) or ``{database_name}`` in single-tenant mode.
When ``AUTO_ONTOLOGY_REQUIRE_VAULT=1`` is set, partial or missing Vault
configuration and failed secret reads fail closed via :class:`VaultSecretError`
instead of falling back to plaintext JSON credentials in Postgres.

This module performs no import-time side effects (it does not load ``.env``);
entrypoints are responsible for loading the environment before use.
"""

from __future__ import annotations

import json
import logging
import os
import re

import hvac

from auto_ontology.server.identity import current_obo_context

logger = logging.getLogger(__name__)

REQUIRED_VAULT_ENV_VARS = (
    "VAULT_ADDR",
    "VAULT_NAMESPACE",
    "VAULT_ROLE_ID",
    "VAULT_SECRET_ID",
)

_SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class VaultSecretError(RuntimeError):
    """Vault is required by policy, misconfigured, or unable to resolve a secret."""


def is_vault_required() -> bool:
    """Return True when deployment policy forbids plaintext Postgres credentials."""
    raw = (os.environ.get("AUTO_ONTOLOGY_REQUIRE_VAULT") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _effective_tenant_id(tenant_id: str | None) -> str:
    if tenant_id is not None:
        return tenant_id.strip()
    ctx = current_obo_context()
    return ctx.tenant_id.strip() if ctx is not None and ctx.tenant_id else ""


def secret_path(database_name: str, *, tenant_id: str | None = None) -> str:
    """Build a validated Vault KV path (``{tenant_id}/{database_name}`` when scoped)."""
    db_clean = database_name.strip()
    if not db_clean or ".." in db_clean or "/" in db_clean:
        raise VaultSecretError(f"Invalid Vault secret identifier: {database_name!r}")

    effective_tenant = _effective_tenant_id(tenant_id)
    if not effective_tenant:
        return db_clean
    if not _SAFE_SEGMENT_RE.match(effective_tenant):
        raise VaultSecretError(f"Invalid Vault tenant identifier: {effective_tenant!r}")
    return f"{effective_tenant}/{db_clean}"


def is_vault_configured() -> bool:
    """Return True if the user provided all required Vault env vars.

    When :func:`is_vault_required` is enabled, any missing Vault variable raises
    :class:`VaultSecretError` rather than silently storing plaintext credentials.
    """
    missing = [var for var in REQUIRED_VAULT_ENV_VARS if not os.environ.get(var)]
    if missing:
        if is_vault_required():
            raise VaultSecretError(
                "AUTO_ONTOLOGY_REQUIRE_VAULT is enabled, but Vault environment "
                f"variables are missing: {', '.join(missing)}"
            )
        if len(missing) < len(REQUIRED_VAULT_ENV_VARS):
            logger.warning(
                "Vault is partially configured; ignoring Vault and treating "
                "connections as unencrypted. Missing env vars: %s",
                ", ".join(missing),
            )
    return not missing


def get_client() -> hvac.Client:
    addr = os.environ["VAULT_ADDR"]
    auth_mount = os.environ.get("VAULT_AUTH_MOUNT", "approle/nvdcs/dc1")
    try:
        client = hvac.Client(
            url=addr,
            namespace=os.environ["VAULT_NAMESPACE"],
        )
        client.auth.approle.login(
            role_id=os.environ["VAULT_ROLE_ID"],
            secret_id=os.environ["VAULT_SECRET_ID"],
            mount_point=auth_mount,
        )
    except Exception:
        logger.exception(
            "Failed to authenticate to Vault at %s (auth_mount=%s)", addr, auth_mount
        )
        raise
    logger.debug("Authenticated to Vault at %s (auth_mount=%s)", addr, auth_mount)
    return client


def write_secret(
    database_name: str,
    secret: dict[str, str],
    *,
    tenant_id: str | None = None,
) -> None:
    is_vault_configured()
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    path = secret_path(database_name, tenant_id=tenant_id)
    try:
        client = get_client()
        client.secrets.kv.v1.create_or_update_secret(
            path=path,
            secret={"connection": json.dumps(secret)},
            mount_point=mount_point,
        )
    except Exception:
        logger.exception(
            "Failed to write Vault secret for path %r (mount=%s)",
            path,
            mount_point,
        )
        raise
    logger.info("Wrote Vault secret for path %r (mount=%s)", path, mount_point)


def read_secret(
    database_name: str, *, tenant_id: str | None = None
) -> dict[str, str] | str:
    if not is_vault_configured():
        return ""
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    path = secret_path(database_name, tenant_id=tenant_id)
    try:
        client = get_client()
        response = client.secrets.kv.v1.read_secret(
            path=path,
            mount_point=mount_point,
        )
        return json.loads(response["data"]["connection"])
    except Exception as exc:
        if is_vault_required():
            raise VaultSecretError(
                f"Failed to read required Vault secret {path!r} (mount={mount_point})"
            ) from exc
        logger.warning(
            "Could not read Vault secret for path %r (mount=%s); treating as no secret",
            path,
            mount_point,
            exc_info=True,
        )
        return ""


def delete_secrets(
    database_name: str | None = None, *, tenant_id: str | None = None
) -> None:
    """Delete a single secret if database_name is given, otherwise delete all."""
    if not is_vault_configured():
        return

    client = get_client()
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")

    def delete(path: str) -> None:
        try:
            client.secrets.kv.v1.delete_secret(path=path, mount_point=mount_point)
        except Exception:
            logger.exception(
                "Failed to delete Vault secret %r (mount=%s)", path, mount_point
            )
            raise
        logger.info("Deleted Vault secret %r (mount=%s)", path, mount_point)

    if database_name:
        delete(secret_path(database_name, tenant_id=tenant_id))
        return

    prefix = _effective_tenant_id(tenant_id)
    list_endpoint = f"v1/{mount_point}/{prefix}" if prefix else f"v1/{mount_point}"
    try:
        # Use GET with ?list=true instead of the LIST HTTP verb, which the
        # Vault endpoint/proxy rejects ("Unsupported HTTP method").
        response = client.adapter.get(
            list_endpoint,
            params={"list": "true"},
        )
        keys = response["data"]["keys"]
    except hvac.exceptions.InvalidPath:
        # Nothing stored under this mount.
        logger.info("No secrets found under mount %r.", mount_point)
        return
    except Exception:
        logger.exception("Failed to list Vault secrets under mount %r", mount_point)
        raise

    logger.info("Deleting %d Vault secret(s) under mount %r", len(keys), mount_point)
    for key in keys:
        delete(f"{prefix}/{key}" if prefix else key)
