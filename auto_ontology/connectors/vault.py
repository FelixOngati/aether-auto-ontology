# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HashiCorp Vault / OpenBao access for UI-managed database connection secrets.

Connection credentials are stored in Vault keyed by ``{tenant_id}/{database_name}``
(when a tenant context is present) or ``{database_name}`` in single-tenant mode.
Supports both OSS Vault / OpenBao (optional ``VAULT_NAMESPACE``), token or
AppRole authentication, and KV v1 / KV v2 secret engines (``VAULT_KV_VERSION``).

When ``AUTO_ONTOLOGY_REQUIRE_VAULT=1`` is set, partial or missing Vault
configuration and failed Vault operations fail closed via
:class:`VaultSecretError` instead of falling back to plaintext JSON credentials
in Postgres.

This module performs no import-time side effects (it does not load ``.env``);
entrypoints are responsible for loading the environment before use.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import hvac

from auto_ontology.server.identity import current_obo_context

logger = logging.getLogger(__name__)

VAULT_ENV_VARS = (
    "VAULT_ADDR",
    "VAULT_NAMESPACE",
    "VAULT_TOKEN",
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


def _kv_version() -> int:
    raw = (os.environ.get("VAULT_KV_VERSION") or "1").strip()
    if raw == "1":
        return 1
    if raw == "2":
        return 2
    raise VaultSecretError(f"Unsupported VAULT_KV_VERSION {raw!r}; expected '1' or '2'")


def _validate_segment(segment: str, *, kind: str) -> str:
    cleaned = segment.strip()
    if (
        not cleaned
        or cleaned in {".", ".."}
        or ".." in cleaned
        or not _SAFE_SEGMENT_RE.match(cleaned)
    ):
        raise VaultSecretError(f"Invalid Vault {kind} identifier: {segment!r}")
    return cleaned


def _effective_tenant_id(tenant_id: str | None) -> str:
    if tenant_id is not None:
        return tenant_id.strip()
    ctx = current_obo_context()
    return ctx.tenant_id.strip() if ctx is not None and ctx.tenant_id else ""


def secret_path(database_name: str, *, tenant_id: str | None = None) -> str:
    """Build a validated Vault KV path (``{tenant_id}/{database_name}`` when scoped)."""
    db_clean = _validate_segment(database_name, kind="secret")
    effective_tenant = _effective_tenant_id(tenant_id)
    if not effective_tenant:
        return db_clean
    tenant_clean = _validate_segment(effective_tenant, kind="tenant")
    return f"{tenant_clean}/{db_clean}"


def _missing_vault_env_vars() -> list[str]:
    addr = (os.environ.get("VAULT_ADDR") or "").strip()
    token = (os.environ.get("VAULT_TOKEN") or "").strip()
    role_id = (os.environ.get("VAULT_ROLE_ID") or "").strip()
    secret_id = (os.environ.get("VAULT_SECRET_ID") or "").strip()

    missing: list[str] = []
    if not addr:
        missing.append("VAULT_ADDR")
    if not token and not (role_id and secret_id):
        if not role_id:
            missing.append("VAULT_ROLE_ID")
        if not secret_id:
            missing.append("VAULT_SECRET_ID")
    return missing


def is_vault_configured() -> bool:
    """Return True if Vault address and auth credentials (token or AppRole) are set.

    ``VAULT_NAMESPACE`` is optional so OSS HashiCorp Vault and OpenBao work
    alongside Vault Enterprise. When :func:`is_vault_required` is enabled, any
    missing required Vault variable raises :class:`VaultSecretError`.
    """
    missing = _missing_vault_env_vars()
    if not missing:
        _kv_version()
        return True

    if is_vault_required():
        raise VaultSecretError(
            "AUTO_ONTOLOGY_REQUIRE_VAULT is enabled, but Vault environment "
            f"variables are missing: {', '.join(missing)}"
        )

    any_set = any((os.environ.get(var) or "").strip() for var in VAULT_ENV_VARS)
    if any_set:
        logger.warning(
            "Vault is partially configured; ignoring Vault and treating "
            "connections as unencrypted. Missing env vars: %s",
            ", ".join(missing),
        )
    return False


def get_client() -> hvac.Client:
    addr = (os.environ.get("VAULT_ADDR") or "").strip()
    if not addr:
        raise VaultSecretError("VAULT_ADDR is not configured")
    namespace = (os.environ.get("VAULT_NAMESPACE") or "").strip() or None
    token = (os.environ.get("VAULT_TOKEN") or "").strip()
    auth_mount = os.environ.get("VAULT_AUTH_MOUNT", "approle/nvdcs/dc1")
    try:
        if token:
            client = hvac.Client(url=addr, namespace=namespace, token=token)
        else:
            client = hvac.Client(url=addr, namespace=namespace)
            client.auth.approle.login(
                role_id=os.environ["VAULT_ROLE_ID"],
                secret_id=os.environ["VAULT_SECRET_ID"],
                mount_point=auth_mount,
            )
    except Exception as exc:
        logger.exception("Failed to authenticate to Vault at %s", addr)
        if is_vault_required():
            raise VaultSecretError("Failed to authenticate to Vault") from exc
        raise
    logger.debug("Authenticated to Vault at %s", addr)
    return client


def write_secret(
    database_name: str,
    secret: dict[str, Any],
    *,
    tenant_id: str | None = None,
) -> None:
    if not is_vault_configured():
        return
    kv_ver = _kv_version()
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    path = secret_path(database_name, tenant_id=tenant_id)
    payload = {"connection": json.dumps(secret)}
    try:
        client = get_client()
        if kv_ver == 2:
            client.secrets.kv.v2.create_or_update_secret(
                path=path,
                secret=payload,
                mount_point=mount_point,
            )
        else:
            client.secrets.kv.v1.create_or_update_secret(
                path=path,
                secret=payload,
                mount_point=mount_point,
            )
    except VaultSecretError:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to write Vault secret for path %r (mount=%s)",
            path,
            mount_point,
        )
        if is_vault_required():
            raise VaultSecretError(
                f"Failed to write required Vault secret {path!r} (mount={mount_point})"
            ) from exc
        raise
    logger.info("Wrote Vault secret for path %r (mount=%s)", path, mount_point)


def read_secret(
    database_name: str, *, tenant_id: str | None = None
) -> dict[str, Any] | str:
    if not is_vault_configured():
        return ""
    kv_ver = _kv_version()
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")
    path = secret_path(database_name, tenant_id=tenant_id)
    try:
        client = get_client()
        if kv_ver == 2:
            response = client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=mount_point,
                raise_on_deleted_version=True,
            )
            secret_data = response["data"]["data"]
        else:
            response = client.secrets.kv.v1.read_secret(
                path=path,
                mount_point=mount_point,
            )
            secret_data = response["data"]

        if isinstance(secret_data, dict) and isinstance(
            secret_data.get("connection"), str
        ):
            return json.loads(secret_data["connection"])
        if isinstance(secret_data, dict):
            return dict(secret_data)
        raise ValueError("Unexpected Vault secret payload structure")
    except hvac.exceptions.InvalidPath:
        logger.debug("No Vault secret stored at path %r (mount=%s)", path, mount_point)
        return ""
    except VaultSecretError:
        raise
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
    """Delete a single secret if database_name is given, otherwise delete tenant secrets."""
    if not is_vault_configured():
        return

    kv_ver = _kv_version()
    mount_point = os.environ.get("VAULT_KV_MOUNT", "auto-ontology")

    if database_name is not None:
        target_path = secret_path(database_name, tenant_id=tenant_id)
    else:
        raw_prefix = _effective_tenant_id(tenant_id)
        if not raw_prefix and is_vault_required():
            raise VaultSecretError(
                "Unscoped bulk Vault secret deletion requires a tenant_id when "
                "AUTO_ONTOLOGY_REQUIRE_VAULT is enabled"
            )
        target_path = _validate_segment(raw_prefix, kind="tenant") if raw_prefix else ""

    try:
        client = get_client()
    except VaultSecretError:
        raise
    except Exception as exc:
        if is_vault_required():
            raise VaultSecretError("Failed to connect to Vault for deletion") from exc
        raise

    def _delete_one(path: str) -> None:
        try:
            if kv_ver == 2:
                client.secrets.kv.v2.delete_metadata_and_all_versions(
                    path=path, mount_point=mount_point
                )
            else:
                client.secrets.kv.v1.delete_secret(path=path, mount_point=mount_point)
        except VaultSecretError:
            raise
        except Exception as exc:
            logger.exception(
                "Failed to delete Vault secret %r (mount=%s)", path, mount_point
            )
            if is_vault_required():
                raise VaultSecretError(
                    f"Failed to delete required Vault secret {path!r} (mount={mount_point})"
                ) from exc
            raise
        logger.info("Deleted Vault secret %r (mount=%s)", path, mount_point)

    if database_name is not None:
        _delete_one(target_path)
        return

    def _collect_leaf_paths(sub_prefix: str) -> list[str]:
        if kv_ver == 2:
            endpoint = (
                f"v1/{mount_point}/metadata/{sub_prefix}"
                if sub_prefix
                else f"v1/{mount_point}/metadata"
            )
        else:
            endpoint = (
                f"v1/{mount_point}/{sub_prefix}" if sub_prefix else f"v1/{mount_point}"
            )
        try:
            response = client.adapter.get(endpoint, params={"list": "true"})
            keys = response["data"]["keys"]
        except hvac.exceptions.InvalidPath:
            return []
        except Exception as exc:
            logger.exception("Failed to list Vault secrets under mount %r", mount_point)
            if is_vault_required():
                raise VaultSecretError(
                    f"Failed to list Vault secrets under mount {mount_point!r}"
                ) from exc
            raise

        leaves: list[str] = []
        for raw_key in keys:
            key = str(raw_key)
            if not key:
                continue
            if key.endswith("/"):
                nested = key.rstrip("/")
                if nested:
                    child_prefix = f"{sub_prefix}/{nested}" if sub_prefix else nested
                    leaves.extend(_collect_leaf_paths(child_prefix))
            else:
                leaves.append(f"{sub_prefix}/{key}" if sub_prefix else key)
        return leaves

    leaf_paths = _collect_leaf_paths(target_path)
    if not leaf_paths:
        logger.info("No secrets found under mount %r.", mount_point)
        return

    logger.info(
        "Deleting %d Vault secret(s) under mount %r", len(leaf_paths), mount_point
    )
    for leaf in leaf_paths:
        _delete_one(leaf)
