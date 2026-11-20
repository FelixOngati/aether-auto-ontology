# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tenant-namespaced and fail-closed Vault secret resolution."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from auto_ontology.connectors import vault
from auto_ontology.dal import connections as connections_dal
from auto_ontology.server.identity import CallerOboContext, bind_obo_context


def _set_vault_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VAULT_ADDR", "https://vault.internal:8200")
    monkeypatch.setenv("VAULT_NAMESPACE", "enterprise")
    monkeypatch.setenv("VAULT_ROLE_ID", "role-1")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-1")


def test_secret_path_namespaces_by_tenant_and_rejects_traversal() -> None:
    assert vault.secret_path("warehouse_prod") == "warehouse_prod"
    assert (
        vault.secret_path("warehouse_prod", tenant_id="tenant-a")
        == "tenant-a/warehouse_prod"
    )

    with bind_obo_context(CallerOboContext(user_id="u1", tenant_id="tenant-obo")):
        assert vault.secret_path("analytics") == "tenant-obo/analytics"

    with pytest.raises(vault.VaultSecretError):
        vault.secret_path("../escape", tenant_id="tenant-a")
    with pytest.raises(vault.VaultSecretError):
        vault.secret_path("analytics", tenant_id="tenant/evil")


def test_require_vault_fails_closed_when_env_missing_or_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.internal:8200")
    monkeypatch.delenv("VAULT_NAMESPACE", raising=False)
    monkeypatch.delenv("VAULT_ROLE_ID", raising=False)
    monkeypatch.delenv("VAULT_SECRET_ID", raising=False)

    with pytest.raises(vault.VaultSecretError, match="VAULT_NAMESPACE"):
        vault.is_vault_configured()

    with pytest.raises(vault.VaultSecretError):
        vault.read_secret("warehouse_prod", tenant_id="tenant-a")


def test_require_vault_fails_closed_on_read_error_and_blocks_plaintext_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    _set_vault_env(monkeypatch)

    broken_client = MagicMock()
    broken_client.secrets.kv.v1.read_secret.side_effect = RuntimeError("vault sealed")
    monkeypatch.setattr(vault, "get_client", lambda: broken_client)

    with pytest.raises(vault.VaultSecretError, match="tenant-a/warehouse_prod"):
        vault.read_secret("warehouse_prod", tenant_id="tenant-a")

    with pytest.raises(vault.VaultSecretError, match="Plaintext connection"):
        connections_dal.insert_connection(
            connection='{"password":"secret"}',
            database_name="warehouse_prod",
        )
