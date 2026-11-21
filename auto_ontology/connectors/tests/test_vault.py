# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tenant-namespaced and fail-closed Vault secret resolution."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any
from unittest.mock import MagicMock

import hvac
import pytest
from fastapi.testclient import TestClient

from auto_ontology.connectors import vault
from auto_ontology.dal import connections as connections_dal
from auto_ontology.server.__main__ import create_app
from auto_ontology.server.connections import service as connections_service
from auto_ontology.server.identity import CallerOboContext, bind_obo_context


def _set_vault_env(
    monkeypatch: pytest.MonkeyPatch, *, include_namespace: bool = False
) -> None:
    monkeypatch.setenv("VAULT_ADDR", "https://vault.internal:8200")
    if include_namespace:
        monkeypatch.setenv("VAULT_NAMESPACE", "enterprise")
    else:
        monkeypatch.delenv("VAULT_NAMESPACE", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    monkeypatch.setenv("VAULT_ROLE_ID", "role-1")
    monkeypatch.setenv("VAULT_SECRET_ID", "secret-1")


def _mint_jwt(claims: dict[str, Any], secret: str = "http-obo-secret") -> str:
    def _enc(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    h_b64 = _enc({"alg": "HS256", "typ": "JWT"})
    p_b64 = _enc(claims)
    sig = hmac.new(
        secret.encode("utf-8"), f"{h_b64}.{p_b64}".encode("ascii"), hashlib.sha256
    ).digest()
    s_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    return f"{h_b64}.{p_b64}.{s_b64}"


def test_secret_path_namespaces_by_tenant_and_rejects_unsafe_characters() -> None:
    assert vault.secret_path("warehouse_prod") == "warehouse_prod"
    assert (
        vault.secret_path("warehouse_prod", tenant_id="tenant-a")
        == "tenant-a/warehouse_prod"
    )

    with bind_obo_context(CallerOboContext(user_id="u1", tenant_id="tenant-obo")):
        assert vault.secret_path("analytics") == "tenant-obo/analytics"

    for bad_db in (
        "",
        ".",
        "..",
        "../escape",
        "db/sub",
        "db?list=true",
        "db#frag",
        "db%2fescape",
        "db\\slash",
        "db with space",
    ):
        with pytest.raises(vault.VaultSecretError):
            vault.secret_path(bad_db, tenant_id="tenant-a")

    for bad_tenant in (".", "..", "tenant/evil", "tenant?x=1"):
        with pytest.raises(vault.VaultSecretError):
            vault.secret_path("analytics", tenant_id=bad_tenant)


def test_oss_vault_without_namespace_supports_token_and_approle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    monkeypatch.setenv("VAULT_ADDR", "http://vault:8200")
    monkeypatch.delenv("VAULT_NAMESPACE", raising=False)
    monkeypatch.delenv("VAULT_ROLE_ID", raising=False)
    monkeypatch.delenv("VAULT_SECRET_ID", raising=False)
    monkeypatch.setenv("VAULT_TOKEN", "s.oss-root-token")

    assert vault.is_vault_configured() is True

    captured_kwargs: dict[str, Any] = {}

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)
            self.auth = MagicMock()

    monkeypatch.setattr(vault.hvac, "Client", _FakeClient)
    vault.get_client()
    assert captured_kwargs == {
        "url": "http://vault:8200",
        "namespace": None,
        "token": "s.oss-root-token",
    }


def test_require_vault_fails_closed_when_env_missing_or_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    monkeypatch.setenv("VAULT_ADDR", "https://vault.internal:8200")
    monkeypatch.delenv("VAULT_NAMESPACE", raising=False)
    monkeypatch.delenv("VAULT_TOKEN", raising=False)
    monkeypatch.delenv("VAULT_ROLE_ID", raising=False)
    monkeypatch.delenv("VAULT_SECRET_ID", raising=False)

    with pytest.raises(vault.VaultSecretError, match="VAULT_ROLE_ID"):
        vault.is_vault_configured()

    with pytest.raises(vault.VaultSecretError):
        vault.read_secret("warehouse_prod", tenant_id="tenant-a")


def test_write_secret_noop_when_unconfigured_and_wraps_errors_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTO_ONTOLOGY_REQUIRE_VAULT", raising=False)
    for name in vault.VAULT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    # Unconfigured Vault is a clean no-op rather than KeyError: 'VAULT_ADDR'.
    vault.write_secret("warehouse_prod", {"database": "warehouse_prod"})

    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    _set_vault_env(monkeypatch)
    broken_client = MagicMock()
    broken_client.secrets.kv.v1.create_or_update_secret.side_effect = RuntimeError(
        "vault readonly"
    )
    monkeypatch.setattr(vault, "get_client", lambda: broken_client)

    with pytest.raises(vault.VaultSecretError, match="tenant-a/warehouse_prod"):
        vault.write_secret(
            "warehouse_prod",
            {"database": "warehouse_prod", "password": "pw"},
            tenant_id="tenant-a",
        )


def test_kv_v2_round_trip_and_bulk_delete_tenant_safety(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    monkeypatch.setenv("VAULT_KV_VERSION", "2")
    monkeypatch.setenv("VAULT_KV_MOUNT", "secret")
    _set_vault_env(monkeypatch)

    kv2_store: dict[str, dict[str, Any]] = {}
    fake_client = MagicMock()
    fake_client.secrets.kv.v2.create_or_update_secret.side_effect = (
        lambda *, path, secret, mount_point: kv2_store.__setitem__(path, dict(secret))
    )

    def _read_v2(*, path: str, mount_point: str, raise_on_deleted_version: bool):
        if path not in kv2_store:
            raise hvac.exceptions.InvalidPath("not found")
        return {"data": {"data": kv2_store[path]}}

    fake_client.secrets.kv.v2.read_secret_version.side_effect = _read_v2
    deleted_paths: list[str] = []
    fake_client.secrets.kv.v2.delete_metadata_and_all_versions.side_effect = (
        lambda *, path, mount_point: deleted_paths.append(path)
    )
    fake_client.adapter.get.side_effect = lambda endpoint, params=None: {
        "data": {"keys": ["warehouse_a", "warehouse_b"]}
    }
    monkeypatch.setattr(vault, "get_client", lambda: fake_client)

    vault.write_secret(
        "warehouse_a",
        {"database": "warehouse_a", "type": "postgres", "password": "s1"},
        tenant_id="tenant-a",
    )
    # Direct dict secret written by Aether control-plane-api (no "connection" wrapper)
    kv2_store["tenant-a/warehouse_b"] = {
        "database": "warehouse_b",
        "type": "postgres",
        "password": "s2",
    }

    assert vault.read_secret("warehouse_a", tenant_id="tenant-a") == {
        "database": "warehouse_a",
        "type": "postgres",
        "password": "s1",
    }
    assert vault.read_secret("warehouse_b", tenant_id="tenant-a") == {
        "database": "warehouse_b",
        "type": "postgres",
        "password": "s2",
    }
    assert vault.read_secret("warehouse_a", tenant_id="tenant-b") == ""

    # Unscoped bulk delete is blocked when AUTO_ONTOLOGY_REQUIRE_VAULT=1.
    with pytest.raises(vault.VaultSecretError, match="Unscoped bulk"):
        vault.delete_secrets(None)

    vault.delete_secrets(None, tenant_id="tenant-a")
    assert deleted_paths == ["tenant-a/warehouse_a", "tenant-a/warehouse_b"]


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


def test_http_connections_obo_middleware_tenant_vault_and_sanitized_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", "http-obo-secret")
    monkeypatch.setenv("AUTO_ONTOLOGY_REQUIRE_VAULT", "1")
    monkeypatch.setenv("VAULT_KV_MOUNT", "internal-secret-mount")
    _set_vault_env(monkeypatch)

    # Stub DAL catalog rows + ingestion/cache side effects so TestClient exercises
    # FastAPI middleware -> connections router -> service -> DAL -> Vault end to end.
    catalog_rows: dict[str, dict[str, Any]] = {}
    vault_kv: dict[str, str] = {}

    fake_store = MagicMock()
    fake_store.query_read.side_effect = lambda _stmt: list(catalog_rows.values())
    monkeypatch.setattr(connections_dal, "store", lambda: fake_store)

    def _fake_insert_connection(*, connection: str, database_name: str):
        if vault.is_vault_required() and connection and connection.strip():
            raise vault.VaultSecretError("Plaintext connection disabled")
        row = {
            "name": database_name,
            "connection": json.loads(connection) if connection.strip() else None,
        }
        catalog_rows[database_name] = row
        return row

    monkeypatch.setattr(
        connections_service, "insert_connection", _fake_insert_connection
    )
    monkeypatch.setattr(connections_service, "_refresh_connection_caches", lambda: None)
    monkeypatch.setattr(connections_service, "trigger_ingest", lambda _conn: None)

    fake_vault = MagicMock()
    fake_vault.secrets.kv.v1.create_or_update_secret.side_effect = (
        lambda *, path, secret, mount_point: vault_kv.__setitem__(
            path, secret["connection"]
        )
    )

    def _fake_read_v1(*, path: str, mount_point: str):
        if path not in vault_kv:
            raise hvac.exceptions.InvalidPath("not found")
        return {"data": {"connection": vault_kv[path]}}

    fake_vault.secrets.kv.v1.read_secret.side_effect = _fake_read_v1
    monkeypatch.setattr(vault, "get_client", lambda: fake_vault)

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    # Public liveness probe stays accessible, while /api/connections requires auth.
    assert client.get("/api/health/live").status_code == 200
    assert client.get("/api/connections").status_code == 401
    assert (
        client.post(
            "/api/connections",
            json={"connection": {"database": "warehouse_prod", "type": "postgres"}},
        ).status_code
        == 401
    )

    now = int(time.time())
    jwt_tenant_a = _mint_jwt(
        {
            "sub": "svc:control-plane",
            "tenant_id": "tenant-a",
            "roles": ["admin"],
            "exp": now + 300,
        }
    )
    jwt_tenant_b = _mint_jwt(
        {
            "sub": "analyst@tenant-b.example",
            "tenant_id": "tenant-b",
            "roles": ["viewer"],
            "exp": now + 300,
        }
    )

    create_resp = client.post(
        "/api/connections",
        headers={"Authorization": f"Bearer {jwt_tenant_a}"},
        json={
            "connection": {
                "database": "warehouse_prod",
                "type": "postgres",
                "host": "pg.internal",
                "port": 5432,
                "user": "readonly",
                "password": "top-secret-password",
            }
        },
    )
    assert create_resp.status_code == 201, create_resp.text
    assert "tenant-a/warehouse_prod" in vault_kv
    assert catalog_rows["warehouse_prod"]["connection"] is None
    assert "password" not in create_resp.json()["data"]["connection"]

    # Tenant A sees its connection; Tenant B does not see Tenant A's secret.
    list_a = client.get(
        "/api/connections", headers={"Authorization": f"Bearer {jwt_tenant_a}"}
    )
    assert list_a.status_code == 200
    assert list_a.json()["count"] == 1
    assert list_a.json()["data"][0]["database_name"] == "warehouse_prod"

    list_b = client.get(
        "/api/connections", headers={"Authorization": f"Bearer {jwt_tenant_b}"}
    )
    assert list_b.status_code == 200
    assert list_b.json() == {"data": [], "count": 0}

    # Simulated Vault outage returns sanitized 503 without mount/path/env details.
    fake_vault.secrets.kv.v1.read_secret.side_effect = RuntimeError("vault sealed")
    outage_resp = client.get(
        "/api/connections", headers={"Authorization": f"Bearer {jwt_tenant_a}"}
    )
    assert outage_resp.status_code == 503
    body_text = outage_resp.text
    assert outage_resp.json() == {
        "detail": "Credential vault is unavailable or misconfigured."
    }
    assert "internal-secret-mount" not in body_text
    assert "tenant-a/warehouse_prod" not in body_text
    assert "VAULT_" not in body_text
