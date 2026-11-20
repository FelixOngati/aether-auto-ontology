# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for offline perimeter kill-switches, optional RERANK, tenant cache, and catalog RLS."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.connectors import registry as connector_registry
from auto_ontology.dal.tenant_rls import (
    apply_catalog_tenant_rls,
    build_tenant_rls_statements,
)
from auto_ontology.retrieval.text_to_sql.react.config import prediction_enabled
from auto_ontology.server.identity import CallerOboContext, bind_obo_context
from auto_ontology.utils import model_config, rerank


def test_offline_perimeter_suppresses_cloud_defaults_and_blocks_cloud_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "DEFAULT_MODELS_ENDPOINT",
        "REASONING_ENDPOINT",
        "NON_REASONING_ENDPOINT",
        "EMBED_ENDPOINT",
        "RERANK_ENDPOINT",
        "BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEFAULT_MODELS_API_KEY", "nvapi-secret")
    monkeypatch.setenv("AUTO_ONTOLOGY_OFFLINE_PERIMETER", "1")

    # Built-in cloud endpoints are suppressed rather than silently targeting integrate.api.nvidia.com
    assert model_config.resolve("REASONING", "ENDPOINT") == ""
    with pytest.raises(model_config.OfflinePerimeterError):
        model_config.enforce_offline_perimeter_config()

    # Explicitly setting a public cloud inference URL raises OfflinePerimeterError
    monkeypatch.setenv("DEFAULT_MODELS_ENDPOINT", "https://integrate.api.nvidia.com/v1")
    with pytest.raises(model_config.OfflinePerimeterError):
        model_config.resolve("REASONING", "ENDPOINT")

    # Pointing at an internal Ollama / vLLM endpoint succeeds
    monkeypatch.setenv("DEFAULT_MODELS_ENDPOINT", "http://ollama:11434/v1")
    assert model_config.resolve("REASONING", "ENDPOINT") == "http://ollama:11434/v1"
    model_config.enforce_offline_perimeter_config()


def test_offline_perimeter_blocks_kumo_rfm_unless_explicit_local_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KUMO_RFM_API_URL", "https://integrate.api.nvidia.com/v1/kumo")
    monkeypatch.setenv("AUTO_ONTOLOGY_OFFLINE_PERIMETER", "1")
    monkeypatch.delenv("AUTO_ONTOLOGY_ALLOW_LOCAL_KUMO", raising=False)
    assert prediction_enabled() is False

    # Even with local override, public cloud host stays blocked
    monkeypatch.setenv("AUTO_ONTOLOGY_ALLOW_LOCAL_KUMO", "1")
    assert prediction_enabled() is False

    # Internal self-hosted endpoint with explicit local override is permitted
    monkeypatch.setenv("KUMO_RFM_API_URL", "http://kumo-nim.internal:8000")
    assert prediction_enabled() is True


def test_disable_rerank_bypasses_network_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_DISABLE_RERANK", "1")
    assert rerank.is_rerank_enabled("https://inference.example/v1/rerank") is False

    hits = [{"id": "first", "text": "alpha"}, {"id": "second", "text": "beta"}]
    with patch.object(
        rerank.httpx, "post", side_effect=AssertionError("network called")
    ):
        result = rerank.rerank_hits(
            "beta",
            hits,
            rerank_invoke_url="https://inference.example/v1/rerank",
            model_name="rerank-model",
            top_n=1,
        )
    assert len(result) == 1
    assert result[0]["id"] == "first"
    assert result[0]["_rerank_score"] == 0.0


def test_connector_registry_partitions_cache_by_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector_registry.invalidate_connectors_cache()

    def _fake_list_connections() -> list[dict[str, str]]:
        from auto_ontology.server.identity import current_obo_context

        obo = current_obo_context()
        tenant = obo.tenant_id if obo else "shared"
        return [{"db_type": "sqlite", "database": f"/tmp/{tenant}.db"}]

    created: list[str] = []

    def _fake_create(cs: str, schemas=None, table_filter=None):
        conn = MagicMock()
        conn.database_name = cs
        created.append(cs)
        return conn

    with (
        patch(
            "auto_ontology.dal.connections.list_connections",
            side_effect=_fake_list_connections,
        ),
        patch.object(
            connector_registry,
            "build_connection_string",
            side_effect=lambda c: f"sqlite:///{c['database']}",
        ),
        patch.object(connector_registry, "create_connector", side_effect=_fake_create),
    ):
        with bind_obo_context(CallerOboContext(user_id="alice", tenant_id="tenant-a")):
            conns_a1 = connector_registry.get_connectors()
            conns_a2 = connector_registry.get_connectors()
        with bind_obo_context(CallerOboContext(user_id="bob", tenant_id="tenant-b")):
            conns_b = connector_registry.get_connectors()

    assert conns_a1[0].database_name == "sqlite:////tmp/tenant-a.db"
    assert conns_a2[0] is conns_a1[0]
    assert conns_b[0].database_name == "sqlite:////tmp/tenant-b.db"
    assert len(created) == 2

    connector_registry.invalidate_connectors_cache("tenant-a")
    conns_a1[0].close.assert_called_once()
    conns_b[0].close.assert_not_called()
    connector_registry.invalidate_connectors_cache()


def test_catalog_tenant_rls_statements_and_applicator() -> None:
    stmts = build_tenant_rls_statements("catalog_table", guc_prefix="aether")
    assert len(stmts) == 5
    assert "ENABLE ROW LEVEL SECURITY" in stmts[1]
    assert "FORCE ROW LEVEL SECURITY" in stmts[2]
    assert "current_setting('aether.tenant_id', true)" in stmts[4]
    assert "IS NOT NULL" in stmts[4]

    with pytest.raises(ValueError, match="Unsafe SQL table identifier"):
        build_tenant_rls_statements("catalog_table; DROP TABLE users")

    conn = MagicMock()
    count = apply_catalog_tenant_rls(conn, tables=("connections", "catalog_table"))
    assert count == 2
    assert conn.execute.call_count == 10
