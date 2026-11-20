# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for headless HS256 workload On-Behalf-Of (OBO) authentication."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
from typing import Any

import httpx
import pytest
from fastmcp.server.auth.auth import AccessToken

from auto_ontology_mcp.auto_ontology_auth import (
    AutoOntologyTokenVerifier,
    CompositeTokenVerifier,
    WorkloadJwtVerifier,
)
from auto_ontology_mcp.config import ConfigError, load_settings
from auto_ontology_mcp.server import (
    INTERNAL_ABAC_HEADER,
    INTERNAL_TENANT_HEADER,
    INTERNAL_USER_HEADER,
    CallerAuth,
)

SECRET = "test-workload-obo-secret-32-bytes"


def _mint_jwt(claims: dict[str, Any], secret: str = SECRET, alg: str = "HS256") -> str:
    def _enc(obj: dict[str, Any]) -> str:
        raw = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    header_b64 = _enc({"alg": alg, "typ": "JWT"})
    payload_b64 = _enc(claims)
    sig = hmac.new(
        secret.encode("utf-8"),
        f"{header_b64}.{payload_b64}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode("ascii").rstrip("=")
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def test_valid_workload_jwt_resolves_subject_tenant_and_abac() -> None:
    verifier = WorkloadJwtVerifier(
        SECRET, issuer="mcp-host", audience="auto-ontology-mcp"
    )
    token = _mint_jwt(
        {
            "sub": "analyst@example.com",
            "tenant_id": "tenant-eu",
            "iss": "mcp-host",
            "aud": "auto-ontology-mcp",
            "exp": int(time.time()) + 300,
            "scope": "ontology:read",
            "abac": {"region": "EMEA", "roles": ["finance_analyst"]},
        }
    )

    access = asyncio.run(verifier.verify_token(token))
    assert access is not None
    assert access.subject == "analyst@example.com"
    assert access.scopes == ["ontology:read"]
    assert access.claims["tenant_id"] == "tenant-eu"
    assert access.claims["abac"] == {"region": "EMEA", "roles": ["finance_analyst"]}


def test_forged_expired_or_tenantless_workload_jwt_fails_closed() -> None:
    verifier = WorkloadJwtVerifier(SECRET)
    now = int(time.time())

    forged = _mint_jwt(
        {"sub": "u1", "tenant_id": "t1", "exp": now + 60}, secret="wrong-secret"
    )
    expired = _mint_jwt({"sub": "u1", "tenant_id": "t1", "exp": now - 5})
    no_tenant = _mint_jwt({"sub": "u1", "exp": now + 60})

    assert asyncio.run(verifier.verify_token(forged)) is None
    assert asyncio.run(verifier.verify_token(expired)) is None
    assert asyncio.run(verifier.verify_token(no_tenant)) is None


def test_composite_verifier_accepts_both_workload_jwt_and_opaque_oauth() -> None:
    workload = WorkloadJwtVerifier(SECRET)
    remote = AutoOntologyTokenVerifier(
        "https://auto_ontology.example",
        timeout_s=5.0,
        transport=httpx.MockTransport(
            lambda _req: httpx.Response(200, json={"sub": "browser-user"})
        ),
    )
    composite = CompositeTokenVerifier(workload, remote)

    jwt_tok = _mint_jwt(
        {"sub": "svc-worker", "tenant_id": "tenant-a", "exp": int(time.time()) + 120}
    )
    jwt_access = asyncio.run(composite.verify_token(jwt_tok))
    assert jwt_access is not None and jwt_access.subject == "svc-worker"

    opaque_access = asyncio.run(composite.verify_token("opaque-session-token"))
    assert opaque_access is not None and opaque_access.subject == "browser-user"


def test_caller_auth_strips_spoofed_headers_and_forwards_verified_obo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "auto_ontology_mcp.server.auto_ontology_access_token", lambda: "verified-jwt"
    )
    monkeypatch.setattr(
        "auto_ontology_mcp.server.auto_ontology_obo_claims",
        lambda: {
            "sub": "alice@example.com",
            "tenant_id": "tenant-a",
            "abac": {"region": "US"},
        },
    )

    req = httpx.Request(
        "GET",
        "http://localhost:3000/api/terms",
        headers={
            "x-api-key": "spoofed-key",
            INTERNAL_TENANT_HEADER: "spoofed-tenant",
        },
    )
    flow = CallerAuth().auth_flow(req)
    authed = next(flow)

    assert "x-api-key" not in authed.headers
    assert authed.headers["authorization"] == "Bearer verified-jwt"
    assert authed.headers[INTERNAL_USER_HEADER] == "alice@example.com"
    assert authed.headers[INTERNAL_TENANT_HEADER] == "tenant-a"
    assert json.loads(authed.headers[INTERNAL_ABAC_HEADER]) == {"region": "US"}


def test_workload_jwt_only_requires_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_WORKLOAD_JWT_ONLY", "1")
    monkeypatch.delenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", raising=False)
    monkeypatch.delenv("OBO_SIGNING_SECRET", raising=False)

    with pytest.raises(ConfigError, match="AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET"):
        load_settings()

    monkeypatch.setenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", SECRET)
    settings = load_settings()
    assert settings.workload_jwt_only is True
    assert isinstance(
        AccessToken(token="t", client_id="c", scopes=[], subject="s"), AccessToken
    )
