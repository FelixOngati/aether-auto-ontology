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
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.auth.auth import AccessToken

from auto_ontology_mcp import server as mcp_server
from auto_ontology_mcp.auto_ontology_auth import (
    AutoOntologyTokenVerifier,
    CompositeTokenVerifier,
    WorkloadJwtVerifier,
)
from auto_ontology_mcp.config import (
    DEFAULT_SPEC_PATH,
    DEFAULT_WORKLOAD_API_URL,
    ConfigError,
    Settings,
    load_settings,
)
from auto_ontology_mcp.server import (
    INTERNAL_ABAC_HEADER,
    INTERNAL_TENANT_HEADER,
    INTERNAL_USER_HEADER,
    CallerAuth,
    build_server,
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


def test_top_level_claims_multi_issuer_leeway_and_max_ttl() -> None:
    verifier = WorkloadJwtVerifier(
        SECRET,
        issuer="aether-mcp-connectors, aether-control-plane",
        audience="auto-ontology-mcp",
        clock_skew_s=30,
        max_ttl_s=600,
    )
    now = int(time.time())
    admin_tok = _mint_jwt(
        {
            "sub": "svc:control-plane",
            "tenant_id": "tenant-cp",
            "iss": "aether-control-plane",
            "aud": "auto-ontology-mcp",
            "iat": now - 5,
            "nbf": now + 10,  # within 30s clock-skew leeway
            "exp": now + 300,
            "region": "GLOBAL",
            "roles": ["admin"],
            "workflow_run_id": "wf-run-42",
        }
    )
    access = asyncio.run(verifier.verify_token(admin_tok))
    assert access is not None
    assert access.claims["region"] == "GLOBAL"
    assert access.claims["roles"] == ["admin"]
    assert access.claims["workflow_run_id"] == "wf-run-42"
    assert access.claims["abac"] == {
        "region": "GLOBAL",
        "roles": ["admin"],
        "workflow_run_id": "wf-run-42",
    }

    over_ttl = _mint_jwt(
        {
            "sub": "svc:control-plane",
            "tenant_id": "tenant-cp",
            "iss": "aether-mcp-connectors",
            "aud": "auto-ontology-mcp",
            "iat": now,
            "exp": now + 3600,
        }
    )
    assert asyncio.run(verifier.verify_token(over_ttl)) is None


def test_forged_expired_or_tenantless_workload_jwt_fails_closed() -> None:
    verifier = WorkloadJwtVerifier(SECRET, clock_skew_s=0)
    now = int(time.time())

    forged = _mint_jwt(
        {"sub": "u1", "tenant_id": "t1", "exp": now + 60}, secret="wrong-secret"
    )
    expired = _mint_jwt({"sub": "u1", "tenant_id": "t1", "exp": now - 5})
    no_tenant = _mint_jwt({"sub": "u1", "exp": now + 60})

    assert asyncio.run(verifier.verify_token(forged)) is None
    assert asyncio.run(verifier.verify_token(expired)) is None
    assert asyncio.run(verifier.verify_token(no_tenant)) is None


def test_composite_verifier_never_falls_through_on_three_segment_jwts() -> None:
    remote_calls: list[str] = []

    def _userinfo_handler(req: httpx.Request) -> httpx.Response:
        remote_calls.append(req.headers.get("authorization", ""))
        return httpx.Response(200, json={"sub": "browser-user"})

    workload = WorkloadJwtVerifier(SECRET, clock_skew_s=0)
    remote = AutoOntologyTokenVerifier(
        "https://auto_ontology.example",
        timeout_s=5.0,
        transport=httpx.MockTransport(_userinfo_handler),
    )
    composite = CompositeTokenVerifier(workload, remote)
    now = int(time.time())

    valid_jwt = _mint_jwt(
        {"sub": "svc-worker", "tenant_id": "tenant-a", "exp": now + 120}
    )
    jwt_access = asyncio.run(composite.verify_token(valid_jwt))
    assert jwt_access is not None and jwt_access.subject == "svc-worker"

    # Forged, expired, tenantless SSO, and malformed 3-segment tokens MUST NOT
    # be leaked outbound to USERINFO_PATH or accepted without tenant_id.
    forged_jwt = _mint_jwt(
        {"sub": "svc-worker", "tenant_id": "tenant-a", "exp": now + 120},
        secret="wrong-secret",
    )
    expired_jwt = _mint_jwt(
        {"sub": "svc-worker", "tenant_id": "tenant-a", "exp": now - 60}
    )
    sso_jwt_without_tenant = _mint_jwt(
        {"sub": "sso-user", "exp": now + 120}, alg="RS256"
    )
    for bad in (forged_jwt, expired_jwt, sso_jwt_without_tenant, "bad.jwt.segments"):
        assert asyncio.run(composite.verify_token(bad)) is None
    assert remote_calls == []

    opaque_access = asyncio.run(composite.verify_token("opaque-session-token"))
    assert opaque_access is not None and opaque_access.subject == "browser-user"
    assert remote_calls == ["Bearer opaque-session-token"]


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
        "http://localhost:3001/api/terms",
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


def test_workload_jwt_only_requires_secret_and_defaults_to_backend_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AUTO_ONTOLOGY_API_URL", raising=False)
    monkeypatch.setenv("AUTO_ONTOLOGY_WORKLOAD_JWT_ONLY", "1")
    monkeypatch.delenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", raising=False)
    monkeypatch.delenv("OBO_SIGNING_SECRET", raising=False)

    with pytest.raises(ConfigError, match="AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET"):
        load_settings()

    monkeypatch.setenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", SECRET)
    settings = load_settings()
    assert settings.workload_jwt_only is True
    assert settings.api_url == DEFAULT_WORKLOAD_API_URL
    assert isinstance(
        AccessToken(token="t", client_id="c", scopes=[], subject="s"), AccessToken
    )


def test_end_to_end_fastmcp_http_bearer_jwt_forwards_obo_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_upstream: list[httpx.Request] = []

    def _upstream_handler(request: httpx.Request) -> httpx.Response:
        captured_upstream.append(request)
        return httpx.Response(
            200,
            json={
                "terms": [],
                "total": 0,
                "column_attribute_counts": [],
                "sql_attribute_counts": [],
                "related_counts": [],
            },
        )

    settings = Settings(
        api_url="http://auto-ontology-api:3001",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=10.0,
        chat_timeout_s=60.0,
        public_url="http://127.0.0.1:3003",
        workload_jwt_secret=SECRET,
        workload_jwt_issuer="aether-mcp-connectors,aether-control-plane",
        workload_jwt_audience="auto-ontology-mcp",
        workload_jwt_only=True,
    )

    monkeypatch.setattr(
        mcp_server,
        "build_client",
        lambda s: httpx.AsyncClient(
            base_url=s.api_url,
            auth=CallerAuth(),
            timeout=s.timeout_s,
            transport=httpx.MockTransport(_upstream_handler),
        ),
    )

    mcp, upstream_client = build_server(settings)
    asgi_app = mcp.http_app(path="/mcp")
    valid_jwt = _mint_jwt(
        {
            "sub": "analyst@tenant-eu.example",
            "tenant_id": "tenant-eu",
            "iss": "aether-mcp-connectors",
            "aud": "auto-ontology-mcp",
            "exp": int(time.time()) + 300,
            "region": "EMEA",
            "roles": ["analyst"],
            "workflow_run_id": "wf-e2e-1",
        }
    )

    async def _exercise() -> None:
        try:
            async with asgi_app.router.lifespan_context(asgi_app):
                transport = StreamableHttpTransport(
                    url="http://mcp.local/mcp",
                    headers={
                        "Authorization": f"Bearer {valid_jwt}",
                        "x-api-key": "spoofed-api-key",
                        INTERNAL_TENANT_HEADER: "spoofed-tenant",
                    },
                    httpx_client_factory=lambda **kwargs: httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=asgi_app),
                        **kwargs,
                    ),
                )
                async with Client(transport) as session:
                    await session.call_tool("search_terms", {})

                # Unauthenticated / forged JWT must be rejected at the MCP HTTP boundary (401).
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=asgi_app),
                    base_url="http://mcp.local",
                ) as raw_http:
                    unauth = await raw_http.post("/mcp", json={"jsonrpc": "2.0"})
                    assert unauth.status_code == 401
        finally:
            await upstream_client.aclose()

    asyncio.run(_exercise())

    assert len(captured_upstream) == 1
    sent = captured_upstream[0]
    assert sent.headers["authorization"] == f"Bearer {valid_jwt}"
    assert sent.headers[INTERNAL_USER_HEADER] == "analyst@tenant-eu.example"
    assert sent.headers[INTERNAL_TENANT_HEADER] == "tenant-eu"
    assert "x-api-key" not in sent.headers
    assert json.loads(sent.headers[INTERNAL_ABAC_HEADER]) == {
        "region": "EMEA",
        "roles": ["analyst"],
        "workflow_run_id": "wf-e2e-1",
    }
