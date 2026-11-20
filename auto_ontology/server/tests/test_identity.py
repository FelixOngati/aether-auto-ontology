# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trusted gateway headers and signed workload OBO JWT verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import pytest
from fastapi import HTTPException, Request

from auto_ontology.server.identity import (
    CallerOboContext,
    bind_obo_context,
    current_obo_context,
    resolve_internal_user,
    resolve_obo_context,
    verify_workload_jwt,
)


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat/completions",
            "headers": [
                (key.lower().encode("ascii"), value.encode("ascii"))
                for key, value in headers.items()
            ],
        }
    )


def _mint_jwt(claims: dict[str, Any], secret: str = "unit-test-secret") -> str:
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


def test_forwarded_header_resolves_user(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", raising=False)
    monkeypatch.delenv("OBO_SIGNING_SECRET", raising=False)
    user_id = resolve_internal_user(
        _request({"x-auto-ontology-user-id": "user-1"}),
        required=True,
    )

    assert user_id == "user-1"


def test_missing_required_identity_is_rejected() -> None:
    with pytest.raises(HTTPException) as exc_info:
        resolve_internal_user(_request({}), required=True)

    assert exc_info.value.status_code == 401


def test_stateless_request_without_identity_remains_supported() -> None:
    assert resolve_internal_user(_request({}), required=False) is None


def test_signed_workload_jwt_resolves_obo_context_and_rejects_spoofed_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET", "unit-test-secret")
    tok = _mint_jwt(
        {
            "sub": "analyst@example.com",
            "tenant_id": "tenant-a",
            "exp": int(time.time()) + 120,
            "abac": {"region": "EMEA", "roles": ["analyst"]},
        }
    )

    obo = resolve_obo_context(
        _request({"Authorization": f"Bearer {tok}"}), required=True
    )
    assert obo is not None
    assert obo.user_id == "analyst@example.com"
    assert obo.tenant_id == "tenant-a"
    assert obo.region == "EMEA"
    assert obo.roles == ("analyst",)

    with pytest.raises(HTTPException) as spoof_exc:
        resolve_obo_context(
            _request({"x-auto-ontology-tenant-id": "tenant-b"}), required=False
        )
    assert spoof_exc.value.status_code == 401

    with pytest.raises(HTTPException) as bad_sig:
        verify_workload_jwt(tok, secret="different-secret")
    assert bad_sig.value.status_code == 401


def test_bind_obo_context_scopes_and_resets() -> None:
    assert current_obo_context() is None
    ctx = CallerOboContext(user_id="u-1", tenant_id="tenant-1", region="APAC")
    with bind_obo_context(ctx):
        assert current_obo_context() == ctx
    assert current_obo_context() is None
