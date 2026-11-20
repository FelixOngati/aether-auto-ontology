# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read identities forwarded by the trusted gateway or signed workload OBO JWTs.

Lives at the top of ``auto_ontology.server`` rather than under ``chat``, where it
started: identity and On-Behalf-Of (OBO) tenant/ABAC context apply across routers
and connector execution.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from fastapi import HTTPException, Request

INTERNAL_USER_HEADER = "x-auto-ontology-user-id"
INTERNAL_TENANT_HEADER = "x-auto-ontology-tenant-id"
INTERNAL_ABAC_HEADER = "x-auto-ontology-abac"


@dataclass(frozen=True)
class CallerOboContext:
    """Verified On-Behalf-Of caller identity and ABAC attributes."""

    user_id: str
    tenant_id: str = ""
    region: str = ""
    roles: tuple[str, ...] = ()
    abac: dict[str, Any] = field(default_factory=dict)


_CURRENT_OBO_CONTEXT: contextvars.ContextVar[CallerOboContext | None] = (
    contextvars.ContextVar("auto_ontology_obo_context", default=None)
)


def current_obo_context() -> CallerOboContext | None:
    """Return the active :class:`CallerOboContext` bound to the current task/thread."""
    return _CURRENT_OBO_CONTEXT.get()


@contextlib.contextmanager
def bind_obo_context(context: CallerOboContext | None) -> Iterator[None]:
    """Bind *context* for the duration of a ``with`` block."""
    token = _CURRENT_OBO_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_OBO_CONTEXT.reset(token)


def _workload_jwt_secret() -> str:
    return (
        os.environ.get("AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET")
        or os.environ.get("OBO_SIGNING_SECRET")
        or ""
    ).strip()


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def verify_workload_jwt(token: str, *, secret: str | None = None) -> CallerOboContext:
    """Verify an HS256 workload OBO JWT and return its :class:`CallerOboContext`.

    Raises :class:`HTTPException` (401 or 403) when the token is malformed,
    unsigned, expired, or missing required ``sub`` / ``tenant_id`` claims.
    """
    effective_secret = (
        secret if secret is not None else _workload_jwt_secret()
    ).strip()
    if not effective_secret:
        raise HTTPException(
            status_code=401, detail="Workload JWT verification secret is not configured"
        )

    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Malformed workload bearer token")

    header_b64, payload_b64, sig_b64 = parts
    try:
        header = json.loads(_b64url_decode(header_b64))
        payload = json.loads(_b64url_decode(payload_b64))
        provided_sig = _b64url_decode(sig_b64)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=401, detail="Malformed workload bearer token"
        ) from exc

    if not isinstance(header, dict) or header.get("alg") != "HS256":
        raise HTTPException(
            status_code=401, detail="Unsupported workload JWT algorithm"
        )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="Invalid workload JWT claims")

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected_sig = hmac.new(
        effective_secret.encode("utf-8"), signing_input, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise HTTPException(status_code=401, detail="Invalid workload JWT signature")

    now = int(time.time())
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)) or int(exp) <= now:
        raise HTTPException(status_code=401, detail="Expired workload JWT")

    subject = str(payload.get("sub") or "").strip()
    tenant_id = str(payload.get("tenant_id") or "").strip()
    if not subject or not tenant_id:
        raise HTTPException(
            status_code=403,
            detail="Workload JWT requires non-empty sub and tenant_id claims",
        )

    raw_abac = payload.get("abac")
    if raw_abac is not None and not isinstance(raw_abac, dict):
        raise HTTPException(status_code=403, detail="Invalid workload ABAC claim")
    abac = dict(raw_abac) if isinstance(raw_abac, dict) else {}
    region = str(abac.get("region") or payload.get("region") or "").strip()
    raw_roles = abac.get("roles") or payload.get("roles") or []
    roles = (
        tuple(str(r) for r in raw_roles if str(r).strip())
        if isinstance(raw_roles, (list, tuple))
        else ()
    )
    return CallerOboContext(
        user_id=subject,
        tenant_id=tenant_id,
        region=region,
        roles=roles,
        abac=abac,
    )


def resolve_obo_context(
    request: Request, *, required: bool = False
) -> CallerOboContext | None:
    """Resolve verified :class:`CallerOboContext` from bearer JWT or gateway headers.

    When ``AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET`` is configured, bearer tokens are
    verified cryptographically and unverified ``x-auto-ontology-tenant-id``
    headers without a matching bearer token are rejected (fail closed).
    """
    secret = _workload_jwt_secret()
    auth_header = (request.headers.get("authorization") or "").strip()
    scheme, _, bearer = auth_header.partition(" ")
    bearer = bearer.strip()

    if secret:
        if scheme.lower() == "bearer" and bearer and bearer.count(".") == 2:
            return verify_workload_jwt(bearer, secret=secret)
        if request.headers.get(INTERNAL_TENANT_HEADER):
            raise HTTPException(
                status_code=401,
                detail="Signed workload bearer token required for tenant context",
            )
        if required:
            raise HTTPException(
                status_code=401, detail="Authenticated workload token required"
            )

    user_id = (request.headers.get(INTERNAL_USER_HEADER) or "").strip()
    tenant_id = (request.headers.get(INTERNAL_TENANT_HEADER) or "").strip()
    raw_abac = (request.headers.get(INTERNAL_ABAC_HEADER) or "").strip()
    abac: dict[str, Any] = {}
    if raw_abac:
        try:
            decoded = json.loads(raw_abac)
            if isinstance(decoded, dict):
                abac = decoded
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Malformed x-auto-ontology-abac header"
            ) from exc

    if user_id or tenant_id:
        region = str(abac.get("region") or "").strip()
        raw_roles = abac.get("roles") or []
        roles = (
            tuple(str(r) for r in raw_roles if str(r).strip())
            if isinstance(raw_roles, list)
            else ()
        )
        return CallerOboContext(
            user_id=user_id or "anonymous",
            tenant_id=tenant_id,
            region=region,
            roles=roles,
            abac=abac,
        )

    if required:
        raise HTTPException(
            status_code=401, detail="Authenticated conversation required"
        )
    return None


def resolve_internal_user(request: Request, *, required: bool) -> str | None:
    """Return the Auto Ontology user id supplied by gateway headers or workload JWT."""
    obo = resolve_obo_context(request, required=False)
    if obo is not None and obo.user_id and obo.user_id != "anonymous":
        return obo.user_id

    user_id = (request.headers.get(INTERNAL_USER_HEADER) or "").strip()
    if user_id:
        return user_id

    if required:
        raise HTTPException(
            status_code=401, detail="Authenticated conversation required"
        )
    return None


__all__ = [
    "CallerOboContext",
    "INTERNAL_ABAC_HEADER",
    "INTERNAL_TENANT_HEADER",
    "INTERNAL_USER_HEADER",
    "bind_obo_context",
    "current_obo_context",
    "resolve_internal_user",
    "resolve_obo_context",
    "verify_workload_jwt",
]
