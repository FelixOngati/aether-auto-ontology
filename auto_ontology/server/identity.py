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
INTERNAL_GATEWAY_SECRET_HEADER = "x-auto-ontology-gateway-secret"

DEFAULT_CLOCK_SKEW_S = 30
DEFAULT_MAX_TTL_S = 86_400


@dataclass(frozen=True)
class CallerOboContext:
    """Verified On-Behalf-Of caller identity and ABAC attributes."""

    user_id: str
    tenant_id: str = ""
    region: str = ""
    roles: tuple[str, ...] = ()
    workflow_run_id: str = ""
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


def is_workload_jwt_only() -> bool:
    """Return True when only signed workload OBO JWTs are accepted."""
    raw = (os.environ.get("AUTO_ONTOLOGY_WORKLOAD_JWT_ONLY") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _gateway_secret() -> str:
    return (os.environ.get("AUTO_ONTOLOGY_INTERNAL_GATEWAY_SECRET") or "").strip()


def _has_verified_gateway_secret(request: Request) -> bool:
    expected = _gateway_secret()
    if not expected:
        return False
    provided = (request.headers.get(INTERNAL_GATEWAY_SECRET_HEADER) or "").strip()
    if not provided:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def is_workload_auth_required(request: Request) -> bool:
    """Return True when FastAPI HTTP middleware must require a verified caller."""
    if is_workload_jwt_only():
        return True
    if _workload_jwt_secret() and not _has_verified_gateway_secret(request):
        return True
    return False


def _positive_int_env(name: str, default: int, *, allow_zero: bool = False) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if val < 0 or (val == 0 and not allow_zero):
        return default
    return val


def _parse_issuer_allowlist(raw: str | None) -> set[str]:
    source = (
        raw
        if raw is not None
        else (os.environ.get("AUTO_ONTOLOGY_WORKLOAD_JWT_ISSUER") or "")
    )
    return {item.strip() for item in source.split(",") if item.strip()}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def verify_workload_jwt(
    token: str,
    *,
    secret: str | None = None,
    issuer: str | None = None,
    audience: str | None = None,
) -> CallerOboContext:
    """Verify an HS256 workload OBO JWT and return its :class:`CallerOboContext`.

    Raises :class:`HTTPException` (401 or 403) when the token is malformed,
    unsigned, expired, not-yet-valid, exceeds max TTL, fails issuer/audience
    checks, or lacks required ``sub`` / ``tenant_id`` claims.
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
    skew = _positive_int_env(
        "AUTO_ONTOLOGY_WORKLOAD_JWT_CLOCK_SKEW_S",
        DEFAULT_CLOCK_SKEW_S,
        allow_zero=True,
    )
    max_ttl = _positive_int_env(
        "AUTO_ONTOLOGY_WORKLOAD_JWT_MAX_TTL_S", DEFAULT_MAX_TTL_S
    )

    exp = payload.get("exp")
    if not _is_number(exp) or int(exp) + skew <= now:
        raise HTTPException(status_code=401, detail="Expired workload JWT")
    if int(exp) - now > max_ttl + skew:
        raise HTTPException(
            status_code=401, detail="Workload JWT lifetime exceeds maximum allowed TTL"
        )

    nbf = payload.get("nbf")
    if nbf is not None and (not _is_number(nbf) or int(nbf) - skew > now):
        raise HTTPException(status_code=401, detail="Workload JWT is not yet valid")

    iat = payload.get("iat")
    if iat is not None:
        if not _is_number(iat) or int(iat) - skew > now:
            raise HTTPException(status_code=401, detail="Invalid workload JWT iat")
        if int(exp) - int(iat) > max_ttl:
            raise HTTPException(
                status_code=401,
                detail="Workload JWT lifetime exceeds maximum allowed TTL",
            )

    allowed_issuers = _parse_issuer_allowlist(issuer)
    if allowed_issuers:
        token_issuer = str(payload.get("iss") or "").strip()
        if token_issuer not in allowed_issuers:
            raise HTTPException(status_code=401, detail="Untrusted workload JWT issuer")

    expected_audience = (
        audience
        if audience is not None
        else (os.environ.get("AUTO_ONTOLOGY_WORKLOAD_JWT_AUDIENCE") or "")
    ).strip()
    if expected_audience:
        raw_aud = payload.get("aud")
        aud_values = raw_aud if isinstance(raw_aud, list) else [raw_aud]
        if expected_audience not in {str(a) for a in aud_values if a is not None}:
            raise HTTPException(status_code=401, detail="Invalid workload JWT audience")

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
    raw_roles = abac.get("roles") if "roles" in abac else payload.get("roles")
    if raw_roles is not None and not isinstance(raw_roles, (list, tuple)):
        raise HTTPException(status_code=403, detail="Invalid workload roles claim")
    roles = (
        tuple(str(r).strip() for r in raw_roles if str(r).strip())
        if isinstance(raw_roles, (list, tuple))
        else ()
    )
    workflow_run_id = str(
        abac.get("workflow_run_id") or payload.get("workflow_run_id") or ""
    ).strip()

    if region and "region" not in abac:
        abac["region"] = region
    if roles and "roles" not in abac:
        abac["roles"] = list(roles)
    if workflow_run_id and "workflow_run_id" not in abac:
        abac["workflow_run_id"] = workflow_run_id

    return CallerOboContext(
        user_id=subject,
        tenant_id=tenant_id,
        region=region,
        roles=roles,
        workflow_run_id=workflow_run_id,
        abac=abac,
    )


def _has_any_internal_identity_header(request: Request) -> bool:
    return any(
        (request.headers.get(name) or "").strip()
        for name in (
            INTERNAL_USER_HEADER,
            INTERNAL_TENANT_HEADER,
            INTERNAL_ABAC_HEADER,
        )
    )


def resolve_obo_context(
    request: Request, *, required: bool = False
) -> CallerOboContext | None:
    """Resolve verified :class:`CallerOboContext` from bearer JWT or gateway headers.

    When ``AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET`` is configured, bearer tokens are
    verified cryptographically and any unverified internal identity headers
    without a verified bearer token (or verified gateway secret) are rejected
    fail-closed.
    """
    secret = _workload_jwt_secret()
    auth_header = (request.headers.get("authorization") or "").strip()
    scheme, _, bearer = auth_header.partition(" ")
    bearer = bearer.strip()

    if is_workload_jwt_only() and not secret:
        raise HTTPException(
            status_code=401,
            detail="Workload JWT verification secret is not configured",
        )

    if secret:
        if auth_header:
            if scheme.lower() != "bearer" or not bearer or bearer.count(".") != 2:
                raise HTTPException(
                    status_code=401, detail="Malformed workload bearer token"
                )
            return verify_workload_jwt(bearer, secret=secret)

        if not _has_verified_gateway_secret(request):
            if _has_any_internal_identity_header(request):
                raise HTTPException(
                    status_code=401,
                    detail="Signed workload bearer token required for internal identity headers",
                )
            if required:
                raise HTTPException(
                    status_code=401, detail="Authenticated workload token required"
                )
            return None

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

    if user_id or tenant_id or abac:
        region = str(abac.get("region") or "").strip()
        raw_roles = abac.get("roles") or []
        roles = (
            tuple(str(r).strip() for r in raw_roles if str(r).strip())
            if isinstance(raw_roles, list)
            else ()
        )
        workflow_run_id = str(abac.get("workflow_run_id") or "").strip()
        return CallerOboContext(
            user_id=user_id or "anonymous",
            tenant_id=tenant_id,
            region=region,
            roles=roles,
            workflow_run_id=workflow_run_id,
            abac=abac,
        )

    if required:
        raise HTTPException(
            status_code=401, detail="Authenticated conversation required"
        )
    return None


def resolve_internal_user(request: Request, *, required: bool) -> str | None:
    """Return the Auto Ontology user id supplied by gateway headers or workload JWT."""
    ctx = current_obo_context()
    obo = ctx if ctx is not None else resolve_obo_context(request, required=False)
    if obo is not None and obo.user_id and obo.user_id != "anonymous":
        return obo.user_id

    if required:
        raise HTTPException(
            status_code=401, detail="Authenticated conversation required"
        )
    return None


__all__ = [
    "CallerOboContext",
    "DEFAULT_CLOCK_SKEW_S",
    "DEFAULT_MAX_TTL_S",
    "INTERNAL_ABAC_HEADER",
    "INTERNAL_GATEWAY_SECRET_HEADER",
    "INTERNAL_TENANT_HEADER",
    "INTERNAL_USER_HEADER",
    "bind_obo_context",
    "current_obo_context",
    "is_workload_auth_required",
    "is_workload_jwt_only",
    "resolve_internal_user",
    "resolve_obo_context",
    "verify_workload_jwt",
]
