# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sign-in, where Auto Ontology itself is the authorization server.

This server could instead sit between the client and the deployment's identity
provider. That works, but it has to be told a client id, a client secret, and a
redirect URI that someone registered with that provider first — a ticket for
every deployment, and a secret to hold and rotate. Auto Ontology already knows how to sign
a person in, including through SSO, so it does, and this server carries no
credentials at all.

What that buys, concretely:

* Nothing to configure beyond pointing at Auto Ontology. Clients register themselves
  through dynamic client registration, which Auto Ontology supports and most corporate
  providers do not, so no redirect URI is ever registered by hand.
* No secret on disk here, and no secret to rotate.
* The registrations and grants live in Auto Ontology's database, so replicas share them
  and a restart does not sign anyone out.
* Revocation is Auto Ontology's: deleting the grant, banning the user, or changing their
  role takes effect on the next call.

The token a caller presents is one Auto Ontology minted, so verifying it is a question
only Auto Ontology can answer — hence the round trip below. There is no local signature to
check: these tokens are opaque by design, which also means they leak nothing if
logged. The same token is then forwarded upstream unchanged, where
``frontend/auth/resolve-user.ts`` resolves it to its owner.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

import httpx
from fastmcp.server.auth.auth import AccessToken, RemoteAuthProvider, TokenVerifier
from fastmcp.utilities.logging import get_logger
from pydantic import AnyHttpUrl

from auto_ontology_mcp.config import Settings

logger = get_logger(__name__)

# Better Auth 1.7's OAuth Provider validates a bearer at UserInfo without
# requiring this resource server to hold client credentials. Auto Ontology adds the
# issuing client and granted scope as private claims for FastMCP's AccessToken.
USERINFO_PATH = "/api/auth/oauth2/userinfo"
CLIENT_ID_CLAIM = "urn:auto-ontology:oauth:client_id"
SCOPE_CLAIM = "urn:auto-ontology:oauth:scope"


class AutoOntologyTokenVerifier(TokenVerifier):
    """Validates a token by asking Auto Ontology whether it still stands.

    A rejected token returns ``None`` so the caller is told to sign in again.
    A Auto Ontology that cannot be reached raises instead: an outage is not an expired
    token, and answering "unauthenticated" would send everyone through a
    sign-in that cannot succeed either.
    """

    def __init__(
        self,
        api_url: str,
        timeout_s: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        super().__init__()
        self._api_url = api_url
        self._timeout_s = timeout_s
        # A short-lived client of its own rather than the one the tools share:
        # that one rewrites the Authorization header to the caller's credential,
        # which is the very thing being checked here. `transport` is a seam for
        # tests.
        self._transport = transport

    async def verify_token(self, token: str) -> AccessToken | None:
        async with httpx.AsyncClient(
            base_url=self._api_url,
            timeout=self._timeout_s,
            transport=self._transport,
        ) as client:
            response = await client.get(
                USERINFO_PATH, headers={"Authorization": f"Bearer {token}"}
            )

        if response.status_code >= 500:
            response.raise_for_status()

        # Unknown, expired, or revoked tokens receive a 4xx response. Anything
        # unparseable is treated the same way: a body we cannot read is not an
        # identity, and raising here would turn a refusal into a server error.
        try:
            user_info = response.json() if response.status_code == 200 else None
        except ValueError:
            user_info = None

        if not isinstance(user_info, dict) or not user_info.get("sub"):
            logger.debug("Auto Ontology rejected an access token")
            return None

        # Space-separated per RFC 6749, and absent when the grant carries none.
        scopes = str(user_info.get(SCOPE_CLAIM) or "").split()

        return AccessToken(
            token=token,
            # The client that was granted the token, which FastMCP uses to
            # attribute the session. Registration guarantees it.
            client_id=str(user_info.get(CLIENT_ID_CLAIM) or "unknown"),
            scopes=scopes,
            subject=str(user_info["sub"]),
        )


def _b64url_decode(segment: str) -> bytes:
    padded = segment + "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


class WorkloadJwtVerifier(TokenVerifier):
    """Verify headless HS256 workload On-Behalf-Of (OBO) JWTs without browser cookies.

    Workflow orchestrators resuming long-running jobs cannot complete an
    interactive browser OAuth redirect. When ``workload_jwt_secret`` is
    configured, callers present a short-lived HS256 JWT carrying ``sub``,
    ``tenant_id``, ``exp``, and optional ``abac`` attributes. Invalid, expired,
    or incomplete tokens fail closed (``None`` -> HTTP 401).
    """

    def __init__(
        self,
        secret: str,
        *,
        issuer: str = "",
        audience: str = "",
    ) -> None:
        super().__init__()
        self._secret = secret.encode("utf-8")
        self._issuer = issuer
        self._audience = audience

    async def verify_token(self, token: str) -> AccessToken | None:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header_b64, payload_b64, sig_b64 = parts
        try:
            header = json.loads(_b64url_decode(header_b64))
            payload = json.loads(_b64url_decode(payload_b64))
            provided_sig = _b64url_decode(sig_b64)
        except (ValueError, UnicodeDecodeError):
            return None

        if not isinstance(header, dict) or header.get("alg") != "HS256":
            return None
        if not isinstance(payload, dict):
            return None

        signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
        expected_sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_sig, provided_sig):
            return None

        now = int(time.time())
        exp = payload.get("exp")
        if not isinstance(exp, (int, float)) or int(exp) <= now:
            return None

        nbf = payload.get("nbf")
        if nbf is not None and (not isinstance(nbf, (int, float)) or int(nbf) > now):
            return None

        if self._issuer and str(payload.get("iss") or "") != self._issuer:
            return None
        if self._audience:
            aud = payload.get("aud")
            aud_values = aud if isinstance(aud, list) else [aud]
            if self._audience not in [str(a) for a in aud_values if a is not None]:
                return None

        subject = str(payload.get("sub") or "").strip()
        tenant_id = str(payload.get("tenant_id") or "").strip()
        if not subject or not tenant_id:
            return None

        abac = payload.get("abac")
        if abac is not None and not isinstance(abac, dict):
            return None

        raw_scopes = payload.get("scope") or payload.get(SCOPE_CLAIM) or ""
        scopes = (
            [str(s) for s in raw_scopes]
            if isinstance(raw_scopes, list)
            else str(raw_scopes).split()
        )

        claims: dict[str, Any] = {
            "sub": subject,
            "tenant_id": tenant_id,
            "abac": dict(abac) if isinstance(abac, dict) else {},
        }
        if "iss" in payload:
            claims["iss"] = payload["iss"]
        if "aud" in payload:
            claims["aud"] = payload["aud"]

        return AccessToken(
            token=token,
            client_id=str(
                payload.get("client_id") or payload.get(CLIENT_ID_CLAIM) or "workload"
            ),
            scopes=scopes,
            expires_at=int(exp),
            subject=subject,
            claims=claims,
        )


class CompositeTokenVerifier(TokenVerifier):
    """Verify signed workload JWTs locally and delegate opaque grants to UserInfo."""

    def __init__(
        self,
        workload: WorkloadJwtVerifier,
        remote: AutoOntologyTokenVerifier,
    ) -> None:
        super().__init__()
        self._workload = workload
        self._remote = remote

    async def verify_token(self, token: str) -> AccessToken | None:
        if token.count(".") == 2:
            verified = await self._workload.verify_token(token)
            if verified is not None:
                return verified
        return await self._remote.verify_token(token)


def build_auto_ontology_auth(settings: Settings) -> RemoteAuthProvider | TokenVerifier:
    """Advertise Auto Ontology as the authorization server for this MCP server.

    When ``workload_jwt_secret`` is set, headless HS256 OBO workload tokens are
    verified locally without browser cookies; with ``workload_jwt_only=True``,
    only signed workload JWTs are accepted.
    """
    workload = (
        WorkloadJwtVerifier(
            settings.workload_jwt_secret,
            issuer=settings.workload_jwt_issuer,
            audience=settings.workload_jwt_audience,
        )
        if settings.workload_jwt_secret
        else None
    )
    if workload is not None and settings.workload_jwt_only:
        return workload

    remote = AutoOntologyTokenVerifier(settings.api_url, timeout_s=settings.timeout_s)
    verifier: TokenVerifier = (
        CompositeTokenVerifier(workload, remote) if workload is not None else remote
    )
    return RemoteAuthProvider(
        token_verifier=verifier,
        authorization_servers=[AnyHttpUrl(settings.api_url)],
        base_url=settings.public_url,
        resource_name="Auto Ontology",
    )


def auto_ontology_obo_claims() -> dict[str, Any]:
    """Return verified OBO claims (``subject``, ``tenant_id``, ``abac``) if present."""
    from fastmcp.server.dependencies import get_access_token

    try:
        token = get_access_token()
    except Exception:
        return {}
    if token is None:
        return {}
    claims = dict(getattr(token, "claims", None) or {})
    subject = getattr(token, "subject", None)
    if subject and "sub" not in claims:
        claims["sub"] = subject
    return claims


def auto_ontology_access_token() -> str | None:
    """The token this request was authenticated with, to forward to Auto Ontology.

    Read back from FastMCP rather than off the incoming header so that what is
    forwarded is the token that actually passed verification.
    """

    # Imported here because it resolves request state: at module scope it would
    # bind at import time, outside any request.
    from fastmcp.server.dependencies import get_access_token

    try:
        token = get_access_token()
    except Exception:
        # Raised when there is no authenticated request in scope, which is a
        # normal state rather than a fault.
        return None

    return getattr(token, "token", None) or None


__all__ = [
    "CLIENT_ID_CLAIM",
    "AutoOntologyTokenVerifier",
    "CompositeTokenVerifier",
    "SCOPE_CLAIM",
    "USERINFO_PATH",
    "WorkloadJwtVerifier",
    "auto_ontology_access_token",
    "auto_ontology_obo_claims",
    "build_auto_ontology_auth",
]
