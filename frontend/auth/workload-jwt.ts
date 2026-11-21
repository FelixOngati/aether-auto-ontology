// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { decodeProtectedHeader, jwtVerify, type JWTPayload } from 'jose';

const DEFAULT_CLOCK_SKEW_S = 30;
const DEFAULT_MAX_TTL_S = 86_400;

export type VerifiedWorkloadPrincipal = {
	token: string;
	subject: string;
	tenantId: string;
	role: 'admin' | 'viewer';
	abac: Record<string, unknown>;
};

const isTruthyEnv = (name: string): boolean => {
	const raw = (process.env[name] ?? '').trim().toLowerCase();
	return raw === '1' || raw === 'true' || raw === 'yes' || raw === 'on';
};

const positiveIntEnv = (name: string, fallback: number, allowZero = false): number => {
	const raw = (process.env[name] ?? '').trim();
	if (!raw) return fallback;
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed) || parsed < 0 || (parsed === 0 && !allowZero)) return fallback;
	return parsed;
};

export const getWorkloadJwtSecret = (): string =>
	(process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET ?? process.env.OBO_SIGNING_SECRET ?? '').trim();

export const isWorkloadJwtOnly = (): boolean => isTruthyEnv('AUTO_ONTOLOGY_WORKLOAD_JWT_ONLY');

const extractBearer = (headers: Headers): string | null => {
	const header = headers.get('authorization');
	if (!header) return null;
	const [scheme, token] = header.split(' ');
	if (scheme?.toLowerCase() !== 'bearer' || !token) return null;
	return token.trim() || null;
};

/**
 * Return true when `headers` carries a bearer credential that must be settled
 * by {@link verifyWorkloadJwt} rather than falling through to OIDC JWKS lookup.
 */
export const isWorkloadJwtCandidate = (headers: Headers): boolean => {
	if (isWorkloadJwtOnly()) return true;
	if (!getWorkloadJwtSecret()) return false;
	const token = extractBearer(headers);
	if (!token || token.split('.').length !== 3) return false;
	try {
		const protectedHeader = decodeProtectedHeader(token);
		return protectedHeader.alg === 'HS256';
	} catch {
		return true;
	}
};

export async function verifyWorkloadJwt(
	headers: Headers,
): Promise<VerifiedWorkloadPrincipal | null> {
	const secret = getWorkloadJwtSecret();
	if (!secret) return null;

	const token = extractBearer(headers);
	if (!token || token.split('.').length !== 3) return null;

	const clockSkew = positiveIntEnv(
		'AUTO_ONTOLOGY_WORKLOAD_JWT_CLOCK_SKEW_S',
		DEFAULT_CLOCK_SKEW_S,
		true,
	);
	const maxTtl = positiveIntEnv('AUTO_ONTOLOGY_WORKLOAD_JWT_MAX_TTL_S', DEFAULT_MAX_TTL_S);
	const expectedAudience = (process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_AUDIENCE ?? '').trim();

	let payload: JWTPayload;
	try {
		({ payload } = await jwtVerify(token, new TextEncoder().encode(secret), {
			algorithms: ['HS256'],
			clockTolerance: clockSkew,
			...(expectedAudience ? { audience: expectedAudience } : {}),
		}));
	} catch {
		return null;
	}

	const now = Math.floor(Date.now() / 1000);
	if (typeof payload.exp !== 'number' || payload.exp + clockSkew <= now) return null;
	if (payload.exp - now > maxTtl + clockSkew) return null;
	if (payload.iat !== undefined) {
		if (typeof payload.iat !== 'number' || payload.iat - clockSkew > now) return null;
		if (payload.exp - payload.iat > maxTtl) return null;
	}

	const issuerRaw = (process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_ISSUER ?? '').trim();
	if (issuerRaw) {
		const allowedIssuers = new Set(
			issuerRaw
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean),
		);
		const tokenIssuer = typeof payload.iss === 'string' ? payload.iss.trim() : '';
		if (allowedIssuers.size > 0 && !allowedIssuers.has(tokenIssuer)) return null;
	}

	const subject = typeof payload.sub === 'string' ? payload.sub.trim() : '';
	const tenantId = typeof payload.tenant_id === 'string' ? payload.tenant_id.trim() : '';
	if (!subject || !tenantId) return null;

	const rawAbac = payload.abac;
	if (
		rawAbac !== undefined &&
		(typeof rawAbac !== 'object' || rawAbac === null || Array.isArray(rawAbac))
	) {
		return null;
	}
	const abac: Record<string, unknown> =
		typeof rawAbac === 'object' && rawAbac !== null
			? { ...(rawAbac as Record<string, unknown>) }
			: {};

	const region =
		typeof abac.region === 'string' && abac.region.trim()
			? abac.region.trim()
			: typeof payload.region === 'string'
				? payload.region.trim()
				: '';
	const rawRoles = 'roles' in abac ? abac.roles : payload.roles;
	if (rawRoles !== undefined && !Array.isArray(rawRoles)) return null;
	const roles = Array.isArray(rawRoles)
		? rawRoles.map((r) => String(r).trim()).filter(Boolean)
		: [];
	const workflowRunId =
		typeof abac.workflow_run_id === 'string' && abac.workflow_run_id.trim()
			? abac.workflow_run_id.trim()
			: typeof payload.workflow_run_id === 'string'
				? payload.workflow_run_id.trim()
				: '';

	if (region && !('region' in abac)) abac.region = region;
	if (roles.length > 0 && !('roles' in abac)) abac.roles = roles;
	if (workflowRunId && !('workflow_run_id' in abac)) abac.workflow_run_id = workflowRunId;

	const role: 'admin' | 'viewer' = roles.includes('admin') ? 'admin' : 'viewer';

	return {
		token,
		subject,
		tenantId,
		role,
		abac,
	};
}
