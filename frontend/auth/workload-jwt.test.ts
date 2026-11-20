// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SignJWT } from 'jose';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { proxyToBackend } from './proxy-backend';
import { isWorkloadJwtCandidate, verifyWorkloadJwt } from './workload-jwt';

const SECRET = 'frontend-workload-obo-secret-32b';

async function mintJwt(
	claims: Record<string, unknown>,
	secret = SECRET,
	expOffsetSec = 300,
): Promise<string> {
	const now = Math.floor(Date.now() / 1000);
	return new SignJWT(claims)
		.setProtectedHeader({ alg: 'HS256', typ: 'JWT' })
		.setIssuedAt(now)
		.setExpirationTime(now + expOffsetSec)
		.sign(new TextEncoder().encode(secret));
}

describe('workload-jwt and proxyToBackend', () => {
	const savedEnv = { ...process.env };

	afterEach(() => {
		process.env = { ...savedEnv };
		vi.restoreAllMocks();
	});

	it('verifies HS256 workload OBO JWTs with multi-issuer allowlist and top-level claims', async () => {
		process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET = SECRET;
		process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_ISSUER =
			'aether-mcp-connectors, aether-control-plane';
		process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_AUDIENCE = 'auto-ontology-mcp';

		const token = await mintJwt({
			sub: 'svc:control-plane',
			tenant_id: 'tenant-a',
			iss: 'aether-control-plane',
			aud: 'auto-ontology-mcp',
			region: 'GLOBAL',
			roles: ['admin'],
			workflow_run_id: 'wf-9',
		});

		const headers = new Headers({ Authorization: `Bearer ${token}` });
		expect(isWorkloadJwtCandidate(headers)).toBe(true);

		const verified = await verifyWorkloadJwt(headers);
		expect(verified).toEqual({
			token,
			subject: 'svc:control-plane',
			tenantId: 'tenant-a',
			role: 'admin',
			abac: {
				region: 'GLOBAL',
				roles: ['admin'],
				workflow_run_id: 'wf-9',
			},
		});
	});

	it('fails closed on forged or tenantless workload JWTs', async () => {
		process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET = SECRET;

		const forged = await mintJwt(
			{ sub: 'u1', tenant_id: 'tenant-a' },
			'wrong-secret-32-bytes-0000000000',
		);
		const noTenant = await mintJwt({ sub: 'u1' });

		expect(
			await verifyWorkloadJwt(new Headers({ Authorization: `Bearer ${forged}` })),
		).toBeNull();
		expect(
			await verifyWorkloadJwt(new Headers({ Authorization: `Bearer ${noTenant}` })),
		).toBeNull();
	});

	it('forwards verified OBO headers and strips spoofed client headers in proxyToBackend', async () => {
		process.env.AUTO_ONTOLOGY_WORKLOAD_JWT_SECRET = SECRET;

		const token = await mintJwt({
			sub: 'analyst@tenant-a.example',
			tenant_id: 'tenant-a',
			abac: { region: 'EMEA', roles: ['analyst'] },
		});

		let capturedHeaders: Record<string, string> = {};
		vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
			capturedHeaders = (init?.headers ?? {}) as Record<string, string>;
			return new Response(JSON.stringify({ data: [] }), {
				status: 200,
				headers: { 'content-type': 'application/json' },
			});
		});

		const req = new Request('http://localhost:3000/api/terms', {
			method: 'GET',
			headers: {
				Authorization: `Bearer ${token}`,
				'x-auto-ontology-tenant-id': 'spoofed-tenant',
				'x-auto-ontology-user-id': 'spoofed-user',
			},
		});

		const resp = await proxyToBackend(req);
		expect(resp.status).toBe(200);
		expect(capturedHeaders.Authorization).toBe(`Bearer ${token}`);
		expect(capturedHeaders['x-auto-ontology-user-id']).toBe('analyst@tenant-a.example');
		expect(capturedHeaders['x-auto-ontology-tenant-id']).toBe('tenant-a');
		expect(JSON.parse(capturedHeaders['x-auto-ontology-abac'])).toEqual({
			region: 'EMEA',
			roles: ['analyst'],
		});
	});
});
