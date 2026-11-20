// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export const INTERNAL_USER_HEADER = 'x-auto-ontology-user-id';
export const INTERNAL_TENANT_HEADER = 'x-auto-ontology-tenant-id';
export const INTERNAL_ABAC_HEADER = 'x-auto-ontology-abac';
export const INTERNAL_GATEWAY_SECRET_HEADER = 'x-auto-ontology-gateway-secret';

export type InternalOboMetadata = {
	tenantId?: string;
	abac?: Record<string, unknown>;
};

export const buildGatewaySecretHeaders = (): Record<string, string> => {
	const gatewaySecret = (process.env.AUTO_ONTOLOGY_INTERNAL_GATEWAY_SECRET ?? '').trim();
	return gatewaySecret ? { [INTERNAL_GATEWAY_SECRET_HEADER]: gatewaySecret } : {};
};

/**
 * Forward the authenticated user identity across the private Next.js →
 * FastAPI hop. FastAPI must not be publicly reachable because this header is
 * trusted without independent verification unless workload JWT auth is active.
 */
export const buildInternalIdentityHeaders = (
	userId: string,
	obo?: InternalOboMetadata,
): Record<string, string> => {
	const result: Record<string, string> = {
		...buildGatewaySecretHeaders(),
		[INTERNAL_USER_HEADER]: userId,
	};
	if (obo?.tenantId) {
		result[INTERNAL_TENANT_HEADER] = obo.tenantId;
	}
	if (obo?.abac && Object.keys(obo.abac).length > 0) {
		result[INTERNAL_ABAC_HEADER] = JSON.stringify(obo.abac);
	}
	return result;
};
