# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""enable fail-closed tenant row level security on catalog tables

Revision ID: a1c4e8b27d09
Revises: 0b6f3c2a9d14
Create Date: 2026-11-22 23:10:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from auto_ontology.dal.tenant_rls import (
    DEFAULT_TENANT_SCOPED_TABLES,
    apply_catalog_tenant_rls,
)

revision: str = "a1c4e8b27d09"
down_revision: Union[str, Sequence[str], None] = "0b6f3c2a9d14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add tenant_id and FORCE ROW LEVEL SECURITY on catalog tables that exist."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names(schema="public"))
    tables = [name for name in DEFAULT_TENANT_SCOPED_TABLES if name in existing]
    if tables:
        apply_catalog_tenant_rls(bind, tables=tables)


def downgrade() -> None:
    """Drop the tenant isolation policies and tenant_id columns."""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names(schema="public"))
    for table_name in DEFAULT_TENANT_SCOPED_TABLES:
        if table_name not in existing:
            continue
        policy = f"{table_name}_tenant_isolation"
        op.execute(
            sa.text(f'ALTER TABLE public."{table_name}" DISABLE ROW LEVEL SECURITY')
        )
        op.execute(
            sa.text(f'DROP POLICY IF EXISTS "{policy}" ON public."{table_name}"')
        )
        op.execute(
            sa.text(
                f'ALTER TABLE public."{table_name}" DROP COLUMN IF EXISTS tenant_id'
            )
        )
