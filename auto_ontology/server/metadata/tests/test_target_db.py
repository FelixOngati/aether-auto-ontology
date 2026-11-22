# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""target_db matching for the entity-coverage payload."""

from __future__ import annotations

import pytest

from auto_ontology.server.metadata.service import (
    PredictionFlowError,
    select_connectors_for_target,
)


class _Connector:
    def __init__(self, name: str, connector_id: str = "") -> None:
        self.database_name = name
        self.id = connector_id


def test_named_database_and_catalog_uuid_select_the_connector() -> None:
    warehouse = _Connector("warehouse_prod", "db-1")
    other = _Connector("analytics", "db-2")
    catalog = [
        {"id": "11111111-1111-1111-1111-111111111111", "name": "warehouse_prod"},
        {"id": "22222222-2222-2222-2222-222222222222", "name": "analytics"},
    ]

    by_name = select_connectors_for_target(
        [warehouse, other], "Warehouse_Prod", catalog
    )
    by_uuid = select_connectors_for_target(
        [warehouse, other],
        "11111111-1111-1111-1111-111111111111",
        catalog,
    )
    by_connector_id = select_connectors_for_target([warehouse, other], "db-2", [])

    assert by_name == [warehouse]
    assert by_uuid == [warehouse]
    assert by_connector_id == [other]


def test_named_database_that_is_not_connected_is_a_distinct_error() -> None:
    with pytest.raises(PredictionFlowError, match="not connected"):
        select_connectors_for_target(
            [_Connector("warehouse_prod")],
            "missing-db",
            [{"id": "db-1", "name": "warehouse_prod"}],
        )


def test_no_connectors_without_a_target_keeps_the_unconfigured_error() -> None:
    with pytest.raises(PredictionFlowError, match="No database connection"):
        select_connectors_for_target([], None, [])
