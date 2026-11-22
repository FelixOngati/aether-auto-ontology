# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration for ``question-entity-coverage``.

``entity_coverage`` runs a fast pipeline that extracts entities, retrieves
semantic candidates, and returns a deterministic coverage grade.

The pipeline shares retriever/connector state and is not safe to run in parallel
(the same constraint that makes chat single-slot), so runs are serialized here.
"""

from __future__ import annotations

import logging
import threading

from auto_ontology.retrieval.entity_coverage.main import get_coverage_response
from auto_ontology.retrieval.entity_coverage.main import (
    llm_client as coverage_llm_client,
)
from auto_ontology.retrieval.entity_coverage.state import (
    DEFAULT_MAX_DISTANCE,
    EntityCoveragePayload,
)
from auto_ontology.server.chat.settings_dal import fetch_acronyms, fetch_custom_prompts
from auto_ontology.connectors.registry import get_connectors
from auto_ontology.utils.retriever import (
    get_data_objects_retriever,
    get_semantic_objects_retriever,
)

logger = logging.getLogger(__name__)

# Retrieval runs share retriever/connector state and a single LLM budget and are
# not safe to run in parallel; serialize like the chat pool's single slot.
_run_lock = threading.Lock()


class PredictionFlowError(RuntimeError):
    """The prediction flow could not produce a result for the question."""


def _connector_keys(connector: object) -> set[str]:
    keys: set[str] = set()
    for attr in ("database_name", "id", "database_id", "connection_id", "catalog_id"):
        value = getattr(connector, attr, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            keys.add(text.casefold())
    return keys


def select_connectors_for_target(
    connectors: list,
    target_db: str | None,
    catalog_databases: list[dict] | None = None,
) -> list:
    """Keep the connectors named by *target_db*.

    *target_db* may be a catalog UUID or a database name. A name that this
    process does not have is a different error from having no connections at
    all.
    """
    if not target_db or not str(target_db).strip():
        if not connectors:
            raise PredictionFlowError("No database connection is configured.")
        return list(connectors)

    requested = str(target_db).strip()
    accepted = {requested.casefold()}
    for database in catalog_databases or []:
        if not isinstance(database, dict):
            continue
        database_id = str(database.get("id") or "").strip()
        database_name = str(database.get("name") or "").strip()
        id_match = database_id and (
            requested == database_id or requested.casefold() == database_id.casefold()
        )
        name_match = database_name and requested.casefold() == database_name.casefold()
        if id_match or name_match:
            if database_id:
                accepted.add(database_id.casefold())
            if database_name:
                accepted.add(database_name.casefold())

    matched = [
        connector for connector in connectors if _connector_keys(connector) & accepted
    ]
    if not matched:
        raise PredictionFlowError(
            f"Database {requested!r} is not connected to this process."
        )
    return matched


def _catalog_databases() -> list[dict]:
    try:
        from auto_ontology.dal.datasources import fetch_databases

        rows = fetch_databases(zone_ids=None)
    except Exception:
        logger.warning("Catalog database list is unavailable for target_db matching")
        return []
    return [row for row in rows if isinstance(row, dict)]


def _build_coverage_payload(
    question: str,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    target_db: str | None = None,
) -> EntityCoveragePayload:
    """Assemble the payload for the entity-coverage pipeline."""
    connectors = select_connectors_for_target(
        get_connectors(), target_db, _catalog_databases()
    )
    custom_prompts = fetch_custom_prompts()
    payload: EntityCoveragePayload = {
        "question": question,
        "data_retriever": get_data_objects_retriever(),
        "semantic_retriever": get_semantic_objects_retriever(),
        "connectors": connectors,
        "acronyms": fetch_acronyms(),
        "custom_prompts": custom_prompts,
        "max_distance": max_distance,
    }
    return payload


def entity_coverage(
    question: str,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    target_db: str | None = None,
) -> dict:
    """Return ranked semantic candidates and a 0–1 entity coverage grade.

    Raises :class:`PredictionFlowError` when the flow cannot produce a result.
    """
    if coverage_llm_client is None:
        raise PredictionFlowError("LLM client is not configured.")
    with _run_lock:
        try:
            return get_coverage_response(
                _build_coverage_payload(
                    question, max_distance=max_distance, target_db=target_db
                )
            )
        except (ValueError, RuntimeError) as exc:
            raise PredictionFlowError(str(exc)) from exc
