# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for executed SQL and column/PII provenance extraction in ReactSQLAgent."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import HumanMessage
import pytest

from auto_ontology.retrieval.text_to_sql.agents.response import ResponseAgent
from auto_ontology.retrieval.text_to_sql.agents.sql_execution import QueryResponse
from auto_ontology.retrieval.text_to_sql.react import agent as agent_module
from auto_ontology.retrieval.text_to_sql.react.loop import EpisodeResult, Submission


def test_react_and_response_agents_propagate_sql_and_column_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_response = QueryResponse(result=['[{"total": 250.0}]'], sliced=False)
    submission = Submission(
        "SELECT billing_email, status FROM public.invoices",
        "Invoice contacts.",
        "",
        db_response,
    )
    trajectory = [
        {
            "role": "ai",
            "tool_calls": [
                {
                    "name": "run_sql",
                    "args": {
                        "sql": "SELECT COUNT(*) FROM public.invoices",
                        "final": False,
                    },
                },
                {
                    "name": "submit",
                    "args": {
                        "sql": "SELECT billing_email, status FROM public.invoices"
                    },
                },
            ],
        }
    ]

    def fake_episode(
        _llm: object, ctx: Any, *_a: object, **_k: object
    ) -> EpisodeResult:
        ctx.seen_tables["public.invoices"] = {
            "schema_name": "public",
            "table_name": "invoices",
            "columns": [
                {"column_name": "amount", "newly_ingested": False},
                {"column_name": "status"},
                {
                    "column_name": "billing_email",
                    "pii_tags": ["EMAIL"],
                    "newly_ingested": True,
                },
                {
                    "column_name": "ssn",
                    "pii_tags": ["SSN"],
                    "newly_ingested": True,
                },
            ],
        }
        return EpisodeResult(submission, "submitted", 1, trajectory)

    from typing import Any

    monkeypatch.setattr(agent_module, "run_episode", fake_episode)

    state = {
        "initial_question": "What is total invoice amount?",
        "evidence": "",
        "llm": SimpleNamespace(),
        "connectors": [SimpleNamespace(database_name="erp", dialect="postgres")],
        "semantic_retriever": object(),
        "messages": [HumanMessage(content="What is total invoice amount?")],
        "path_state": {"target_db": "erp"},
    }
    step1 = agent_module.ReactSQLAgent().execute(state)
    step2 = ResponseAgent().execute({**state, "path_state": step1["path_state"]})
    final = step2["path_state"]["final_response"]

    assert final["target_db"] == "erp"
    assert final["executed_queries"] == [
        "SELECT COUNT(*) FROM public.invoices",
        "SELECT billing_email, status FROM public.invoices",
    ]
    assert final["referenced_columns"] == [
        "public.invoices.status",
        "public.invoices.billing_email",
    ]
    assert final["pii_tags"] == {"public.invoices.billing_email": ["EMAIL"]}
    assert "public.invoices.ssn" not in final["referenced_columns"]
    assert final["newly_ingested_columns"] == ["public.invoices.billing_email"]


def test_unclassified_columns_leave_the_pii_map_absent() -> None:
    referenced, pii, newly = agent_module._governance_from_queries(
        ["SELECT amount FROM public.invoices"],
        [
            {
                "schema_name": "public",
                "table_name": "invoices",
                "columns": [
                    {"column_name": "amount"},
                    {"column_name": "billing_email", "pii_tags": ["EMAIL"]},
                ],
            }
        ],
        "postgres",
    )
    assert referenced == ["public.invoices.amount"]
    assert pii is None
    assert newly == []
