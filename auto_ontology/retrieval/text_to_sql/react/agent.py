# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pipeline step that answers a question with one ReAct episode.

It hands a query to ``format_and_respond`` in the shape that node reads, ends
the run itself with a KumoRFM prediction's response, or hands off to
``unconstructable_sql_response`` when the episode ends without an answer.
"""

from __future__ import annotations

from typing import Any, Dict

import sqlglot
from langchain_core.messages import AIMessage
from sqlglot import exp

from auto_ontology.retrieval.text_to_sql.base import BaseAgent, record_thought
from auto_ontology.retrieval.text_to_sql.models import SQLGenerationModel
from auto_ontology.retrieval.text_to_sql.react.context import ToolContext
from auto_ontology.retrieval.text_to_sql.react.loop import run_episode
from auto_ontology.retrieval.text_to_sql.react.mode import ToolName, episode_mode
from auto_ontology.retrieval.text_to_sql.react.run_sql import connector_for
from auto_ontology.retrieval.text_to_sql.state import (
    AgentState,
    get_question_for_processing,
    get_standalone_question,
)

STEP_NAME = "react_sql_agent"

_UNANSWERED = (
    "I couldn't write a query that answers this question from the data I could "
    "find. Try rephrasing it, or name the table or measure you mean."
)

_NO_ASSUMPTIONS = "No assumptions were needed beyond what the question states."


def _extract_executed_queries(
    trajectory: list[dict[str, Any]], final_sql: str = ""
) -> list[str]:
    seen: list[str] = []
    for step in trajectory:
        for call in step.get("tool_calls") or []:
            if call.get("name") in (ToolName.RUN_SQL, ToolName.SUBMIT):
                sql = str((call.get("args") or {}).get("sql") or "").strip()
                if sql and sql not in seen:
                    seen.append(sql)
    if final_sql and final_sql.strip() and final_sql.strip() not in seen:
        seen.append(final_sql.strip())
    return seen


def _catalog_columns(
    tables: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Return ``(table, column, qualified_name)`` for catalog columns."""
    found: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for tbl in tables:
        if not isinstance(tbl, dict):
            continue
        table_name = str(tbl.get("table_name") or tbl.get("name") or "").strip()
        schema_name = str(tbl.get("schema_name") or tbl.get("schema") or "").strip()
        prefix = f"{schema_name}.{table_name}" if schema_name else table_name
        for col in tbl.get("columns") or []:
            if not isinstance(col, dict):
                continue
            col_name = str(col.get("column_name") or col.get("name") or "").strip()
            if not col_name:
                continue
            qualified = f"{prefix}.{col_name}" if prefix else col_name
            found.append((tbl, col, qualified))
    return found


def _statement_column_names(sql: str, dialect: str | None) -> set[str] | None:
    """Column names used by *sql*, or every column of its tables when it stars.

    ``None`` means the statement did not parse. ``COUNT(*)`` does not name a
    column. A bare ``SELECT *`` returns ``{"*"}`` so the caller expands the
    tables that appear in the statement.
    """
    try:
        parsed = sqlglot.parse_one(sql, read=dialect or None)
    except sqlglot.errors.SqlglotError:
        return None
    names: set[str] = set()
    for column in parsed.find_all(exp.Column):
        if column.name:
            names.add(column.name.casefold())
    for star in parsed.find_all(exp.Star):
        if isinstance(star.parent, exp.Count):
            continue
        names.add("*")
    return names


def _statement_table_names(sql: str, dialect: str | None) -> set[str] | None:
    try:
        parsed = sqlglot.parse_one(sql, read=dialect or None)
    except sqlglot.errors.SqlglotError:
        return None
    names: set[str] = set()
    for table in parsed.find_all(exp.Table):
        if table.name:
            names.add(table.name.casefold())
    return names


def _governance_from_queries(
    queries: list[str],
    tables: list[dict[str, Any]],
    dialect: str | None,
) -> tuple[list[str], dict[str, list[str]] | None, list[str]]:
    """PII and column lists for columns that appear in *queries*.

    Columns the agent only opened in the catalog are omitted. A selected
    column with no catalog classification is omitted from the PII map, and
    the map itself is ``None`` when no referenced column is classified.
    """
    catalog = _catalog_columns(tables)
    referenced: list[str] = []
    pii_tags: dict[str, list[str]] = {}
    newly_ingested: list[str] = []
    for sql in queries:
        column_names = _statement_column_names(sql, dialect)
        table_names = _statement_table_names(sql, dialect)
        if column_names is None or table_names is None:
            continue
        expand_star = "*" in column_names
        for tbl, col, qualified in catalog:
            table_name = str(tbl.get("table_name") or tbl.get("name") or "").casefold()
            if table_names and table_name not in table_names:
                continue
            col_name = str(col.get("column_name") or col.get("name") or "")
            if not expand_star and col_name.casefold() not in column_names:
                continue
            if qualified not in referenced:
                referenced.append(qualified)
            raw_tags = col.get("pii_tags") or col.get("tags") or []
            tags = (
                [str(tag) for tag in raw_tags if str(tag).strip()]
                if isinstance(raw_tags, list)
                else []
            )
            if tags:
                pii_tags[qualified] = tags
            if col.get("newly_ingested") is True and qualified not in newly_ingested:
                newly_ingested.append(qualified)
    return referenced, pii_tags or None, newly_ingested


def _tool_thought(call: dict[str, Any]) -> str:
    args = call.get("args") or {}
    name = call.get("name")
    if name == ToolName.RETRIEVE and args.get("purpose") == "information":
        return f"Looking up the catalog: {args.get('query') or 'everything'}"
    if name == ToolName.RETRIEVE:
        return f"Searching for: {args.get('query', '')}"
    if name == ToolName.RUN_SQL and str(args.get("final")).lower() != "true":
        return "Running a query to check the data."
    if name == ToolName.PREDICT:
        return "Predicting with KumoRFM."
    if name == ToolName.GREP:
        return f"Searching a saved result for: {args.get('pattern', '')}"
    if name == ToolName.SUBMIT and not str(args.get("sql") or "").strip():
        return "Answering an information question."
    return "Submitting the final query."


class ReactSQLAgent(BaseAgent):
    """Answer the question by letting the model search, query and submit."""

    def __init__(self) -> None:
        super().__init__("react_sql_agent", failure_decision="unconstructable")

    def validate_input(self, state: AgentState) -> bool:
        return bool(state.get("initial_question")) and bool(state.get("connectors"))

    def execute(self, state: AgentState) -> Dict[str, Any]:
        path_state = dict(state.get("path_state") or {})
        ctx = ToolContext(
            retriever=state.get("semantic_retriever") or state.get("data_retriever"),
            data_retriever=state.get("data_retriever"),
            connectors=list(state.get("connectors") or []),
            database_name=path_state.get("target_db"),
            numeric_type_alignment=bool(state.get("numeric_type_alignment")),
            information_answers=not state.get("calculation_only"),
            question=get_standalone_question(state),
            prediction_llm=state.get("llm"),
            custom_prompts=state.get("custom_prompts") or "",
        )
        connector = connector_for(ctx)
        # A connector that charges for queries says so through cost_status.
        cost_status = getattr(connector, "cost_status", None)
        if callable(cost_status):
            ctx.cost_status = cost_status
        can_run_queries = getattr(connector, "can_run_queries", None)
        if callable(can_run_queries):
            ctx.can_run_queries = can_run_queries
        result = run_episode(
            state.get("reasoning_llm") or state["llm"],
            ctx,
            question=get_question_for_processing(state),
            evidence=state.get("evidence"),
            dialect=getattr(connector, "dialect", None),
            mode=episode_mode(state.get("prediction_override")),
            custom_prompts=state.get("custom_prompts"),
            sql_examples=state.get("sql_examples"),
            value_anchors=state.get("value_anchors"),
            shorten_answer=bool(state.get("shorten_answer")),
        )

        for step in result.trajectory:
            for call in step.get("tool_calls") or []:
                record_thought(path_state, STEP_NAME, _tool_thought(call))
        path_state["react_trajectory"] = result.trajectory
        path_state["react_end_reason"] = result.reason
        relevant_tables = list(ctx.seen_tables.values())
        path_state["relevant_tables"] = relevant_tables
        dialect = getattr(connector, "dialect", None)
        provisional_sql = ""
        if result.submission is not None:
            provisional_sql = result.submission.sql
        queries = _extract_executed_queries(result.trajectory, provisional_sql)
        ref_cols, pii_map, new_cols = _governance_from_queries(
            queries, relevant_tables, dialect
        )
        path_state["referenced_columns"] = ref_cols
        path_state.pop("pii_tags", None)
        if pii_map:
            path_state["pii_tags"] = pii_map
        path_state["newly_ingested_columns"] = new_cols

        if result.prediction is not None:
            response = result.prediction.response
            markdown = str(response.get("response") or "")
            path_state["formatted_response"] = markdown
            path_state["final_response"] = response
            return {
                "decision": "predicted",
                "messages": list(state.get("messages") or [])
                + [AIMessage(content=markdown)],
                "path_state": path_state,
            }

        submission = result.submission
        if submission is None:
            self.logger.info("ReAct episode ended without an answer: %s", result.reason)
            path_state["unconstructable_explanation"] = (
                ctx.prediction_refusal or _UNANSWERED
            )
            return {"decision": "unconstructable", "path_state": path_state}

        if submission.db_response is None:
            path_state["formatted_response"] = submission.response
            path_state["final_response"] = {
                "response": submission.response,
                "sql_code": "",
                "sql_columns": [],
                "custom_analyses_used": [],
                "sql_response_from_db": None,
            }
            return {
                "decision": "information_answer",
                "messages": list(state.get("messages") or [])
                + [AIMessage(content=submission.response)],
                "path_state": path_state,
            }

        path_state["sql_code"] = submission.sql
        path_state["executed_queries"] = _extract_executed_queries(
            result.trajectory, submission.sql
        )
        path_state["sql_generation_result"] = SQLGenerationModel(
            thought=submission.assumptions or _NO_ASSUMPTIONS,
            sql_code=submission.sql,
            response=submission.response or "Here is the result of the query.",
        )
        path_state["sql_response_from_db"] = submission.db_response.result
        path_state.setdefault("candidates", [])
        if ctx.numeric_type_alignment:
            path_state["result_numeric_kinds"] = submission.db_response.numeric_kinds
            path_state["result_numeric_kinds_sql"] = submission.sql
            path_state["result_values_digest"] = submission.db_response.values_digest
        return {"decision": "valid_sql", "path_state": path_state}
