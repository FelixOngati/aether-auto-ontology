# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for extended DataAnswer provenance and >100-row BlobRef handoff."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from fastmcp import Client, FastMCP

from auto_ontology_mcp import chat
from auto_ontology_mcp.config import DEFAULT_SPEC_PATH, Settings


def _sse(*events: dict[str, Any]) -> str:
    body = ""
    for event in events:
        body += f"data: {json.dumps(event)}\n\n"
    body += "data: [DONE]\n\n"
    return body


def test_ask_question_forwards_target_db_and_populates_provenance_and_blob_ref(
    tmp_path: Path,
) -> None:
    captured_body: dict[str, Any] = {}
    large_rows = [{"id": idx, "amount": float(idx * 10)} for idx in range(135)]

    def handler(request: httpx.Request) -> httpx.Response:
        captured_body.update(json.loads(request.content.decode("utf-8")))
        stream = _sse(
            {"type": "sql", "sql": "SELECT COUNT(*) FROM sales.orders"},
            {
                "type": "result",
                "answer": {
                    "response": "Returned 135 order rows.",
                    "sql_code": "SELECT id, amount FROM sales.orders",
                    "sql_response_from_db": [json.dumps(large_rows)],
                    "target_db": "warehouse_prod",
                    "executed_queries": [
                        "SELECT COUNT(*) FROM sales.orders",
                        "SELECT id, amount FROM sales.orders",
                    ],
                    "referenced_columns": ["sales.orders.id", "sales.orders.amount"],
                    "pii_tags": {"sales.orders.customer_email": ["EMAIL"]},
                    "newly_ingested_columns": ["sales.orders.discount_tier"],
                },
            },
        )
        return httpx.Response(
            200,
            content=stream.encode("utf-8"),
            headers={"content-type": "text/event-stream"},
        )

    settings = Settings(
        api_url="http://auto_ontology.test",
        spec_path=DEFAULT_SPEC_PATH,
        host="127.0.0.1",
        port=3003,
        timeout_s=30.0,
        chat_timeout_s=900.0,
        blob_store_dir=tmp_path,
    )
    client = httpx.AsyncClient(
        base_url=settings.api_url, transport=httpx.MockTransport(handler)
    )
    mcp: FastMCP = FastMCP(name="test")
    chat.register(mcp, settings, client)

    async def run() -> Any:
        async with Client(mcp) as session:
            return await session.call_tool(
                "ask_question",
                {
                    "question": "List order amounts",
                    "target_db": "warehouse_prod",
                },
            )

    res = asyncio.run(run())
    assert captured_body["target_db"] == "warehouse_prod"
    assert res.data.target_db == "warehouse_prod"
    assert res.data.row_count == 135
    assert len(res.data.rows) == chat.MAX_ROWS
    assert res.data.truncated is True
    assert res.data.executed_queries == [
        "SELECT COUNT(*) FROM sales.orders",
        "SELECT id, amount FROM sales.orders",
    ]
    assert res.data.referenced_columns == ["sales.orders.id", "sales.orders.amount"]
    assert res.data.pii_tags == {"sales.orders.customer_email": ["EMAIL"]}
    assert res.data.newly_ingested_columns == ["sales.orders.discount_tier"]
    assert res.data.blob_ref is not None
    assert res.data.blob_ref.row_count == 135
    rendered = res.content[0].text
    assert "file://" not in rendered
    assert str(tmp_path) not in rendered
    assert res.data.blob_ref.id == res.data.blob_ref.sha256
    stored = list(tmp_path.rglob("*.json"))
    assert len(stored) == 1
    assert stored[0].parent.name == "warehouse_prod"
    assert len(json.loads(stored[0].read_text(encoding="utf-8"))) == 135
