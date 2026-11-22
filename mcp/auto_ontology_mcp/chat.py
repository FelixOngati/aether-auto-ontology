# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ask_question`` — the text-to-SQL agent as a single MCP tool.

Hand-written rather than generated, because ``POST /api/chat/completions``
answers with ``text/event-stream``: the agent emits a step event per graph node
and only then a result. ``FastMCP.from_openapi`` produces request/response
tools and has nothing useful to say about a stream.

Streaming is not merely an obstacle here. A run is many sequential model calls
and routinely takes tens of seconds, which is long enough that a silent tool
looks hung, so the step events are forwarded as MCP progress notifications and
the caller sees the same reasoning trace the web UI shows.

The response the UI receives is markdown with fenced SQL, shaped for a renderer.
This returns the parts separately instead, so a calling agent can use the SQL
and the rows without parsing prose.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, model_serializer

from auto_ontology_mcp.config import Settings

logger = logging.getLogger(__name__)

# An agent's context is finite and a question can legitimately match thousands
# of rows. Return a readable prefix and say how much was withheld, rather than
# flooding the caller or silently truncating.
MAX_ROWS = 100
_BLOB_TTL = timedelta(hours=1)

_SSE_DATA_PREFIX = "data:"
_SSE_DONE = "[DONE]"


class BlobRef(BaseModel):
    """Opaque handle for a full row payload kept on the server."""

    id: str = Field(
        description="Opaque id of the stored payload. Not a filesystem path."
    )
    sha256: str = Field(description="Hex SHA-256 digest of the stored JSON bytes.")
    size_bytes: int = Field(description="Byte length of the stored JSON payload.")
    row_count: int = Field(description="Total number of rows stored in the blob.")
    expires_at: str = Field(
        description="UTC time after which the server deletes the payload."
    )


class DataAnswer(BaseModel):
    """Structured result of one ``ask_question`` call."""

    answer: str = Field(description="Natural-language answer to the question.")
    sql: str = Field(default="", description="The SQL that produced the rows.")
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description=f"Result rows, at most {MAX_ROWS} of them.",
    )
    row_count: int = Field(
        default=0, description="Rows returned by the query before truncation."
    )
    truncated: bool = Field(
        default=False, description=f"True when more than {MAX_ROWS} rows matched."
    )
    reasoning: str = Field(
        default="",
        description="What the agent did, one line per step it took.",
    )
    target_db: str | None = Field(
        default=None,
        description="Connected database pinned for this question, if specified.",
    )
    executed_queries: list[str] = Field(
        default_factory=list,
        description="Ordered SQL statements executed during the episode.",
    )
    referenced_columns: list[str] = Field(
        default_factory=list,
        description="Qualified columns inspected during semantic retrieval.",
    )
    pii_tags: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Catalog PII tags for columns in the executed SQL. Absent when "
            "the catalog has no classification for those columns."
        ),
    )
    newly_ingested_columns: list[str] = Field(
        default_factory=list,
        description="Referenced columns flagged as newly ingested.",
    )
    blob_ref: BlobRef | None = Field(
        default=None,
        description="Opaque server-side handle for the full row payload when truncated.",
    )

    @model_serializer(mode="wrap")
    def _omit_unclassified_pii(self, handler):
        payload = handler(self)
        if isinstance(payload, dict) and payload.get("pii_tags") is None:
            payload.pop("pii_tags", None)
        return payload


def _parse_rows(raw: Any) -> list[dict[str, Any]]:
    """Normalise the agent's result payload into a list of row dicts.

    The backend sends either a one-item list holding a JSON-records string or
    the records themselves, depending on which path produced the answer.
    """
    if raw is None:
        return []

    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], str):
        try:
            raw = json.loads(raw[0])
        except json.JSONDecodeError:
            logger.warning("Result payload was a string but not JSON; dropping rows")
            return []

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [row for row in raw if isinstance(row, dict)]


def _blob_scope(scope: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in scope.strip())
    return cleaned or "unscoped"


def _purge_expired_blobs(root: Path, now: float) -> None:
    if not root.exists():
        return
    cutoff = now - _BLOB_TTL.total_seconds()
    for path in root.rglob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            logger.warning("Could not expire blob %s", path.name)


def _write_blob_if_needed(
    rows: list[dict[str, Any]],
    blob_store_dir: Path | None,
    *,
    scope: str,
) -> BlobRef | None:
    """Store oversized rows on the server and return an opaque id.

    The tool result never includes a filesystem path. The file lives under a
    scope directory (tenant or target database) and is deleted after
    ``_BLOB_TTL``.
    """
    if len(rows) <= MAX_ROWS or blob_store_dir is None:
        return None
    now = time.time()
    _purge_expired_blobs(blob_store_dir, now)
    directory = blob_store_dir / _blob_scope(scope)
    directory.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(rows, separators=(",", ":"), default=str).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    (directory / f"{digest}.json").write_bytes(raw)
    expires = datetime.fromtimestamp(now, tz=timezone.utc) + _BLOB_TTL
    return BlobRef(
        id=digest,
        sha256=digest,
        size_bytes=len(raw),
        row_count=len(rows),
        expires_at=expires.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def _build_answer(
    payload: dict[str, Any],
    *,
    requested_target_db: str | None = None,
    streamed_queries: list[str] | None = None,
    blob_store_dir: Path | None = None,
) -> DataAnswer:
    # A successful run reports rows under `sql_response_from_db`. `result` is
    # only present in the failure shape `_extract_answer` falls back to, where
    # it is None — so reading it first would discard every real answer.
    rows = _parse_rows(payload.get("sql_response_from_db") or payload.get("result"))
    sql_code = str(payload.get("sql_code") or "")
    queries: list[str] = []
    for candidate in [
        *(streamed_queries or []),
        *(payload.get("executed_queries") or []),
        sql_code,
    ]:
        cleaned = str(candidate or "").strip()
        if cleaned and cleaned not in queries:
            queries.append(cleaned)

    raw_pii = payload.get("pii_tags")
    pii_tags: dict[str, list[str]] | None = None
    if isinstance(raw_pii, dict):
        parsed = {
            str(col): [str(t) for t in tags if str(t).strip()]
            for col, tags in raw_pii.items()
            if isinstance(tags, list)
        }
        parsed = {col: tags for col, tags in parsed.items() if tags}
        if parsed:
            pii_tags = parsed
    scope = str(
        payload.get("tenant_id")
        or requested_target_db
        or payload.get("target_db")
        or ""
    )
    return DataAnswer(
        answer=str(payload.get("response") or ""),
        sql=sql_code,
        rows=rows[:MAX_ROWS],
        row_count=len(rows),
        truncated=len(rows) > MAX_ROWS,
        reasoning=str(payload.get("thoughts") or ""),
        target_db=str(payload.get("target_db") or requested_target_db or "") or None,
        executed_queries=queries,
        referenced_columns=[
            str(c) for c in (payload.get("referenced_columns") or []) if str(c).strip()
        ],
        pii_tags=pii_tags,
        newly_ingested_columns=[
            str(c)
            for c in (payload.get("newly_ingested_columns") or [])
            if str(c).strip()
        ],
        blob_ref=_write_blob_if_needed(rows, blob_store_dir, scope=scope),
    )


def _decode_event(line: str) -> dict[str, Any] | None:
    """Return the JSON object on an SSE ``data:`` line, if there is one.

    Heartbeats arrive as ``: ping`` comments and blank lines separate events;
    both are skipped. A malformed payload is dropped rather than failing the
    run — losing one progress update is better than losing the answer.
    """
    if not line.startswith(_SSE_DATA_PREFIX):
        return None
    payload = line[len(_SSE_DATA_PREFIX) :].strip()
    if not payload or payload == _SSE_DONE:
        return None
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        logger.warning("Skipping unparseable SSE payload: %.200s", payload)
        return None
    return event if isinstance(event, dict) else None


async def _raise_for_status(response: httpx.Response) -> None:
    """Turn a non-200 chat response into a ToolError the caller can act on."""
    if response.status_code == 200:
        return

    await response.aread()
    try:
        detail = response.json().get("detail") or response.json().get("error")
    except (json.JSONDecodeError, ValueError):
        detail = response.text.strip()[:300]

    if response.status_code in (401, 403):
        raise ToolError(
            "Auto Ontology rejected the credentials for this question. Sign in again, "
            "and check that the account is allowed to use chat. "
            f"({detail})"
        )
    if response.status_code == 409:
        # Either the semantic layer was never built, or this conversation
        # already has a turn in flight. The backend's detail distinguishes them.
        raise ToolError(f"Auto Ontology cannot answer right now: {detail}")
    raise ToolError(f"Auto Ontology returned HTTP {response.status_code}: {detail}")


def register(mcp: FastMCP, settings: Settings, client: httpx.AsyncClient) -> None:
    """Attach ``ask_question`` to *mcp*."""

    @mcp.tool(
        name="ask_question",
        description=(
            "Ask a natural-language question about the data connected to this "
            "Auto Ontology deployment. Auto Ontology resolves the question against its semantic "
            "layer, writes SQL, runs it, and returns the answer together with "
            "the SQL it used.\n\n"
            "This is the primary tool and the reason Auto Ontology exists. It is also "
            "slow — many sequential model calls, typically tens of seconds — "
            "so use check_answerable first when you are unsure the question is "
            "in scope, and prefer one well-formed question over several "
            "narrow ones."
        ),
    )
    async def ask_question(
        question: str,
        ctx: Context,
        conversation_id: str | None = None,
        evidence: str | None = None,
        target_db: str | None = None,
    ) -> DataAnswer:
        """Run one text-to-SQL turn and return its structured result.

        Args:
            question: The question, in plain language.
            conversation_id: Continue an existing thread, so follow-up
                questions resolve against earlier turns. Requires the caller
                to hold conversation-write permission, and rejects a second
                question while the first is still running.
            evidence: Optional authoritative evidence supplied separately
                from the question.
            target_db: Optional catalog database name or UUID to pin retrieval
                and SQL execution to a single connected database.
        """
        body: dict[str, Any] = {"question": question}
        if conversation_id:
            body["conversation_id"] = conversation_id
        if evidence:
            body["evidence"] = evidence
        if target_db and target_db.strip():
            body["target_db"] = target_db.strip()

        answer: DataAnswer | None = None
        streamed_queries: list[str] = []
        steps = 0

        try:
            async with client.stream(
                "POST",
                "/api/chat/completions",
                json=body,
                headers={"Accept": "text/event-stream"},
                timeout=settings.chat_timeout_s,
            ) as response:
                await _raise_for_status(response)

                async for line in response.aiter_lines():
                    event = _decode_event(line)
                    if event is None:
                        continue

                    kind = event.get("type")
                    if kind == "step":
                        steps += 1
                        label = event.get("label") or event.get("node") or "working"
                        thought = event.get("thought")
                        # No total: the graph's path depends on the question,
                        # so the step count is not known ahead of time.
                        await ctx.report_progress(
                            progress=steps,
                            message=f"{label}: {thought}" if thought else str(label),
                        )
                    elif kind == "sql":
                        sql_stmt = str(event.get("sql") or "").strip()
                        if sql_stmt and sql_stmt not in streamed_queries:
                            streamed_queries.append(sql_stmt)
                    elif kind == "result":
                        answer = _build_answer(
                            event.get("answer") or {},
                            requested_target_db=target_db,
                            streamed_queries=streamed_queries,
                            blob_store_dir=settings.blob_store_dir,
                        )
                    elif kind == "error":
                        raise ToolError(
                            f"Auto Ontology agent failed: {event.get('message') or 'unknown'}"
                        )
        except httpx.TimeoutException as exc:
            raise ToolError(
                f"Auto Ontology did not answer within {settings.chat_timeout_s:.0f}s. "
                "Raise AUTO_ONTOLOGY_MCP_CHAT_TIMEOUT_S, or ask a narrower question."
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolError(
                f"Could not reach Auto Ontology at {settings.api_url}: {exc}"
            ) from exc

        if answer is None:
            # The stream ended with no result and no error. This is what a
            # cancelled run looks like from the outside.
            raise ToolError(
                "Auto Ontology ended the run without producing an answer. It may have "
                "been cancelled, or the question may not map to the data."
            )
        return answer


__all__ = ["BlobRef", "DataAnswer", "MAX_ROWS", "register"]
