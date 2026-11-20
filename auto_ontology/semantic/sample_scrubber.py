# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic PII scrubbing for sampled warehouse values before persistence.

Columns explicitly tagged ``PII`` in the catalog are excluded from sampling SQL
altogether (:func:`auto_ontology.semantic.visit_enter.pii_column_names`).
However, during initial ingestion or on free-text columns (e.g. ``notes``,
``comments``, ``audit_payload``) not yet tagged ``PII``, sampled strings can
carry structured identifiers or credentials.

When ``AUTO_ONTOLOGY_SCRUB_SAMPLE_VALUES=1`` (or
``AUTO_ONTOLOGY_OFFLINE_PERIMETER=1``) is enabled, every sampled string value is
passed through deterministic high-precision detectors (API/cloud keys, US SSN,
Luhn-validated payment cards, email addresses, and phone numbers) before it is
persisted to ``catalog_column`` or embedded into ``pgvector``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_API_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|nvapi-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{20,})\b"
)
_SSN_RE = re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b")
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,24}\b")
_PAN_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
)


def is_sample_scrubbing_enabled() -> bool:
    """Return ``True`` when automatic sample-value PII scrubbing is enabled."""
    explicit = os.environ.get("AUTO_ONTOLOGY_SCRUB_SAMPLE_VALUES", "").strip().lower()
    if explicit:
        return explicit in _TRUTHY
    offline = os.environ.get("AUTO_ONTOLOGY_OFFLINE_PERIMETER", "").strip().lower()
    return offline in _TRUTHY


def _passes_luhn(digits: str) -> bool:
    if not (13 <= len(digits) <= 19) or not digits.isdigit():
        return False
    if len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for idx, ch in enumerate(digits):
        d = ord(ch) - 48
        if idx % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _replace_luhn_pan(match: re.Match[str]) -> str:
    raw = match.group(0)
    digits = "".join(ch for ch in raw if ch.isdigit())
    if _passes_luhn(digits):
        return "[REDACTED:CREDIT_CARD]"
    return raw


def scrub_sample_text(text: str) -> str:
    """Redact structured credentials and PII spans inside *text*."""
    if not text or text.startswith("[WARNING mixed date formats observed:"):
        return text
    scrubbed = _API_KEY_RE.sub("[REDACTED:API_KEY]", text)
    scrubbed = _SSN_RE.sub("[REDACTED:SSN]", scrubbed)
    scrubbed = _PAN_CANDIDATE_RE.sub(_replace_luhn_pan, scrubbed)
    scrubbed = _EMAIL_RE.sub("[REDACTED:EMAIL]", scrubbed)
    scrubbed = _PHONE_RE.sub("[REDACTED:PHONE]", scrubbed)
    return scrubbed


def scrub_sample_value(value: Any) -> Any:
    """Scrub *value* if it is a string; preserve numeric/boolean/null types."""
    if isinstance(value, str):
        return scrub_sample_text(value)
    return value


def scrub_sample_list(values: Iterable[Any]) -> list[Any]:
    """Return a list with every string element scrubbed."""
    return [scrub_sample_value(v) for v in values]


def scrub_sample_mapping(samples: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Return a shallow-copied column->values mapping with string samples scrubbed."""
    return {col: scrub_sample_list(vals) for col, vals in samples.items()}
