# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for deterministic sample-value PII scrubbing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from auto_ontology.semantic.sample_scrubber import (
    is_sample_scrubbing_enabled,
    scrub_sample_list,
    scrub_sample_mapping,
    scrub_sample_text,
    scrub_sample_value,
)
from auto_ontology.semantic.tests.test_pii_canaries import (
    CANARY,
    STALE_CANARY,
    _compile,
    _leaks,
    _SqliteWarehouse,
)


def test_scrub_sample_text_redacts_structured_pii_and_keys() -> None:
    raw = (
        "Contact jane.doe@acme.example or (415) 555-0199; "
        "SSN 123-45-6789; card 4532 0151 1283 0366; "
        "token sk-abcdefghijklmnopqrstuvwx"
    )
    scrubbed = scrub_sample_text(raw)
    assert "jane.doe@acme.example" not in scrubbed
    assert "415" not in scrubbed
    assert "123-45-6789" not in scrubbed
    assert "4532 0151 1283 0366" not in scrubbed
    assert "sk-abcdefghijklmnopqrstuvwx" not in scrubbed
    assert "[REDACTED:EMAIL]" in scrubbed
    assert "[REDACTED:PHONE]" in scrubbed
    assert "[REDACTED:SSN]" in scrubbed
    assert "[REDACTED:CREDIT_CARD]" in scrubbed
    assert "[REDACTED:API_KEY]" in scrubbed


def test_non_luhn_numbers_and_clean_categories_remain_intact() -> None:
    # 16 digits failing Luhn check are not falsely redacted as payment cards.
    assert scrub_sample_text("ORDER-1234567890123451") == "ORDER-1234567890123451"
    assert scrub_sample_value(42) == 42
    assert scrub_sample_value(3.14) == 3.14
    assert scrub_sample_value(True) is True
    assert scrub_sample_value(None) is None
    assert scrub_sample_list(["ACTIVE", "CLOSED", 10]) == ["ACTIVE", "CLOSED", 10]


def test_scrub_sample_mapping_preserves_date_warning_banner() -> None:
    warning = "[WARNING mixed date formats observed: YYYY-MM-DD, YYYY/MM/DD]"
    mapped = scrub_sample_mapping(
        {
            "notes": ["user@corp.example", warning],
            "amount": [100, 200],
        }
    )
    assert mapped == {
        "notes": ["[REDACTED:EMAIL]", warning],
        "amount": [100, 200],
    }


def test_enablement_follows_env_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUTO_ONTOLOGY_SCRUB_SAMPLE_VALUES", raising=False)
    monkeypatch.delenv("AUTO_ONTOLOGY_OFFLINE_PERIMETER", raising=False)
    assert is_sample_scrubbing_enabled() is False

    monkeypatch.setenv("AUTO_ONTOLOGY_OFFLINE_PERIMETER", "1")
    assert is_sample_scrubbing_enabled() is True

    monkeypatch.setenv("AUTO_ONTOLOGY_SCRUB_SAMPLE_VALUES", "0")
    assert is_sample_scrubbing_enabled() is False


def test_process_table_scrubs_untagged_email_samples_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTO_ONTOLOGY_SCRUB_SAMPLE_VALUES", "1")
    warehouse = _SqliteWarehouse()

    recorded = _compile(warehouse, MagicMock(return_value=set()))

    # Warehouse SQL still ran because columns were untagged, but no email canary
    # survived into persisted sample_values, distinct_values, or FK prompts!
    sink_leaks = [
        place
        for place in _leaks(warehouse, recorded, f"{CANARY}-0@example.invalid")
        if not place.startswith("warehouse result")
    ]
    assert sink_leaks == []
    assert _leaks(warehouse, recorded, STALE_CANARY) == []
    assert "open" in repr(recorded["store_column_sample_values"].call_args_list)


def test_dal_store_sample_values_scrubs_on_write_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from auto_ontology.dal import datasources

    monkeypatch.setenv("AUTO_ONTOLOGY_SCRUB_SAMPLE_VALUES", "1")
    with patch.object(datasources, "_set_column_property") as mock_set:
        datasources.store_column_sample_values(
            "t-1",
            {"comments": ["reach alice@corp.io", "clean-status"]},
        )
    mock_set.assert_called_once()
    encoded = mock_set.call_args[0][1]
    assert "alice@corp.io" not in encoded["comments"]
    assert "[REDACTED:EMAIL]" in encoded["comments"]
    assert "clean-status" in encoded["comments"]
