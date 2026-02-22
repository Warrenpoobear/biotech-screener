"""Tests for scripts/build_catalyst_events_pit.py — PIT-safe catalyst events."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from scripts.build_catalyst_events_pit import (
    DEFAULT_CATALYST_MID_DAYS,
    DEFAULT_CATALYST_NEAR_DAYS,
    DEFAULT_DECAY_MODE,
    SCHEMA_VERSION,
    _count_ledger_sources,
    _count_source_mix,
    _derive_catalyst_decay_w,
    _derive_catalyst_features,
    _derive_catalyst_mode,
    _derive_catalyst_strength,
    build_catalyst_events,
    validate_catalyst_events_schema,
)


# ---------------------------------------------------------------------------
# Minimal LedgerEntry stub for testing (avoids importing event_ledger)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FakeLedgerEntry:
    """Minimal LedgerEntry for test fixtures."""
    event_id: str = "evt_001"
    ticker: str = "ACAD"
    event_type: str = "DATA_READOUT"
    event_date: Optional[str] = "2026-03-01"
    event_date_end: Optional[str] = None
    date_precision: str = "DAY"
    disclosed_at: str = "2026-01-10"
    source: str = "SEC_8K"
    source_uid: str = "0001234567-26-000001"
    confidence: str = "HIGH"
    extractor_version: str = "v1"
    event_name: str = "8-K: Phase 3 data expected H1 2026"
    field_changed: str = ""
    value: str = ""
    tags: Tuple[str, ...] = ()


AS_OF = "2026-01-15"
AS_OF_DATE = date(2026, 1, 15)


def _fake_query_nearest(ledger, ticker, as_of_date):
    """Simplified query_nearest_catalyst for testing."""
    as_of_str = as_of_date.isoformat()
    best_days = None
    best_entry = None
    for entry in ledger:
        if entry.ticker != ticker:
            continue
        if entry.confidence == "LOW":
            continue
        if not entry.event_date or entry.event_date <= as_of_str:
            continue
        days = (date.fromisoformat(entry.event_date) - as_of_date).days
        if best_days is None or days < best_days:
            best_days = days
            best_entry = entry
    if best_entry is None:
        return None
    return {
        "days_to_catalyst": best_days,
        "event_id": best_entry.event_id,
        "disclosed_at": best_entry.disclosed_at,
        "source": best_entry.source,
        "source_uid": best_entry.source_uid,
        "event_type": best_entry.event_type,
        "event_name": best_entry.event_name,
        "event_date": best_entry.event_date,
    }


# ===================================================================
# TestCatalystModeDerivation
# ===================================================================

class TestCatalystModeDerivation:
    """Catalyst mode derivation from days_to_catalyst."""

    def test_positive_days_specific(self):
        """days_to_catalyst > 0 → 'specific_days'."""
        assert _derive_catalyst_mode(45) == "specific_days"

    def test_zero_days_blended(self):
        """days_to_catalyst = 0 → 'blended_window'."""
        assert _derive_catalyst_mode(0) == "blended_window"

    def test_none_days_missing(self):
        """No nearest catalyst → 'missing'."""
        assert _derive_catalyst_mode(None) == "missing"

    def test_strength_near(self):
        """days <= near_days → 'near'."""
        s = _derive_catalyst_strength("specific_days", 60, 120, 180)
        assert s == "near"

    def test_strength_mid(self):
        """near_days < days <= mid_days → 'mid'."""
        s = _derive_catalyst_strength("specific_days", 150, 120, 180)
        assert s == "mid"

    def test_strength_far(self):
        """days > mid_days → 'far'."""
        s = _derive_catalyst_strength("specific_days", 200, 120, 180)
        assert s == "far"

    def test_strength_blended_always_near(self):
        """blended_window → always 'near'."""
        s = _derive_catalyst_strength("blended_window", 0, 120, 180)
        assert s == "near"

    def test_strength_missing_mode(self):
        """missing mode → 'missing' strength."""
        s = _derive_catalyst_strength("missing", None, 120, 180)
        assert s == "missing"


# ===================================================================
# TestDecayWeight
# ===================================================================

class TestDecayWeight:
    """Catalyst decay weight tests."""

    def test_step_near(self):
        """Step mode: near → 1.0."""
        w = _derive_catalyst_decay_w("specific_days", "near", 60, "step")
        assert w == 1.0

    def test_step_mid(self):
        """Step mode: mid → 0.7."""
        w = _derive_catalyst_decay_w("specific_days", "mid", 150, "step")
        assert w == 0.7

    def test_step_far(self):
        """Step mode: far → 0.3."""
        w = _derive_catalyst_decay_w("specific_days", "far", 200, "step")
        assert w == 0.3

    def test_step_missing(self):
        """Step mode: missing → 0.0."""
        w = _derive_catalyst_decay_w("missing", "missing", None, "step")
        assert w == 0.0


# ===================================================================
# TestDeriveFeatures
# ===================================================================

class TestDeriveFeatures:
    """Full feature derivation from nearest catalyst."""

    def test_with_nearest(self):
        """Nearest catalyst → populated features."""
        nearest = {
            "days_to_catalyst": 45,
            "event_id": "evt_001",
            "disclosed_at": "2026-01-10",
            "source": "SEC_8K",
            "source_uid": "acc123",
            "event_type": "DATA_READOUT",
            "event_name": "Phase 3 data",
            "event_date": "2026-03-01",
        }
        f = _derive_catalyst_features(nearest, near_days=120, mid_days=180)
        assert f["days_to_catalyst"] == 45
        assert f["catalyst_mode"] == "specific_days"
        assert f["catalyst_strength"] == "near"
        assert f["catalyst_decay_w"] == 1.0
        assert f["nearest_event_type"] == "DATA_READOUT"
        assert f["nearest_event_source"] == "SEC_8K"

    def test_without_nearest(self):
        """No nearest catalyst → missing features."""
        f = _derive_catalyst_features(None)
        assert f["days_to_catalyst"] is None
        assert f["catalyst_mode"] == "missing"
        assert f["catalyst_strength"] == "missing"
        assert f["catalyst_decay_w"] == 0.0
        assert f["nearest_event_type"] == ""


# ===================================================================
# TestSourceMix
# ===================================================================

class TestSourceMix:
    """Source mix counting tests."""

    def test_multiple_sources(self):
        """Multiple sources counted correctly per ticker."""
        ledger = [
            FakeLedgerEntry(ticker="ACAD", source="SEC_8K"),
            FakeLedgerEntry(ticker="ACAD", source="SEC_8K"),
            FakeLedgerEntry(ticker="ACAD", source="CTGOV"),
            FakeLedgerEntry(ticker="AARD", source="CTGOV"),
        ]
        mix = _count_source_mix(ledger, "ACAD")
        assert mix == {"SEC_8K": 2, "CTGOV": 1}

    def test_zero_events(self):
        """Zero events → empty source_mix dict."""
        ledger = [FakeLedgerEntry(ticker="AARD")]
        mix = _count_source_mix(ledger, "ACAD")
        assert mix == {}

    def test_event_count_equals_sum(self):
        """Event count = sum of source_mix values."""
        ledger = [
            FakeLedgerEntry(ticker="ACAD", source="SEC_8K"),
            FakeLedgerEntry(ticker="ACAD", source="CTGOV"),
        ]
        mix = _count_source_mix(ledger, "ACAD")
        assert sum(mix.values()) == 2

    def test_ledger_sources_total(self):
        """Total ledger source counts across all tickers."""
        ledger = [
            FakeLedgerEntry(ticker="ACAD", source="SEC_8K"),
            FakeLedgerEntry(ticker="AARD", source="CTGOV"),
            FakeLedgerEntry(ticker="ALNY", source="SEC_8K"),
        ]
        sources = _count_ledger_sources(ledger)
        assert sources == {"SEC_8K": 2, "CTGOV": 1}


# ===================================================================
# TestBuildCatalystEvents (integration with fake ledger)
# ===================================================================

class TestBuildCatalystEvents:
    """Integration tests using fake ledger entries."""

    def _build_with_fake_ledger(self, ledger, universe, **kwargs):
        """Build catalyst events with a fake ledger and query function."""
        with patch(
            "scripts.build_catalyst_events_pit.build_catalyst_events",
            wraps=build_catalyst_events,
        ):
            # We need to mock query_nearest_catalyst to use our fake
            with patch(
                "event_ledger.query_nearest_catalyst",
                side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
            ):
                return build_catalyst_events(
                    as_of_date=AS_OF,
                    universe_tickers=universe,
                    ledger=ledger,
                    **kwargs,
                )

    def test_populated_ledger(self):
        """Ledger with entries → correct per-ticker features."""
        ledger = [
            FakeLedgerEntry(
                ticker="ACAD", event_date="2026-03-01",
                source="SEC_8K", event_type="DATA_READOUT",
                disclosed_at="2026-01-10",
            ),
            FakeLedgerEntry(
                ticker="ACAD", event_date="2026-06-15",
                source="CTGOV", event_type="CT_PRIMARY_COMPLETION",
                disclosed_at="2025-06-01",
            ),
        ]
        features = self._build_with_fake_ledger(
            ledger, {"ACAD", "AARD"},
        )
        # ACAD should have nearest catalyst at 2026-03-01
        acad = features["tickers"]["ACAD"]
        assert acad["days_to_catalyst"] == (date(2026, 3, 1) - AS_OF_DATE).days
        assert acad["catalyst_mode"] == "specific_days"
        assert acad["event_count"] == 2
        assert acad["source_mix"] == {"SEC_8K": 1, "CTGOV": 1}

        # AARD should be missing
        aard = features["tickers"]["AARD"]
        assert aard["catalyst_mode"] == "missing"
        assert aard["event_count"] == 0

    def test_empty_ledger(self):
        """Empty ledger → all tickers have 'missing' catalyst_mode."""
        features = self._build_with_fake_ledger([], {"ACAD", "AARD"})
        for tk, td in features["tickers"].items():
            assert td["catalyst_mode"] == "missing"
        assert features["tickers_with_catalyst"] == 0

    def test_ticker_not_in_ledger(self):
        """Ticker not in ledger → 'missing' mode."""
        ledger = [FakeLedgerEntry(ticker="ACAD", event_date="2026-03-01")]
        features = self._build_with_fake_ledger(
            ledger, {"AARD"},
        )
        assert features["tickers"]["AARD"]["catalyst_mode"] == "missing"

    def test_low_confidence_skipped(self):
        """LOW confidence events skipped by query_nearest_catalyst."""
        ledger = [
            FakeLedgerEntry(
                ticker="ACAD", event_date="2026-03-01", confidence="LOW",
            ),
        ]
        features = self._build_with_fake_ledger(
            ledger, {"ACAD"},
        )
        assert features["tickers"]["ACAD"]["catalyst_mode"] == "missing"

    def test_multiple_events_nearest_selected(self):
        """Multiple events same ticker → nearest selected."""
        ledger = [
            FakeLedgerEntry(
                ticker="ACAD", event_date="2026-06-15",
                event_id="far",
            ),
            FakeLedgerEntry(
                ticker="ACAD", event_date="2026-02-01",
                event_id="near",
            ),
        ]
        features = self._build_with_fake_ledger(
            ledger, {"ACAD"},
        )
        acad = features["tickers"]["ACAD"]
        expected_days = (date(2026, 2, 1) - AS_OF_DATE).days
        assert acad["days_to_catalyst"] == expected_days


# ===================================================================
# TestSchemaValidation
# ===================================================================

class TestSchemaValidation:
    """Schema validation tests."""

    def test_full_output_passes(self):
        """Full output passes schema validation."""
        with patch(
            "event_ledger.query_nearest_catalyst",
            side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
        ):
            features = build_catalyst_events(
                AS_OF, {"ACAD"}, ledger=[],
            )
        ok, reason = validate_catalyst_events_schema(features)
        assert ok, reason

    def test_coverage_consistency(self):
        """Coverage consistency check."""
        with patch(
            "event_ledger.query_nearest_catalyst",
            side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
        ):
            features = build_catalyst_events(
                AS_OF, {"ACAD", "AARD"}, ledger=[],
            )
        n_uni = features["tickers_in_universe"]
        n_cat = features["tickers_with_catalyst"]
        expected = round(n_cat / n_uni * 100, 1) if n_uni > 0 else 0.0
        assert abs(features["catalyst_coverage_pct"] - expected) < 0.2

    def test_provenance_has_required_fields(self):
        """Provenance has catalyst_near_days / catalyst_mid_days."""
        with patch(
            "event_ledger.query_nearest_catalyst",
            side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
        ):
            features = build_catalyst_events(
                AS_OF, {"ACAD"}, ledger=[],
            )
        prov = features["provenance"]
        assert prov["catalyst_near_days"] == DEFAULT_CATALYST_NEAR_DAYS
        assert prov["catalyst_mid_days"] == DEFAULT_CATALYST_MID_DAYS
        assert prov["decay_mode"] == DEFAULT_DECAY_MODE

    def test_missing_field_detected(self):
        """Missing required field → validation fails."""
        with patch(
            "event_ledger.query_nearest_catalyst",
            side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
        ):
            features = build_catalyst_events(
                AS_OF, {"ACAD"}, ledger=[],
            )
        del features["schema_version"]
        ok, reason = validate_catalyst_events_schema(features)
        assert not ok
        assert "schema_version" in reason


# ===================================================================
# TestEdgeCases
# ===================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_empty_universe(self):
        """Empty universe → zero coverage."""
        with patch(
            "event_ledger.query_nearest_catalyst",
            side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
        ):
            features = build_catalyst_events(AS_OF, set(), ledger=[])
        assert features["tickers_in_universe"] == 0
        assert features["tickers_with_catalyst"] == 0
        assert features["catalyst_coverage_pct"] == 0.0

    def test_no_universe_tickers_in_ledger(self):
        """No universe tickers in ledger → zero coverage."""
        ledger = [FakeLedgerEntry(ticker="ZZZZ", event_date="2026-03-01")]
        with patch(
            "event_ledger.query_nearest_catalyst",
            side_effect=lambda lg, tk, d: _fake_query_nearest(lg, tk, d),
        ):
            features = build_catalyst_events(
                AS_OF, {"ACAD"}, ledger=ledger,
            )
        assert features["tickers_with_catalyst"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
