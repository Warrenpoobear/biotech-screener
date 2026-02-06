"""
Tests for Morningstar Signal Engine: PIT clipping and ID collision resolution.

Uses synthetic data injected directly into engine internals (no filesystem dependency).

Tests cover:
- PIT price clipping (as_of_date filtering)
- Staleness rejection (7-day boundary)
- score_ticker integration with historical prices
- ID collision resolution (broadcast from donor to recipient)

Author: Wake Robin Capital Management
"""
import json
import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from morningstar_signal_engine import MorningstarSignalEngine


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def engine():
    """Create a fresh MorningstarSignalEngine with synthetic data."""
    e = MorningstarSignalEngine()
    # Inject synthetic fundamental data
    e._data = {
        "TESTCO": {
            "QV009": "50.00",   # Quantitative Fair Value = $50
            "STA4Z": "12.5",    # ROIC
            "HS08F": "8.0",     # ROE
            "ST389": "0.5",     # D/E
            "HS06U": "25.0",    # D/Capital
            "HS035": "15.0",    # Sales Growth
            "HS08D": "5.0",     # Net Margin
        },
    }
    e._snapshot_date = "2026-06-30"
    return e


@pytest.fixture
def engine_with_prices(engine):
    """Engine with synthetic price history time series."""
    engine._price_history = {
        "TESTCO": {
            "HS377": {
                "time_series": [
                    {"date": "2026-06-10", "value": "42.00"},
                    {"date": "2026-06-15", "value": "44.50"},
                    {"date": "2026-06-20", "value": "45.00"},
                    {"date": "2026-06-28", "value": "46.00"},
                    {"date": "2026-07-05", "value": "48.00"},
                    {"date": "2026-07-10", "value": "49.50"},
                ],
            },
        },
    }
    return engine


# =============================================================================
# PIT CLIPPING TESTS
# =============================================================================

class TestPITClipping:
    """Tests for point-in-time price clipping logic."""

    def test_clips_to_as_of_date(self, engine_with_prices):
        """as_of=Jun30 should return Jun28 price, ignoring July entries."""
        price = engine_with_prices._get_latest_price("TESTCO", as_of_date=date(2026, 6, 30))
        assert price == Decimal("46.00"), f"Expected Jun28 price 46.00, got {price}"

    def test_exact_date_match(self, engine_with_prices):
        """as_of=Jun15 should return exactly Jun15 price."""
        price = engine_with_prices._get_latest_price("TESTCO", as_of_date=date(2026, 6, 15))
        assert price == Decimal("44.50"), f"Expected Jun15 price 44.50, got {price}"

    def test_staleness_rejection_7_days(self, engine_with_prices):
        """Price 9 days old should be rejected (beyond 7-day staleness window)."""
        # as_of = Jun 29, nearest price is Jun 20 (9 days gap)
        # But Jun 28 is only 1 day old - need a gap.
        # Use as_of = Jun 28, data only at Jun 10,15,20 - Jun 20 is 8 days old = stale
        engine_with_prices._price_history["STALE_CO"] = {
            "HS377": {
                "time_series": [
                    {"date": "2026-06-10", "value": "30.00"},
                    {"date": "2026-06-15", "value": "31.00"},
                ],
            },
        }
        # as_of = Jun 25: nearest is Jun 15 which is 10 days old → stale
        price = engine_with_prices._get_latest_price("STALE_CO", as_of_date=date(2026, 6, 25))
        assert price is None, f"Expected None for stale price (10 days), got {price}"

    def test_staleness_boundary_exactly_7_days(self, engine_with_prices):
        """Price exactly 7 days old should still be accepted."""
        engine_with_prices._price_history["BOUNDARY_CO"] = {
            "HS377": {
                "time_series": [
                    {"date": "2026-06-18", "value": "35.00"},
                ],
            },
        }
        # as_of = Jun 25: Jun 18 is exactly 7 days old → accepted
        price = engine_with_prices._get_latest_price("BOUNDARY_CO", as_of_date=date(2026, 6, 25))
        assert price == Decimal("35.00"), f"Expected 35.00 at 7-day boundary, got {price}"

    def test_no_data_before_as_of_date(self, engine_with_prices):
        """as_of before all data entries should return None."""
        price = engine_with_prices._get_latest_price("TESTCO", as_of_date=date(2026, 1, 1))
        assert price is None, f"Expected None for date before all data, got {price}"

    def test_none_as_of_date_returns_last(self, engine_with_prices):
        """None as_of → last entry in time series (no clipping)."""
        price = engine_with_prices._get_latest_price("TESTCO", as_of_date=None)
        assert price == Decimal("49.50"), f"Expected last entry 49.50, got {price}"


# =============================================================================
# SCORING WITH PIT PRICES
# =============================================================================

class TestScoringWithPITPrices:
    """Tests that score_ticker respects PIT price clipping."""

    def test_scoring_uses_clipped_price(self, engine_with_prices):
        """score_ticker with historical as_of_date should use PIT-clipped price."""
        result = engine_with_prices.score_ticker(
            "TESTCO",
            current_price=None,  # Force use of price history
            as_of_date=date(2026, 6, 30),
        )
        assert result["status"] == "SUCCESS"
        # Should have used Jun28 price (46.00) from history
        assert "ms_price_from_history" in result["flags"]
        # FV discount should be based on QV=50, price=46 → ~8% discount
        fv_score = result["fair_value_discount_score"]
        assert fv_score is not None, "Expected FV score with PIT price"

    def test_scoring_stale_price_skips_fv_from_history(self, engine_with_prices):
        """score_ticker where price history is stale → no FV from history."""
        engine_with_prices._price_history["STALE_ONLY"] = {
            "HS377": {
                "time_series": [
                    {"date": "2026-01-01", "value": "20.00"},
                ],
            },
        }
        engine_with_prices._data["STALE_ONLY"] = {
            "QV009": "30.00",
        }
        result = engine_with_prices.score_ticker(
            "STALE_ONLY",
            current_price=None,
            as_of_date=date(2026, 6, 30),
        )
        assert result["status"] == "SUCCESS"
        # Price is ~180 days old → stale → no price from history
        assert "ms_price_from_history" not in result["flags"]


# =============================================================================
# ID COLLISION RESOLUTION TESTS
# =============================================================================

class TestIDCollisionResolution:
    """Tests for _resolve_id_collisions() method."""

    @pytest.fixture
    def collision_engine(self, tmp_path):
        """Engine with ID collision scenario (INBX has data, IKT does not)."""
        e = MorningstarSignalEngine()
        # INBX has data, IKT does not
        e._data = {
            "INBX": {
                "QV009": "25.00",
                "STA4Z": "5.0",
                "HS08F": "-3.0",
            },
        }
        e._price_history = {
            "INBX": {
                "HS377": {
                    "time_series": [
                        {"date": "2026-06-15", "value": "22.00"},
                        {"date": "2026-06-20", "value": "23.50"},
                    ],
                },
            },
        }

        # Write ID map with collision
        id_map = {
            "ticker_to_id": {
                "INBX": "0P0001SXEI",
                "IKT": "0P0001SXEI",
                "SOLO": "0P000UNIQUE",
            }
        }
        id_map_file = tmp_path / "morningstar_id_map.json"
        id_map_file.write_text(json.dumps(id_map))

        return e, id_map_file

    def test_missing_ticker_gets_donor_data(self, collision_engine):
        """IKT (no data) gets INBX's fundamentals via broadcast."""
        engine, id_map_file = collision_engine
        assert "IKT" not in engine._data

        engine._resolve_id_collisions(id_map_file)

        assert "IKT" in engine._data
        assert engine._data["IKT"]["QV009"] == "25.00"
        assert engine._data["IKT"]["STA4Z"] == "5.0"

    def test_both_present_no_overwrite(self, collision_engine):
        """Existing data should NOT be clobbered by broadcast."""
        engine, id_map_file = collision_engine

        # Give IKT its own data before resolution
        engine._data["IKT"] = {
            "QV009": "99.99",
            "STA4Z": "20.0",
        }

        engine._resolve_id_collisions(id_map_file)

        # IKT's data should be untouched
        assert engine._data["IKT"]["QV009"] == "99.99"
        assert engine._data["IKT"]["STA4Z"] == "20.0"

    def test_price_history_also_broadcast(self, collision_engine):
        """Price history should be broadcast alongside fundamentals."""
        engine, id_map_file = collision_engine
        assert "IKT" not in engine._price_history

        engine._resolve_id_collisions(id_map_file)

        assert "IKT" in engine._price_history
        ts = engine._price_history["IKT"]["HS377"]["time_series"]
        assert len(ts) == 2
        assert ts[0]["value"] == "22.00"
