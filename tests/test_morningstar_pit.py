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


# =============================================================================
# FV UNCERTAINTY GATING (ST201)
# =============================================================================

class TestFVUncertaintyGating:
    """Tests for ST201 Fair Value Uncertainty → confidence multiplier."""

    @pytest.fixture
    def commercial_engine(self):
        """Engine with a commercial-stage ticker that has full analyst coverage."""
        e = MorningstarSignalEngine()
        e._data = {
            "COVERED": {
                "QV009": "100.00",
                "ST202": "110.00",
                "ST201": "Medium",
                "LT181": "Narrow",
                "STA4Z": "12.5",
                "HS08F": "8.0",
                "ST389": "0.5",
                "HS06U": "25.0",
                "HS035": "15.0",
                "HS08D": "5.0",
            },
        }
        e._snapshot_date = "2026-02-06"
        return e

    def test_low_uncertainty_boosts_fv_weight(self, commercial_engine):
        """Low uncertainty should multiply FV weight by 1.2x."""
        commercial_engine._data["COVERED"]["ST201"] = "Low"
        result = commercial_engine.score_ticker("COVERED", current_price=Decimal("90"), as_of_date=date(2026, 2, 6))
        assert result["status"] == "SUCCESS"
        fv_details = result["sub_details"]["fair_value_discount"]
        assert fv_details["uncertainty_multiplier"] == "1.20"
        assert "ms_fv_uncertainty_low" in result["flags"]

    def test_high_uncertainty_reduces_fv_weight(self, commercial_engine):
        """High uncertainty should multiply FV weight by 0.7x."""
        commercial_engine._data["COVERED"]["ST201"] = "High"
        result = commercial_engine.score_ticker("COVERED", current_price=Decimal("90"), as_of_date=date(2026, 2, 6))
        fv_details = result["sub_details"]["fair_value_discount"]
        assert fv_details["uncertainty_multiplier"] == "0.70"
        assert "ms_fv_uncertainty_high" in result["flags"]

    def test_very_high_uncertainty_strongly_reduces(self, commercial_engine):
        """Very High uncertainty should multiply FV weight by 0.4x."""
        commercial_engine._data["COVERED"]["ST201"] = "Very High"
        result = commercial_engine.score_ticker("COVERED", current_price=Decimal("90"), as_of_date=date(2026, 2, 6))
        fv_details = result["sub_details"]["fair_value_discount"]
        assert fv_details["uncertainty_multiplier"] == "0.40"
        assert "ms_fv_uncertainty_very_high" in result["flags"]

    def test_no_uncertainty_defaults_to_1x(self, commercial_engine):
        """Missing ST201 should default to 1.0x multiplier."""
        del commercial_engine._data["COVERED"]["ST201"]
        result = commercial_engine.score_ticker("COVERED", current_price=Decimal("90"), as_of_date=date(2026, 2, 6))
        fv_details = result["sub_details"]["fair_value_discount"]
        assert fv_details["uncertainty_multiplier"] == "1.00"

    def test_low_vs_high_changes_composite(self, commercial_engine):
        """Low uncertainty should produce a different composite than High."""
        commercial_engine._data["COVERED"]["ST201"] = "Low"
        r_low = commercial_engine.score_ticker("COVERED", current_price=Decimal("90"), as_of_date=date(2026, 2, 6))

        commercial_engine._data["COVERED"]["ST201"] = "High"
        r_high = commercial_engine.score_ticker("COVERED", current_price=Decimal("90"), as_of_date=date(2026, 2, 6))

        # With a discount (price < FV), Low uncertainty boosts FV weight,
        # giving the (high) FV score more influence → higher composite
        assert r_low["morningstar_score"] != r_high["morningstar_score"]


# =============================================================================
# ANALYST FV BLENDING (ST202 + QV009)
# =============================================================================

class TestAnalystFVBlending:
    """Tests for ST202 (analyst FV) blending with QV009 (quant FV)."""

    @pytest.fixture
    def blend_engine(self):
        """Engine with a ticker that has both analyst and quant FV."""
        e = MorningstarSignalEngine()
        e._data = {
            "BLEND": {
                "QV009": "100.00",
                "ST202": "120.00",
            },
        }
        e._snapshot_date = "2026-02-06"
        return e

    def test_blend_formula(self, blend_engine):
        """Effective FV should be 0.6*analyst + 0.4*quant."""
        result = blend_engine.score_ticker("BLEND", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        fv = result["sub_details"]["fair_value_discount"]
        # 0.6 * 120 + 0.4 * 100 = 72 + 40 = 112
        assert Decimal(fv["effective_fv"]) == Decimal("112")
        assert fv["fv_source"] == "analyst_quant_blend"
        assert "ms_fv_analyst_blend" in result["flags"]

    def test_quant_only_when_no_analyst(self, blend_engine):
        """Without ST202, should use QV009 alone."""
        del blend_engine._data["BLEND"]["ST202"]
        result = blend_engine.score_ticker("BLEND", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        fv = result["sub_details"]["fair_value_discount"]
        assert fv["effective_fv"] == "100.00"
        assert fv["fv_source"] == "quant_only"
        assert "ms_fv_analyst_blend" not in result["flags"]

    def test_divergence_flag_over_30pct(self, blend_engine):
        """Divergence > 30% between analyst and quant should flag."""
        blend_engine._data["BLEND"]["ST202"] = "140.00"  # 40% above QV009
        result = blend_engine.score_ticker("BLEND", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        assert "ms_fv_divergence" in result["flags"]
        fv = result["sub_details"]["fair_value_discount"]
        assert Decimal(fv["fv_divergence_pct"]) > Decimal("30")

    def test_no_divergence_flag_under_30pct(self, blend_engine):
        """Divergence < 30% should not flag."""
        blend_engine._data["BLEND"]["ST202"] = "110.00"  # 10% above QV009
        result = blend_engine.score_ticker("BLEND", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        assert "ms_fv_divergence" not in result["flags"]

    def test_analyst_fv_zero_falls_back_to_quant(self, blend_engine):
        """Analyst FV of 0 should be ignored (fall back to quant only)."""
        blend_engine._data["BLEND"]["ST202"] = "0"
        result = blend_engine.score_ticker("BLEND", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        fv = result["sub_details"]["fair_value_discount"]
        assert fv["fv_source"] == "quant_only"


# =============================================================================
# ECONOMIC MOAT SCORING (LT181)
# =============================================================================

class TestMoatQuality:
    """Tests for LT181 (Economic Moat) quality signal."""

    @pytest.fixture
    def moat_engine(self):
        """Engine with tickers in different moat/regime configurations."""
        e = MorningstarSignalEngine()
        e._data = {
            "WIDE_MOAT": {
                "QV009": "100.00",
                "LT181": "Wide",
                "STA4Z": "12.5",
                "HS08F": "8.0",
                "ST389": "0.5",
                "HS06U": "25.0",
                "HS035": "15.0",
                "HS08D": "5.0",
            },
            "NARROW_MOAT": {
                "QV009": "50.00",
                "LT181": "Narrow",
                "STA4Z": "12.5",
                "HS08F": "8.0",
                "ST389": "0.5",
                "HS06U": "25.0",
                "HS035": "15.0",
                "HS08D": "5.0",
            },
            "NO_MOAT": {
                "QV009": "30.00",
                "LT181": "None",
                "STA4Z": "12.5",
                "HS08F": "8.0",
                "ST389": "0.5",
                "HS06U": "25.0",
                "HS035": "15.0",
                "HS08D": "5.0",
            },
            "DEV_STAGE": {
                "QV009": "10.00",
                "LT181": "Wide",
                "HS08D": "-500.0",  # Deeply negative margin → dev regime
            },
            "NO_MOAT_DATA": {
                "QV009": "20.00",
                "STA4Z": "12.5",
                "HS08F": "8.0",
                "ST389": "0.5",
                "HS06U": "25.0",
                "HS035": "15.0",
                "HS08D": "5.0",
            },
        }
        e._snapshot_date = "2026-02-06"
        return e

    def test_wide_moat_scores_85(self, moat_engine):
        """Wide moat should score 85."""
        result = moat_engine.score_ticker("WIDE_MOAT", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        assert result["sub_details"]["moat_quality"]["score"] == Decimal("85.00")
        assert result["sub_details"]["moat_quality"]["moat"] == "Wide"
        assert "ms_wide_moat" in result["flags"]

    def test_narrow_moat_scores_65(self, moat_engine):
        """Narrow moat should score 65."""
        result = moat_engine.score_ticker("NARROW_MOAT", current_price=Decimal("50"), as_of_date=date(2026, 2, 6))
        assert result["sub_details"]["moat_quality"]["score"] == Decimal("65.00")

    def test_no_moat_scores_50(self, moat_engine):
        """None moat should score 50 (neutral)."""
        result = moat_engine.score_ticker("NO_MOAT", current_price=Decimal("30"), as_of_date=date(2026, 2, 6))
        assert result["sub_details"]["moat_quality"]["score"] == Decimal("50.00")
        assert "ms_no_moat" in result["flags"]

    def test_dev_stage_gets_neutral_moat(self, moat_engine):
        """Dev-stage ticker should get neutral moat score regardless of LT181."""
        result = moat_engine.score_ticker("DEV_STAGE", current_price=Decimal("10"), as_of_date=date(2026, 2, 6))
        assert result["sub_details"]["moat_quality"]["score"] == Decimal("50")
        assert "ms_moat_neutral_dev_stage" in result["flags"]

    def test_missing_moat_returns_none(self, moat_engine):
        """Ticker without LT181 in commercial regime should return None (redistributed)."""
        result = moat_engine.score_ticker("NO_MOAT_DATA", current_price=Decimal("20"), as_of_date=date(2026, 2, 6))
        assert result["sub_details"]["moat_quality"]["score"] is None
        assert "ms_moat_no_data" in result["flags"]

    def test_moat_quality_in_return_dict(self, moat_engine):
        """moat_quality_score should appear in the result dict."""
        result = moat_engine.score_ticker("WIDE_MOAT", current_price=Decimal("100"), as_of_date=date(2026, 2, 6))
        assert "moat_quality_score" in result
        assert result["moat_quality_score"] == Decimal("85.00")
