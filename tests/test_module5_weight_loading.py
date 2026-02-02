"""
Tests for Module 5 calibrated weight loading from JSON.

Tests the _load_module5_weights helper in run_screen.py.
No production_data used.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from run_screen import (
    _load_module5_weights,
    _load_module5_weights_bundle,
    _normalize_m3_regime,
    _select_module5_weights,
)


class TestLoadModule5Weights:
    def _write_weights(self, path: Path, feature_weights: dict, extra: dict | None = None):
        data = {"feature_weights": feature_weights}
        if extra:
            data.update(extra)
        path.write_text(json.dumps(data, sort_keys=True, indent=2))

    def test_loads_valid_weights(self, tmp_path):
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_normalized": -0.35,
            "momentum_normalized": -0.25,
            "financial_normalized": -0.15,
            "clinical_normalized": 0.25,
        })
        result = _load_module5_weights(path)
        assert result is not None
        # Should map to component names (strip _normalized suffix)
        assert "valuation" in result
        assert "momentum" in result
        assert "financial" in result
        assert "clinical" in result

    def test_negative_weights_preserved(self, tmp_path):
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_normalized": -0.50,
            "momentum_normalized": -0.30,
        })
        result = _load_module5_weights(path)
        assert result["valuation"] < 0
        assert result["momentum"] < 0

    def test_decimal_conversion(self, tmp_path):
        path = tmp_path / "w.json"
        self._write_weights(path, {"valuation_normalized": -0.35})
        result = _load_module5_weights(path)
        assert isinstance(result["valuation"], Decimal)
        assert result["valuation"] == Decimal("-0.35")

    def test_zero_weights_skipped(self, tmp_path):
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_normalized": -0.50,
            "catalyst_normalized": 0.0,
        })
        result = _load_module5_weights(path)
        assert "valuation" in result
        assert "catalyst" not in result  # zero weights are skipped

    def test_fallback_on_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        result = _load_module5_weights(path)
        assert result is None

    def test_fallback_on_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        result = _load_module5_weights(path)
        assert result is None

    def test_fallback_on_missing_feature_weights_key(self, tmp_path):
        path = tmp_path / "w.json"
        path.write_text(json.dumps({"method": "ridge"}))
        result = _load_module5_weights(path)
        assert result is None

    def test_fallback_on_all_zero_weights(self, tmp_path):
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_normalized": 0.0,
            "momentum_normalized": 0.0,
        })
        result = _load_module5_weights(path)
        assert result is None

    def test_suffix_stripping(self, tmp_path):
        """Various suffixes are stripped to get component name."""
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_normalized": -0.3,
            "clinical_raw": 0.2,
            "financial_contribution": 0.1,
            "momentum_confidence": -0.1,
        })
        result = _load_module5_weights(path)
        assert set(result.keys()) == {"valuation", "clinical", "financial", "momentum"}

    def test_non_numeric_weight_skipped(self, tmp_path):
        """Non-numeric weight values are skipped gracefully."""
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_normalized": -0.35,
            "momentum_normalized": "not_a_number",
        })
        result = _load_module5_weights(path)
        assert result is not None
        assert "valuation" in result
        assert "momentum" not in result

    def test_duplicate_component_keeps_normalized(self, tmp_path):
        """If two features map to same component, prefer _normalized suffix."""
        path = tmp_path / "w.json"
        self._write_weights(path, {
            "valuation_contribution": 0.2,
            "valuation_normalized": -0.5,
        })
        result = _load_module5_weights(path)
        # _normalized has priority over _contribution
        assert result["valuation"] == Decimal("-0.5")


# ── Bundle loading (schema v2) ───────────────────────────────────────

def _make_bundle_json(
    path: Path,
    global_fw: dict,
    by_regime: dict | None = None,
    blend_k: int = 25,
    min_fit_weeks: int = 5,
):
    """Write a schema v2 bundle JSON."""
    data = {
        "schema_version": 2,
        "blend": {"k": blend_k, "min_fit_weeks": min_fit_weeks},
        "global": {"feature_weights": global_fw},
        "by_regime": by_regime or {},
    }
    path.write_text(json.dumps(data, indent=2))


def _make_v1_json(path: Path, feature_weights: dict):
    """Write a schema v1 (legacy) JSON."""
    path.write_text(json.dumps({"feature_weights": feature_weights}, indent=2))


class TestLoadBundle:
    def test_v1_loaded_as_global_only(self, tmp_path):
        """Legacy v1 format wraps into global-only bundle."""
        path = tmp_path / "w.json"
        _make_v1_json(path, {"valuation_normalized": -0.35, "momentum_normalized": -0.25})
        bundle = _load_module5_weights_bundle(path)
        assert bundle is not None
        assert "valuation" in bundle["global"]
        assert "momentum" in bundle["global"]
        assert bundle["by_regime"] == {}
        assert bundle["blend_k"] == 25
        assert bundle["min_fit_weeks"] == 5

    def test_v2_loads_global_and_regime(self, tmp_path):
        """Schema v2 loads global + per-regime weights."""
        path = tmp_path / "w.json"
        _make_bundle_json(
            path,
            global_fw={"valuation_normalized": -0.35, "clinical_normalized": -0.10},
            by_regime={
                "BULL": {"n_weeks": 20, "feature_weights": {
                    "valuation_normalized": -0.40, "clinical_normalized": -0.05,
                }},
                "BEAR": {"n_weeks": 12, "feature_weights": {
                    "valuation_normalized": -0.20, "clinical_normalized": -0.30,
                }},
            },
            blend_k=30,
            min_fit_weeks=8,
        )
        bundle = _load_module5_weights_bundle(path)
        assert bundle is not None
        assert "valuation" in bundle["global"]
        assert "BULL" in bundle["by_regime"]
        assert "BEAR" in bundle["by_regime"]
        assert bundle["n_weeks"]["BULL"] == 20
        assert bundle["n_weeks"]["BEAR"] == 12
        assert bundle["blend_k"] == 30
        assert bundle["min_fit_weeks"] == 8

    def test_v2_decimal_output(self, tmp_path):
        """Bundle weights are Decimal."""
        path = tmp_path / "w.json"
        _make_bundle_json(
            path,
            global_fw={"valuation_normalized": -0.35},
            by_regime={"BULL": {"n_weeks": 10, "feature_weights": {
                "valuation_normalized": -0.40,
            }}},
        )
        bundle = _load_module5_weights_bundle(path)
        assert isinstance(bundle["global"]["valuation"], Decimal)
        assert isinstance(bundle["by_regime"]["BULL"]["valuation"], Decimal)

    def test_missing_file_returns_none(self, tmp_path):
        bundle = _load_module5_weights_bundle(tmp_path / "missing.json")
        assert bundle is None

    def test_unknown_schema_returns_none(self, tmp_path):
        path = tmp_path / "w.json"
        path.write_text(json.dumps({"schema_version": 99, "global": {}}))
        bundle = _load_module5_weights_bundle(path)
        assert bundle is None

    def test_v2_missing_global_returns_none(self, tmp_path):
        path = tmp_path / "w.json"
        path.write_text(json.dumps({"schema_version": 2, "by_regime": {}}))
        bundle = _load_module5_weights_bundle(path)
        assert bundle is None


class TestSelectWeights:
    @pytest.fixture
    def bundle(self):
        return {
            "global": {"valuation": Decimal("-0.35"), "clinical": Decimal("-0.10")},
            "by_regime": {
                "BULL": {"valuation": Decimal("-0.40"), "clinical": Decimal("-0.05")},
                "BEAR": {"valuation": Decimal("-0.20"), "clinical": Decimal("-0.30")},
            },
            "n_weeks": {"BULL": 20, "BEAR": 12},
            "blend_k": 25,
            "min_fit_weeks": 5,
        }

    def test_global_mode_returns_global(self, bundle):
        result = _select_module5_weights(bundle, regime="BULL", mode="global")
        assert result == bundle["global"]

    def test_regime_mode_returns_regime(self, bundle):
        result = _select_module5_weights(bundle, regime="BULL", mode="regime")
        assert result == bundle["by_regime"]["BULL"]

    def test_regime_mode_missing_regime_falls_back(self, bundle):
        result = _select_module5_weights(bundle, regime="CHOP", mode="regime")
        assert result == bundle["global"]

    def test_regime_mode_none_regime_falls_back(self, bundle):
        result = _select_module5_weights(bundle, regime=None, mode="regime")
        assert result == bundle["global"]

    def test_regime_mode_insufficient_weeks_falls_back(self, bundle):
        bundle["n_weeks"]["BEAR"] = 3  # below min_fit_weeks=5
        result = _select_module5_weights(bundle, regime="BEAR", mode="regime")
        assert result == bundle["global"]

    def test_blended_mode_blends_correctly(self, bundle):
        result = _select_module5_weights(bundle, regime="BULL", mode="blended")
        # alpha = 20 / (20 + 25) = 0.4444...
        alpha = Decimal("20") / Decimal("45")
        expected_val = alpha * Decimal("-0.40") + (1 - alpha) * Decimal("-0.35")
        assert abs(result["valuation"] - expected_val) < Decimal("1e-10")
        expected_clin = alpha * Decimal("-0.05") + (1 - alpha) * Decimal("-0.10")
        assert abs(result["clinical"] - expected_clin) < Decimal("1e-10")

    def test_blended_mode_insufficient_weeks_falls_back(self, bundle):
        bundle["n_weeks"]["BEAR"] = 3
        result = _select_module5_weights(bundle, regime="BEAR", mode="blended")
        assert result == bundle["global"]

    def test_all_outputs_are_decimal(self, bundle):
        for mode in ("global", "regime", "blended"):
            result = _select_module5_weights(bundle, regime="BULL", mode=mode)
            for v in result.values():
                assert isinstance(v, Decimal), f"mode={mode}: {type(v)} is not Decimal"


class TestNormalizeM3Regime:
    def test_direct_labels_pass_through(self):
        assert _normalize_m3_regime("BULL") == "BULL"
        assert _normalize_m3_regime("BEAR") == "BEAR"
        assert _normalize_m3_regime("CHOP") == "CHOP"

    def test_chop_sublabels_pass_through(self):
        assert _normalize_m3_regime("CHOP_LOWVOL") == "CHOP_LOWVOL"
        assert _normalize_m3_regime("CHOP_HIGHVOL") == "CHOP_HIGHVOL"

    def test_unknown_returns_none(self):
        assert _normalize_m3_regime("UNKNOWN") is None
        assert _normalize_m3_regime(None) is None
        assert _normalize_m3_regime("") is None

    def test_volatility_spike_maps_to_bear(self):
        assert _normalize_m3_regime("VOLATILITY_SPIKE") == "BEAR"

    def test_recession_risk_maps_to_bear(self):
        assert _normalize_m3_regime("RECESSION_RISK") == "BEAR"

    def test_credit_crisis_maps_to_bear(self):
        assert _normalize_m3_regime("CREDIT_CRISIS") == "BEAR"

    def test_sector_rotation_maps_to_chop(self):
        assert _normalize_m3_regime("SECTOR_ROTATION") == "CHOP"

    def test_sector_dislocation_maps_to_chop(self):
        assert _normalize_m3_regime("SECTOR_DISLOCATION") == "CHOP"

    def test_case_insensitive(self):
        assert _normalize_m3_regime("bull") == "BULL"
        assert _normalize_m3_regime("Bear") == "BEAR"
