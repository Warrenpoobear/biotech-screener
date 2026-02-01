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

from run_screen import _load_module5_weights


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
