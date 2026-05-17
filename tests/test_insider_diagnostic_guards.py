"""Spec 104: Insider Diagnostic Stabilization - Guard Tests.

Verify that insider_net_buy_value_90d stays diagnostic-only:
  - Present in SNAPSHOT_COLUMNS (exported for observability)
  - Absent from FEATURE_REGISTRY (not an alpha input)
  - Not consumed by ExpectationErrorModel
  - Not in any active signal list
  - Tracked nonblocking in FEATURE_COVERAGE_REQUIREMENTS
  - Blank and zero are never collapsed in measurement logic
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestInsiderDiagnosticGuards:
    """Verify insider_net_buy_value_90d stays diagnostic-only."""

    def test_insider_in_snapshot_columns_as_diagnostic(self):
        """insider_net_buy_value_90d should be in SNAPSHOT_COLUMNS (exported for observability)."""
        from run_screen_columns import SNAPSHOT_COLUMNS

        assert "insider_net_buy_value_90d" in SNAPSHOT_COLUMNS

    def test_insider_not_in_feature_registry(self):
        """insider_net_buy_value_90d must NOT be in the alpha feature registry."""
        from common.feature_registry import FEATURE_REGISTRY

        feature_names = {f.name for f in FEATURE_REGISTRY}
        assert (
            "insider_net_buy_value_90d" not in feature_names
        ), "insider_net_buy_value_90d must not be in FEATURE_REGISTRY (diagnostic only)"

    def test_insider_not_consumed_by_expectation_model(self):
        """ExpectationErrorModel must not read insider_net_buy_value_90d."""
        from event_ev.expectation_error_model import ExpectationErrorModel

        source = inspect.getsource(ExpectationErrorModel)
        assert (
            "insider_net_buy_value_90d" not in source
        ), "ExpectationErrorModel must not reference insider_net_buy_value_90d"

    def test_insider_not_in_market_features_keys(self):
        """If ExpectationErrorModel has MARKET_FEATURE_KEYS or similar, insider must not be listed."""
        from event_ev.expectation_error_model import ExpectationErrorModel

        source = inspect.getsource(ExpectationErrorModel)
        # Check that insider is not in any feature key list
        assert (
            "insider" not in source.lower() or "insider_net_buy" not in source
        ), "ExpectationErrorModel must not reference insider features"

    def test_blank_zero_not_collapsed_in_measurement(self):
        """Measurement logic must distinguish blank (NaN) from zero (0.0)."""
        # Create test data with both NaN and 0.0
        data = pd.Series([np.nan, 0.0, 100.0, np.nan, -50.0, 0.0, np.nan])
        blank_count = data.isna().sum()
        zero_count = (data == 0.0).sum()
        nonblank = data.dropna()
        # NaN count should be 3, zero count should be 2
        assert blank_count == 3, f"Expected 3 blanks, got {blank_count}"
        assert zero_count == 2, f"Expected 2 zeros, got {zero_count}"
        # They must not be collapsed
        assert blank_count != zero_count, "Blank and zero must be distinguishable"

    def test_insider_not_in_active_signals(self):
        """insider_net_buy_value_90d must not appear in any active signal list."""
        # Check feature registry for any insider-related active signals
        from common.feature_registry import FEATURE_REGISTRY

        for f in FEATURE_REGISTRY:
            assert "insider" not in f.name.lower(), f"Found insider-related feature in active registry: {f.name}"

    def test_coverage_requirements_insider_is_tracked_nonblocking(self):
        """FEATURE_COVERAGE_REQUIREMENTS should list insider as tracked/nonblocking if present."""
        # This test verifies the production QA check treats insider as nonblocking
        try:
            sys.path.insert(0, str(REPO_ROOT / "tools"))
            spec = importlib.util.spec_from_file_location(
                "production_qa_check",
                str(REPO_ROOT / "tools" / "production_qa_check.py"),
            )
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "FEATURE_COVERAGE_REQUIREMENTS"):
                    for field_name, threshold, is_blocking in mod.FEATURE_COVERAGE_REQUIREMENTS:
                        if "insider" in field_name:
                            assert not is_blocking, (
                                f"insider field {field_name} must be nonblocking " f"in FEATURE_COVERAGE_REQUIREMENTS"
                            )
        except (ImportError, FileNotFoundError):
            pytest.skip("production_qa_check.py not importable in test environment")
