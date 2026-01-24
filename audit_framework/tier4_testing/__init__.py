"""
Tier 4: Testing, Validation & Regression Prevention.

Validates testing infrastructure and historical validation capabilities.

Queries Implemented:
    4.1 - Test Coverage & Regression Suite
    4.2 - Backtesting Validation Framework
    4.3 - Model Explainability & Audit Trail
"""

from audit_framework.tier4_testing.coverage import (
    TestCoverageValidator,
    validate_test_coverage,
)

from audit_framework.tier4_testing.backtesting import (
    BacktestValidator,
    validate_backtest_capability,
)

from audit_framework.tier4_testing.explainability import (
    ExplainabilityValidator,
    validate_explainability,
)

from audit_framework.tier4_testing.runner import (
    run_tier4_audit,
    Tier4Result,
)

__all__ = [
    "TestCoverageValidator",
    "validate_test_coverage",
    "BacktestValidator",
    "validate_backtest_capability",
    "ExplainabilityValidator",
    "validate_explainability",
    "run_tier4_audit",
    "Tier4Result",
]
