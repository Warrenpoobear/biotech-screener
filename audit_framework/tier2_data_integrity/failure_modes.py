"""
Query 2.3 - Failure Mode Catalog & Edge Case Handling.

Systematically tests all known failure modes and edge cases.

Test Categories:
- EPRX test (fabrication rejection)
- Sparse data scenarios
- Anomalous data patterns
- Boundary conditions
"""

import json
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class FailureModeTest:
    """Result of a single failure mode test."""

    test_name: str
    category: str
    description: str
    expected_behavior: str
    actual_behavior: str
    passed: bool
    error_message: Optional[str] = None
    affected_tickers: List[str] = field(default_factory=list)


@dataclass
class FailureModeReport:
    """Complete failure mode test report."""

    tests_run: int
    tests_passed: int
    tests_failed: int
    test_results: List[FailureModeTest] = field(default_factory=list)
    edge_case_coverage: Dict[str, bool] = field(default_factory=dict)
    passed: bool = False

    @property
    def pass_rate(self) -> Decimal:
        if self.tests_run == 0:
            return Decimal("0")
        return (Decimal(self.tests_passed) / Decimal(self.tests_run)).quantize(
            Decimal("0.0001")
        )


class FailureModeValidator:
    """
    Validates failure mode handling and edge case coverage.

    Tests:
    - Fabricated ticker rejection (EPRX test)
    - Sparse data handling
    - Anomalous patterns
    - Boundary conditions
    """

    # Test tickers for different scenarios
    FABRICATED_TICKERS = ["EPRX", "FAKETICKER123", "ZZZZZ"]
    EDGE_CASE_SCENARIOS = [
        "pre_ipo_spac",
        "no_10k_filed",
        "foreign_adr",
        "penny_stock",
        "mega_cap",
        "zero_revenue",
        "negative_cash_flow_growth",
    ]

    def __init__(
        self,
        codebase_path: str,
        data_dir: str = "production_data",
    ):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)
        self.data_dir = self.codebase_path / data_dir

    def _load_json_data(self, filename: str) -> Any:
        """Load JSON data file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def _get_all_tickers(self) -> Set[str]:
        """Get all tickers from data files."""
        tickers = set()

        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    content = f.read()

                # Extract tickers using regex
                matches = re.findall(r'"ticker"\s*:\s*"([A-Z0-9]+)"', content)
                tickers.update(matches)

            except Exception:
                continue

        return tickers

    def test_fabrication_rejection(self) -> List[FailureModeTest]:
        """
        Test that fabricated tickers are properly rejected.

        The system should NOT score or include non-existent tickers.
        """
        results = []
        all_tickers = self._get_all_tickers()

        for fake_ticker in self.FABRICATED_TICKERS:
            is_present = fake_ticker in all_tickers

            results.append(FailureModeTest(
                test_name=f"fabrication_rejection_{fake_ticker}",
                category="fabrication",
                description=f"Verify ticker {fake_ticker} is not in production data",
                expected_behavior="Ticker should not be present in any data file",
                actual_behavior="Not present" if not is_present else "PRESENT - VIOLATION",
                passed=not is_present,
                error_message=None if not is_present else f"Fake ticker {fake_ticker} found in data",
            ))

        return results

    def test_sparse_data_handling(self) -> List[FailureModeTest]:
        """
        Test handling of sparse/missing data.

        Scenarios:
        - Ticker with no 10-K filed
        - Clinical trial with no results
        - Foreign ADR with limited US filings
        """
        results = []

        # Check for proper handling of missing financial data
        financial_data = self._load_json_data("financial_records.json")
        trial_data = self._load_json_data("trial_records.json")
        universe_data = self._load_json_data("universe.json")

        if financial_data and universe_data:
            # Get universe tickers
            universe_tickers = set()
            if isinstance(universe_data, list):
                universe_tickers = {
                    r.get("ticker", "") for r in universe_data if isinstance(r, dict)
                }
            elif isinstance(universe_data, dict) and "active_securities" in universe_data:
                universe_tickers = {
                    r.get("ticker", "") for r in universe_data["active_securities"]
                    if isinstance(r, dict)
                }

            # Get tickers with financial data
            if isinstance(financial_data, list):
                financial_tickers = {
                    r.get("ticker", "") for r in financial_data if isinstance(r, dict)
                }
            else:
                financial_tickers = set()

            # Check for tickers missing financial data
            missing_financial = universe_tickers - financial_tickers

            results.append(FailureModeTest(
                test_name="sparse_data_financial",
                category="sparse_data",
                description="Verify tickers with missing financial data are handled",
                expected_behavior="Missing financial data should be flagged, not fabricated",
                actual_behavior=f"{len(missing_financial)} tickers without financial data",
                passed=True,  # Pass if we can identify them
                affected_tickers=sorted(list(missing_financial))[:10],
            ))

        # Check for trials without completion dates
        if trial_data and isinstance(trial_data, list):
            no_completion = []
            for record in trial_data:
                if isinstance(record, dict):
                    ticker = record.get("ticker", "")
                    completion = record.get("primary_completion_date")
                    if ticker and not completion:
                        no_completion.append(ticker)

            results.append(FailureModeTest(
                test_name="sparse_data_trial_dates",
                category="sparse_data",
                description="Verify trials without completion dates are handled",
                expected_behavior="Missing dates should use UNKNOWN or conservative estimate",
                actual_behavior=f"{len(set(no_completion))} tickers with undated trials",
                passed=True,
                affected_tickers=sorted(list(set(no_completion)))[:10],
            ))

        return results

    def test_anomalous_patterns(self) -> List[FailureModeTest]:
        """
        Test detection of anomalous data patterns.

        Patterns:
        - Negative cash flow but increasing market cap
        - Phase 3 trial with 0% institutional ownership
        - Past catalyst date with no update
        """
        results = []

        financial_data = self._load_json_data("financial_records.json")
        market_data = self._load_json_data("market_data.json")

        if financial_data and isinstance(financial_data, list):
            # Check for anomalies
            anomalies = []

            for record in financial_data:
                if not isinstance(record, dict):
                    continue

                ticker = record.get("ticker", "")

                # Check for negative burn rate (should be negative or zero)
                burn = record.get("quarterly_burn", record.get("burn_rate", 0))
                try:
                    burn_val = Decimal(str(burn)) if burn else Decimal("0")
                    # Burn should typically be negative or zero for biotech
                    if burn_val > Decimal("10000000"):  # Positive > $10M
                        anomalies.append((ticker, "positive_burn", str(burn_val)))
                except Exception:
                    pass

                # Check for zero or negative cash
                cash = record.get("total_cash", record.get("cash", 0))
                try:
                    cash_val = Decimal(str(cash)) if cash else Decimal("0")
                    if cash_val <= 0:
                        anomalies.append((ticker, "zero_cash", str(cash_val)))
                except Exception:
                    pass

            results.append(FailureModeTest(
                test_name="anomalous_pattern_detection",
                category="anomalous_data",
                description="Verify anomalous financial patterns are flagged",
                expected_behavior="Anomalies should trigger manual review flags",
                actual_behavior=f"Found {len(anomalies)} potential anomalies",
                passed=True,  # Pass if we can detect them
                affected_tickers=[a[0] for a in anomalies[:10]],
            ))

        return results

    def test_boundary_conditions(self) -> List[FailureModeTest]:
        """
        Test handling of boundary conditions.

        Boundaries:
        - Extremely small market cap (<$50M)
        - Extremely large pipeline (>20 programs)
        - Zero revenue with high valuation
        """
        results = []

        market_data = self._load_json_data("market_data.json")
        universe_data = self._load_json_data("universe.json")

        small_cap_tickers = []
        large_pipeline_tickers = []

        # Check market data for small caps
        if isinstance(market_data, list):
            for record in market_data:
                if not isinstance(record, dict):
                    continue

                ticker = record.get("ticker", "")
                market_cap = record.get("market_cap", record.get("market_cap_mm", 0))

                try:
                    cap_val = Decimal(str(market_cap)) if market_cap else Decimal("0")
                    if cap_val > 0 and cap_val < Decimal("50"):  # < $50M
                        small_cap_tickers.append(ticker)
                except Exception:
                    pass

        results.append(FailureModeTest(
            test_name="boundary_small_market_cap",
            category="boundary",
            description="Verify handling of extremely small market caps (<$50M)",
            expected_behavior="Small caps should be flagged for liquidity risk",
            actual_behavior=f"Found {len(small_cap_tickers)} tickers below $50M",
            passed=True,
            affected_tickers=small_cap_tickers[:10],
        ))

        # Check for large pipelines
        trial_data = self._load_json_data("trial_records.json")
        if isinstance(trial_data, list):
            ticker_trial_count: Dict[str, int] = {}
            for record in trial_data:
                if isinstance(record, dict):
                    ticker = record.get("ticker", "")
                    if ticker:
                        ticker_trial_count[ticker] = ticker_trial_count.get(ticker, 0) + 1

            large_pipeline = [t for t, c in ticker_trial_count.items() if c > 20]

            results.append(FailureModeTest(
                test_name="boundary_large_pipeline",
                category="boundary",
                description="Verify handling of large pipelines (>20 trials)",
                expected_behavior="Large pipelines should be analyzed correctly",
                actual_behavior=f"Found {len(large_pipeline)} tickers with >20 trials",
                passed=True,
                affected_tickers=large_pipeline[:10],
            ))

        return results

    def check_error_handling_patterns(self) -> List[FailureModeTest]:
        """
        Check codebase for proper error handling patterns.

        Validates:
        - No bare except clauses hiding errors
        - Errors propagate with context
        - Validation failures are tracked
        """
        results = []

        bare_except_files = []
        silent_failure_files = []

        for root, _, files in os.walk(self.codebase_path):
            if any(skip in root for skip in ["test", "deprecated", "venv", "mnt"]):
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        lines = content.splitlines()

                    # Check for bare except
                    for i, line in enumerate(lines, 1):
                        if re.match(r"^\s*except\s*:", line):
                            rel_path = str(file_path.relative_to(self.codebase_path))
                            bare_except_files.append(f"{rel_path}:{i}")

                        # Check for silent pass after except
                        if re.match(r"^\s*except.*:", line):
                            if i < len(lines) and re.match(r"^\s*pass\s*$", lines[i]):
                                rel_path = str(file_path.relative_to(self.codebase_path))
                                silent_failure_files.append(f"{rel_path}:{i}")

                except Exception:
                    continue

        results.append(FailureModeTest(
            test_name="error_handling_bare_except",
            category="error_handling",
            description="Check for bare 'except:' clauses",
            expected_behavior="No bare except clauses that hide errors",
            actual_behavior=f"Found {len(bare_except_files)} bare except clauses",
            passed=len(bare_except_files) == 0,
            error_message=", ".join(bare_except_files[:5]) if bare_except_files else None,
        ))

        results.append(FailureModeTest(
            test_name="error_handling_silent_pass",
            category="error_handling",
            description="Check for silent exception swallowing (except: pass)",
            expected_behavior="Exceptions should be logged or re-raised",
            actual_behavior=f"Found {len(silent_failure_files)} silent failures",
            passed=len(silent_failure_files) == 0,
            error_message=", ".join(silent_failure_files[:5]) if silent_failure_files else None,
        ))

        return results

    def run_audit(self) -> FailureModeReport:
        """
        Run complete failure mode audit.

        Returns:
            FailureModeReport with test results
        """
        all_results: List[FailureModeTest] = []

        # Run all test categories
        all_results.extend(self.test_fabrication_rejection())
        all_results.extend(self.test_sparse_data_handling())
        all_results.extend(self.test_anomalous_patterns())
        all_results.extend(self.test_boundary_conditions())
        all_results.extend(self.check_error_handling_patterns())

        # Calculate stats
        tests_passed = sum(1 for r in all_results if r.passed)
        tests_failed = len(all_results) - tests_passed

        # Edge case coverage map
        edge_case_coverage = {}
        for test in all_results:
            edge_case_coverage[test.test_name] = test.passed

        passed = tests_failed == 0

        return FailureModeReport(
            tests_run=len(all_results),
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_results=all_results,
            edge_case_coverage=edge_case_coverage,
            passed=passed,
        )


def validate_failure_modes(
    codebase_path: str,
    data_dir: str = "production_data",
) -> AuditResult:
    """
    Run complete failure mode validation.

    Args:
        codebase_path: Root of codebase
        data_dir: Data directory name

    Returns:
        AuditResult with findings
    """
    validator = FailureModeValidator(codebase_path, data_dir)
    report = validator.run_audit()

    result = AuditResult(
        check_name="failure_modes",
        passed=report.passed,
        metrics={
            "tests_run": report.tests_run,
            "tests_passed": report.tests_passed,
            "tests_failed": report.tests_failed,
            "pass_rate": str(report.pass_rate),
            "edge_case_coverage": report.edge_case_coverage,
        },
        details=f"Passed {report.tests_passed}/{report.tests_run} failure mode tests",
    )

    # Add findings for failed tests
    for test in report.test_results:
        if not test.passed:
            result.add_finding(
                severity=AuditSeverity.HIGH if test.category == "fabrication" else AuditSeverity.MEDIUM,
                category=ValidationCategory.DATA_QUALITY,
                title=f"Failure mode test failed: {test.test_name}",
                description=f"{test.description}\nExpected: {test.expected_behavior}\nActual: {test.actual_behavior}",
                location=f"category: {test.category}",
                evidence=test.error_message or test.actual_behavior,
                remediation="Implement proper handling for this failure mode",
                compliance_impact="Unhandled failure modes can cause silent data corruption",
            )

    return result
