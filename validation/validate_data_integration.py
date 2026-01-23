#!/usr/bin/env python3
"""
validation/validate_data_integration.py - Data Integration Validation Harness

End-to-end validation of data integration in the biotech screening pipeline.
Runs all data integration contract checks and produces a comprehensive report.

Usage:
    # Validate production data
    python validation/validate_data_integration.py --data-dir production_data --as-of-date 2026-01-15

    # Validate with strict mode (fail on any issue)
    python validation/validate_data_integration.py --data-dir production_data --as-of-date 2026-01-15 --strict

    # Run with fixture data (for testing)
    python validation/validate_data_integration.py --fixture

Checks performed:
1. Schema validation for all input files
2. Join invariants (ticker uniqueness, case consistency, coverage)
3. PIT admissibility for all datasets
4. Coverage guardrails
5. Deterministic output hash

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.data_integration_contracts import (
    # Schema validators
    validate_market_data_schema,
    validate_financial_records_schema,
    validate_trial_records_schema,
    validate_holdings_schema,
    validate_short_interest_schema,
    # Join validation
    validate_join_invariants,
    normalize_ticker_set,
    check_ticker_uniqueness,
    check_ticker_case_consistency,
    # PIT validation
    validate_pit_admissibility,
    validate_dataset_pit,
    PITValidationResult,
    # Coverage guardrails
    validate_coverage_guardrails,
    CoverageConfig,
    CoverageReport,
    # Determinism
    compute_deterministic_hash,
    # Numeric safety
    safe_numeric_check,
    # Exceptions
    DataIntegrationError,
    SchemaValidationError,
    JoinInvariantError,
    PITViolationError,
    CoverageGuardrailError,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURE DATA FOR TESTING
# =============================================================================

def create_fixture_data() -> Dict[str, Any]:
    """
    Create minimal fixture data for integration testing.

    This fixture includes:
    - Edge cases for numeric falsy values (momentum_score=0)
    - Mixed case tickers to test normalization
    - Future-dated records to test PIT filtering
    - Missing fields to test schema validation
    """
    return {
        "universe": [
            {"ticker": "ACME", "status": "active", "market_cap_mm": 1000},
            {"ticker": "BETA", "status": "active", "market_cap_mm": 500},
            {"ticker": "GAMMA", "status": "active", "market_cap_mm": 2000},
            {"ticker": "DELTA", "status": "active", "market_cap_mm": 750},
            {"ticker": "ZERO", "status": "active", "market_cap_mm": 100},  # For testing zero values
        ],
        "financial_records": [
            {"ticker": "ACME", "Cash": 100000000, "NetIncome": -5000000, "source_date": "2025-12-01"},
            {"ticker": "beta", "Cash": 50000000, "NetIncome": -2000000, "source_date": "2025-12-15"},  # lowercase
            {"ticker": "GAMMA", "Cash": 200000000, "NetIncome": 10000000, "source_date": "2025-11-01"},
            {"ticker": "DELTA", "Cash": 75000000, "NetIncome": -3000000, "source_date": "2025-12-10"},
            {"ticker": "ZERO", "Cash": 0, "NetIncome": 0, "source_date": "2025-12-20"},  # Zero values
        ],
        "market_data": [
            {"ticker": "ACME", "price": 25.50, "market_cap": 1000000000, "return_60d": 0.15, "source_date": "2026-01-14"},
            {"ticker": "BETA", "price": 12.00, "market_cap": 500000000, "return_60d": -0.05, "source_date": "2026-01-14"},
            {"ticker": "GAMMA", "price": 45.00, "market_cap": 2000000000, "return_60d": 0.0, "source_date": "2026-01-14"},  # Zero return
            {"ticker": "Delta", "price": 18.75, "market_cap": 750000000, "return_60d": 0.08, "source_date": "2026-01-14"},  # Mixed case
            {"ticker": "ZERO", "price": 5.00, "market_cap": 100000000, "return_60d": 0, "source_date": "2026-01-14"},  # Zero as int
        ],
        "trial_records": [
            {"ticker": "ACME", "nct_id": "NCT12345678", "phase": "Phase 2", "status": "Recruiting", "first_posted": "2024-06-15"},
            {"ticker": "ACME", "nct_id": "NCT12345679", "phase": "Phase 3", "status": "Active", "first_posted": "2025-01-10"},
            {"ticker": "BETA", "nct_id": "NCT23456789", "phase": "Phase 1", "status": "Recruiting", "first_posted": "2025-09-01"},
            {"ticker": "GAMMA", "nct_id": "NCT34567890", "phase": "Phase 3", "status": "Completed", "first_posted": "2023-03-20"},
            {"ticker": "DELTA", "nct_id": "NCT45678901", "phase": "Phase 2", "status": "Recruiting", "first_posted": "2025-06-01"},
            # Future record (should be PIT-filtered)
            {"ticker": "ACME", "nct_id": "NCT99999999", "phase": "Phase 1", "status": "Not yet recruiting", "first_posted": "2026-02-01"},
        ],
        "holdings_snapshots": {
            "ACME": {
                "holdings": {
                    "current": {
                        "0001263508": {"value_kusd": 50000},
                        "0001346824": {"value_kusd": 30000},
                    }
                }
            },
            "BETA": {
                "holdings": {
                    "current": {
                        "0001263508": {"value_kusd": 20000},
                    }
                }
            },
        },
        "short_interest": [
            {"ticker": "ACME", "short_interest": 5000000, "days_to_cover": 3.5, "source_date": "2026-01-10"},
            {"ticker": "GAMMA", "short_interest": 2000000, "days_to_cover": 1.2, "source_date": "2026-01-10"},
        ],
        "momentum_results": {
            "rankings": [
                {"ticker": "ACME", "momentum_score": 75},
                {"ticker": "BETA", "momentum_score": 50},  # Neutral score
                {"ticker": "GAMMA", "momentum_score": 0},  # CRITICAL: Zero score (falsy but valid!)
                {"ticker": "DELTA", "momentum_score": 25},
            ],
            "summary": {
                "coordinated_buys": ["ACME"],
                "coordinated_sells": ["DELTA"],
            }
        },
        "as_of_date": "2026-01-15",
    }


# =============================================================================
# VALIDATION HARNESS
# =============================================================================

@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    severity: str  # "critical", "high", "medium", "low"
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    passed: bool
    checks: List[ValidationCheck] = field(default_factory=list)
    critical_failures: int = 0
    high_failures: int = 0
    medium_failures: int = 0
    low_failures: int = 0
    data_hash: str = ""

    def add_check(self, check: ValidationCheck):
        self.checks.append(check)
        if not check.passed:
            if check.severity == "critical":
                self.critical_failures += 1
            elif check.severity == "high":
                self.high_failures += 1
            elif check.severity == "medium":
                self.medium_failures += 1
            else:
                self.low_failures += 1

    def is_valid(self, allow_medium: bool = True, allow_low: bool = True) -> bool:
        if self.critical_failures > 0 or self.high_failures > 0:
            return False
        if not allow_medium and self.medium_failures > 0:
            return False
        if not allow_low and self.low_failures > 0:
            return False
        return True


class DataIntegrationValidator:
    """
    Validates data integration across the biotech screening pipeline.
    """

    def __init__(
        self,
        universe: List[Dict],
        financial_records: List[Dict],
        market_data: List[Dict],
        trial_records: List[Dict],
        as_of_date: str,
        holdings_snapshots: Optional[Dict] = None,
        short_interest: Optional[List[Dict]] = None,
        momentum_results: Optional[Dict] = None,
    ):
        self.universe = universe
        self.financial_records = financial_records
        self.market_data = market_data
        self.trial_records = trial_records
        self.as_of_date = as_of_date
        self.holdings_snapshots = holdings_snapshots or {}
        self.short_interest = short_interest or []
        self.momentum_results = momentum_results or {}

        self.report = ValidationReport(passed=True)

    def validate_all(self) -> ValidationReport:
        """Run all validation checks and return report."""
        logger.info("=" * 60)
        logger.info("DATA INTEGRATION VALIDATION")
        logger.info("=" * 60)
        logger.info(f"As-of date: {self.as_of_date}")
        logger.info(f"Universe size: {len(self.universe)}")
        logger.info("")

        # 1. Schema validation
        self._validate_schemas()

        # 2. Ticker normalization and uniqueness
        self._validate_ticker_consistency()

        # 3. Join invariants
        self._validate_joins()

        # 4. PIT admissibility
        self._validate_pit()

        # 5. Coverage guardrails
        self._validate_coverage()

        # 6. Numeric falsy value handling
        self._validate_numeric_handling()

        # 7. Compute deterministic hash
        self._compute_data_hash()

        # Finalize
        self.report.passed = self.report.is_valid()

        # Print summary
        self._print_summary()

        return self.report

    def _validate_schemas(self):
        """Validate all input data schemas."""
        logger.info("[1/7] Validating schemas...")

        # Market data
        try:
            is_valid, invalid = validate_market_data_schema(
                self.market_data, strict=False, raise_on_error=False
            )
            self.report.add_check(ValidationCheck(
                name="market_data_schema",
                passed=is_valid,
                severity="high" if not is_valid else "low",
                message=f"Market data schema: {len(invalid)} invalid records" if invalid else "Market data schema valid",
                details={"invalid_count": len(invalid), "sample": invalid[:3]},
            ))
        except Exception as e:
            self.report.add_check(ValidationCheck(
                name="market_data_schema",
                passed=False,
                severity="critical",
                message=f"Market data schema validation failed: {e}",
            ))

        # Financial records
        try:
            is_valid, invalid = validate_financial_records_schema(
                self.financial_records, strict=False, raise_on_error=False
            )
            self.report.add_check(ValidationCheck(
                name="financial_records_schema",
                passed=is_valid,
                severity="high" if not is_valid else "low",
                message=f"Financial records schema: {len(invalid)} invalid records" if invalid else "Financial records schema valid",
                details={"invalid_count": len(invalid), "sample": invalid[:3]},
            ))
        except Exception as e:
            self.report.add_check(ValidationCheck(
                name="financial_records_schema",
                passed=False,
                severity="critical",
                message=f"Financial records schema validation failed: {e}",
            ))

        # Trial records
        try:
            is_valid, invalid = validate_trial_records_schema(
                self.trial_records, strict=False, raise_on_error=False
            )
            self.report.add_check(ValidationCheck(
                name="trial_records_schema",
                passed=is_valid,
                severity="high" if not is_valid else "low",
                message=f"Trial records schema: {len(invalid)} invalid records" if invalid else "Trial records schema valid",
                details={"invalid_count": len(invalid), "sample": invalid[:3]},
            ))
        except Exception as e:
            self.report.add_check(ValidationCheck(
                name="trial_records_schema",
                passed=False,
                severity="critical",
                message=f"Trial records schema validation failed: {e}",
            ))

        # Holdings (optional)
        if self.holdings_snapshots:
            try:
                is_valid, errors = validate_holdings_schema(
                    self.holdings_snapshots, raise_on_error=False
                )
                self.report.add_check(ValidationCheck(
                    name="holdings_schema",
                    passed=is_valid,
                    severity="medium" if not is_valid else "low",
                    message=f"Holdings schema: {len(errors)} errors" if errors else "Holdings schema valid",
                    details={"errors": errors[:5]},
                ))
            except Exception as e:
                self.report.add_check(ValidationCheck(
                    name="holdings_schema",
                    passed=False,
                    severity="medium",
                    message=f"Holdings schema validation failed: {e}",
                ))

    def _validate_ticker_consistency(self):
        """Validate ticker normalization and uniqueness."""
        logger.info("[2/7] Validating ticker consistency...")

        # Check uniqueness in each dataset
        for name, records in [
            ("financial", self.financial_records),
            ("market", self.market_data),
        ]:
            is_unique, duplicates = check_ticker_uniqueness(records)
            self.report.add_check(ValidationCheck(
                name=f"{name}_ticker_uniqueness",
                passed=is_unique,
                severity="high" if not is_unique else "low",
                message=f"{name.title()} ticker uniqueness: {len(duplicates)} duplicates" if duplicates else f"{name.title()} tickers are unique",
                details={"duplicates": list(duplicates)[:5]},
            ))

        # Check case consistency
        for name, records in [
            ("financial", self.financial_records),
            ("market", self.market_data),
        ]:
            is_consistent, variants = check_ticker_case_consistency(records)
            self.report.add_check(ValidationCheck(
                name=f"{name}_ticker_case",
                passed=is_consistent,
                severity="medium" if not is_consistent else "low",
                message=f"{name.title()} ticker case: {len(variants)} inconsistencies" if variants else f"{name.title()} ticker case is consistent",
                details={"variants": {k: list(v) for k, v in list(variants.items())[:5]}},
            ))

    def _validate_joins(self):
        """Validate join invariants."""
        logger.info("[3/7] Validating join invariants...")

        universe_tickers = normalize_ticker_set([r.get("ticker") for r in self.universe])
        financial_tickers = normalize_ticker_set([r.get("ticker") for r in self.financial_records])
        market_tickers = normalize_ticker_set([r.get("ticker") for r in self.market_data])
        clinical_tickers = normalize_ticker_set([r.get("ticker") for r in self.trial_records])

        result = validate_join_invariants(
            universe_tickers=universe_tickers,
            financial_tickers=financial_tickers,
            clinical_tickers=clinical_tickers,
            market_tickers=market_tickers,
            min_financial_coverage_pct=80.0,
            min_clinical_coverage_pct=50.0,
            min_market_coverage_pct=50.0,
        )

        self.report.add_check(ValidationCheck(
            name="join_invariants",
            passed=result["is_valid"],
            severity="high" if not result["is_valid"] else "low",
            message=f"Join invariants: {result['coverage_failures']}" if result["coverage_failures"] else "Join invariants valid",
            details=result,
        ))

    def _validate_pit(self):
        """Validate PIT admissibility for all datasets."""
        logger.info("[4/7] Validating PIT admissibility...")

        # Market data
        pit_result = validate_dataset_pit(
            "market", self.market_data, self.as_of_date
        )
        self.report.add_check(ValidationCheck(
            name="market_data_pit",
            passed=pit_result.is_valid,
            severity="critical" if not pit_result.is_valid else "low",
            message=f"Market data PIT: {pit_result.pit_violated} future records" if pit_result.pit_violated else "Market data PIT valid",
            details={
                "total": pit_result.total_records,
                "compliant": pit_result.pit_compliant,
                "violated": pit_result.pit_violated,
                "cutoff": pit_result.pit_cutoff,
            },
        ))

        # Financial records
        pit_result = validate_dataset_pit(
            "financial", self.financial_records, self.as_of_date
        )
        self.report.add_check(ValidationCheck(
            name="financial_records_pit",
            passed=pit_result.is_valid,
            severity="critical" if not pit_result.is_valid else "low",
            message=f"Financial records PIT: {pit_result.pit_violated} future records" if pit_result.pit_violated else "Financial records PIT valid",
            details={
                "total": pit_result.total_records,
                "compliant": pit_result.pit_compliant,
                "violated": pit_result.pit_violated,
            },
        ))

        # Trial records
        pit_result = validate_dataset_pit(
            "trial", self.trial_records, self.as_of_date
        )
        self.report.add_check(ValidationCheck(
            name="trial_records_pit",
            passed=pit_result.is_valid,
            severity="high" if not pit_result.is_valid else "low",
            message=f"Trial records PIT: {pit_result.pit_violated} future records" if pit_result.pit_violated else "Trial records PIT valid",
            details={
                "total": pit_result.total_records,
                "compliant": pit_result.pit_compliant,
                "violated": pit_result.pit_violated,
                "future_sample": pit_result.future_records[:3],
            },
        ))

    def _validate_coverage(self):
        """Validate coverage guardrails."""
        logger.info("[5/7] Validating coverage guardrails...")

        universe_tickers = normalize_ticker_set([r.get("ticker") for r in self.universe])
        financial_tickers = normalize_ticker_set([r.get("ticker") for r in self.financial_records])
        market_tickers = normalize_ticker_set([r.get("ticker") for r in self.market_data])
        clinical_tickers = normalize_ticker_set([r.get("ticker") for r in self.trial_records])

        report = validate_coverage_guardrails(
            universe_size=len(universe_tickers),
            financial_count=len(financial_tickers & universe_tickers),
            clinical_count=len(clinical_tickers & universe_tickers),
            market_count=len(market_tickers & universe_tickers),
        )

        self.report.add_check(ValidationCheck(
            name="coverage_guardrails",
            passed=report.is_valid,
            severity="high" if not report.is_valid else "low",
            message=f"Coverage guardrails: {report.failures}" if report.failures else "Coverage guardrails passed",
            details={
                "coverage": report.component_coverage,
                "failures": report.failures,
                "warnings": report.warnings,
            },
        ))

    def _validate_numeric_handling(self):
        """Validate numeric falsy value handling."""
        logger.info("[6/7] Validating numeric falsy value handling...")

        # Check momentum scores for zero values
        if self.momentum_results:
            rankings = self.momentum_results.get("rankings", [])
            zero_scores = []
            falsy_issues = []

            for ranking in rankings:
                ticker = ranking.get("ticker")
                score = ranking.get("momentum_score")

                # Check if score is zero (falsy but valid)
                if score == 0 or score == 0.0:
                    zero_scores.append(ticker)

                # Check if the code would incorrectly skip this
                # The bug: `if ticker and momentum_score` would skip score=0
                if ticker and not score:  # This would incorrectly trigger for score=0
                    if safe_numeric_check(score):  # But safe_numeric_check knows it's valid
                        falsy_issues.append({
                            "ticker": ticker,
                            "score": score,
                            "issue": "score=0 would be skipped by `if score` check",
                        })

            self.report.add_check(ValidationCheck(
                name="numeric_falsy_handling",
                passed=len(falsy_issues) == 0,
                severity="critical" if falsy_issues else "low",
                message=f"Numeric falsy handling: {len(falsy_issues)} zero scores that could be skipped" if falsy_issues else "Numeric falsy handling OK",
                details={
                    "zero_scores": zero_scores,
                    "falsy_issues": falsy_issues,
                },
            ))

        # Check market data for zero returns
        zero_returns = []
        for record in self.market_data:
            return_60d = record.get("return_60d")
            if return_60d == 0 or return_60d == 0.0:
                zero_returns.append(record.get("ticker"))

        if zero_returns:
            self.report.add_check(ValidationCheck(
                name="zero_returns_handling",
                passed=True,  # Just informational
                severity="low",
                message=f"Found {len(zero_returns)} tickers with return_60d=0 (valid but requires careful handling)",
                details={"tickers": zero_returns},
            ))

    def _compute_data_hash(self):
        """Compute deterministic hash of all input data."""
        logger.info("[7/7] Computing deterministic data hash...")

        all_data = {
            "universe": sorted(self.universe, key=lambda x: x.get("ticker", "")),
            "financial_records": sorted(self.financial_records, key=lambda x: x.get("ticker", "")),
            "market_data": sorted(self.market_data, key=lambda x: x.get("ticker", "")),
            "trial_records": sorted(self.trial_records, key=lambda x: (x.get("ticker", ""), x.get("nct_id", ""))),
            "as_of_date": self.as_of_date,
        }

        self.report.data_hash = compute_deterministic_hash(all_data)

        self.report.add_check(ValidationCheck(
            name="deterministic_hash",
            passed=True,
            severity="low",
            message=f"Data hash computed: {self.report.data_hash[:40]}...",
            details={"hash": self.report.data_hash},
        ))

    def _print_summary(self):
        """Print validation summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        total_checks = len(self.report.checks)
        passed_checks = sum(1 for c in self.report.checks if c.passed)

        logger.info(f"Total checks: {total_checks}")
        logger.info(f"Passed: {passed_checks}")
        logger.info(f"Failed: {total_checks - passed_checks}")
        logger.info("")
        logger.info(f"  Critical failures: {self.report.critical_failures}")
        logger.info(f"  High failures: {self.report.high_failures}")
        logger.info(f"  Medium failures: {self.report.medium_failures}")
        logger.info(f"  Low failures: {self.report.low_failures}")
        logger.info("")

        if self.report.passed:
            logger.info("RESULT: PASSED")
        else:
            logger.info("RESULT: FAILED")

            # Show failed checks
            logger.info("")
            logger.info("Failed checks:")
            for check in self.report.checks:
                if not check.passed:
                    logger.info(f"  [{check.severity.upper()}] {check.name}: {check.message}")


def load_data_from_dir(data_dir: Path, as_of_date: str) -> Dict[str, Any]:
    """Load all data files from a directory."""
    data = {"as_of_date": as_of_date}

    # Required files
    with open(data_dir / "universe.json") as f:
        raw = json.load(f)
        if isinstance(raw, dict):
            data["universe"] = raw.get("active_securities", raw.get("securities", []))
        else:
            data["universe"] = raw

    with open(data_dir / "financial_records.json") as f:
        data["financial_records"] = json.load(f)

    with open(data_dir / "market_data.json") as f:
        data["market_data"] = json.load(f)

    with open(data_dir / "trial_records.json") as f:
        data["trial_records"] = json.load(f)

    # Optional files
    holdings_file = data_dir / "holdings_snapshots.json"
    if holdings_file.exists():
        with open(holdings_file) as f:
            data["holdings_snapshots"] = json.load(f)

    short_interest_file = data_dir / "short_interest.json"
    if short_interest_file.exists():
        with open(short_interest_file) as f:
            data["short_interest"] = json.load(f)

    momentum_file = data_dir / "momentum_results.json"
    if momentum_file.exists():
        with open(momentum_file) as f:
            data["momentum_results"] = json.load(f)

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Validate data integration in biotech screening pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Path to data directory",
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        help="Analysis date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use built-in fixture data for testing",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error code 1 on any failure",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for JSON report",
    )

    args = parser.parse_args()

    if args.fixture:
        logger.info("Using fixture data for testing")
        data = create_fixture_data()
    elif args.data_dir and args.as_of_date:
        logger.info(f"Loading data from {args.data_dir}")
        data = load_data_from_dir(args.data_dir, args.as_of_date)
    else:
        parser.error("Either --fixture or both --data-dir and --as-of-date are required")

    # Create validator
    validator = DataIntegrationValidator(
        universe=data["universe"],
        financial_records=data["financial_records"],
        market_data=data["market_data"],
        trial_records=data["trial_records"],
        as_of_date=data["as_of_date"],
        holdings_snapshots=data.get("holdings_snapshots"),
        short_interest=data.get("short_interest"),
        momentum_results=data.get("momentum_results"),
    )

    # Run validation
    report = validator.validate_all()

    # Output report
    if args.output:
        report_dict = {
            "passed": report.passed,
            "critical_failures": report.critical_failures,
            "high_failures": report.high_failures,
            "medium_failures": report.medium_failures,
            "low_failures": report.low_failures,
            "data_hash": report.data_hash,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details,
                }
                for c in report.checks
            ],
        }
        with open(args.output, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info(f"Report written to {args.output}")

    # Exit with appropriate code
    if args.strict and not report.passed:
        sys.exit(1)
    elif report.critical_failures > 0 or report.high_failures > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
