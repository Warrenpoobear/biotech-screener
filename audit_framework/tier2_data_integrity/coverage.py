"""
Query 2.2 - Data Quality & Coverage Analysis.

Generates comprehensive data quality metrics for institutional investors.

Metrics:
- Coverage statistics across universe
- Freshness metrics
- Quality gates and warnings
- Automated coverage dashboard data
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class CoverageMetrics:
    """Coverage metrics for a specific data type."""

    data_type: str
    total_universe: int
    covered_count: int
    missing_count: int
    coverage_percent: Decimal
    missing_tickers: List[str] = field(default_factory=list)


@dataclass
class FreshnessMetrics:
    """Data freshness metrics."""

    data_type: str
    median_age_days: int
    max_age_days: int
    stale_count: int  # > 90 days for financial data
    stale_tickers: List[str] = field(default_factory=list)


@dataclass
class QualityGate:
    """A quality gate check result."""

    gate_name: str
    threshold: str
    actual_value: str
    passed: bool
    warning_level: str  # "info", "warning", "critical"
    affected_tickers: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Complete data quality and coverage report."""

    as_of_date: str
    universe_size: int
    coverage_metrics: List[CoverageMetrics] = field(default_factory=list)
    freshness_metrics: List[FreshnessMetrics] = field(default_factory=list)
    quality_gates: List[QualityGate] = field(default_factory=list)
    overall_coverage: Decimal = Decimal("0")
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "as_of_date": self.as_of_date,
            "universe_size": self.universe_size,
            "overall_coverage": str(self.overall_coverage),
            "passed": self.passed,
            "coverage_metrics": [
                {
                    "data_type": m.data_type,
                    "total_universe": m.total_universe,
                    "covered_count": m.covered_count,
                    "coverage_percent": str(m.coverage_percent),
                }
                for m in self.coverage_metrics
            ],
            "freshness_metrics": [
                {
                    "data_type": f.data_type,
                    "median_age_days": f.median_age_days,
                    "max_age_days": f.max_age_days,
                    "stale_count": f.stale_count,
                }
                for f in self.freshness_metrics
            ],
            "quality_gates": [
                {
                    "gate_name": g.gate_name,
                    "threshold": g.threshold,
                    "actual_value": g.actual_value,
                    "passed": g.passed,
                    "warning_level": g.warning_level,
                }
                for g in self.quality_gates
            ],
        }


class CoverageValidator:
    """
    Validates data quality and coverage across the screening universe.

    Generates metrics suitable for:
    - Weekly investment committee review
    - Automated monitoring dashboards
    - Regulatory compliance reporting
    """

    # Freshness thresholds (days)
    STALE_FINANCIAL_DAYS = 90  # > 90 days = stale
    STALE_TRIAL_DAYS = 30      # > 30 days = stale
    STALE_13F_DAYS = 60        # > 60 days (quarterly + lag)

    # Coverage thresholds
    MIN_FINANCIAL_COVERAGE = Decimal("0.90")  # 90%
    MIN_POS_COVERAGE = Decimal("0.70")        # 70%
    MIN_CATALYST_COVERAGE = Decimal("0.70")   # 70%

    def __init__(
        self,
        codebase_path: str,
        data_dir: str = "production_data",
        as_of_date: Optional[str] = None,
    ):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)
        self.data_dir = self.codebase_path / data_dir
        self.as_of_date = as_of_date or date.today().isoformat()

    def _load_universe(self) -> Set[str]:
        """Load the screening universe."""
        universe_file = self.data_dir / "universe.json"
        if not universe_file.exists():
            return set()

        try:
            with open(universe_file, "r") as f:
                data = json.load(f)

            # Handle different structures
            if isinstance(data, list):
                return {
                    r.get("ticker", r.get("symbol", ""))
                    for r in data
                    if isinstance(r, dict)
                }
            elif isinstance(data, dict):
                # Check for active_securities
                if "active_securities" in data:
                    return {
                        r.get("ticker", "")
                        for r in data["active_securities"]
                        if isinstance(r, dict)
                    }
                # Check for records
                if "records" in data:
                    return {
                        r.get("ticker", "")
                        for r in data["records"]
                        if isinstance(r, dict)
                    }

        except Exception:
            pass

        return set()

    def _load_json_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load and normalize JSON data file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check for various record containers
                for key in ["records", "results", "securities", "scores"]:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                return [data]

        except Exception:
            pass

        return []

    def _extract_tickers(self, records: List[Dict[str, Any]]) -> Set[str]:
        """Extract ticker symbols from records."""
        tickers = set()
        for record in records:
            if isinstance(record, dict):
                ticker = record.get("ticker", record.get("symbol", ""))
                if ticker:
                    tickers.add(ticker)
        return tickers

    def _calculate_freshness(
        self,
        records: List[Dict[str, Any]],
        date_field: str,
    ) -> FreshnessMetrics:
        """Calculate freshness metrics for a dataset."""
        ages = []
        stale_tickers = []

        as_of = datetime.fromisoformat(self.as_of_date).date()

        for record in records:
            if not isinstance(record, dict):
                continue

            date_str = record.get(date_field, record.get("source_date", ""))
            if not date_str:
                continue

            try:
                record_date = datetime.fromisoformat(str(date_str)[:10]).date()
                age = (as_of - record_date).days
                ages.append(age)

                ticker = record.get("ticker", record.get("symbol", ""))
                if age > self.STALE_FINANCIAL_DAYS and ticker:
                    stale_tickers.append(ticker)

            except Exception:
                continue

        if not ages:
            return FreshnessMetrics(
                data_type="unknown",
                median_age_days=0,
                max_age_days=0,
                stale_count=0,
            )

        ages.sort()
        median_age = ages[len(ages) // 2]
        max_age = max(ages)
        stale_count = len(stale_tickers)

        return FreshnessMetrics(
            data_type="",  # Set by caller
            median_age_days=median_age,
            max_age_days=max_age,
            stale_count=stale_count,
            stale_tickers=stale_tickers[:20],
        )

    def check_financial_coverage(self, universe: Set[str]) -> CoverageMetrics:
        """Check financial data coverage."""
        records = self._load_json_data("financial_records.json")
        covered = self._extract_tickers(records)

        covered_count = len(universe & covered)
        missing = universe - covered

        coverage = Decimal("0")
        if universe:
            coverage = (Decimal(covered_count) / Decimal(len(universe))).quantize(
                Decimal("0.0001")
            )

        return CoverageMetrics(
            data_type="financial",
            total_universe=len(universe),
            covered_count=covered_count,
            missing_count=len(missing),
            coverage_percent=coverage,
            missing_tickers=sorted(list(missing))[:20],
        )

    def check_clinical_coverage(self, universe: Set[str]) -> CoverageMetrics:
        """Check clinical/PoS data coverage."""
        records = self._load_json_data("trial_records.json")

        # Check for tickers with PoS scores
        covered = set()
        for record in records:
            if isinstance(record, dict):
                ticker = record.get("ticker", "")
                # Check for PoS or clinical fields
                has_pos = record.get("pos_score") is not None
                has_phase = record.get("phase") or record.get("lead_phase")
                if ticker and (has_pos or has_phase):
                    covered.add(ticker)

        covered_count = len(universe & covered)
        missing = universe - covered

        coverage = Decimal("0")
        if universe:
            coverage = (Decimal(covered_count) / Decimal(len(universe))).quantize(
                Decimal("0.0001")
            )

        return CoverageMetrics(
            data_type="clinical_pos",
            total_universe=len(universe),
            covered_count=covered_count,
            missing_count=len(missing),
            coverage_percent=coverage,
            missing_tickers=sorted(list(missing))[:20],
        )

    def check_catalyst_coverage(self, universe: Set[str]) -> CoverageMetrics:
        """Check catalyst event coverage."""
        records = self._load_json_data("trial_records.json")

        # Check for tickers with dated catalyst events
        covered = set()
        for record in records:
            if isinstance(record, dict):
                ticker = record.get("ticker", "")
                has_catalyst = (
                    record.get("primary_completion_date")
                    or record.get("next_catalyst_date")
                    or record.get("expected_readout")
                )
                if ticker and has_catalyst:
                    covered.add(ticker)

        covered_count = len(universe & covered)
        missing = universe - covered

        coverage = Decimal("0")
        if universe:
            coverage = (Decimal(covered_count) / Decimal(len(universe))).quantize(
                Decimal("0.0001")
            )

        return CoverageMetrics(
            data_type="catalyst",
            total_universe=len(universe),
            covered_count=covered_count,
            missing_count=len(missing),
            coverage_percent=coverage,
            missing_tickers=sorted(list(missing))[:20],
        )

    def check_13f_coverage(self, universe: Set[str]) -> CoverageMetrics:
        """Check 13F institutional holdings coverage."""
        records = self._load_json_data("holdings_snapshots.json")

        covered = self._extract_tickers(records)
        covered_count = len(universe & covered)
        missing = universe - covered

        coverage = Decimal("0")
        if universe:
            coverage = (Decimal(covered_count) / Decimal(len(universe))).quantize(
                Decimal("0.0001")
            )

        return CoverageMetrics(
            data_type="13f_holdings",
            total_universe=len(universe),
            covered_count=covered_count,
            missing_count=len(missing),
            coverage_percent=coverage,
            missing_tickers=sorted(list(missing))[:20],
        )

    def run_quality_gates(
        self,
        coverage_metrics: List[CoverageMetrics],
        freshness_metrics: List[FreshnessMetrics],
    ) -> List[QualityGate]:
        """Run quality gate checks."""
        gates = []

        # Financial coverage gate
        financial = next(
            (m for m in coverage_metrics if m.data_type == "financial"), None
        )
        if financial:
            passed = financial.coverage_percent >= self.MIN_FINANCIAL_COVERAGE
            gates.append(QualityGate(
                gate_name="financial_coverage",
                threshold=f">= {self.MIN_FINANCIAL_COVERAGE*100}%",
                actual_value=f"{financial.coverage_percent*100:.1f}%",
                passed=passed,
                warning_level="critical" if not passed else "info",
                affected_tickers=financial.missing_tickers,
            ))

        # PoS coverage gate
        pos = next(
            (m for m in coverage_metrics if m.data_type == "clinical_pos"), None
        )
        if pos:
            passed = pos.coverage_percent >= self.MIN_POS_COVERAGE
            gates.append(QualityGate(
                gate_name="pos_coverage",
                threshold=f">= {self.MIN_POS_COVERAGE*100}%",
                actual_value=f"{pos.coverage_percent*100:.1f}%",
                passed=passed,
                warning_level="warning" if not passed else "info",
                affected_tickers=pos.missing_tickers,
            ))

        # Catalyst coverage gate
        catalyst = next(
            (m for m in coverage_metrics if m.data_type == "catalyst"), None
        )
        if catalyst:
            passed = catalyst.coverage_percent >= self.MIN_CATALYST_COVERAGE
            gates.append(QualityGate(
                gate_name="catalyst_coverage",
                threshold=f">= {self.MIN_CATALYST_COVERAGE*100}%",
                actual_value=f"{catalyst.coverage_percent*100:.1f}%",
                passed=passed,
                warning_level="warning" if not passed else "info",
                affected_tickers=catalyst.missing_tickers,
            ))

        # Staleness gates
        for freshness in freshness_metrics:
            if freshness.stale_count > 0:
                stale_pct = 0
                if coverage_metrics:
                    total = coverage_metrics[0].total_universe
                    if total > 0:
                        stale_pct = (freshness.stale_count / total) * 100

                gates.append(QualityGate(
                    gate_name=f"{freshness.data_type}_freshness",
                    threshold="< 90 days",
                    actual_value=f"max {freshness.max_age_days} days, {stale_pct:.1f}% stale",
                    passed=freshness.stale_count == 0,
                    warning_level="warning" if freshness.stale_count > 0 else "info",
                    affected_tickers=freshness.stale_tickers,
                ))

        return gates

    def run_audit(self) -> CoverageReport:
        """
        Run complete coverage audit.

        Returns:
            CoverageReport with findings
        """
        universe = self._load_universe()

        # Calculate coverage metrics
        coverage_metrics = [
            self.check_financial_coverage(universe),
            self.check_clinical_coverage(universe),
            self.check_catalyst_coverage(universe),
            self.check_13f_coverage(universe),
        ]

        # Calculate freshness metrics
        financial_records = self._load_json_data("financial_records.json")
        trial_records = self._load_json_data("trial_records.json")

        freshness_metrics = []

        fin_fresh = self._calculate_freshness(financial_records, "filed_date")
        fin_fresh.data_type = "financial"
        freshness_metrics.append(fin_fresh)

        trial_fresh = self._calculate_freshness(trial_records, "last_update_posted")
        trial_fresh.data_type = "clinical"
        freshness_metrics.append(trial_fresh)

        # Run quality gates
        quality_gates = self.run_quality_gates(coverage_metrics, freshness_metrics)

        # Calculate overall coverage
        if coverage_metrics:
            overall = sum(
                m.coverage_percent for m in coverage_metrics
            ) / Decimal(len(coverage_metrics))
        else:
            overall = Decimal("0")

        # Determine pass/fail
        critical_gates_passed = all(
            g.passed for g in quality_gates if g.warning_level == "critical"
        )
        passed = critical_gates_passed and overall >= Decimal("0.75")

        return CoverageReport(
            as_of_date=self.as_of_date,
            universe_size=len(universe),
            coverage_metrics=coverage_metrics,
            freshness_metrics=freshness_metrics,
            quality_gates=quality_gates,
            overall_coverage=overall.quantize(Decimal("0.0001")),
            passed=passed,
        )


def validate_data_coverage(
    codebase_path: str,
    data_dir: str = "production_data",
    as_of_date: Optional[str] = None,
) -> AuditResult:
    """
    Run complete data coverage validation.

    Args:
        codebase_path: Root of codebase
        data_dir: Data directory name
        as_of_date: Reference date

    Returns:
        AuditResult with findings
    """
    validator = CoverageValidator(codebase_path, data_dir, as_of_date)
    report = validator.run_audit()

    result = AuditResult(
        check_name="data_coverage",
        passed=report.passed,
        metrics=report.to_dict(),
        details=f"Overall coverage: {report.overall_coverage*100:.1f}%, "
                f"Universe size: {report.universe_size}",
    )

    # Add findings for failed quality gates
    for gate in report.quality_gates:
        if not gate.passed:
            severity = (
                AuditSeverity.CRITICAL if gate.warning_level == "critical"
                else AuditSeverity.HIGH if gate.warning_level == "warning"
                else AuditSeverity.MEDIUM
            )

            result.add_finding(
                severity=severity,
                category=ValidationCategory.COVERAGE,
                title=f"Quality gate failed: {gate.gate_name}",
                description=f"Expected {gate.threshold}, got {gate.actual_value}",
                location="production_data/",
                evidence=f"Affected tickers: {', '.join(gate.affected_tickers[:10])}...",
                remediation=f"Increase {gate.gate_name} to meet threshold",
                compliance_impact="Incomplete coverage may lead to missed opportunities or biased screening",
            )

    return result
