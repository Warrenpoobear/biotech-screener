"""
Query 4.2 - Backtesting Validation Framework.

Validates historical validation infrastructure capabilities.

Checks:
- Walk-forward testing architecture
- Regime testing capability
- Statistical validation metrics
- Overfitting detection
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
class BacktestCapability:
    """Assessment of a backtesting capability."""

    capability: str
    present: bool
    implementation_files: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class BacktestReport:
    """Complete backtesting capability report."""

    capabilities: List[BacktestCapability] = field(default_factory=list)
    has_walk_forward: bool = False
    has_pit_snapshots: bool = False
    has_regime_testing: bool = False
    has_statistical_validation: bool = False
    has_overfitting_detection: bool = False
    readiness_score: int = 0  # 0-100
    passed: bool = False

    @property
    def capability_count(self) -> int:
        return sum(1 for c in self.capabilities if c.present)


class BacktestValidator:
    """
    Validates backtesting infrastructure.

    Checks:
    - Historical data reconstruction capability
    - Point-in-time data snapshots
    - Regime-specific testing
    - Statistical metrics calculation
    - Overfitting detection mechanisms
    """

    # Required backtesting capabilities
    REQUIRED_CAPABILITIES: Dict[str, List[str]] = {
        "walk_forward": [
            r"walk_forward",
            r"walk-forward",
            r"expanding_window",
            r"rolling_window",
        ],
        "pit_reconstruction": [
            r"as_of_date",
            r"pit_cutoff",
            r"point_in_time",
            r"historical_snapshot",
        ],
        "regime_testing": [
            r"regime",
            r"bull.*bear",
            r"market_condition",
            r"volatility.*regime",
        ],
        "statistical_metrics": [
            r"sharpe.*ratio",
            r"information.*coefficient",
            r"\bIC\b",
            r"hit.*rate",
            r"drawdown",
        ],
        "overfitting_detection": [
            r"out.*of.*sample",
            r"cross.*validation",
            r"holdout",
            r"overfit",
            r"in.*sample.*out.*sample",
        ],
    }

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)
        self.backtest_dir = self.codebase_path / "backtest"

    def _check_capability(
        self,
        capability_name: str,
        patterns: List[str],
    ) -> BacktestCapability:
        """Check if a capability is implemented."""
        implementation_files = []

        search_dirs = [
            self.codebase_path,
            self.backtest_dir,
            self.codebase_path / "validation",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for py_file in search_dir.glob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read().lower()

                    for pattern in patterns:
                        if re.search(pattern.lower(), content):
                            rel_path = str(py_file.relative_to(self.codebase_path))
                            if rel_path not in implementation_files:
                                implementation_files.append(rel_path)
                            break

                except Exception:
                    continue

        present = len(implementation_files) > 0

        return BacktestCapability(
            capability=capability_name,
            present=present,
            implementation_files=implementation_files,
            description=f"{capability_name.replace('_', ' ').title()} capability",
        )

    def check_historical_data(self) -> Dict[str, Any]:
        """Check for historical data availability."""
        data_info = {
            "has_snapshots": False,
            "snapshot_count": 0,
            "date_range": None,
        }

        # Check for snapshot directories
        snapshot_dirs = [
            self.codebase_path / "data" / "aact_snapshots",
            self.codebase_path / "production_data" / "ctgov_state",
        ]

        for snap_dir in snapshot_dirs:
            if snap_dir.exists():
                data_info["has_snapshots"] = True
                snapshot_files = list(snap_dir.glob("*.json")) + list(snap_dir.glob("*.jsonl"))
                data_info["snapshot_count"] += len(snapshot_files)

        return data_info

    def check_metrics_implementation(self) -> Dict[str, bool]:
        """Check for statistical metrics implementations."""
        metrics = {
            "sharpe_ratio": False,
            "information_coefficient": False,
            "hit_rate": False,
            "maximum_drawdown": False,
            "rank_correlation": False,
        }

        metrics_file = self.backtest_dir / "metrics.py"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                metrics["sharpe_ratio"] = "sharpe" in content
                metrics["information_coefficient"] = "ic" in content or "information_coefficient" in content
                metrics["hit_rate"] = "hit_rate" in content or "hit rate" in content
                metrics["maximum_drawdown"] = "drawdown" in content
                metrics["rank_correlation"] = "spearman" in content or "rank" in content

            except Exception:
                pass

        return metrics

    def run_audit(self) -> BacktestReport:
        """
        Run complete backtesting capability audit.

        Returns:
            BacktestReport with findings
        """
        capabilities = []

        # Check each required capability
        for cap_name, patterns in self.REQUIRED_CAPABILITIES.items():
            cap = self._check_capability(cap_name, patterns)
            capabilities.append(cap)

        # Check historical data
        hist_data = self.check_historical_data()

        # Check metrics
        metrics = self.check_metrics_implementation()

        # Determine specific capabilities
        has_walk_forward = any(
            c.present for c in capabilities if c.capability == "walk_forward"
        )
        has_pit = any(
            c.present for c in capabilities if c.capability == "pit_reconstruction"
        )
        has_regime = any(
            c.present for c in capabilities if c.capability == "regime_testing"
        )
        has_stats = any(
            c.present for c in capabilities if c.capability == "statistical_metrics"
        )
        has_overfit = any(
            c.present for c in capabilities if c.capability == "overfitting_detection"
        )

        # Calculate readiness score
        score = 0
        score += 20 if has_walk_forward else 0
        score += 20 if has_pit else 0
        score += 20 if has_regime else 0
        score += 20 if has_stats else 0
        score += 10 if has_overfit else 0
        score += 10 if hist_data["has_snapshots"] else 0

        passed = score >= 60 and has_pit and has_stats

        return BacktestReport(
            capabilities=capabilities,
            has_walk_forward=has_walk_forward,
            has_pit_snapshots=has_pit and hist_data["has_snapshots"],
            has_regime_testing=has_regime,
            has_statistical_validation=has_stats,
            has_overfitting_detection=has_overfit,
            readiness_score=score,
            passed=passed,
        )


def validate_backtest_capability(codebase_path: str) -> AuditResult:
    """
    Run complete backtesting capability validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = BacktestValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="backtest_capability",
        passed=report.passed,
        metrics={
            "readiness_score": report.readiness_score,
            "capability_count": report.capability_count,
            "has_walk_forward": report.has_walk_forward,
            "has_pit_snapshots": report.has_pit_snapshots,
            "has_regime_testing": report.has_regime_testing,
            "has_statistical_validation": report.has_statistical_validation,
            "has_overfitting_detection": report.has_overfitting_detection,
        },
        details=f"Backtest readiness score: {report.readiness_score}/100",
    )

    # Add findings for missing capabilities
    for cap in report.capabilities:
        if not cap.present:
            severity = (
                AuditSeverity.HIGH
                if cap.capability in ["pit_reconstruction", "statistical_metrics"]
                else AuditSeverity.MEDIUM
            )

            result.add_finding(
                severity=severity,
                category=ValidationCategory.TEST_COVERAGE,
                title=f"Missing backtest capability: {cap.capability}",
                description=f"{cap.description} is not implemented",
                location="backtest/",
                evidence="No matching implementation found",
                remediation=f"Implement {cap.capability.replace('_', ' ')} functionality",
                compliance_impact="Missing capability limits historical validation",
            )

    return result
