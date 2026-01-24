"""
Query 4.3 - Model Explainability & Audit Trail.

Validates that screening decisions are explainable for investment
committee review.

Checks:
- Score decomposition capability
- Audit trail documentation
- Sensitivity analysis
- Fiduciary duty compliance
"""

import ast
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
class ExplainabilityFeature:
    """Assessment of an explainability feature."""

    feature: str
    present: bool
    implementation_files: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AuditTrailCheck:
    """Audit trail compliance check."""

    check_name: str
    passed: bool
    evidence: str = ""
    location: str = ""


@dataclass
class ExplainabilityReport:
    """Complete explainability audit report."""

    features: List[ExplainabilityFeature] = field(default_factory=list)
    audit_trail_checks: List[AuditTrailCheck] = field(default_factory=list)
    has_score_decomposition: bool = False
    has_governance_metadata: bool = False
    has_sensitivity_analysis: bool = False
    has_decision_logging: bool = False
    explainability_score: int = 0  # 0-100
    passed: bool = False


class ExplainabilityValidator:
    """
    Validates model explainability and audit trail.

    Checks:
    - Score component breakdown
    - Governance metadata in outputs
    - Sensitivity analysis capability
    - Decision rationale logging
    """

    # Explainability patterns
    EXPLAINABILITY_PATTERNS: Dict[str, List[str]] = {
        "score_decomposition": [
            r"component.*score",
            r"score.*breakdown",
            r"weight.*contribution",
            r"factor.*attribution",
            r"sub_score",
        ],
        "governance_metadata": [
            r"_governance",
            r"provenance",
            r"run_id",
            r"schema_version",
            r"audit_trail",
        ],
        "sensitivity_analysis": [
            r"sensitivity",
            r"parameter.*impact",
            r"weight.*change",
            r"monte.*carlo",
            r"confidence.*interval",
        ],
        "decision_logging": [
            r"audit.*log",
            r"decision.*reason",
            r"exclusion.*reason",
            r"score.*rationale",
            r"flag.*reason",
        ],
    }

    def __init__(self, codebase_path: str):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)

    def _check_feature(
        self,
        feature_name: str,
        patterns: List[str],
    ) -> ExplainabilityFeature:
        """Check if an explainability feature is implemented."""
        implementation_files = []

        exclude_dirs = {"tests", "deprecated", "venv", "mnt", "audit_framework"}

        for root, dirs, files in os.walk(self.codebase_path):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().lower()

                    for pattern in patterns:
                        if re.search(pattern.lower(), content):
                            rel_path = str(file_path.relative_to(self.codebase_path))
                            if rel_path not in implementation_files:
                                implementation_files.append(rel_path)
                            break

                except Exception:
                    continue

        return ExplainabilityFeature(
            feature=feature_name,
            present=len(implementation_files) > 0,
            implementation_files=implementation_files[:5],
            description=f"{feature_name.replace('_', ' ').title()}",
        )

    def check_governance_in_outputs(self) -> AuditTrailCheck:
        """Check for _governance blocks in output schemas."""
        data_dir = self.codebase_path / "production_data"
        has_governance = False
        evidence = ""

        if data_dir.exists():
            for json_file in data_dir.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        # Check first 1000 chars for governance
                        content = f.read(1000)
                        if "_governance" in content:
                            has_governance = True
                            evidence = f"Found in {json_file.name}"
                            break
                except Exception:
                    continue

        return AuditTrailCheck(
            check_name="governance_metadata_in_outputs",
            passed=has_governance,
            evidence=evidence if has_governance else "No _governance blocks found",
            location="production_data/",
        )

    def check_audit_log_infrastructure(self) -> AuditTrailCheck:
        """Check for audit logging infrastructure."""
        audit_log_file = self.codebase_path / "governance" / "audit_log.py"

        if audit_log_file.exists():
            try:
                with open(audit_log_file, "r", encoding="utf-8") as f:
                    content = f.read()

                has_jsonl = "jsonl" in content.lower()
                has_stage = "stage" in content.lower()

                return AuditTrailCheck(
                    check_name="audit_log_infrastructure",
                    passed=has_jsonl and has_stage,
                    evidence="JSONL audit log with stage tracking" if has_jsonl else "Partial implementation",
                    location="governance/audit_log.py",
                )

            except Exception:
                pass

        return AuditTrailCheck(
            check_name="audit_log_infrastructure",
            passed=False,
            evidence="No audit_log.py found",
            location="governance/",
        )

    def check_score_transparency(self) -> AuditTrailCheck:
        """Check for score calculation transparency."""
        # Look for composite scoring module with component breakdown
        composite_files = list(self.codebase_path.glob("module_5*.py"))
        has_transparency = False
        evidence = ""

        for comp_file in composite_files:
            try:
                with open(comp_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for component weights or breakdown
                if re.search(r"weight.*=|component.*score|sub_score", content, re.IGNORECASE):
                    has_transparency = True
                    evidence = f"Found in {comp_file.name}"
                    break

            except Exception:
                continue

        return AuditTrailCheck(
            check_name="score_transparency",
            passed=has_transparency,
            evidence=evidence if has_transparency else "No score component breakdown found",
            location="module_5*.py",
        )

    def check_exclusion_reasons(self) -> AuditTrailCheck:
        """Check for exclusion reason tracking."""
        has_reasons = False
        evidence = ""

        for module in ["module_1_universe.py", "module_5_composite.py"]:
            module_file = self.codebase_path / module
            if not module_file.exists():
                continue

            try:
                with open(module_file, "r", encoding="utf-8") as f:
                    content = f.read()

                if re.search(r"exclusion.*reason|reason.*excluded|excluded.*securities", content, re.IGNORECASE):
                    has_reasons = True
                    evidence = f"Found in {module}"
                    break

            except Exception:
                continue

        return AuditTrailCheck(
            check_name="exclusion_reasons",
            passed=has_reasons,
            evidence=evidence if has_reasons else "No exclusion reason tracking",
            location="module_1_universe.py",
        )

    def run_audit(self) -> ExplainabilityReport:
        """
        Run complete explainability audit.

        Returns:
            ExplainabilityReport with findings
        """
        features = []

        # Check each explainability feature
        for feat_name, patterns in self.EXPLAINABILITY_PATTERNS.items():
            feat = self._check_feature(feat_name, patterns)
            features.append(feat)

        # Run audit trail checks
        audit_checks = [
            self.check_governance_in_outputs(),
            self.check_audit_log_infrastructure(),
            self.check_score_transparency(),
            self.check_exclusion_reasons(),
        ]

        # Determine capabilities
        has_decomp = any(f.present for f in features if f.feature == "score_decomposition")
        has_gov = any(f.present for f in features if f.feature == "governance_metadata")
        has_sens = any(f.present for f in features if f.feature == "sensitivity_analysis")
        has_log = any(f.present for f in features if f.feature == "decision_logging")

        # Calculate score
        score = 0
        score += sum(25 for f in features if f.present)
        score = min(100, score)

        passed = score >= 50 and has_gov and has_decomp

        return ExplainabilityReport(
            features=features,
            audit_trail_checks=audit_checks,
            has_score_decomposition=has_decomp,
            has_governance_metadata=has_gov,
            has_sensitivity_analysis=has_sens,
            has_decision_logging=has_log,
            explainability_score=score,
            passed=passed,
        )


def validate_explainability(codebase_path: str) -> AuditResult:
    """
    Run complete explainability validation.

    Args:
        codebase_path: Root of codebase

    Returns:
        AuditResult with findings
    """
    validator = ExplainabilityValidator(codebase_path)
    report = validator.run_audit()

    result = AuditResult(
        check_name="explainability",
        passed=report.passed,
        metrics={
            "explainability_score": report.explainability_score,
            "has_score_decomposition": report.has_score_decomposition,
            "has_governance_metadata": report.has_governance_metadata,
            "has_sensitivity_analysis": report.has_sensitivity_analysis,
            "has_decision_logging": report.has_decision_logging,
            "audit_trail_checks_passed": sum(1 for c in report.audit_trail_checks if c.passed),
        },
        details=f"Explainability score: {report.explainability_score}/100",
    )

    # Add findings for missing features
    for feat in report.features:
        if not feat.present:
            severity = (
                AuditSeverity.HIGH
                if feat.feature in ["governance_metadata", "score_decomposition"]
                else AuditSeverity.MEDIUM
            )

            result.add_finding(
                severity=severity,
                category=ValidationCategory.DOCUMENTATION,
                title=f"Missing explainability feature: {feat.feature}",
                description=f"{feat.description} is not implemented",
                location="scoring modules",
                evidence="No matching implementation found",
                remediation=f"Implement {feat.feature.replace('_', ' ')} capability",
                compliance_impact="Missing explainability limits investment committee review",
            )

    # Add findings for failed audit trail checks
    for check in report.audit_trail_checks:
        if not check.passed:
            result.add_finding(
                severity=AuditSeverity.HIGH,
                category=ValidationCategory.DOCUMENTATION,
                title=f"Audit trail check failed: {check.check_name}",
                description=check.evidence,
                location=check.location,
                evidence=check.evidence,
                remediation="Implement proper audit trail tracking",
                compliance_impact="Missing audit trail violates fiduciary documentation requirements",
            )

    return result
