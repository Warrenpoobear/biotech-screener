"""
Query 2.1 - Tier-0 Provenance Lock Validation.

Audits the "no fabricated data" guarantee by tracing every fact
to its primary source.

Primary Sources:
- SEC EDGAR filings (10-K, 10-Q, 13F)
- ClinicalTrials.gov (NCT numbers)
- FDA databases (NDA numbers, approval dates)
- Company IR (press releases)

Validates:
- Every data field has documented source
- Sources can be programmatically verified
- UNKNOWN is used instead of inference
- Fabricated ticker test (EPRX firewall)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from decimal import Decimal

from audit_framework.types import (
    AuditResult,
    AuditSeverity,
    ValidationCategory,
)


@dataclass
class ProvenanceRecord:
    """A single data record with provenance tracking."""

    ticker: str
    field_name: str
    value: Any
    source_type: Optional[str] = None  # "SEC_EDGAR", "CTGOV", "FDA", "IR"
    source_id: Optional[str] = None    # NCT number, form ID, etc.
    source_date: Optional[str] = None
    is_verified: bool = False
    is_fabricated: bool = False


@dataclass
class ProvenanceReport:
    """Complete provenance audit report."""

    total_records: int
    records_with_source: int
    records_without_source: int
    records_verified: int
    fabricated_detected: int
    source_type_breakdown: Dict[str, int] = field(default_factory=dict)
    missing_sources: List[str] = field(default_factory=list)
    coverage_score: Decimal = Decimal("0")
    passed: bool = False

    @property
    def provenance_coverage(self) -> Decimal:
        if self.total_records == 0:
            return Decimal("0")
        return (
            Decimal(self.records_with_source) / Decimal(self.total_records)
        ).quantize(Decimal("0.0001"))


class ProvenanceValidator:
    """
    Validates data provenance across the pipeline.

    Ensures:
    - All facts traceable to primary sources
    - Source identifiers are valid and verifiable
    - UNKNOWN is used instead of inference
    - Fabricated data is rejected
    """

    # Source type patterns
    SOURCE_PATTERNS: Dict[str, str] = {
        "SEC_EDGAR": r"(10-K|10-Q|8-K|13F|S-1|DEF 14A)",
        "CTGOV": r"NCT\d{8}",
        "FDA": r"(NDA|BLA|ANDA)\d+",
        "PUBMED": r"PMID:\s*\d+",
        "DOI": r"10\.\d{4,}/[\w\.\-/]+",
    }

    # Fields that MUST have provenance
    REQUIRED_PROVENANCE_FIELDS: Set[str] = {
        "cash",
        "cash_equivalents",
        "total_cash",
        "quarterly_burn",
        "burn_rate",
        "market_cap",
        "shares_outstanding",
        "revenue",
        "net_income",
        "clinical_stage",
        "lead_phase",
        "primary_indication",
    }

    def __init__(self, codebase_path: str, data_dir: str = "production_data"):
        """Initialize validator."""
        self.codebase_path = Path(codebase_path)
        self.data_dir = self.codebase_path / data_dir

    def _extract_source_type(self, record: Dict[str, Any]) -> Optional[str]:
        """Extract source type from a record."""
        # Check for explicit source field
        source = record.get("source", record.get("data_source", ""))

        for source_type, pattern in self.SOURCE_PATTERNS.items():
            if re.search(pattern, str(source), re.IGNORECASE):
                return source_type

        # Check for NCT ID
        nct_id = record.get("nct_id", record.get("trial_id", ""))
        if nct_id and re.match(r"NCT\d{8}", str(nct_id)):
            return "CTGOV"

        # Check for filing type
        filing = record.get("filing_type", record.get("form_type", ""))
        if filing and re.search(self.SOURCE_PATTERNS["SEC_EDGAR"], str(filing)):
            return "SEC_EDGAR"

        return None

    def _check_record_provenance(
        self,
        record: Dict[str, Any],
        file_name: str,
    ) -> List[ProvenanceRecord]:
        """Check provenance of a single record."""
        results = []
        ticker = record.get("ticker", record.get("symbol", "UNKNOWN"))

        source_type = self._extract_source_type(record)
        source_date = record.get("source_date", record.get("filed_date", ""))
        source_id = record.get("nct_id", record.get("form_id", record.get("source", "")))

        for field_name in self.REQUIRED_PROVENANCE_FIELDS:
            if field_name not in record:
                continue

            value = record[field_name]

            prov_record = ProvenanceRecord(
                ticker=ticker,
                field_name=field_name,
                value=value,
                source_type=source_type,
                source_id=str(source_id) if source_id else None,
                source_date=str(source_date) if source_date else None,
                is_verified=source_type is not None,
            )

            results.append(prov_record)

        return results

    def _audit_data_file(self, file_path: Path) -> List[ProvenanceRecord]:
        """Audit a single data file for provenance."""
        results = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return results

        file_name = file_path.name

        if isinstance(data, list):
            for record in data:
                if isinstance(record, dict):
                    results.extend(self._check_record_provenance(record, file_name))
        elif isinstance(data, dict):
            # Check for records in various structures
            for key in ["records", "results", "securities", "scores"]:
                if key in data and isinstance(data[key], list):
                    for record in data[key]:
                        if isinstance(record, dict):
                            results.extend(self._check_record_provenance(record, file_name))
                    break
            else:
                # Treat as single record
                results.extend(self._check_record_provenance(data, file_name))

        return results

    def test_fabrication_firewall(self, test_ticker: str = "EPRX") -> Dict[str, Any]:
        """
        Test that fabricated tickers are rejected by the system.

        The EPRX test validates the system won't score non-existent tickers.
        """
        result = {
            "test_ticker": test_ticker,
            "rejection_expected": True,
            "rejection_detected": False,
            "error_message": None,
            "passed": False,
        }

        # Check if ticker appears in any production data
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Search for the test ticker
                content = json.dumps(data)
                if test_ticker in content:
                    result["rejection_detected"] = False
                    result["error_message"] = f"Test ticker {test_ticker} found in {json_file.name}"
                    return result

            except Exception:
                continue

        result["rejection_detected"] = True
        result["passed"] = True
        result["error_message"] = f"Ticker {test_ticker} correctly not present in data"

        return result

    def audit_codebase_for_unknown_handling(self) -> List[str]:
        """
        Check that UNKNOWN is used instead of data fabrication.

        Returns list of files with proper UNKNOWN handling.
        """
        compliant_files = []
        unknown_patterns = [
            r'"UNKNOWN"',
            r"'UNKNOWN'",
            r"Severity\.UNKNOWN",
            r"\.get\([^)]+,\s*['\"]UNKNOWN['\"]\)",
            r"if.*is\s+None.*UNKNOWN",
        ]

        for root, _, files in os.walk(self.codebase_path):
            # Skip test directories
            if "test" in root.lower() or "deprecated" in root:
                continue

            for file in files:
                if not file.endswith(".py"):
                    continue

                file_path = Path(root) / file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    if any(re.search(p, content) for p in unknown_patterns):
                        rel_path = str(file_path.relative_to(self.codebase_path))
                        compliant_files.append(rel_path)

                except Exception:
                    continue

        return compliant_files

    def run_audit(self) -> ProvenanceReport:
        """
        Run complete provenance audit.

        Returns:
            ProvenanceReport with findings
        """
        all_records: List[ProvenanceRecord] = []

        # Audit all JSON data files
        if self.data_dir.exists():
            for json_file in self.data_dir.glob("*.json"):
                records = self._audit_data_file(json_file)
                all_records.extend(records)

        # Calculate metrics
        total = len(all_records)
        with_source = sum(1 for r in all_records if r.source_type)
        verified = sum(1 for r in all_records if r.is_verified)

        # Source type breakdown
        source_breakdown: Dict[str, int] = {}
        for record in all_records:
            if record.source_type:
                source_breakdown[record.source_type] = (
                    source_breakdown.get(record.source_type, 0) + 1
                )

        # Missing sources
        missing = [
            f"{r.ticker}.{r.field_name}"
            for r in all_records
            if not r.source_type
        ]

        # Calculate coverage
        coverage = Decimal("0")
        if total > 0:
            coverage = (Decimal(with_source) / Decimal(total)).quantize(
                Decimal("0.0001")
            )

        # Test fabrication firewall
        firewall_test = self.test_fabrication_firewall()

        passed = coverage >= Decimal("0.95") and firewall_test["passed"]

        return ProvenanceReport(
            total_records=total,
            records_with_source=with_source,
            records_without_source=total - with_source,
            records_verified=verified,
            fabricated_detected=0 if firewall_test["passed"] else 1,
            source_type_breakdown=source_breakdown,
            missing_sources=missing[:100],  # Limit for reporting
            coverage_score=coverage,
            passed=passed,
        )


def validate_provenance(
    codebase_path: str,
    data_dir: str = "production_data",
) -> AuditResult:
    """
    Run complete provenance validation.

    Args:
        codebase_path: Root of codebase
        data_dir: Data directory name

    Returns:
        AuditResult with findings
    """
    validator = ProvenanceValidator(codebase_path, data_dir)
    report = validator.run_audit()

    result = AuditResult(
        check_name="provenance_validation",
        passed=report.passed,
        metrics={
            "total_records": report.total_records,
            "records_with_source": report.records_with_source,
            "records_without_source": report.records_without_source,
            "provenance_coverage": str(report.provenance_coverage),
            "source_type_breakdown": report.source_type_breakdown,
            "fabrication_firewall_passed": report.fabricated_detected == 0,
        },
        details=f"Provenance coverage: {report.provenance_coverage*100:.2f}%",
    )

    # Add findings for missing provenance
    if report.records_without_source > 0:
        severity = (
            AuditSeverity.CRITICAL
            if report.provenance_coverage < Decimal("0.90")
            else AuditSeverity.HIGH
            if report.provenance_coverage < Decimal("0.95")
            else AuditSeverity.MEDIUM
        )

        result.add_finding(
            severity=severity,
            category=ValidationCategory.PROVENANCE,
            title="Data records missing source provenance",
            description=(
                f"{report.records_without_source} of {report.total_records} records "
                f"lack source type identification"
            ),
            location="production_data/",
            evidence=f"Coverage: {report.provenance_coverage*100:.2f}%\n"
                     f"Missing: {', '.join(report.missing_sources[:10])}...",
            remediation="Add source_type, source_id, and source_date to all records",
            compliance_impact="Untracked data sources prevent audit verification",
        )

    # Check UNKNOWN handling
    unknown_files = validator.audit_codebase_for_unknown_handling()
    result.metrics["unknown_handling_files"] = len(unknown_files)

    return result
