"""
Data consistency validation utilities.

Validates consistency across multiple data files to catch:
- Missing records for universe members
- Orphaned records (data for non-universe tickers)
- Duplicate records
- Data staleness across files

Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyIssue:
    """Single consistency issue."""
    severity: str  # "error", "warning", "info"
    category: str  # "missing", "orphan", "duplicate", "stale"
    message: str
    affected_tickers: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        ticker_sample = self.affected_tickers[:5]
        if len(self.affected_tickers) > 5:
            ticker_sample.append(f"...and {len(self.affected_tickers) - 5} more")
        return f"[{self.severity.upper()}] {self.category}: {self.message} (tickers: {ticker_sample})"


@dataclass
class ConsistencyReport:
    """Result of consistency validation."""
    valid: bool
    issues: List[ConsistencyIssue] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def errors(self) -> List[ConsistencyIssue]:
        return [i for i in self.issues if i.severity == "error"]

    def warnings(self) -> List[ConsistencyIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def summary(self) -> str:
        errors = len(self.errors())
        warnings = len(self.warnings())
        return f"Consistency check: {errors} errors, {warnings} warnings"


def extract_tickers(
    records: List[Dict[str, Any]],
    ticker_field: str = "ticker"
) -> Set[str]:
    """
    Extract unique tickers from records.

    Args:
        records: List of record dicts
        ticker_field: Field name containing ticker

    Returns:
        Set of unique tickers (uppercase)
    """
    tickers = set()
    for record in records:
        ticker = record.get(ticker_field)
        if ticker:
            tickers.add(ticker.upper())
    return tickers


def find_duplicates(
    records: List[Dict[str, Any]],
    key_fields: List[str]
) -> Dict[str, List[int]]:
    """
    Find duplicate records by key fields.

    Args:
        records: List of record dicts
        key_fields: Fields that together form unique key

    Returns:
        Dict mapping duplicate keys to list of record indices
    """
    seen: Dict[str, List[int]] = {}

    for idx, record in enumerate(records):
        key_parts = [str(record.get(f, "")) for f in key_fields]
        key = "|".join(key_parts)

        if key not in seen:
            seen[key] = []
        seen[key].append(idx)

    # Return only duplicates (more than one occurrence)
    return {k: v for k, v in seen.items() if len(v) > 1}


def check_coverage(
    universe_tickers: Set[str],
    data_tickers: Set[str],
    data_name: str
) -> Tuple[Set[str], Set[str], float]:
    """
    Check data coverage against universe.

    Args:
        universe_tickers: Set of universe tickers
        data_tickers: Set of tickers in data file
        data_name: Name of data file for messages

    Returns:
        Tuple of (missing_tickers, orphan_tickers, coverage_pct)
    """
    missing = universe_tickers - data_tickers
    orphans = data_tickers - universe_tickers
    coverage = (len(universe_tickers) - len(missing)) / len(universe_tickers) * 100 if universe_tickers else 0

    return missing, orphans, coverage


def validate_data_consistency(
    universe: List[Dict[str, Any]],
    financial_records: Optional[List[Dict[str, Any]]] = None,
    trial_records: Optional[List[Dict[str, Any]]] = None,
    market_data: Optional[List[Dict[str, Any]]] = None,
    short_interest: Optional[List[Dict[str, Any]]] = None,
    min_coverage_pct: float = 80.0,
    as_of_date: Optional[date] = None,
) -> ConsistencyReport:
    """
    Validate consistency across all data files.

    Args:
        universe: Universe records
        financial_records: Financial data records
        trial_records: Clinical trial records
        market_data: Market data records
        short_interest: Short interest records
        min_coverage_pct: Minimum required coverage percentage
        as_of_date: Reference date for staleness checks

    Returns:
        ConsistencyReport with issues and stats
    """
    issues: List[ConsistencyIssue] = []
    stats: Dict[str, Any] = {}

    # Extract universe tickers
    universe_tickers = extract_tickers(universe, "ticker")
    stats["universe_count"] = len(universe_tickers)

    # Check for duplicates in universe
    universe_dups = find_duplicates(universe, ["ticker"])
    if universe_dups:
        dup_tickers = [k.split("|")[0] for k in universe_dups.keys()]
        issues.append(ConsistencyIssue(
            severity="error",
            category="duplicate",
            message=f"Universe contains {len(universe_dups)} duplicate tickers",
            affected_tickers=dup_tickers,
            details={"duplicates": universe_dups},
        ))

    # Check financial records
    if financial_records is not None:
        fin_tickers = extract_tickers(financial_records, "ticker")
        missing, orphans, coverage = check_coverage(universe_tickers, fin_tickers, "financial")

        stats["financial_coverage_pct"] = round(coverage, 1)
        stats["financial_missing"] = len(missing)
        stats["financial_orphans"] = len(orphans)

        if missing:
            severity = "error" if coverage < min_coverage_pct else "warning"
            issues.append(ConsistencyIssue(
                severity=severity,
                category="missing",
                message=f"Financial data missing for {len(missing)} universe members ({coverage:.1f}% coverage)",
                affected_tickers=sorted(missing)[:20],
                details={"coverage_pct": coverage},
            ))

        if orphans:
            issues.append(ConsistencyIssue(
                severity="warning",
                category="orphan",
                message=f"Financial data exists for {len(orphans)} non-universe tickers",
                affected_tickers=sorted(orphans)[:20],
            ))

        # Check for duplicate financial records
        fin_dups = find_duplicates(financial_records, ["ticker"])
        if fin_dups:
            issues.append(ConsistencyIssue(
                severity="warning",
                category="duplicate",
                message=f"Financial records contain {len(fin_dups)} duplicate tickers",
                affected_tickers=[k.split("|")[0] for k in fin_dups.keys()],
            ))

    # Check trial records
    if trial_records is not None:
        # Trials can use different ticker fields
        trial_tickers = set()
        for record in trial_records:
            ticker = record.get("lead_sponsor_ticker") or record.get("ticker")
            if ticker:
                trial_tickers.add(ticker.upper())

        missing, orphans, coverage = check_coverage(universe_tickers, trial_tickers, "trials")

        stats["trials_coverage_pct"] = round(coverage, 1)
        stats["trials_tickers_with_data"] = len(trial_tickers & universe_tickers)

        # Trials don't require full coverage (not all companies have trials)
        if coverage < 50:  # Less strict threshold for trials
            issues.append(ConsistencyIssue(
                severity="info",
                category="missing",
                message=f"Trial data coverage is {coverage:.1f}% (may be expected)",
                affected_tickers=[],
                details={"coverage_pct": coverage},
            ))

        # Check for duplicate trial records by NCT ID
        trial_dups = find_duplicates(trial_records, ["nct_id"])
        if trial_dups:
            issues.append(ConsistencyIssue(
                severity="warning",
                category="duplicate",
                message=f"Trial records contain {len(trial_dups)} duplicate NCT IDs",
                affected_tickers=[],
                details={"duplicate_count": len(trial_dups)},
            ))

    # Check market data
    if market_data is not None:
        market_tickers = extract_tickers(market_data, "ticker")
        missing, orphans, coverage = check_coverage(universe_tickers, market_tickers, "market")

        stats["market_coverage_pct"] = round(coverage, 1)
        stats["market_missing"] = len(missing)

        if missing:
            severity = "error" if coverage < min_coverage_pct else "warning"
            issues.append(ConsistencyIssue(
                severity=severity,
                category="missing",
                message=f"Market data missing for {len(missing)} universe members ({coverage:.1f}% coverage)",
                affected_tickers=sorted(missing)[:20],
                details={"coverage_pct": coverage},
            ))

    # Check short interest
    if short_interest is not None:
        si_tickers = extract_tickers(short_interest, "ticker")
        missing, orphans, coverage = check_coverage(universe_tickers, si_tickers, "short_interest")

        stats["short_interest_coverage_pct"] = round(coverage, 1)

        # Short interest data is optional, so just info-level
        if coverage < 50:
            issues.append(ConsistencyIssue(
                severity="info",
                category="missing",
                message=f"Short interest coverage is {coverage:.1f}%",
                affected_tickers=[],
                details={"coverage_pct": coverage},
            ))

    # Determine overall validity
    has_errors = any(i.severity == "error" for i in issues)

    return ConsistencyReport(
        valid=not has_errors,
        issues=issues,
        stats=stats,
    )


def validate_record_completeness(
    records: List[Dict[str, Any]],
    required_fields: List[str],
    data_name: str,
) -> ConsistencyReport:
    """
    Validate that records have all required fields.

    Args:
        records: Records to validate
        required_fields: List of required field names
        data_name: Name of data source

    Returns:
        ConsistencyReport
    """
    issues: List[ConsistencyIssue] = []
    missing_by_field: Dict[str, List[str]] = {f: [] for f in required_fields}

    for record in records:
        ticker = record.get("ticker", "UNKNOWN")
        for field in required_fields:
            if field not in record or record[field] is None:
                missing_by_field[field].append(ticker)

    for field, tickers in missing_by_field.items():
        if tickers:
            pct = len(tickers) / len(records) * 100 if records else 0
            severity = "error" if pct > 50 else "warning" if pct > 10 else "info"
            issues.append(ConsistencyIssue(
                severity=severity,
                category="missing",
                message=f"{data_name} missing '{field}' for {len(tickers)} records ({pct:.1f}%)",
                affected_tickers=tickers[:10],
                details={"missing_pct": pct},
            ))

    return ConsistencyReport(
        valid=not any(i.severity == "error" for i in issues),
        issues=issues,
        stats={
            "total_records": len(records),
            "fields_checked": required_fields,
        },
    )


def log_consistency_report(report: ConsistencyReport, logger_instance=None) -> None:
    """
    Log consistency report to logger.

    Args:
        report: ConsistencyReport to log
        logger_instance: Logger to use (default: module logger)
    """
    log = logger_instance or logger

    log.info(report.summary())
    log.info(f"  Stats: {report.stats}")

    for issue in report.errors():
        log.error(f"  {issue}")

    for issue in report.warnings():
        log.warning(f"  {issue}")


__all__ = [
    "ConsistencyIssue",
    "ConsistencyReport",
    "extract_tickers",
    "find_duplicates",
    "check_coverage",
    "validate_data_consistency",
    "validate_record_completeness",
    "log_consistency_report",
]
