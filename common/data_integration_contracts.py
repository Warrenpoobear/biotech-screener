"""
common/data_integration_contracts.py - Data Integration Contract Layer

Provides explicit validation contracts for data integration across the screening pipeline.
Enforces:
- Schema validation for each dataset
- Join key invariants (uniqueness, case normalization, no duplicates)
- PIT (Point-in-Time) admissibility for each data source
- Coverage guardrails with configurable thresholds
- Deterministic behavior validation

Usage:
    from common.data_integration_contracts import (
        validate_market_data_schema,
        validate_financial_records_schema,
        validate_trial_records_schema,
        validate_join_invariants,
        validate_pit_admissibility,
        validate_coverage_guardrails,
        DataIntegrationError,
    )

    # Schema validation
    validate_market_data_schema(market_records, strict=True)

    # Join invariants
    validate_join_invariants(
        universe_tickers=active_tickers,
        financial_tickers=set(f['ticker'] for f in financial_data),
        as_of_date=as_of_date,
    )

    # PIT validation
    validate_pit_admissibility(records, as_of_date, date_field="source_date")

    # Coverage guardrails
    validate_coverage_guardrails(
        universe_size=100,
        financial_coverage=85,
        clinical_coverage=92,
    )

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from common.date_utils import normalize_date, to_date_object
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible

__all__ = [
    # Exceptions
    "DataIntegrationError",
    "SchemaValidationError",
    "JoinInvariantError",
    "PITViolationError",
    "CoverageGuardrailError",
    # Schema validators
    "validate_market_data_schema",
    "validate_financial_records_schema",
    "validate_trial_records_schema",
    "validate_holdings_schema",
    "validate_short_interest_schema",
    # Join validation
    "validate_join_invariants",
    "normalize_ticker_set",
    "check_ticker_uniqueness",
    "check_ticker_case_consistency",
    # PIT validation
    "validate_pit_admissibility",
    "validate_dataset_pit",
    "PITValidationResult",
    # Coverage guardrails
    "validate_coverage_guardrails",
    "CoverageConfig",
    "CoverageReport",
    # Determinism
    "compute_deterministic_hash",
    "validate_output_determinism",
    # Numeric safety
    "safe_numeric_check",
    "validate_numeric_field",
]

logger = logging.getLogger(__name__)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class DataIntegrationError(Exception):
    """Base exception for data integration errors."""
    pass


class SchemaValidationError(DataIntegrationError):
    """Raised when data doesn't match expected schema."""
    def __init__(self, message: str, field: str = None, record: Any = None):
        super().__init__(message)
        self.field = field
        self.record = record


class JoinInvariantError(DataIntegrationError):
    """Raised when join key invariants are violated."""
    def __init__(
        self,
        message: str,
        missing_tickers: Set[str] = None,
        duplicate_tickers: Set[str] = None,
        case_mismatches: Set[str] = None,
    ):
        super().__init__(message)
        self.missing_tickers = missing_tickers or set()
        self.duplicate_tickers = duplicate_tickers or set()
        self.case_mismatches = case_mismatches or set()


class PITViolationError(DataIntegrationError):
    """Raised when point-in-time safety is violated."""
    def __init__(self, message: str, future_records: List[Dict] = None):
        super().__init__(message)
        self.future_records = future_records or []


class CoverageGuardrailError(DataIntegrationError):
    """Raised when coverage falls below configured thresholds."""
    def __init__(
        self,
        message: str,
        component: str,
        actual_pct: float,
        threshold_pct: float,
    ):
        super().__init__(message)
        self.component = component
        self.actual_pct = actual_pct
        self.threshold_pct = threshold_pct


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

# Required fields for market_data.json records
MARKET_DATA_REQUIRED_FIELDS = frozenset({"ticker"})
MARKET_DATA_OPTIONAL_FIELDS = frozenset({
    "price", "market_cap", "volume", "average_30d",
    "return_60d", "xbi_return_60d", "volatility_252d",
    "source_date", "as_of_date",
})

# Required fields for financial_records.json
FINANCIAL_REQUIRED_FIELDS = frozenset({"ticker"})
FINANCIAL_OPTIONAL_FIELDS = frozenset({
    "Cash", "NetIncome", "TotalAssets", "TotalLiabilities",
    "source_date", "filing_date", "fiscal_period",
    "quarterly_burn", "runway_months", "market_cap_mm",
})

# Required fields for trial_records.json
TRIAL_REQUIRED_FIELDS = frozenset({"ticker", "nct_id"})
TRIAL_OPTIONAL_FIELDS = frozenset({
    "phase", "status", "conditions", "start_date", "completion_date",
    "first_posted", "last_update_posted", "source_date",
    "enrollment", "primary_endpoint", "randomized", "blinded",
})

# Required fields for holdings_snapshots.json
HOLDINGS_REQUIRED_FIELDS = frozenset({"ticker"})  # Top-level key

# Required fields for short_interest.json
SHORT_INTEREST_REQUIRED_FIELDS = frozenset({"ticker"})
SHORT_INTEREST_OPTIONAL_FIELDS = frozenset({
    "short_interest", "days_to_cover", "short_pct_float",
    "source_date", "settlement_date",
})


# =============================================================================
# SCHEMA VALIDATORS
# =============================================================================

def validate_market_data_schema(
    records: List[Dict[str, Any]],
    strict: bool = False,
    raise_on_error: bool = True,
) -> Tuple[bool, List[Dict]]:
    """
    Validate market_data.json records against expected schema.

    Args:
        records: List of market data records
        strict: If True, fail on any missing optional field
        raise_on_error: If True, raise SchemaValidationError on failure

    Returns:
        (is_valid, list of invalid records with error details)

    Raises:
        SchemaValidationError: If raise_on_error and validation fails
    """
    invalid_records = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_records.append({
                "index": i,
                "record": record,
                "error": "Record is not a dict",
            })
            continue

        # Check required fields
        missing_required = MARKET_DATA_REQUIRED_FIELDS - set(record.keys())
        if missing_required:
            invalid_records.append({
                "index": i,
                "ticker": record.get("ticker"),
                "error": f"Missing required fields: {missing_required}",
            })
            continue

        # Validate ticker is non-empty string
        ticker = record.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            invalid_records.append({
                "index": i,
                "ticker": ticker,
                "error": "Ticker must be non-empty string",
            })
            continue

        # Validate numeric fields are actually numeric (or None)
        numeric_fields = ["price", "market_cap", "volume", "return_60d", "volatility_252d"]
        for field_name in numeric_fields:
            value = record.get(field_name)
            if value is not None:
                if not isinstance(value, (int, float, Decimal)):
                    try:
                        # Allow string representations of numbers
                        float(str(value))
                    except (ValueError, TypeError):
                        invalid_records.append({
                            "index": i,
                            "ticker": ticker,
                            "error": f"Field '{field_name}' is not numeric: {value}",
                        })

    is_valid = len(invalid_records) == 0

    if not is_valid and raise_on_error:
        raise SchemaValidationError(
            f"Market data schema validation failed: {len(invalid_records)} invalid records",
            record=invalid_records[:5],  # First 5 errors
        )

    return is_valid, invalid_records


def validate_financial_records_schema(
    records: List[Dict[str, Any]],
    strict: bool = False,
    raise_on_error: bool = True,
) -> Tuple[bool, List[Dict]]:
    """
    Validate financial_records.json against expected schema.

    Args:
        records: List of financial records
        strict: If True, require optional fields
        raise_on_error: If True, raise on failure

    Returns:
        (is_valid, list of invalid records)
    """
    invalid_records = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_records.append({
                "index": i,
                "error": "Record is not a dict",
            })
            continue

        # Check required fields
        missing_required = FINANCIAL_REQUIRED_FIELDS - set(record.keys())
        if missing_required:
            invalid_records.append({
                "index": i,
                "ticker": record.get("ticker"),
                "error": f"Missing required fields: {missing_required}",
            })
            continue

        # Validate ticker
        ticker = record.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            invalid_records.append({
                "index": i,
                "error": "Ticker must be non-empty string",
            })

    is_valid = len(invalid_records) == 0

    if not is_valid and raise_on_error:
        raise SchemaValidationError(
            f"Financial records schema validation failed: {len(invalid_records)} invalid records",
            record=invalid_records[:5],
        )

    return is_valid, invalid_records


def validate_trial_records_schema(
    records: List[Dict[str, Any]],
    strict: bool = False,
    raise_on_error: bool = True,
) -> Tuple[bool, List[Dict]]:
    """
    Validate trial_records.json against expected schema.

    Args:
        records: List of trial records
        strict: If True, require nct_id format validation
        raise_on_error: If True, raise on failure

    Returns:
        (is_valid, list of invalid records)
    """
    invalid_records = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_records.append({
                "index": i,
                "error": "Record is not a dict",
            })
            continue

        # Check required fields
        missing_required = TRIAL_REQUIRED_FIELDS - set(record.keys())
        if missing_required:
            invalid_records.append({
                "index": i,
                "ticker": record.get("ticker"),
                "nct_id": record.get("nct_id"),
                "error": f"Missing required fields: {missing_required}",
            })
            continue

        # Validate ticker
        ticker = record.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            invalid_records.append({
                "index": i,
                "error": "Ticker must be non-empty string",
            })
            continue

        # Validate nct_id format (NCT + 8 digits)
        nct_id = record.get("nct_id")
        if strict:
            if not isinstance(nct_id, str):
                invalid_records.append({
                    "index": i,
                    "ticker": ticker,
                    "error": f"nct_id must be string, got {type(nct_id)}",
                })
            elif not nct_id.startswith("NCT") or len(nct_id) != 11:
                invalid_records.append({
                    "index": i,
                    "ticker": ticker,
                    "nct_id": nct_id,
                    "error": "nct_id must match format NCT + 8 digits",
                })

    is_valid = len(invalid_records) == 0

    if not is_valid and raise_on_error:
        raise SchemaValidationError(
            f"Trial records schema validation failed: {len(invalid_records)} invalid records",
            record=invalid_records[:5],
        )

    return is_valid, invalid_records


def validate_holdings_schema(
    data: Dict[str, Any],
    raise_on_error: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Validate holdings_snapshots.json format.

    Expected format:
    {
        "TICKER": {
            "holdings": {
                "current": {
                    "MANAGER_CIK": {"value_kusd": 123456},
                }
            }
        }
    }

    Returns:
        (is_valid, list of error messages)
    """
    errors = []

    if not isinstance(data, dict):
        errors.append(f"Holdings data must be dict, got {type(data)}")
        if raise_on_error:
            raise SchemaValidationError("Holdings data must be dict")
        return False, errors

    for ticker, ticker_data in data.items():
        if not isinstance(ticker, str) or not ticker.strip():
            errors.append(f"Invalid ticker key: {ticker}")
            continue

        if not isinstance(ticker_data, dict):
            errors.append(f"Ticker {ticker} data must be dict")
            continue

        holdings = ticker_data.get("holdings")
        if holdings is not None and not isinstance(holdings, dict):
            errors.append(f"Ticker {ticker} holdings must be dict")

    is_valid = len(errors) == 0

    if not is_valid and raise_on_error:
        raise SchemaValidationError(
            f"Holdings schema validation failed: {errors[:5]}"
        )

    return is_valid, errors


def validate_short_interest_schema(
    records: List[Dict[str, Any]],
    raise_on_error: bool = True,
) -> Tuple[bool, List[Dict]]:
    """
    Validate short_interest.json against expected schema.

    Returns:
        (is_valid, list of invalid records)
    """
    invalid_records = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            invalid_records.append({
                "index": i,
                "error": "Record is not a dict",
            })
            continue

        # Check required fields
        missing_required = SHORT_INTEREST_REQUIRED_FIELDS - set(record.keys())
        if missing_required:
            invalid_records.append({
                "index": i,
                "ticker": record.get("ticker"),
                "error": f"Missing required fields: {missing_required}",
            })

    is_valid = len(invalid_records) == 0

    if not is_valid and raise_on_error:
        raise SchemaValidationError(
            f"Short interest schema validation failed: {len(invalid_records)} invalid records"
        )

    return is_valid, invalid_records


# =============================================================================
# JOIN VALIDATION
# =============================================================================

def normalize_ticker_set(
    tickers: Union[Set[str], List[str]],
    to_upper: bool = True,
) -> Set[str]:
    """
    Normalize a collection of tickers to consistent case.

    Args:
        tickers: Set or list of ticker symbols
        to_upper: If True, normalize to uppercase (recommended)

    Returns:
        Set of normalized tickers
    """
    if to_upper:
        return {t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()}
    else:
        return {t.strip() for t in tickers if isinstance(t, str) and t.strip()}


def check_ticker_uniqueness(
    records: List[Dict[str, Any]],
    ticker_field: str = "ticker",
) -> Tuple[bool, Set[str]]:
    """
    Check that ticker values are unique in the dataset.

    Args:
        records: List of records to check
        ticker_field: Name of the ticker field

    Returns:
        (is_unique, set of duplicate tickers)
    """
    seen = set()
    duplicates = set()

    for record in records:
        ticker = record.get(ticker_field)
        if ticker:
            normalized = ticker.upper()
            if normalized in seen:
                duplicates.add(normalized)
            seen.add(normalized)

    return len(duplicates) == 0, duplicates


def check_ticker_case_consistency(
    records: List[Dict[str, Any]],
    ticker_field: str = "ticker",
) -> Tuple[bool, Dict[str, Set[str]]]:
    """
    Check for case inconsistencies in ticker values.

    Returns:
        (is_consistent, dict mapping canonical ticker -> set of variants)
    """
    ticker_variants: Dict[str, Set[str]] = {}

    for record in records:
        ticker = record.get(ticker_field)
        if isinstance(ticker, str) and ticker.strip():
            canonical = ticker.upper()
            if canonical not in ticker_variants:
                ticker_variants[canonical] = set()
            ticker_variants[canonical].add(ticker)

    # Find tickers with multiple case variants
    inconsistent = {
        canonical: variants
        for canonical, variants in ticker_variants.items()
        if len(variants) > 1
    }

    return len(inconsistent) == 0, inconsistent


def validate_join_invariants(
    universe_tickers: Set[str],
    financial_tickers: Optional[Set[str]] = None,
    clinical_tickers: Optional[Set[str]] = None,
    market_tickers: Optional[Set[str]] = None,
    min_financial_coverage_pct: float = 80.0,
    min_clinical_coverage_pct: float = 50.0,
    min_market_coverage_pct: float = 50.0,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """
    Validate join key invariants across datasets.

    Checks:
    1. All ticker sets have consistent casing (normalized to uppercase)
    2. No duplicate tickers within each dataset
    3. Coverage meets minimum thresholds
    4. No orphan tickers (tickers in downstream data not in universe)

    Args:
        universe_tickers: Active tickers from Module 1
        financial_tickers: Tickers in financial data
        clinical_tickers: Tickers in clinical/trial data
        market_tickers: Tickers in market data
        min_*_coverage_pct: Minimum coverage percentages
        raise_on_error: If True, raise JoinInvariantError on failure

    Returns:
        Dict with validation results and diagnostics
    """
    # Normalize all ticker sets
    universe_norm = normalize_ticker_set(universe_tickers)
    financial_norm = normalize_ticker_set(financial_tickers or set())
    clinical_norm = normalize_ticker_set(clinical_tickers or set())
    market_norm = normalize_ticker_set(market_tickers or set())

    n_universe = len(universe_norm)
    if n_universe == 0:
        return {
            "is_valid": True,
            "warning": "Empty universe - no join validation performed",
            "coverage": {},
        }

    # Calculate coverage
    financial_coverage = len(financial_norm & universe_norm) / n_universe * 100 if financial_tickers else None
    clinical_coverage = len(clinical_norm & universe_norm) / n_universe * 100 if clinical_tickers else None
    market_coverage = len(market_norm & universe_norm) / n_universe * 100 if market_tickers else None

    # Find missing tickers
    missing_financial = universe_norm - financial_norm if financial_tickers else set()
    missing_clinical = universe_norm - clinical_norm if clinical_tickers else set()
    missing_market = universe_norm - market_norm if market_tickers else set()

    # Find orphan tickers (in data but not in universe)
    orphan_financial = financial_norm - universe_norm if financial_tickers else set()
    orphan_clinical = clinical_norm - universe_norm if clinical_tickers else set()
    orphan_market = market_norm - universe_norm if market_tickers else set()

    # Check thresholds
    coverage_failures = []
    if financial_coverage is not None and financial_coverage < min_financial_coverage_pct:
        coverage_failures.append(f"Financial coverage {financial_coverage:.1f}% < {min_financial_coverage_pct}%")
    if clinical_coverage is not None and clinical_coverage < min_clinical_coverage_pct:
        coverage_failures.append(f"Clinical coverage {clinical_coverage:.1f}% < {min_clinical_coverage_pct}%")
    if market_coverage is not None and market_coverage < min_market_coverage_pct:
        coverage_failures.append(f"Market coverage {market_coverage:.1f}% < {min_market_coverage_pct}%")

    is_valid = len(coverage_failures) == 0

    result = {
        "is_valid": is_valid,
        "universe_size": n_universe,
        "coverage": {
            "financial_pct": financial_coverage,
            "clinical_pct": clinical_coverage,
            "market_pct": market_coverage,
        },
        "missing": {
            "financial": list(sorted(missing_financial)[:10]),  # Sample
            "clinical": list(sorted(missing_clinical)[:10]),
            "market": list(sorted(missing_market)[:10]),
        },
        "orphans": {
            "financial": list(sorted(orphan_financial)[:10]),
            "clinical": list(sorted(orphan_clinical)[:10]),
            "market": list(sorted(orphan_market)[:10]),
        },
        "missing_counts": {
            "financial": len(missing_financial),
            "clinical": len(missing_clinical),
            "market": len(missing_market),
        },
        "orphan_counts": {
            "financial": len(orphan_financial),
            "clinical": len(orphan_clinical),
            "market": len(orphan_market),
        },
        "coverage_failures": coverage_failures,
    }

    if not is_valid and raise_on_error:
        raise JoinInvariantError(
            f"Join invariant validation failed: {coverage_failures}",
            missing_tickers=missing_financial | missing_clinical | missing_market,
        )

    return result


# =============================================================================
# PIT VALIDATION
# =============================================================================

@dataclass
class PITValidationResult:
    """Result of PIT validation for a dataset."""
    is_valid: bool
    total_records: int
    pit_compliant: int
    pit_violated: int
    future_records: List[Dict] = field(default_factory=list)
    missing_date_records: int = 0
    date_field_used: str = ""
    pit_cutoff: str = ""


def validate_pit_admissibility(
    records: List[Dict[str, Any]],
    as_of_date: Union[str, date],
    date_field: str = "source_date",
    fallback_fields: Optional[List[str]] = None,
    raise_on_error: bool = False,
    max_future_records: int = 0,
) -> PITValidationResult:
    """
    Validate that all records are PIT-admissible (no future data).

    PIT Rule: source_date <= as_of_date - 1 day (cutoff)

    Args:
        records: List of records to validate
        as_of_date: Analysis date
        date_field: Primary date field to check
        fallback_fields: Alternative date fields to try if primary is missing
        raise_on_error: If True, raise on any future records
        max_future_records: Maximum allowed future records (0 = none allowed)

    Returns:
        PITValidationResult with validation details
    """
    pit_cutoff = compute_pit_cutoff(as_of_date)
    fallback_fields = fallback_fields or ["last_update_posted", "first_posted"]

    pit_compliant = 0
    pit_violated = 0
    missing_date = 0
    future_records = []

    for record in records:
        # Try primary date field first
        date_value = record.get(date_field)
        field_used = date_field

        # Try fallback fields if primary is missing
        if date_value is None:
            for fallback in fallback_fields:
                date_value = record.get(fallback)
                if date_value is not None:
                    field_used = fallback
                    break

        if date_value is None:
            missing_date += 1
            continue

        # Check PIT admissibility
        if is_pit_admissible(date_value, pit_cutoff):
            pit_compliant += 1
        else:
            pit_violated += 1
            future_records.append({
                "ticker": record.get("ticker"),
                "nct_id": record.get("nct_id"),
                "date_field": field_used,
                "date_value": str(date_value),
                "pit_cutoff": str(pit_cutoff),
            })

    is_valid = pit_violated <= max_future_records

    result = PITValidationResult(
        is_valid=is_valid,
        total_records=len(records),
        pit_compliant=pit_compliant,
        pit_violated=pit_violated,
        future_records=future_records[:10],  # Sample
        missing_date_records=missing_date,
        date_field_used=date_field,
        pit_cutoff=str(pit_cutoff),
    )

    if not is_valid and raise_on_error:
        raise PITViolationError(
            f"PIT validation failed: {pit_violated} future records detected (max allowed: {max_future_records})",
            future_records=future_records[:10],
        )

    return result


def validate_dataset_pit(
    dataset_name: str,
    records: List[Dict[str, Any]],
    as_of_date: Union[str, date],
    date_field_priority: List[str] = None,
) -> PITValidationResult:
    """
    Validate PIT for a specific dataset type with appropriate date field priority.

    Args:
        dataset_name: "market", "financial", "trial", "holdings", "short_interest"
        records: Records to validate
        as_of_date: Analysis date

    Returns:
        PITValidationResult
    """
    # Dataset-specific date field priorities
    DATE_FIELD_PRIORITY = {
        "market": ["as_of_date", "source_date", "date"],
        "financial": ["source_date", "filing_date", "report_date"],
        "trial": ["first_posted", "last_update_posted", "source_date"],
        "holdings": ["as_of_date", "report_date", "source_date"],
        "short_interest": ["settlement_date", "source_date", "as_of_date"],
    }

    fields = date_field_priority or DATE_FIELD_PRIORITY.get(dataset_name, ["source_date"])
    primary_field = fields[0] if fields else "source_date"
    fallback_fields = fields[1:] if len(fields) > 1 else None

    return validate_pit_admissibility(
        records=records,
        as_of_date=as_of_date,
        date_field=primary_field,
        fallback_fields=fallback_fields,
    )


# =============================================================================
# COVERAGE GUARDRAILS
# =============================================================================

@dataclass(frozen=True)
class CoverageConfig:
    """Configuration for coverage guardrails."""
    # Minimum coverage percentages (of universe)
    min_financial_pct: float = 80.0
    min_clinical_pct: float = 50.0
    min_market_pct: float = 50.0
    min_catalyst_pct: float = 10.0
    min_pos_pct: float = 0.0  # Optional component
    min_momentum_pct: float = 0.0  # Optional component

    # Behavior
    strict_mode: bool = False  # Raise exception vs log warning
    warn_threshold_pct: float = 10.0  # Warn if within this % of threshold


@dataclass
class CoverageReport:
    """Report of coverage validation."""
    is_valid: bool
    universe_size: int
    component_coverage: Dict[str, float]  # component -> pct
    failures: List[str]
    warnings: List[str]


def validate_coverage_guardrails(
    universe_size: int,
    financial_count: int = 0,
    clinical_count: int = 0,
    market_count: int = 0,
    catalyst_count: int = 0,
    pos_count: int = 0,
    momentum_count: int = 0,
    config: Optional[CoverageConfig] = None,
    raise_on_error: bool = False,
) -> CoverageReport:
    """
    Validate that component coverage meets minimum thresholds.

    Args:
        universe_size: Total active tickers
        *_count: Number of tickers with data for each component
        config: Coverage configuration
        raise_on_error: If True, raise CoverageGuardrailError on failure

    Returns:
        CoverageReport with validation details
    """
    config = config or CoverageConfig()

    if universe_size == 0:
        return CoverageReport(
            is_valid=True,
            universe_size=0,
            component_coverage={},
            failures=[],
            warnings=["Empty universe - coverage validation skipped"],
        )

    # Calculate coverage percentages
    coverage = {
        "financial": (financial_count / universe_size) * 100 if financial_count else 0.0,
        "clinical": (clinical_count / universe_size) * 100 if clinical_count else 0.0,
        "market": (market_count / universe_size) * 100 if market_count else 0.0,
        "catalyst": (catalyst_count / universe_size) * 100 if catalyst_count else 0.0,
        "pos": (pos_count / universe_size) * 100 if pos_count else 0.0,
        "momentum": (momentum_count / universe_size) * 100 if momentum_count else 0.0,
    }

    # Check against thresholds
    thresholds = {
        "financial": config.min_financial_pct,
        "clinical": config.min_clinical_pct,
        "market": config.min_market_pct,
        "catalyst": config.min_catalyst_pct,
        "pos": config.min_pos_pct,
        "momentum": config.min_momentum_pct,
    }

    failures = []
    warnings = []

    for component, pct in coverage.items():
        threshold = thresholds[component]
        if pct < threshold:
            failures.append(f"{component}: {pct:.1f}% < {threshold}% threshold")
        elif pct < threshold + config.warn_threshold_pct:
            warnings.append(f"{component}: {pct:.1f}% approaching threshold ({threshold}%)")

    is_valid = len(failures) == 0

    report = CoverageReport(
        is_valid=is_valid,
        universe_size=universe_size,
        component_coverage=coverage,
        failures=failures,
        warnings=warnings,
    )

    if not is_valid and raise_on_error:
        raise CoverageGuardrailError(
            f"Coverage guardrails failed: {failures}",
            component=failures[0].split(":")[0] if failures else "unknown",
            actual_pct=coverage.get(failures[0].split(":")[0], 0) if failures else 0,
            threshold_pct=thresholds.get(failures[0].split(":")[0], 0) if failures else 0,
        )

    return report


# =============================================================================
# NUMERIC SAFETY
# =============================================================================

def safe_numeric_check(value: Any) -> bool:
    """
    Safely check if a value is numeric (including zero and negative).

    CRITICAL: Handles the common bug of `if value` being falsy for 0.

    Args:
        value: Value to check

    Returns:
        True if value is a valid number (including 0, negative numbers)
    """
    if value is None:
        return False

    if isinstance(value, bool):
        return False  # Booleans are not numeric for our purposes

    if isinstance(value, (int, float, Decimal)):
        return True

    # Try parsing string representation
    if isinstance(value, str):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    return False


def validate_numeric_field(
    value: Any,
    field_name: str,
    allow_zero: bool = True,
    allow_negative: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a numeric field with detailed error messages.

    Args:
        value: Value to validate
        field_name: Name of the field (for error messages)
        allow_zero: If True, zero is valid
        allow_negative: If True, negative values are valid
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        (is_valid, error_message or None)
    """
    if value is None:
        return False, f"{field_name} is None"

    if not safe_numeric_check(value):
        return False, f"{field_name} is not numeric: {value}"

    try:
        num_value = float(str(value))
    except (ValueError, TypeError) as e:
        return False, f"{field_name} cannot be converted to number: {e}"

    if not allow_zero and num_value == 0:
        return False, f"{field_name} cannot be zero"

    if not allow_negative and num_value < 0:
        return False, f"{field_name} cannot be negative: {num_value}"

    if min_value is not None and num_value < min_value:
        return False, f"{field_name} {num_value} is below minimum {min_value}"

    if max_value is not None and num_value > max_value:
        return False, f"{field_name} {num_value} is above maximum {max_value}"

    return True, None


# =============================================================================
# DETERMINISM VALIDATION
# =============================================================================

def compute_deterministic_hash(
    data: Any,
    include_types: bool = False,
) -> str:
    """
    Compute a deterministic hash for any data structure.

    Ensures:
    - Dict keys are sorted
    - Decimal values are normalized
    - Dates are ISO formatted
    - Sets are sorted lists

    Args:
        data: Data to hash
        include_types: If True, include type information in hash

    Returns:
        SHA256 hash string
    """
    def normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [normalize(item) for item in obj]
        elif isinstance(obj, set):
            return sorted([normalize(item) for item in obj])
        elif isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, date):
            return obj.isoformat()
        elif include_types:
            return {"__type__": type(obj).__name__, "__value__": str(obj)}
        else:
            return obj

    normalized = normalize(data)
    json_str = json.dumps(normalized, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"


def validate_output_determinism(
    output1: Dict[str, Any],
    output2: Dict[str, Any],
    exclude_fields: Optional[Set[str]] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that two outputs are deterministically identical.

    Args:
        output1: First output
        output2: Second output
        exclude_fields: Fields to exclude from comparison (e.g., timestamps)

    Returns:
        (is_deterministic, difference description or None)
    """
    exclude = exclude_fields or {"timestamp", "run_id", "provenance"}

    def filter_dict(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if k not in exclude}

    hash1 = compute_deterministic_hash(filter_dict(output1))
    hash2 = compute_deterministic_hash(filter_dict(output2))

    if hash1 == hash2:
        return True, None
    else:
        return False, f"Hash mismatch: {hash1[:16]}... != {hash2[:16]}..."
