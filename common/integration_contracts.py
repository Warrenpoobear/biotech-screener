"""
Integration Contracts for Biotech Screener Pipeline.

This module defines:
1. Type aliases and protocols for module boundaries
2. Schema validation functions for inter-module data
3. Standardized data structures for pipeline communication

Design Philosophy:
- Explicit type hints at all module boundaries
- Schema validation for fail-loud behavior
- Standardized field names across modules

Version: 1.0.0
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TypedDict,
    Union,
    runtime_checkable,
)


__version__ = "1.0.0"


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Standardized ticker collection type - prefer Set for uniqueness guarantees
TickerSet = Set[str]
TickerList = List[str]
TickerCollection = Union[Set[str], List[str]]

# Date types - modules should accept both
DateLike = Union[str, date]


# =============================================================================
# DATE NORMALIZATION
# =============================================================================

def normalize_date_input(date_input: DateLike) -> date:
    """
    Normalize date input to date object.

    Accepts:
    - date object (returned as-is)
    - ISO format string "YYYY-MM-DD"

    Raises:
        ValueError: If date format is invalid
    """
    if isinstance(date_input, date):
        return date_input
    if isinstance(date_input, str):
        return date.fromisoformat(date_input)
    raise ValueError(f"Invalid date type: {type(date_input)}. Expected date or str.")


def normalize_date_string(date_input: DateLike) -> str:
    """
    Normalize date input to ISO string.

    Accepts:
    - date object
    - ISO format string "YYYY-MM-DD"

    Returns:
        ISO format string "YYYY-MM-DD"
    """
    if isinstance(date_input, date):
        return date_input.isoformat()
    if isinstance(date_input, str):
        # Validate by parsing
        date.fromisoformat(date_input)
        return date_input
    raise ValueError(f"Invalid date type: {type(date_input)}. Expected date or str.")


def normalize_ticker_set(tickers: TickerCollection) -> Set[str]:
    """
    Normalize ticker collection to Set[str].

    Accepts:
    - Set[str] (returned as-is)
    - List[str] (converted to set)

    Returns:
        Set[str] of unique tickers
    """
    if isinstance(tickers, set):
        return tickers
    if isinstance(tickers, list):
        return set(tickers)
    raise ValueError(f"Invalid ticker collection type: {type(tickers)}. Expected Set or List.")


# =============================================================================
# STANDARDIZED SCORE FIELD NAMES
# =============================================================================

class ScoreFieldNames:
    """
    Standardized score field names across modules.

    Use these constants when accessing scores from module outputs.
    """
    # Module 2 - Financial
    FINANCIAL_SCORE = "financial_score"
    FINANCIAL_NORMALIZED = "financial_normalized"  # Legacy alias

    # Module 3 - Catalyst
    CATALYST_SCORE = "catalyst_score"
    CATALYST_SCORE_BLENDED = "score_blended"  # V2 field
    CATALYST_SCORE_NET = "catalyst_score_net"  # Legacy field

    # Module 4 - Clinical
    CLINICAL_SCORE = "clinical_score"

    # Module 5 - Composite
    COMPOSITE_SCORE = "composite_score"


# =============================================================================
# MODULE OUTPUT SCHEMAS (TypedDict for validation)
# =============================================================================

class Module1SecurityRecord(TypedDict, total=False):
    """Schema for Module 1 active security record."""
    ticker: str
    status: str
    market_cap_mm: Optional[float]
    sector: Optional[str]
    subsector: Optional[str]


class Module1Output(TypedDict):
    """Schema for Module 1 output."""
    active_securities: List[Module1SecurityRecord]
    excluded_securities: List[Dict[str, Any]]
    diagnostic_counts: Dict[str, int]


class Module2ScoreRecord(TypedDict, total=False):
    """Schema for Module 2 financial score record."""
    ticker: str
    financial_score: float  # Standardized name
    financial_normalized: float  # Legacy alias (same value)
    runway_months: Optional[float]
    cash_to_mcap: Optional[float]
    market_cap_mm: Optional[float]  # Added for Module 5 integration
    severity: str
    flags: List[str]


class Module2Output(TypedDict):
    """Schema for Module 2 output."""
    scores: List[Module2ScoreRecord]
    diagnostic_counts: Dict[str, int]


class Module3SummaryRecord(TypedDict, total=False):
    """Schema for Module 3 catalyst summary record."""
    ticker: str
    catalyst_score: float  # Standardized name
    score_blended: float  # V2 detail
    score_override: float  # V2 detail
    severe_negative_flag: bool
    next_catalyst_date: Optional[str]
    catalyst_window_bucket: str


class Module3Output(TypedDict):
    """Schema for Module 3 output."""
    summaries: Dict[str, Any]  # ticker -> TickerCatalystSummaryV2
    diagnostic_counts: Dict[str, int]
    as_of_date: str
    schema_version: str
    score_version: str
    # Deprecated
    summaries_legacy: Optional[Dict[str, Any]]
    diagnostic_counts_legacy: Optional[Dict[str, int]]


class Module4ScoreRecord(TypedDict, total=False):
    """Schema for Module 4 clinical score record."""
    ticker: str
    clinical_score: str  # Decimal as string
    lead_phase: str
    lead_trial_nct_id: Optional[str]
    n_trials_unique: int
    severity: str
    flags: List[str]


class Module4Output(TypedDict):
    """Schema for Module 4 output."""
    as_of_date: str
    scores: List[Module4ScoreRecord]
    diagnostic_counts: Dict[str, int]
    provenance: Dict[str, Any]


class Module5RankedRecord(TypedDict, total=False):
    """Schema for Module 5 ranked security record."""
    ticker: str
    composite_score: float
    rank: int
    clinical_normalized: Optional[float]
    financial_normalized: Optional[float]
    catalyst_normalized: Optional[float]
    severity: str
    flags: List[str]


class Module5Output(TypedDict):
    """Schema for Module 5 output."""
    ranked_securities: List[Module5RankedRecord]
    excluded_securities: List[Dict[str, Any]]
    diagnostic_counts: Dict[str, int]
    governance: Dict[str, Any]


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

class SchemaValidationError(Exception):
    """Raised when module output fails schema validation."""
    pass


def validate_module_1_output(output: Dict[str, Any]) -> None:
    """
    Validate Module 1 output schema.

    Raises:
        SchemaValidationError: If validation fails
    """
    required_keys = {"active_securities", "excluded_securities", "diagnostic_counts"}
    missing = required_keys - set(output.keys())
    if missing:
        raise SchemaValidationError(f"Module 1 output missing keys: {missing}")

    if not isinstance(output["active_securities"], list):
        raise SchemaValidationError("active_securities must be a list")

    for i, sec in enumerate(output["active_securities"]):
        if "ticker" not in sec:
            raise SchemaValidationError(f"active_securities[{i}] missing 'ticker' field")


def validate_module_2_output(output: Dict[str, Any]) -> None:
    """
    Validate Module 2 output schema.

    Raises:
        SchemaValidationError: If validation fails
    """
    required_keys = {"scores", "diagnostic_counts"}
    missing = required_keys - set(output.keys())
    if missing:
        raise SchemaValidationError(f"Module 2 output missing keys: {missing}")

    if not isinstance(output["scores"], list):
        raise SchemaValidationError("scores must be a list")

    for i, score in enumerate(output["scores"]):
        if "ticker" not in score:
            raise SchemaValidationError(f"scores[{i}] missing 'ticker' field")
        # Check for standardized or legacy field name
        if "financial_score" not in score and "financial_normalized" not in score:
            raise SchemaValidationError(
                f"scores[{i}] missing 'financial_score' or 'financial_normalized' field"
            )


def validate_module_3_output(output: Dict[str, Any]) -> None:
    """
    Validate Module 3 output schema.

    Raises:
        SchemaValidationError: If validation fails
    """
    required_keys = {"summaries", "diagnostic_counts", "as_of_date"}
    missing = required_keys - set(output.keys())
    if missing:
        raise SchemaValidationError(f"Module 3 output missing keys: {missing}")

    if not isinstance(output["summaries"], dict):
        raise SchemaValidationError("summaries must be a dict")

    # Warn about deprecated summaries_legacy
    if "summaries_legacy" in output and output["summaries_legacy"]:
        warnings.warn(
            "summaries_legacy is deprecated and will be removed in a future version. "
            "Use summaries instead.",
            DeprecationWarning,
            stacklevel=2
        )


def validate_module_4_output(output: Dict[str, Any]) -> None:
    """
    Validate Module 4 output schema.

    Raises:
        SchemaValidationError: If validation fails
    """
    required_keys = {"scores", "diagnostic_counts", "as_of_date"}
    missing = required_keys - set(output.keys())
    if missing:
        raise SchemaValidationError(f"Module 4 output missing keys: {missing}")

    if not isinstance(output["scores"], list):
        raise SchemaValidationError("scores must be a list")

    for i, score in enumerate(output["scores"]):
        if "ticker" not in score:
            raise SchemaValidationError(f"scores[{i}] missing 'ticker' field")
        if "clinical_score" not in score:
            raise SchemaValidationError(f"scores[{i}] missing 'clinical_score' field")


def validate_module_5_output(output: Dict[str, Any]) -> None:
    """
    Validate Module 5 output schema.

    Raises:
        SchemaValidationError: If validation fails
    """
    required_keys = {"ranked_securities", "excluded_securities", "diagnostic_counts"}
    missing = required_keys - set(output.keys())
    if missing:
        raise SchemaValidationError(f"Module 5 output missing keys: {missing}")

    if not isinstance(output["ranked_securities"], list):
        raise SchemaValidationError("ranked_securities must be a list")


def validate_pipeline_handoff(
    source_module: str,
    target_module: str,
    output: Dict[str, Any]
) -> None:
    """
    Validate output when passing between modules.

    Args:
        source_module: Name of source module (e.g., "module_1")
        target_module: Name of target module (e.g., "module_2")
        output: Output data from source module

    Raises:
        SchemaValidationError: If validation fails
    """
    validators = {
        "module_1": validate_module_1_output,
        "module_2": validate_module_2_output,
        "module_3": validate_module_3_output,
        "module_4": validate_module_4_output,
        "module_5": validate_module_5_output,
    }

    validator = validators.get(source_module)
    if validator:
        try:
            validator(output)
        except SchemaValidationError as e:
            raise SchemaValidationError(
                f"Handoff from {source_module} to {target_module} failed: {e}"
            ) from e


# =============================================================================
# SCORE EXTRACTION HELPERS
# =============================================================================

def extract_financial_score(score_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract financial score from Module 2 score record.

    Handles both standardized and legacy field names.
    """
    # Try standardized name first
    if "financial_score" in score_record:
        return float(score_record["financial_score"])
    # Fall back to legacy name
    if "financial_normalized" in score_record:
        return float(score_record["financial_normalized"])
    return None


def extract_catalyst_score(summary: Any) -> Optional[float]:
    """
    Extract catalyst score from Module 3 summary.

    Handles both dataclass objects and dict formats.
    """
    # Try dataclass attribute first (V2)
    if hasattr(summary, "score_blended"):
        return float(summary.score_blended)
    # Try dict access
    if isinstance(summary, dict):
        if "catalyst_score" in summary:
            return float(summary["catalyst_score"])
        if "score_blended" in summary:
            return float(summary["score_blended"])
        if "catalyst_score_net" in summary:
            return float(summary["catalyst_score_net"])
    # Try legacy dataclass
    if hasattr(summary, "catalyst_score_net"):
        return float(summary.catalyst_score_net)
    return None


def extract_clinical_score(score_record: Dict[str, Any]) -> Optional[float]:
    """
    Extract clinical score from Module 4 score record.
    """
    if "clinical_score" in score_record:
        val = score_record["clinical_score"]
        # May be Decimal string
        return float(val) if val is not None else None
    return None


def extract_market_cap_mm(record: Dict[str, Any]) -> Optional[float]:
    """
    Extract market_cap_mm from various record formats.

    Checks multiple field names for compatibility.
    """
    # Direct field
    if "market_cap_mm" in record and record["market_cap_mm"] is not None:
        return float(record["market_cap_mm"])
    # Market cap in dollars (convert to millions)
    if "market_cap" in record and record["market_cap"] is not None:
        return float(record["market_cap"]) / 1e6
    return None


# =============================================================================
# DEPRECATION HELPERS
# =============================================================================

def warn_legacy_field(field_name: str, replacement: str, module: str) -> None:
    """Emit deprecation warning for legacy field names."""
    warnings.warn(
        f"{module}: Field '{field_name}' is deprecated. Use '{replacement}' instead.",
        DeprecationWarning,
        stacklevel=3
    )


def warn_summaries_legacy() -> None:
    """Emit deprecation warning for summaries_legacy."""
    warnings.warn(
        "Module 3: 'summaries_legacy' is deprecated and will be removed in v2.0. "
        "Use 'summaries' with TickerCatalystSummaryV2 objects instead.",
        DeprecationWarning,
        stacklevel=2
    )
