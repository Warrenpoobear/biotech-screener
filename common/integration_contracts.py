"""
Integration Contracts for Biotech Screener Pipeline.

This module defines:
1. Type aliases and protocols for module boundaries
2. Schema validation functions for inter-module data
3. Standardized data structures for pipeline communication
4. Re-exports of key types from specialized schema modules

Design Philosophy:
- Explicit type hints at all module boundaries
- Schema validation for fail-loud behavior
- Standardized field names across modules
- Single source of truth via re-exports

Version: 1.1.0
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    runtime_checkable,
)

# Re-export Module 3 schema types for cross-module usage
from module_3_schema import (
    # Enums
    EventType,
    EventSeverity,
    ConfidenceLevel,
    CatalystWindowBucket,
    # Dataclasses
    CatalystEventV2,
    TickerCatalystSummaryV2,
    DiagnosticCounts as Module3DiagnosticCounts,
    # Version constants
    SCHEMA_VERSION as MODULE_3_SCHEMA_VERSION,
    SCORE_VERSION as MODULE_3_SCORE_VERSION,
    # Validation functions
    validate_summary_schema as validate_m3_summary_schema,
)


__version__ = "1.3.0"  # Added diagnostic normalization, improved extraction functions, strict validation default

# =============================================================================
# VALIDATION MODE
# =============================================================================

import os

# Validation modes: "strict" (raise), "warn" (log warning), "off" (skip)
# Default is "strict" for fail-loud behavior on schema violations
_DEFAULT_VALIDATION_MODE = "strict"


def get_validation_mode() -> str:
    """Get current validation mode from environment."""
    return os.getenv("IC_VALIDATION_MODE", _DEFAULT_VALIDATION_MODE)


def set_validation_mode(mode: str) -> None:
    """
    Set validation mode programmatically.

    Args:
        mode: One of "strict", "warn", "off"
    """
    if mode not in ("strict", "warn", "off"):
        raise ValueError(f"Invalid validation mode: {mode}. Must be 'strict', 'warn', or 'off'")
    os.environ["IC_VALIDATION_MODE"] = mode


def is_strict_mode() -> bool:
    """Check if strict validation mode is enabled."""
    return get_validation_mode() == "strict"


def is_validation_enabled() -> bool:
    """Check if validation is enabled (not 'off')."""
    return get_validation_mode() != "off"


# =============================================================================
# CONTRACT VERSIONING
# =============================================================================

# Supported schema versions for each module (for forward/backward compat)
SUPPORTED_SCHEMA_VERSIONS = {
    "module_1": {"1.0.0"},
    "module_2": {"1.0.0"},
    "module_3": {MODULE_3_SCHEMA_VERSION, "m3catalyst_vnext_20260111"},
    "module_4": {"1.0.0"},
    "module_5": {"1.0.0", "1.1.0"},
}


def check_schema_version(
    module_name: str,
    result: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Check if result schema version is supported.

    Args:
        module_name: Name of module (e.g., "module_3")
        result: Module output to check
        strict: If True, raise on unsupported version; else warn

    Returns:
        Tuple of (is_supported, version_found)
    """
    supported = SUPPORTED_SCHEMA_VERSIONS.get(module_name, set())
    if not supported:
        return True, None  # No version checking for this module

    # Look for schema version in common locations
    version = None
    if "schema_version" in result:
        version = result["schema_version"]
    elif "_schema" in result and "schema_version" in result["_schema"]:
        version = result["_schema"]["schema_version"]

    if version is None:
        # No version found - allow for backwards compatibility
        return True, None

    is_supported = version in supported
    if not is_supported:
        msg = f"{module_name} schema version '{version}' not in supported: {supported}"
        if strict:
            raise SchemaValidationError(msg)
        else:
            warnings.warn(msg, DeprecationWarning, stacklevel=3)

    return is_supported, version


# =============================================================================
# SCHEMA MIGRATION HELPERS
# =============================================================================

def migrate_module_output(
    result: Dict[str, Any],
    module_name: str,
    target_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Migrate module output to target schema version.

    Args:
        result: Module output to migrate
        module_name: Name of module (e.g., "module_2")
        target_version: Target version (defaults to latest supported)

    Returns:
        Migrated result dict
    """
    if module_name == "module_2":
        return _migrate_module_2_output(result, target_version)
    elif module_name == "module_3":
        return _migrate_module_3_output(result, target_version)
    # Other modules don't need migration yet
    return result


def _migrate_module_2_output(
    result: Dict[str, Any],
    target_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Migrate Module 2 output.

    Handles:
    - financial_normalized -> financial_score field rename
    """
    migrated = dict(result)

    # Ensure both field names exist for backwards compatibility
    if "scores" in migrated:
        for score in migrated["scores"]:
            if "financial_normalized" in score and "financial_score" not in score:
                score["financial_score"] = score["financial_normalized"]
            elif "financial_score" in score and "financial_normalized" not in score:
                score["financial_normalized"] = score["financial_score"]

    return migrated


def _migrate_module_3_output(
    result: Dict[str, Any],
    target_version: Optional[str] = None
) -> Dict[str, Any]:
    """
    Migrate Module 3 output.

    Handles:
    - Legacy TickerCatalystSummary -> TickerCatalystSummaryV2
    - summaries_legacy deprecation
    """
    migrated = dict(result)

    # If summaries_legacy exists but summaries doesn't, copy it
    if "summaries_legacy" in migrated and "summaries" not in migrated:
        migrated["summaries"] = migrated["summaries_legacy"]

    # Migrate individual summaries if needed
    if "summaries" in migrated:
        for ticker, summary in migrated["summaries"].items():
            if isinstance(summary, dict):
                # Check if it's legacy format (has catalyst_score_net but no score_blended)
                if "catalyst_score_net" in summary and "score_blended" not in summary:
                    # Migrate to vNext format
                    net_score = summary.get("catalyst_score_net", 0)
                    # Convert legacy net score to 0-100 scale
                    blended = 50 + (float(net_score) * 10)
                    blended = max(0, min(100, blended))

                    migrated["summaries"][ticker] = {
                        **summary,
                        "score_blended": blended,
                        "scores": {
                            "score_blended": str(blended),
                            "score_override": str(blended),
                        },
                        "flags": {
                            "severe_negative_flag": summary.get("severe_negative_flag", False),
                        },
                    }

    return migrated


def ensure_dual_field_names(result: Dict[str, Any], module_name: str) -> Dict[str, Any]:
    """
    Ensure both standard and legacy field names exist for backwards compatibility.

    This is useful when returning data that may be consumed by both old and new code.
    """
    return migrate_module_output(result, module_name)


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
# NESTED TYPE DEFINITIONS FOR MODULE OUTPUTS
# =============================================================================

class ExcludedSecurityRecord(TypedDict, total=False):
    """Schema for excluded security record."""
    ticker: str
    reason: str
    exclusion_type: str
    details: str


class DiagnosticCountsDict(TypedDict, total=False):
    """Schema for diagnostic counts in module outputs."""
    scored: int
    missing: int
    excluded: int
    total: int
    errors: int
    warnings: int


class ProvenanceDict(TypedDict, total=False):
    """Schema for provenance/audit trail metadata."""
    run_id: str
    schema_version: str
    score_version: str
    parameters_hash: str
    pit_cutoff: str
    input_hashes: Dict[str, str]
    timestamp: str
    module_version: str


class GovernanceDict(TypedDict, total=False):
    """Schema for governance metadata in Module 5 output."""
    run_id: str
    schema_version: str
    score_version: str
    parameters_hash: str
    pit_cutoff: str
    weights: Dict[str, float]
    config: Dict[str, Union[str, int, float, bool]]


# Module-specific result type alias
ModuleResultDict = Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any], None]]


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


class Module1Output(TypedDict, total=False):
    """Schema for Module 1 output."""
    as_of_date: str  # ISO date string
    active_securities: List[Module1SecurityRecord]
    excluded_securities: List[ExcludedSecurityRecord]
    diagnostic_counts: DiagnosticCountsDict
    provenance: ProvenanceDict


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
    diagnostic_counts: DiagnosticCountsDict


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
    summaries: Dict[str, Union[TickerCatalystSummaryV2, Module3SummaryRecord]]
    diagnostic_counts: DiagnosticCountsDict
    as_of_date: str
    schema_version: str
    score_version: str
    # Deprecated
    summaries_legacy: Optional[Dict[str, Module3SummaryRecord]]
    diagnostic_counts_legacy: Optional[DiagnosticCountsDict]


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
    diagnostic_counts: DiagnosticCountsDict
    provenance: ProvenanceDict


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
    excluded_securities: List[ExcludedSecurityRecord]
    diagnostic_counts: DiagnosticCountsDict
    governance: GovernanceDict


# Type variable for generic module output type
ModuleOutput = Union[Module1Output, Module2Output, Module3Output, Module4Output, Module5Output]

# Score record types
ScoreRecord = Union[Module2ScoreRecord, Module4ScoreRecord]
FinancialScoreRecord = Module2ScoreRecord
ClinicalScoreRecord = Module4ScoreRecord
CatalystSummary = Union[TickerCatalystSummaryV2, Module3SummaryRecord, Dict[str, Any]]


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


def validate_module_3_output(output: Dict[str, Any], deep: bool = False) -> None:
    """
    Validate Module 3 output schema.

    Args:
        output: Module 3 output dict
        deep: If True, validate individual summary schemas (slower)

    Raises:
        SchemaValidationError: If validation fails
    """
    required_keys = {"summaries", "diagnostic_counts", "as_of_date"}
    missing = required_keys - set(output.keys())
    if missing:
        raise SchemaValidationError(f"Module 3 output missing keys: {missing}")

    if not isinstance(output["summaries"], dict):
        raise SchemaValidationError("summaries must be a dict")

    # Check schema version compatibility
    check_schema_version("module_3", output)

    # Warn about deprecated summaries_legacy
    if "summaries_legacy" in output and output["summaries_legacy"]:
        warnings.warn(
            "summaries_legacy is deprecated and will be removed in a future version. "
            "Use summaries instead.",
            DeprecationWarning,
            stacklevel=2
        )

    # Deep validation using module_3_schema validator
    if deep:
        for ticker, summary in output["summaries"].items():
            # Handle both dataclass and dict
            if hasattr(summary, "to_dict"):
                summary_dict = summary.to_dict()
            elif isinstance(summary, dict):
                summary_dict = summary
            else:
                raise SchemaValidationError(
                    f"summaries[{ticker}] must be dict or TickerCatalystSummaryV2"
                )

            is_valid, errors = validate_m3_summary_schema(summary_dict)
            if not is_valid:
                raise SchemaValidationError(
                    f"summaries[{ticker}] schema validation failed: {errors}"
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

def extract_financial_score(
    score_record: Union[Module2ScoreRecord, Mapping[str, Any]],
    warn_legacy: bool = True
) -> Optional[float]:
    """
    Extract financial score from Module 2 score record.

    Handles both standardized and legacy field names.

    Args:
        score_record: Module 2 score record dict
        warn_legacy: If True, emit deprecation warning when using legacy field

    Returns:
        Financial score as float, or None if not found
    """
    # Try standardized name first
    if "financial_score" in score_record:
        return float(score_record["financial_score"])
    # Fall back to legacy name
    if "financial_normalized" in score_record:
        if warn_legacy:
            warn_legacy_field("financial_normalized", "financial_score", "Module 2")
        return float(score_record["financial_normalized"])
    return None


def extract_catalyst_score(
    summary: CatalystSummary,
    warn_legacy: bool = True
) -> Optional[float]:
    """
    Extract catalyst score from Module 3 summary.

    Handles both dataclass objects and dict formats, including:
    - Standardized "catalyst_score" field
    - V2 format with "score_blended" attribute or dict key
    - V2 nested format with scores["score_blended"]
    - Legacy "catalyst_score_net" field

    Args:
        summary: Module 3 summary (dataclass or dict)
        warn_legacy: If True, emit deprecation warning when using legacy fields

    Returns:
        Catalyst score as float (0-100 scale), or None if not found
    """
    if summary is None:
        return None

    try:
        # 1. Try standardized name first
        if isinstance(summary, dict) and "catalyst_score" in summary:
            val = summary["catalyst_score"]
            if val is not None:
                return float(val)

        # 2. Try dataclass attribute (V2 uses score_blended)
        if hasattr(summary, "score_blended") and summary.score_blended is not None:
            return float(summary.score_blended)

        # 3. Try V2 nested format: scores dict with score_blended
        if isinstance(summary, dict):
            scores = summary.get("scores", {})
            if isinstance(scores, dict) and "score_blended" in scores:
                val = scores["score_blended"]
                if val is not None:
                    return float(val)

            # 4. Try flat V2 format: direct score_blended key
            if "score_blended" in summary:
                val = summary["score_blended"]
                if val is not None:
                    return float(val)

        # 5. Fall back to legacy field names
        if isinstance(summary, dict) and "catalyst_score_net" in summary:
            if warn_legacy:
                warn_legacy_field("catalyst_score_net", "catalyst_score", "Module 3")
            val = summary["catalyst_score_net"]
            if val is not None:
                return float(val)

        # 6. Try legacy dataclass attribute
        if hasattr(summary, "catalyst_score_net") and summary.catalyst_score_net is not None:
            if warn_legacy:
                warn_legacy_field("catalyst_score_net", "catalyst_score", "Module 3")
            return float(summary.catalyst_score_net)

        return None

    except (ValueError, TypeError, InvalidOperation) as e:
        warnings.warn(
            f"Failed to extract catalyst_score from summary: {e}",
            RuntimeWarning,
            stacklevel=2
        )
        return None


def extract_clinical_score(
    score_record: Union[Module4ScoreRecord, Mapping[str, Any]]
) -> Optional[float]:
    """
    Extract clinical score from Module 4 score record.

    Handles:
    - Decimal string (from Module 4 output, e.g., "75.50")
    - float value
    - Decimal object
    - None values

    Returns:
        Clinical score as float, or None if not found or invalid
    """
    if "clinical_score" not in score_record:
        return None

    val = score_record["clinical_score"]
    if val is None:
        return None

    try:
        # Handle string (from Module 4 Decimal serialization), float, or Decimal
        if isinstance(val, str):
            return float(val)
        elif isinstance(val, (int, float)):
            return float(val)
        elif isinstance(val, Decimal):
            return float(val)
        else:
            # Attempt conversion for any other type
            return float(val)
    except (ValueError, TypeError, InvalidOperation) as e:
        warnings.warn(
            f"Failed to extract clinical_score: {val!r} ({type(val).__name__}): {e}",
            RuntimeWarning,
            stacklevel=2
        )
        return None


def extract_market_cap_mm(
    record: Union[Module1SecurityRecord, Module2ScoreRecord, Mapping[str, Any]]
) -> Optional[float]:
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
# DIAGNOSTIC COUNTS NORMALIZATION
# =============================================================================

# Standardized diagnostic count field names
class DiagnosticFields:
    """Standardized diagnostic count field names across modules."""
    # Common fields (all modules)
    SCORED = "scored"  # Number of items successfully scored
    TOTAL = "total"  # Total input items
    EXCLUDED = "excluded"  # Items excluded from scoring
    MISSING = "missing"  # Items with missing data

    # Module 3 specific
    EVENTS_TOTAL = "events_detected_total"
    TICKERS_WITH_EVENTS = "tickers_with_events"
    TICKERS_ANALYZED = "tickers_analyzed"
    TICKERS_SEVERE_NEG = "tickers_with_severe_negative"

    # Module 4 specific
    TRIALS_RAW = "total_trials_raw"
    TRIALS_UNIQUE = "total_trials_unique"
    PIT_FILTERED = "pit_filtered"


def normalize_diagnostic_counts(
    module_name: str,
    diagnostic_counts: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize diagnostic counts to standardized schema.

    Ensures all modules have consistent field names for common diagnostics.
    Module-specific fields are preserved as-is.

    Args:
        module_name: Name of module (e.g., "module_2")
        diagnostic_counts: Raw diagnostic counts from module

    Returns:
        Normalized diagnostic counts dict
    """
    normalized = dict(diagnostic_counts)

    # Ensure common fields exist with defaults
    if DiagnosticFields.SCORED not in normalized:
        # Try to infer from module-specific fields
        if module_name == "module_3":
            normalized[DiagnosticFields.SCORED] = normalized.get(
                DiagnosticFields.TICKERS_ANALYZED, 0
            )
        elif "total_scored" in normalized:
            normalized[DiagnosticFields.SCORED] = normalized["total_scored"]

    if DiagnosticFields.TOTAL not in normalized:
        # Try to infer
        if "total_input" in normalized:
            normalized[DiagnosticFields.TOTAL] = normalized["total_input"]
        elif DiagnosticFields.TICKERS_ANALYZED in normalized:
            normalized[DiagnosticFields.TOTAL] = normalized[DiagnosticFields.TICKERS_ANALYZED]

    return normalized


def validate_diagnostic_counts(
    module_name: str,
    diagnostic_counts: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate diagnostic counts contain expected fields.

    Args:
        module_name: Name of module (e.g., "module_2")
        diagnostic_counts: Diagnostic counts dict to validate
        strict: If True, require all expected fields

    Returns:
        Tuple of (is_valid, list of warnings/errors)
    """
    warnings_list = []

    # Module-specific expected fields
    expected_by_module = {
        "module_1": {DiagnosticFields.SCORED, DiagnosticFields.EXCLUDED},
        "module_2": {DiagnosticFields.SCORED, DiagnosticFields.MISSING},
        "module_3": {DiagnosticFields.TICKERS_ANALYZED, DiagnosticFields.EVENTS_TOTAL},
        "module_4": {DiagnosticFields.SCORED, DiagnosticFields.PIT_FILTERED},
        "module_5": {DiagnosticFields.TOTAL, "rankable", DiagnosticFields.EXCLUDED},
    }

    expected = expected_by_module.get(module_name, set())
    present = set(diagnostic_counts.keys())
    missing = expected - present

    if missing:
        msg = f"{module_name} diagnostic_counts missing expected fields: {missing}"
        warnings_list.append(msg)
        if strict:
            return False, warnings_list

    return True, warnings_list


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
