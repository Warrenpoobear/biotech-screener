#!/usr/bin/env python3
"""
alpha_signal_contract.py - Alpha Signal Contract + Schema Enforcement

Defines the required/optional fields the decision engine must have to make
clinical+catalyst-aware decisions.  Validates records at the DE boundary to
prevent silent schema drift.

Contract version is pinned: any change to required fields bumps the version
so downstream consumers (drift reports, backtests) can detect incompatibility.

Conditional requiredness (v1.1.0):
    clinical_score_z is required for clinical-cohort archetypes
    (drug_developer, commercial_biotech, commercial_pharma) and
    recommended-only for platform archetypes (platform_diagnostics,
    platform_devices, platform_services) which have no clinical trials
    to score.

Usage:
    from alpha_signal_contract import validate_alpha_inputs, validate_alpha_outputs

    # Before DE loop:
    diag = validate_alpha_inputs(rec_by_ticker, csv_rows, schema_mode="warn")

    # After alpha cohort scoring:
    diag = validate_alpha_outputs(csv_rows, schema_mode="warn")
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# CONTRACT VERSION
# =============================================================================

ALPHA_CONTRACT_VERSION = "v1.1.0"

# =============================================================================
# FIELD GROUPS
# =============================================================================

# --- Catalyst input fields (read from rec at DE boundary) ---
# These are the fields _compute_overlays() in decision_engine.py reads.
CATALYST_REQUIRED_FIELDS: List[Tuple[str, ...]] = [
    # (top-level key, sub-key, ...)  — path into rec dict
    ("catalyst_decay", "days_to_catalyst"),
    ("catalyst_decay", "in_optimal_window"),
]

CATALYST_RECOMMENDED_FIELDS: List[Tuple[str, ...]] = [
    # Traceability from event ledger (populated via Module 3 → rec propagation).
    # These are optional until the propagation path is wired end-to-end.
    # NOTE: this list is the source of truth; CATALYST_TRACE_KEYS below derives
    # flat dot-path keys from it for telemetry counters and test assertions.
    ("catalyst_decay", "nearest_catalyst_event_id"),
    ("catalyst_decay", "nearest_catalyst_disclosed_at"),
    ("catalyst_decay", "nearest_catalyst_source_uid"),
]

# Flat dot-path keys derived from CATALYST_RECOMMENDED_FIELDS above
# (used in to_dict() n_missing_catalyst_trace counter and tests)
CATALYST_TRACE_KEYS: Tuple[str, ...] = tuple(
    ".".join(path) for path in CATALYST_RECOMMENDED_FIELDS
)

# --- Clinical input fields (read from csv_row at DE boundary) ---
# These are on the csv_rows dict, not the Module 5 rec dict.
# clinical_score_z is conditionally required: required for archetypes that
# have clinical trial data (drug_developer, commercial_biotech, commercial_pharma),
# recommended-only for platform archetypes (no clinical trials to score).
CLINICAL_REQUIRED_FIELDS: List[str] = []  # unconditionally required (none currently)

CLINICAL_CONDITIONAL_REQUIRED_FIELDS: List[str] = [
    "clinical_score_z",
]

# Archetypes for which clinical_score_z is required (have clinical trial data)
CLINICAL_REQUIRED_ARCHETYPES = frozenset({
    "drug_developer",
    "commercial_biotech",
    "commercial_pharma",
})

CLINICAL_RECOMMENDED_FIELDS: List[str] = [
    "clinical_alpha_z",
    "clinical_readout_days",
    "clinical_coverage_flag",
]

# --- Eligibility/overlay input fields (read from rec) ---
OVERLAY_REQUIRED_FIELDS: List[Tuple[str, ...]] = [
    ("defensive_features",),            # dict must exist
    ("severity",),                      # string
]

OVERLAY_RECOMMENDED_FIELDS: List[Tuple[str, ...]] = [
    ("smart_money_signal",),
    ("coinvest",),
    ("score_breakdown", "enhancements", "momentum"),
    ("confidence_overall",),
]

# --- Alpha output fields (on csv_row after alpha cohort scoring) ---
ALPHA_OUTPUT_REQUIRED_FIELDS: List[str] = [
    "alpha_cohort_key",
    "alpha_cohort_raw",
    "alpha_cohort_pct",
]

ALPHA_OUTPUT_RECOMMENDED_FIELDS: List[str] = [
    "clinical_score_z_tier",
]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def _resolve_nested(d: Dict, path: Tuple[str, ...]) -> Tuple[bool, Any]:
    """Walk a nested dict along `path`. Returns (found, value)."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return False, None
        if key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def _check_rec_fields(
    rec: Dict,
    required: List[Tuple[str, ...]],
    recommended: List[Tuple[str, ...]],
) -> Tuple[List[str], List[str]]:
    """Check required and recommended nested field paths on a rec dict.

    Returns (errors, warnings).
    """
    errors: List[str] = []
    warnings: List[str] = []

    for path in required:
        found, val = _resolve_nested(rec, path)
        # The parent dict must exist; the leaf value may be None (meaning "no data")
        # but the key itself must be present.
        if not found:
            errors.append(".".join(path))

    for path in recommended:
        found, val = _resolve_nested(rec, path)
        if not found:
            warnings.append(".".join(path))

    return errors, warnings


def _check_row_fields(
    row: Dict,
    required: List[str],
    recommended: List[str],
) -> Tuple[List[str], List[str]]:
    """Check required and recommended flat field names on a csv_row dict.

    A field is "present" if the key exists and value is not empty string.
    None is acceptable (explicit "no data").
    """
    errors: List[str] = []
    warnings: List[str] = []

    for field in required:
        if field not in row:
            errors.append(field)
        # Note: value="" or value=None is OK — the field key must exist.
        # The DE uses blank-when-missing semantics, so "" is valid.

    for field in recommended:
        if field not in row:
            warnings.append(field)

    return errors, warnings


# =============================================================================
# INPUT VALIDATION (before DE loop)
# =============================================================================

class AlphaContractDiagnostics:
    """Accumulates validation results across all tickers."""

    def __init__(self) -> None:
        self.contract_version: str = ALPHA_CONTRACT_VERSION
        self.n_tickers_checked: int = 0
        self.n_tickers_with_errors: int = 0
        self.n_tickers_with_warnings: int = 0
        self.total_errors: int = 0
        self.total_warnings: int = 0
        self.error_field_counts: Dict[str, int] = {}
        self.warning_field_counts: Dict[str, int] = {}
        self.sample_errors: List[Dict[str, Any]] = []  # first N tickers with errors

    def record(
        self,
        ticker: str,
        errors: List[str],
        warnings: List[str],
        max_samples: int = 5,
    ) -> None:
        self.n_tickers_checked += 1
        if errors:
            self.n_tickers_with_errors += 1
            self.total_errors += len(errors)
            for f in errors:
                self.error_field_counts[f] = self.error_field_counts.get(f, 0) + 1
            if len(self.sample_errors) < max_samples:
                self.sample_errors.append({"ticker": ticker, "errors": errors})
        if warnings:
            self.n_tickers_with_warnings += 1
            self.total_warnings += len(warnings)
            for f in warnings:
                self.warning_field_counts[f] = self.warning_field_counts.get(f, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        n_missing_rec = self.warning_field_counts.get("alpha_contract.missing_rec", 0)
        # Sum of the 3 nearest-catalyst traceability field warnings
        # (event_id, disclosed_at, source_uid from event ledger propagation)
        n_missing_catalyst_trace = sum(
            self.warning_field_counts.get(k, 0) for k in CATALYST_TRACE_KEYS
        )
        return {
            "alpha_contract_version": self.contract_version,
            "n_tickers_checked": self.n_tickers_checked,
            "n_tickers_with_errors": self.n_tickers_with_errors,
            "n_tickers_with_warnings": self.n_tickers_with_warnings,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "n_missing_rec": n_missing_rec,
            "n_missing_catalyst_trace": n_missing_catalyst_trace,
            "top_missing_required": dict(
                sorted(self.error_field_counts.items(), key=lambda x: -x[1])[:10]
            ),
            "top_missing_recommended": dict(
                sorted(self.warning_field_counts.items(), key=lambda x: -x[1])[:10]
            ),
            "sample_errors": self.sample_errors,
        }

    @property
    def has_errors(self) -> bool:
        return self.total_errors > 0


def validate_alpha_inputs(
    rec_by_ticker: Dict[str, Dict],
    csv_rows: List[Dict],
    schema_mode: str = "warn",
) -> AlphaContractDiagnostics:
    """Validate all records at the DE input boundary.

    Checks that each (rec, csv_row) pair has the required fields for
    clinical+catalyst-aware decision making.

    Missing rec behavior: When a csv_row ticker has no matching entry in
    rec_by_ticker, it is recorded as a warning ("alpha_contract.missing_rec") rather than
    an error.  This is intentional — the DE loop itself skips these rows,
    so missing rec is a data-quality signal, not a contract violation.
    schema_mode="fail" does NOT raise on missing-rec warnings alone.

    Args:
        rec_by_ticker: Module 5 ranked_securities keyed by ticker
        csv_rows: The csv_rows list (enriched with clinical scores)
        schema_mode: "warn" (log + continue) or "fail" (raise on errors)

    Returns:
        AlphaContractDiagnostics with aggregated results
    """
    diag = AlphaContractDiagnostics()

    for row in csv_rows:
        ticker = (row.get("ticker") or "").strip()
        if not ticker:
            continue

        rec = rec_by_ticker.get(ticker)
        if rec is None:
            diag.record(ticker, [], ["alpha_contract.missing_rec"])
            continue

        all_errors: List[str] = []
        all_warnings: List[str] = []

        # Catalyst fields on rec
        errs, warns = _check_rec_fields(
            rec, CATALYST_REQUIRED_FIELDS, CATALYST_RECOMMENDED_FIELDS,
        )
        all_errors.extend(errs)
        all_warnings.extend(warns)

        # Overlay fields on rec
        errs, warns = _check_rec_fields(
            rec, OVERLAY_REQUIRED_FIELDS, OVERLAY_RECOMMENDED_FIELDS,
        )
        all_errors.extend(errs)
        all_warnings.extend(warns)

        # Clinical fields on row (conditional requiredness by archetype)
        archetype = (row.get("archetype") or "").strip().lower()
        if archetype in CLINICAL_REQUIRED_ARCHETYPES:
            clinical_req = CLINICAL_REQUIRED_FIELDS + CLINICAL_CONDITIONAL_REQUIRED_FIELDS
            clinical_rec = CLINICAL_RECOMMENDED_FIELDS
        else:
            # Platform archetypes: conditional fields demoted to recommended
            clinical_req = CLINICAL_REQUIRED_FIELDS
            clinical_rec = CLINICAL_RECOMMENDED_FIELDS + CLINICAL_CONDITIONAL_REQUIRED_FIELDS

        errs, warns = _check_row_fields(row, clinical_req, clinical_rec)
        all_errors.extend(errs)
        all_warnings.extend(warns)

        diag.record(ticker, all_errors, all_warnings)

    # Logging
    if diag.has_errors:
        msg = (
            f"Alpha contract {ALPHA_CONTRACT_VERSION}: "
            f"{diag.total_errors} errors across {diag.n_tickers_with_errors} tickers "
            f"(top missing: {list(diag.error_field_counts.keys())[:5]})"
        )
        if schema_mode == "fail":
            raise AlphaContractViolation(msg, diag)
        else:
            logger.warning(f"ALPHA_CONTRACT_WARN: {msg}")
    else:
        logger.info(
            f"Alpha contract {ALPHA_CONTRACT_VERSION}: "
            f"{diag.n_tickers_checked} tickers checked, "
            f"{diag.total_warnings} warnings"
        )

    return diag


def validate_alpha_outputs(
    csv_rows: List[Dict],
    schema_mode: str = "warn",
    alpha_cohort_enabled: bool = True,
) -> AlphaContractDiagnostics:
    """Validate records after alpha cohort scoring.

    Checks that alpha output fields are populated.

    Args:
        csv_rows: The csv_rows list (enriched with alpha scores)
        schema_mode: "warn" or "fail"
        alpha_cohort_enabled: Whether alpha cohort scoring was run

    Returns:
        AlphaContractDiagnostics with aggregated results
    """
    diag = AlphaContractDiagnostics()

    if not alpha_cohort_enabled:
        # If alpha cohort is off, output fields are not expected
        logger.info(
            f"Alpha contract {ALPHA_CONTRACT_VERSION}: "
            "alpha cohort disabled, skipping output validation"
        )
        return diag

    for row in csv_rows:
        ticker = row.get("ticker", "")
        errs, warns = _check_row_fields(
            row, ALPHA_OUTPUT_REQUIRED_FIELDS, ALPHA_OUTPUT_RECOMMENDED_FIELDS,
        )
        diag.record(ticker, errs, warns)

    if diag.has_errors:
        msg = (
            f"Alpha contract {ALPHA_CONTRACT_VERSION} (outputs): "
            f"{diag.total_errors} errors across {diag.n_tickers_with_errors} tickers "
            f"(top missing: {list(diag.error_field_counts.keys())[:5]})"
        )
        if schema_mode == "fail":
            raise AlphaContractViolation(msg, diag)
        else:
            logger.warning(f"ALPHA_CONTRACT_WARN: {msg}")
    else:
        logger.info(
            f"Alpha contract {ALPHA_CONTRACT_VERSION} (outputs): "
            f"{diag.n_tickers_checked} tickers OK, "
            f"{diag.total_warnings} warnings"
        )

    return diag


# =============================================================================
# EXCEPTION
# =============================================================================

class AlphaContractViolation(Exception):
    """Raised when schema_mode='fail' and contract is violated."""

    def __init__(self, message: str, diagnostics: AlphaContractDiagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics
