"""
PATCH 003: Schema Validation at Load Points
============================================

HIGH FIX: JSON files are loaded without schema validation at critical points,
allowing malformed data to corrupt scoring silently.

This patch provides:
1. Schema validation wrappers for data loading
2. Fail-fast validation with clear error messages
3. Integration with existing common/integration_contracts.py

Files affected:
- module_3_catalyst.py:398-399 (json.load without validation)
- run_screen.py:996 (direct bracket access)
- validation/validate_data_integration.py:618-650

Usage:
    from patches.patch_003_schema_validation import (
        load_and_validate_trial_records,
        load_and_validate_financial_records,
        validate_module_output,
    )
"""

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import logging

logger = logging.getLogger(__name__)


class SchemaValidationError(Exception):
    """Raised when data fails schema validation."""

    def __init__(self, message: str, errors: List[str], partial_data: Optional[Any] = None):
        self.message = message
        self.errors = errors
        self.partial_data = partial_data
        super().__init__(f"{message}: {errors}")


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    record_count: int
    valid_count: int


# =============================================================================
# TRIAL RECORDS SCHEMA
# =============================================================================

TRIAL_RECORD_REQUIRED_FIELDS = {
    "nct_id": str,
    "ticker": str,
    "overall_status": str,
}

TRIAL_RECORD_OPTIONAL_FIELDS = {
    "phase": str,
    "condition": str,
    "primary_completion_date": str,
    "study_completion_date": str,
    "enrollment": (int, type(None)),
    "source_date": str,
}


def validate_trial_record(record: Dict[str, Any], index: int) -> List[str]:
    """
    Validate a single trial record against schema.

    Returns list of error messages (empty if valid).
    """
    errors = []
    ticker = record.get("ticker", f"record_{index}")

    # Check required fields
    for field, expected_type in TRIAL_RECORD_REQUIRED_FIELDS.items():
        value = record.get(field)
        if value is None:
            errors.append(f"{ticker}: Missing required field '{field}'")
        elif not isinstance(value, expected_type):
            errors.append(
                f"{ticker}: Field '{field}' has wrong type "
                f"(expected {expected_type.__name__}, got {type(value).__name__})"
            )

    # Validate NCT ID format
    nct_id = record.get("nct_id", "")
    if nct_id and not nct_id.startswith("NCT"):
        errors.append(f"{ticker}: Invalid NCT ID format: {nct_id}")

    # Validate status
    valid_statuses = {
        "ACTIVE", "RECRUITING", "COMPLETED", "TERMINATED", "SUSPENDED",
        "WITHDRAWN", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "UNKNOWN"
    }
    status = record.get("overall_status", "").upper()
    if status and status not in valid_statuses:
        # Warning, not error - new statuses may appear
        logger.warning(f"{ticker}: Unrecognized status '{status}'")

    # Validate dates if present
    for date_field in ["primary_completion_date", "study_completion_date", "source_date"]:
        date_value = record.get(date_field)
        if date_value:
            try:
                # Accept YYYY-MM-DD format
                date.fromisoformat(date_value[:10])
            except (ValueError, TypeError):
                errors.append(f"{ticker}: Invalid date format in '{date_field}': {date_value}")

    return errors


def load_and_validate_trial_records(
    file_path: Path,
    strict: bool = True,
) -> tuple[List[Dict[str, Any]], ValidationResult]:
    """
    Load trial records with schema validation.

    Args:
        file_path: Path to trial_records.json
        strict: If True, raise exception on any validation errors

    Returns:
        (valid_records, validation_result)

    Raises:
        SchemaValidationError: If strict=True and validation fails
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    logger.info(f"Loading trial records from {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Trial records file not found: {file_path}")

    with open(file_path) as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(
                f"Invalid JSON in {file_path}",
                [str(e)],
            )

    if not isinstance(records, list):
        raise SchemaValidationError(
            f"Expected list of records, got {type(records).__name__}",
            ["Root element must be a list"],
        )

    all_errors = []
    all_warnings = []
    valid_records = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            all_errors.append(f"Record {i}: Expected dict, got {type(record).__name__}")
            continue

        errors = validate_trial_record(record, i)

        if errors:
            all_errors.extend(errors)
        else:
            valid_records.append(record)

    result = ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
        record_count=len(records),
        valid_count=len(valid_records),
    )

    if strict and all_errors:
        raise SchemaValidationError(
            f"Trial records validation failed: {len(all_errors)} errors",
            all_errors,
            partial_data=valid_records,
        )

    if all_errors:
        logger.warning(
            f"Trial records validation: {len(valid_records)}/{len(records)} valid, "
            f"{len(all_errors)} errors"
        )
    else:
        logger.info(f"Trial records validated: {len(valid_records)} records OK")

    return valid_records, result


# =============================================================================
# FINANCIAL RECORDS SCHEMA
# =============================================================================

FINANCIAL_RECORD_REQUIRED_FIELDS = {
    "ticker": str,
}

FINANCIAL_RECORD_NUMERIC_FIELDS = {
    "cash_mm", "Cash", "burn_rate_mm", "NetIncome", "R&D",
    "CFO", "FCF", "market_cap_mm", "market_cap",
}


def validate_financial_record(record: Dict[str, Any], index: int) -> List[str]:
    """Validate a single financial record."""
    errors = []
    ticker = record.get("ticker", f"record_{index}")

    # Check required fields
    if "ticker" not in record:
        errors.append(f"Record {index}: Missing required field 'ticker'")
        return errors

    # Check numeric fields are actually numeric
    for field in FINANCIAL_RECORD_NUMERIC_FIELDS:
        value = record.get(field)
        if value is not None:
            if not isinstance(value, (int, float, Decimal)):
                try:
                    float(value)  # Allow string numbers
                except (ValueError, TypeError):
                    errors.append(
                        f"{ticker}: Field '{field}' is not numeric: {value}"
                    )

    # Validate source_date if present
    source_date = record.get("source_date")
    if source_date:
        try:
            date.fromisoformat(source_date[:10])
        except (ValueError, TypeError):
            errors.append(f"{ticker}: Invalid source_date format: {source_date}")

    return errors


def load_and_validate_financial_records(
    file_path: Path,
    strict: bool = True,
) -> tuple[List[Dict[str, Any]], ValidationResult]:
    """
    Load financial records with schema validation.

    Similar to load_and_validate_trial_records.
    """
    logger.info(f"Loading financial records from {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"Financial records file not found: {file_path}")

    with open(file_path) as f:
        try:
            records = json.load(f)
        except json.JSONDecodeError as e:
            raise SchemaValidationError(
                f"Invalid JSON in {file_path}",
                [str(e)],
            )

    if not isinstance(records, list):
        raise SchemaValidationError(
            f"Expected list of records, got {type(records).__name__}",
            ["Root element must be a list"],
        )

    all_errors = []
    valid_records = []

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            all_errors.append(f"Record {i}: Expected dict, got {type(record).__name__}")
            continue

        errors = validate_financial_record(record, i)

        if errors:
            all_errors.extend(errors)
        else:
            valid_records.append(record)

    result = ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=[],
        record_count=len(records),
        valid_count=len(valid_records),
    )

    if strict and all_errors:
        raise SchemaValidationError(
            f"Financial records validation failed: {len(all_errors)} errors",
            all_errors,
            partial_data=valid_records,
        )

    return valid_records, result


# =============================================================================
# MODULE OUTPUT VALIDATION
# =============================================================================

def validate_module_output(
    output: Dict[str, Any],
    module_name: str,
    required_fields: Set[str],
) -> ValidationResult:
    """
    Validate module output has required structure.

    Args:
        output: Module output dict
        module_name: Name of module for error messages
        required_fields: Set of required top-level fields

    Returns:
        ValidationResult
    """
    errors = []

    if not isinstance(output, dict):
        errors.append(f"{module_name}: Output must be dict, got {type(output).__name__}")
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=[],
            record_count=0,
            valid_count=0,
        )

    # Check required fields
    for field in required_fields:
        if field not in output:
            errors.append(f"{module_name}: Missing required field '{field}'")

    # Module-specific validation
    if module_name == "module_2":
        scores = output.get("scores", [])
        if not isinstance(scores, list):
            errors.append(f"{module_name}: 'scores' must be a list")
        else:
            for i, score in enumerate(scores):
                if not isinstance(score, dict):
                    errors.append(f"{module_name}: scores[{i}] must be a dict")
                elif "ticker" not in score:
                    errors.append(f"{module_name}: scores[{i}] missing 'ticker'")
                elif "financial_score" not in score:
                    errors.append(f"{module_name}: scores[{i}] missing 'financial_score'")

    elif module_name == "module_3":
        summaries = output.get("summaries", {})
        if not isinstance(summaries, dict):
            errors.append(f"{module_name}: 'summaries' must be a dict")

    elif module_name == "module_5":
        ranked = output.get("ranked_securities", [])
        if not isinstance(ranked, list):
            errors.append(f"{module_name}: 'ranked_securities' must be a list")
        else:
            for i, sec in enumerate(ranked):
                if not isinstance(sec, dict):
                    errors.append(f"{module_name}: ranked_securities[{i}] must be a dict")
                elif "ticker" not in sec:
                    errors.append(f"{module_name}: ranked_securities[{i}] missing 'ticker'")
                elif "composite_score" not in sec:
                    errors.append(f"{module_name}: ranked_securities[{i}] missing 'composite_score'")

    record_count = len(output.get("scores", output.get("ranked_securities", [])))

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=[],
        record_count=record_count,
        valid_count=record_count if len(errors) == 0 else 0,
    )


# =============================================================================
# SAFE DICT ACCESS
# =============================================================================

def safe_get(
    data: Dict[str, Any],
    key: str,
    default: Any = None,
    expected_type: Optional[type] = None,
) -> Any:
    """
    Get value from dict with type checking.

    Unlike dict.get(), this logs warnings for type mismatches.

    Args:
        data: Dictionary to access
        key: Key to get
        default: Default value if key missing
        expected_type: If provided, validate type

    Returns:
        Value or default
    """
    value = data.get(key)

    if value is None:
        return default

    if expected_type is not None and not isinstance(value, expected_type):
        logger.warning(
            f"Type mismatch for key '{key}': "
            f"expected {expected_type.__name__}, got {type(value).__name__}"
        )
        return default

    return value


def safe_list_access(
    data: List[Any],
    index: int,
    default: Any = None,
) -> Any:
    """
    Safely access list element.

    Args:
        data: List to access
        index: Index to get (supports negative indexing)
        default: Default value if index out of range

    Returns:
        Value or default
    """
    try:
        if abs(index) < len(data):
            return data[index]
        return default
    except (TypeError, IndexError):
        return default


if __name__ == "__main__":
    print("=" * 70)
    print("PATCH 003: Schema Validation at Load Points")
    print("=" * 70)
    print()
    print("This patch adds schema validation when loading JSON data.")
    print()
    print("PROBLEM with original implementation:")
    print("  - json.load() without validation")
    print("  - Direct bracket access without checks")
    print("  - Malformed data silently corrupts scoring")
    print()
    print("SOLUTION (this patch):")
    print("  - Validate schema on load")
    print("  - Clear error messages for each violation")
    print("  - Safe accessor functions with type checking")
    print()
    print("Usage:")
    print("  from patches.patch_003_schema_validation import (")
    print("      load_and_validate_trial_records,")
    print("      load_and_validate_financial_records,")
    print("      safe_get,")
    print("  )")
    print()
    print("  # Load with validation")
    print("  records, result = load_and_validate_trial_records(")
    print("      Path('production_data/trial_records.json'),")
    print("      strict=True  # Raise on errors")
    print("  )")
    print()
    print("  # Safe dictionary access")
    print("  score = safe_get(record, 'financial_score', default=0.0, expected_type=float)")
    print()
