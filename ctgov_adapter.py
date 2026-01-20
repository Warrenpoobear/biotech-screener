#!/usr/bin/env python3
"""
ctgov_adapter.py - Production CT.gov Data Adapter

Converts trial_records.json entries to canonical Module 3A format.
Handles multiple input variants with deterministic field extraction.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Optional
from enum import Enum
import re
import logging
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class CTGovStatus(Enum):
    """
    Normalized CT.gov status values (ordered by trial lifecycle progression).

    Ordering rationale:
    - Lower values = worse outcomes / earlier termination
    - Higher values = trial progressing / completed successfully

    Non-trial statuses (APPROVED_FOR_MARKETING, AVAILABLE, etc.) are mapped
    to specific values based on their signal characteristics.
    """
    WITHDRAWN = 0
    TERMINATED = 1
    SUSPENDED = 2
    WITHHELD = 3              # Results withheld - negative signal
    NO_LONGER_AVAILABLE = 4   # Expanded access ended - neutral/slightly negative
    UNKNOWN = 5
    ENROLLING_BY_INVITATION = 6
    NOT_YET_RECRUITING = 7
    RECRUITING = 8
    AVAILABLE = 9             # Expanded access / compassionate use - positive signal
    ACTIVE_NOT_RECRUITING = 10
    APPROVED_FOR_MARKETING = 11  # Drug approved - very positive (post-trial)
    COMPLETED = 12

    @classmethod
    def from_string(cls, status_str: str) -> 'CTGovStatus':
        """Normalize status string to enum"""
        if not status_str:
            return cls.UNKNOWN

        # Normalize: uppercase, replace separators
        normalized = status_str.upper()
        normalized = re.sub(r'[,\-\s]+', '_', normalized)
        normalized = re.sub(r'_+', '_', normalized)
        normalized = normalized.strip('_')

        # Handle known CT.gov status variants
        STATUS_ALIASES = {
            # Standard statuses (exact match after normalization)
            'WITHDRAWN': cls.WITHDRAWN,
            'TERMINATED': cls.TERMINATED,
            'SUSPENDED': cls.SUSPENDED,
            'WITHHELD': cls.WITHHELD,
            'NO_LONGER_AVAILABLE': cls.NO_LONGER_AVAILABLE,
            'UNKNOWN_STATUS': cls.UNKNOWN,
            'ENROLLING_BY_INVITATION': cls.ENROLLING_BY_INVITATION,
            'NOT_YET_RECRUITING': cls.NOT_YET_RECRUITING,
            'RECRUITING': cls.RECRUITING,
            'AVAILABLE': cls.AVAILABLE,
            'ACTIVE_NOT_RECRUITING': cls.ACTIVE_NOT_RECRUITING,
            'APPROVED_FOR_MARKETING': cls.APPROVED_FOR_MARKETING,
            'COMPLETED': cls.COMPLETED,
            # Common aliases
            'ACTIVE': cls.ACTIVE_NOT_RECRUITING,
            'APPROVED': cls.APPROVED_FOR_MARKETING,
            'ENROLL_BY_INVITATION': cls.ENROLLING_BY_INVITATION,
            'INVITATION_ONLY': cls.ENROLLING_BY_INVITATION,
        }

        if normalized in STATUS_ALIASES:
            return STATUS_ALIASES[normalized]

        # Try direct enum lookup
        try:
            return cls[normalized]
        except KeyError:
            logger.warning(f"Unknown status: '{status_str}' → UNKNOWN")
            return cls.UNKNOWN

    @property
    def is_terminal_negative(self) -> bool:
        """True if status indicates trial stopped/failed"""
        return self in {
            CTGovStatus.WITHDRAWN,
            CTGovStatus.TERMINATED,
            CTGovStatus.SUSPENDED,
        }

    @property
    def is_terminal_positive(self) -> bool:
        """True if status indicates successful completion/approval"""
        return self in {
            CTGovStatus.COMPLETED,
            CTGovStatus.APPROVED_FOR_MARKETING,
        }

    @property
    def is_active(self) -> bool:
        """True if trial is actively recruiting or progressing"""
        return self in {
            CTGovStatus.RECRUITING,
            CTGovStatus.ACTIVE_NOT_RECRUITING,
            CTGovStatus.ENROLLING_BY_INVITATION,
            CTGovStatus.NOT_YET_RECRUITING,
            CTGovStatus.AVAILABLE,
        }


class CompletionType(Enum):
    """Date completion type"""
    ACTUAL = "ACTUAL"
    ANTICIPATED = "ANTICIPATED"
    ESTIMATED = "ESTIMATED"  # CT.gov also uses ESTIMATED
    
    @classmethod
    def from_string(cls, type_str: str | None) -> Optional['CompletionType']:
        if not type_str:
            return None
        try:
            return cls[type_str.upper()]
        except KeyError:
            logger.warning(f"Unknown completion type: '{type_str}'")
            return None


# ============================================================================
# CANONICAL RECORD
# ============================================================================

@dataclass(frozen=True)
class CanonicalTrialRecord:
    """Canonical Module 3A trial record"""
    ticker: str
    nct_id: str
    overall_status: CTGovStatus
    last_update_posted: date  # PIT anchor
    primary_completion_date: Optional[date]
    primary_completion_type: Optional[CompletionType]
    completion_date: Optional[date]
    completion_type: Optional[CompletionType]
    results_first_posted: Optional[date]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSONL storage"""
        return {
            'ticker': self.ticker,
            'nct_id': self.nct_id,
            'overall_status': self.overall_status.name,
            'last_update_posted': self.last_update_posted.isoformat(),
            'primary_completion_date': self.primary_completion_date.isoformat() if self.primary_completion_date else None,
            'primary_completion_type': self.primary_completion_type.value if self.primary_completion_type else None,
            'completion_date': self.completion_date.isoformat() if self.completion_date else None,
            'completion_type': self.completion_type.value if self.completion_type else None,
            'results_first_posted': self.results_first_posted.isoformat() if self.results_first_posted else None
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'CanonicalTrialRecord':
        """Deserialize from JSONL"""
        return cls(
            ticker=data['ticker'],
            nct_id=data['nct_id'],
            overall_status=CTGovStatus[data['overall_status']],
            last_update_posted=date.fromisoformat(data['last_update_posted']),
            primary_completion_date=date.fromisoformat(data['primary_completion_date']) if data['primary_completion_date'] else None,
            primary_completion_type=CompletionType[data['primary_completion_type']] if data['primary_completion_type'] else None,
            completion_date=date.fromisoformat(data['completion_date']) if data['completion_date'] else None,
            completion_type=CompletionType[data['completion_type']] if data['completion_type'] else None,
            results_first_posted=date.fromisoformat(data['results_first_posted']) if data['results_first_posted'] else None
        )
    
    def compute_hash(self) -> str:
        """Compute deterministic hash for delta detection"""
        canonical_json = str(self.to_dict())
        return hashlib.sha256(canonical_json.encode()).hexdigest()[:16]


# ============================================================================
# EXCEPTIONS
# ============================================================================

class AdapterError(Exception):
    """Base exception for adapter errors"""
    pass


class MissingRequiredFieldError(AdapterError):
    """Required field cannot be extracted"""
    pass


class FutureDataError(AdapterError):
    """last_update_posted > as_of_date (leakage)"""
    pass


# ============================================================================
# VALIDATION STATISTICS
# ============================================================================

@dataclass
class AdapterStats:
    """Validation statistics"""
    total_records: int = 0
    successful_extractions: int = 0
    missing_ticker: int = 0
    missing_nct_id: int = 0
    missing_overall_status: int = 0
    missing_last_update_posted: int = 0
    failed_date_parses: int = 0
    unknown_status_strings: int = 0
    future_data_violations: int = 0
    
    def log_summary(self):
        """Log validation summary"""
        logger.info("="*80)
        logger.info("Adapter Validation Summary")
        logger.info("="*80)
        logger.info(f"Total: {self.total_records}")
        logger.info(f"Success: {self.successful_extractions} ({self.success_rate:.1%})")
        
        if self.missing_last_update_posted > 0:
            logger.error(f"CRITICAL: Missing last_update_posted: {self.missing_last_update_posted}")
        
        if self.future_data_violations > 0:
            logger.error(f"CRITICAL: Future data violations: {self.future_data_violations}")
        
        if self.pct_missing_overall_status > 0.05:
            logger.warning(f"Missing overall_status: {self.missing_overall_status} ({self.pct_missing_overall_status:.1%})")
    
    @property
    def success_rate(self) -> float:
        return self.successful_extractions / self.total_records if self.total_records > 0 else 0.0
    
    @property
    def pct_missing_overall_status(self) -> float:
        return self.missing_overall_status / self.total_records if self.total_records > 0 else 0.0


# ============================================================================
# ADAPTER CONFIGURATION
# ============================================================================

@dataclass
class AdapterConfig:
    """Adapter behavior configuration

    Note on fail_on_future_data:
        - False (default): Filter out future records with a warning, continue processing
        - True: Crash on any future data (strict mode for production)

    The default is False to allow historical/backtest runs where as_of_date
    may be before the data's last_update_date. The adapter will still report
    how many records were filtered and fail if >50% are filtered.
    """
    max_missing_overall_status: float = 0.05  # 5% threshold
    allow_partial_dates: bool = False
    fail_on_future_data: bool = False  # Changed: filter (not crash) by default
    log_unknown_statuses: bool = True
    max_future_data_ratio: float = 0.50  # Fail if >50% of data is from future


# ============================================================================
# CT.GOV ADAPTER
# ============================================================================

class CTGovAdapter:
    """
    Adapter for converting trial_records.json to canonical form
    
    Handles three input variants:
    - Form A: Raw CT.gov v2 JSON (nested protocolSection)
    - Form B: Flattened schema (pre-extracted fields)
    - Form C: Hybrid (ticker + nct_id + embedded ctgov_record)
    """
    
    def __init__(self, config: AdapterConfig = AdapterConfig()):
        self.config = config
        self.stats = AdapterStats()
    
    def extract_canonical_record(
        self, 
        record: dict[str, Any],
        as_of_date: date
    ) -> CanonicalTrialRecord:
        """Extract canonical record from any input variant"""
        self.stats.total_records += 1
        
        # Root selection: try ctgov_record, study, or record itself
        root = record.get("ctgov_record") or record.get("study") or record
        
        try:
            # Required: ticker
            ticker = self._extract_ticker(record)
            
            # Required: nct_id
            nct_id = self._extract_nct_id(record, root)
            
            # Required: last_update_posted (PIT anchor)
            last_update_posted = self._extract_last_update_posted(record, root)
            
            # PIT validation - always raise FutureDataError for future records
            # The batch processor decides whether to fail or filter based on config
            if last_update_posted > as_of_date:
                self.stats.future_data_violations += 1
                raise FutureDataError(
                    f"Future data: {nct_id} has last_update_posted={last_update_posted} > as_of_date={as_of_date}"
                )
            
            # Optional: overall_status
            overall_status = self._extract_overall_status(record, root)
            if overall_status is None:
                self.stats.missing_overall_status += 1
                overall_status = CTGovStatus.UNKNOWN
            
            # Optional: dates
            primary_completion_date = self._extract_primary_completion_date(record, root)
            primary_completion_type = self._extract_primary_completion_type(record, root)
            completion_date = self._extract_completion_date(record, root)
            completion_type = self._extract_completion_type(record, root)
            results_first_posted = self._extract_results_first_posted(record, root)
            
            canonical = CanonicalTrialRecord(
                ticker=ticker,
                nct_id=nct_id,
                overall_status=overall_status,
                last_update_posted=last_update_posted,
                primary_completion_date=primary_completion_date,
                primary_completion_type=primary_completion_type,
                completion_date=completion_date,
                completion_type=completion_type,
                results_first_posted=results_first_posted
            )
            
            self.stats.successful_extractions += 1
            return canonical
            
        except MissingRequiredFieldError:
            raise
        except FutureDataError:
            raise  # Let FutureDataError propagate for batch processor to handle
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise AdapterError(f"Extraction failed: {e}")
    
    # ========================================================================
    # FIELD EXTRACTION
    # ========================================================================
    
    def _extract_ticker(self, record: dict[str, Any]) -> str:
        """Extract ticker with deterministic try order"""
        candidates = [
            record.get("ticker"),
            record.get("symbol")
        ]
        
        for candidate in candidates:
            if candidate:
                return str(candidate).upper().strip()
        
        self.stats.missing_ticker += 1
        raise MissingRequiredFieldError("ticker: Could not extract")
    
    def _extract_nct_id(self, record: dict[str, Any], root: dict[str, Any]) -> str:
        """Extract NCT ID with deterministic try order"""
        candidates = [
            record.get("nct_id"),
            self._safe_get(root, ["protocolSection", "identificationModule", "nctId"]),
            root.get("nctId")
        ]
        
        for candidate in candidates:
            if candidate:
                return str(candidate).strip()
        
        self.stats.missing_nct_id += 1
        raise MissingRequiredFieldError("nct_id: Could not extract")
    
    def _extract_last_update_posted(self, record: dict[str, Any], root: dict[str, Any]) -> date:
        """Extract last_update_posted (CRITICAL PIT anchor)"""
        candidates = [
            record.get("last_update_posted"),
            self._safe_get(root, ["protocolSection", "statusModule", "lastUpdatePostDateStruct", "date"]),
            self._safe_get(root, ["statusModule", "lastUpdatePostDateStruct", "date"])
        ]
        
        for candidate in candidates:
            if candidate:
                parsed = self._parse_date(candidate)
                if parsed:
                    return parsed
        
        self.stats.missing_last_update_posted += 1
        raise MissingRequiredFieldError("last_update_posted: CRITICAL field missing")
    
    def _extract_overall_status(self, record: dict[str, Any], root: dict[str, Any]) -> Optional[CTGovStatus]:
        """Extract overall_status"""
        candidates = [
            record.get("overall_status"),
            record.get("status"),  # User's data uses "status" not "overall_status"
            self._safe_get(root, ["protocolSection", "statusModule", "overallStatus"]),
            self._safe_get(root, ["statusModule", "overallStatus"])
        ]
        
        for candidate in candidates:
            if candidate:
                status = CTGovStatus.from_string(str(candidate))
                if status == CTGovStatus.UNKNOWN and self.config.log_unknown_statuses:
                    self.stats.unknown_status_strings += 1
                return status
        
        return None
    
    def _extract_primary_completion_date(self, record: dict[str, Any], root: dict[str, Any]) -> Optional[date]:
        """Extract primary_completion_date"""
        candidates = [
            record.get("primary_completion_date"),
            self._safe_get(root, ["protocolSection", "statusModule", "primaryCompletionDateStruct", "date"]),
            self._safe_get(root, ["statusModule", "primaryCompletionDateStruct", "date"])
        ]
        
        for candidate in candidates:
            if candidate:
                return self._parse_date(candidate)
        
        return None
    
    def _extract_primary_completion_type(self, record: dict[str, Any], root: dict[str, Any]) -> Optional[CompletionType]:
        """Extract primary_completion_type"""
        candidates = [
            record.get("primary_completion_type"),
            self._safe_get(root, ["protocolSection", "statusModule", "primaryCompletionDateStruct", "type"]),
            self._safe_get(root, ["statusModule", "primaryCompletionDateStruct", "type"])
        ]
        
        for candidate in candidates:
            if candidate:
                return CompletionType.from_string(str(candidate))
        
        return None
    
    def _extract_completion_date(self, record: dict[str, Any], root: dict[str, Any]) -> Optional[date]:
        """Extract completion_date"""
        candidates = [
            record.get("completion_date"),
            self._safe_get(root, ["protocolSection", "statusModule", "completionDateStruct", "date"]),
            self._safe_get(root, ["statusModule", "completionDateStruct", "date"])
        ]
        
        for candidate in candidates:
            if candidate:
                return self._parse_date(candidate)
        
        return None
    
    def _extract_completion_type(self, record: dict[str, Any], root: dict[str, Any]) -> Optional[CompletionType]:
        """Extract completion_type"""
        candidates = [
            record.get("completion_type"),
            self._safe_get(root, ["protocolSection", "statusModule", "completionDateStruct", "type"]),
            self._safe_get(root, ["statusModule", "completionDateStruct", "type"])
        ]
        
        for candidate in candidates:
            if candidate:
                return CompletionType.from_string(str(candidate))
        
        return None
    
    def _extract_results_first_posted(self, record: dict[str, Any], root: dict[str, Any]) -> Optional[date]:
        """Extract results_first_posted"""
        candidates = [
            record.get("results_first_posted"),
            self._safe_get(root, ["resultsSection", "resultsFirstPostDateStruct", "date"])
        ]
        
        for candidate in candidates:
            if candidate:
                return self._parse_date(candidate)
        
        return None
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    @staticmethod
    def _safe_get(obj: dict[str, Any], path: list[str]) -> Any:
        """Safely traverse nested dict"""
        current = obj
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        return current
    
    def _parse_date(self, date_str: str | None) -> Optional[date]:
        """Parse date with strict validation (ISO YYYY-MM-DD only)"""
        if not date_str:
            return None
        
        date_str = str(date_str).strip()
        
        # Strict: only full ISO dates
        if not self.config.allow_partial_dates:
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                self.stats.failed_date_parses += 1
                return None
        
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            self.stats.failed_date_parses += 1
            return None
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_batch(self) -> bool:
        """Run validation gates on batch statistics"""
        self.stats.log_summary()

        failures = []
        warnings = []

        # Gate 1: last_update_posted must be 100%
        if self.stats.missing_last_update_posted > 0:
            failures.append(f"Missing last_update_posted: {self.stats.missing_last_update_posted}")

        # Gate 2: overall_status coverage
        if self.stats.pct_missing_overall_status > 0.10:
            failures.append(f"Missing overall_status: {self.stats.pct_missing_overall_status:.1%} (threshold: 10%)")

        # Gate 3: future data handling
        if self.stats.future_data_violations > 0:
            future_ratio = self.stats.future_data_violations / max(self.stats.total_records, 1)

            if self.config.fail_on_future_data:
                # Strict mode: any future data is a failure
                failures.append(f"Future data violations: {self.stats.future_data_violations}")
            elif future_ratio > self.config.max_future_data_ratio:
                # Too many filtered: fail (probably wrong as_of_date)
                failures.append(
                    f"Too many future data records filtered: {self.stats.future_data_violations}/{self.stats.total_records} "
                    f"({future_ratio:.1%} > {self.config.max_future_data_ratio:.0%} threshold). "
                    f"Check if as_of_date is correct."
                )
            else:
                # Acceptable: warn but continue
                warnings.append(
                    f"Filtered {self.stats.future_data_violations} future-dated records "
                    f"({future_ratio:.1%} of total)"
                )

        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")

        if failures:
            logger.error("VALIDATION GATES FAILED:")
            for failure in failures:
                logger.error(f"  - {failure}")
            return False

        logger.info("✅ VALIDATION GATES PASSED")
        return True


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_trial_records_batch(
    records: list[dict[str, Any]],
    as_of_date: date,
    config: AdapterConfig = AdapterConfig()
) -> tuple[list[CanonicalTrialRecord], AdapterStats]:
    """
    Process batch of trial records into canonical form
    
    Returns: (canonical_records, stats)
    Raises: AdapterError if validation gates fail
    """
    adapter = CTGovAdapter(config)
    canonical_records = []
    
    for i, record in enumerate(records):
        try:
            canonical = adapter.extract_canonical_record(record, as_of_date)
            canonical_records.append(canonical)
        except MissingRequiredFieldError as e:
            logger.warning(f"Skipping record {i}: {e}")
        except FutureDataError as e:
            if config.fail_on_future_data:
                # Strict mode: log error and crash
                logger.error(f"PIT violation in record {i}: {e}")
                raise
            else:
                # Filter mode: skip silently (count is tracked in stats)
                logger.debug(f"Filtering future-dated record {i}: {e}")
    
    # Run validation gates
    if not adapter.validate_batch():
        raise AdapterError("Batch validation failed")
    
    return canonical_records, adapter.stats


if __name__ == "__main__":
    # Example usage
    print("CT.gov Adapter loaded successfully")
    print("Use process_trial_records_batch() to convert trial_records.json")
