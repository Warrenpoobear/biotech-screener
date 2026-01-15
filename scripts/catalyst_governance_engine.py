#!/usr/bin/env python3
"""
catalyst_governance_engine.py - Governance Engine for Catalyst Module

Implements:
- Fail-closed validation (6 hard rules)
- Black swan detection
- Data quality gates
- Audit trail generation
- Composite scoring with governance overlay

Design Philosophy:
- FAIL LOUDLY: Invalid data should halt, not proceed silently
- GOVERNED: Every decision traceable
- DETERMINISTIC: Same inputs → same outputs
- PIT-SAFE: No future data leakage

Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set
import hashlib
import json


# =============================================================================
# GOVERNANCE CONSTANTS
# =============================================================================

# Staleness thresholds (days)
STALENESS_WARNING_DAYS = 90
STALENESS_CRITICAL_DAYS = 180
STALENESS_REJECT_DAYS = 365

# Data quality thresholds
MIN_TRIAL_COVERAGE_PCT = Decimal("0.50")  # 50% of expected trials must be present
MAX_MISSING_FIELDS_PCT = Decimal("0.20")  # 20% missing fields allowed


# =============================================================================
# VALIDATION RULES
# =============================================================================

class ValidationSeverity(str, Enum):
    """Severity of validation failure."""
    FATAL = "FATAL"  # Halt processing
    ERROR = "ERROR"  # Flag but continue
    WARNING = "WARNING"  # Log only
    INFO = "INFO"  # Informational


class ValidationRule(str, Enum):
    """Governance validation rules."""
    # === FATAL RULES (Halt processing) ===
    RULE_PIT_VIOLATION = "RULE_PIT_VIOLATION"
    RULE_FUTURE_DATA_LEAK = "RULE_FUTURE_DATA_LEAK"
    RULE_CRITICAL_FIELD_MISSING = "RULE_CRITICAL_FIELD_MISSING"

    # === ERROR RULES (Flag but continue) ===
    RULE_DATA_STALE = "RULE_DATA_STALE"
    RULE_COVERAGE_LOW = "RULE_COVERAGE_LOW"
    RULE_DUPLICATE_DETECTED = "RULE_DUPLICATE_DETECTED"

    # === WARNING RULES (Log only) ===
    RULE_FIELD_MISSING = "RULE_FIELD_MISSING"
    RULE_DATA_QUALITY_LOW = "RULE_DATA_QUALITY_LOW"
    RULE_UNUSUAL_PATTERN = "RULE_UNUSUAL_PATTERN"

    # === BLACK SWAN RULES ===
    RULE_BLACK_SWAN_DETECTED = "RULE_BLACK_SWAN_DETECTED"


RULE_SEVERITY: Dict[ValidationRule, ValidationSeverity] = {
    ValidationRule.RULE_PIT_VIOLATION: ValidationSeverity.FATAL,
    ValidationRule.RULE_FUTURE_DATA_LEAK: ValidationSeverity.FATAL,
    ValidationRule.RULE_CRITICAL_FIELD_MISSING: ValidationSeverity.FATAL,
    ValidationRule.RULE_DATA_STALE: ValidationSeverity.ERROR,
    ValidationRule.RULE_COVERAGE_LOW: ValidationSeverity.ERROR,
    ValidationRule.RULE_DUPLICATE_DETECTED: ValidationSeverity.ERROR,
    ValidationRule.RULE_FIELD_MISSING: ValidationSeverity.WARNING,
    ValidationRule.RULE_DATA_QUALITY_LOW: ValidationSeverity.WARNING,
    ValidationRule.RULE_UNUSUAL_PATTERN: ValidationSeverity.WARNING,
    ValidationRule.RULE_BLACK_SWAN_DETECTED: ValidationSeverity.ERROR,
}


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationViolation:
    """A single validation violation."""
    rule: ValidationRule
    severity: ValidationSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    @property
    def violation_id(self) -> str:
        """Stable violation ID."""
        canonical = f"{self.rule.value}|{self.message}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "rule": self.rule.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
        }


@dataclass
class ValidationResult:
    """Complete validation result."""
    is_valid: bool
    n_fatal: int = 0
    n_error: int = 0
    n_warning: int = 0
    violations: List[ValidationViolation] = field(default_factory=list)
    validation_hash: str = ""

    @property
    def passed(self) -> bool:
        """True if no fatal violations."""
        return self.n_fatal == 0

    def add_violation(self, violation: ValidationViolation) -> None:
        """Add a violation and update counts."""
        self.violations.append(violation)

        if violation.severity == ValidationSeverity.FATAL:
            self.n_fatal += 1
            self.is_valid = False
        elif violation.severity == ValidationSeverity.ERROR:
            self.n_error += 1
        elif violation.severity == ValidationSeverity.WARNING:
            self.n_warning += 1

    def compute_hash(self) -> str:
        """Compute validation result hash."""
        violations_json = json.dumps(
            [v.to_dict() for v in sorted(self.violations, key=lambda v: v.violation_id)],
            sort_keys=True,
        )
        return hashlib.sha256(violations_json.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "passed": self.passed,
            "n_fatal": self.n_fatal,
            "n_error": self.n_error,
            "n_warning": self.n_warning,
            "validation_hash": self.compute_hash(),
            "violations": [v.to_dict() for v in self.violations],
        }


# =============================================================================
# BLACK SWAN DETECTION
# =============================================================================

class BlackSwanType(str, Enum):
    """Types of black swan events."""
    CLINICAL_HOLD = "CLINICAL_HOLD"
    TRIAL_TERMINATED = "TRIAL_TERMINATED"
    CRL_RECEIVED = "CRL_RECEIVED"
    SAFETY_SIGNAL = "SAFETY_SIGNAL"
    REGULATORY_REJECTION = "REGULATORY_REJECTION"
    SPONSOR_BANKRUPTCY = "SPONSOR_BANKRUPTCY"
    LEADERSHIP_EXODUS = "LEADERSHIP_EXODUS"
    MANUFACTURING_FAILURE = "MANUFACTURING_FAILURE"


@dataclass
class BlackSwanEvent:
    """A detected black swan event."""
    ticker: str
    event_type: BlackSwanType
    event_date: str
    description: str
    severity_score: Decimal  # 0-100, higher = more severe
    is_confirmed: bool = False
    source: str = "INFERRED"

    @property
    def event_id(self) -> str:
        """Stable event ID."""
        canonical = f"{self.ticker}|{self.event_type.value}|{self.event_date}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "ticker": self.ticker,
            "event_type": self.event_type.value,
            "event_date": self.event_date,
            "description": self.description,
            "severity_score": str(self.severity_score),
            "is_confirmed": self.is_confirmed,
            "source": self.source,
        }


# Black swan severity scores
BLACK_SWAN_SEVERITY: Dict[BlackSwanType, Decimal] = {
    BlackSwanType.CLINICAL_HOLD: Decimal("90"),
    BlackSwanType.TRIAL_TERMINATED: Decimal("85"),
    BlackSwanType.CRL_RECEIVED: Decimal("80"),
    BlackSwanType.SAFETY_SIGNAL: Decimal("75"),
    BlackSwanType.REGULATORY_REJECTION: Decimal("95"),
    BlackSwanType.SPONSOR_BANKRUPTCY: Decimal("100"),
    BlackSwanType.LEADERSHIP_EXODUS: Decimal("50"),
    BlackSwanType.MANUFACTURING_FAILURE: Decimal("70"),
}


# =============================================================================
# GOVERNANCE ENGINE
# =============================================================================

class GovernanceEngine:
    """
    Governance engine for catalyst module.

    Implements fail-closed validation with 6 hard rules:
    1. PIT Violation: No future data in analysis
    2. Future Data Leak: Source dates must be <= as_of_date
    3. Critical Field Missing: NCT ID, ticker, status required
    4. Data Stale: Source data > threshold old
    5. Coverage Low: Trial coverage below threshold
    6. Duplicate Detected: Same event_id seen multiple times

    Also implements:
    - Black swan detection
    - Data quality scoring
    - Audit trail generation
    """

    def __init__(self, as_of_date: date):
        self.as_of_date = as_of_date
        self.black_swans: List[BlackSwanEvent] = []
        self.audit_log: List[Dict[str, Any]] = []

    def validate_pit_compliance(
        self,
        events: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate PIT (Point-in-Time) compliance.

        Rule: No event should have source_date > as_of_date
        """
        result = ValidationResult(is_valid=True)

        for event in events:
            source_date_str = event.get("source_date") or event.get("disclosed_at")
            if not source_date_str:
                continue

            try:
                source_date = date.fromisoformat(source_date_str)
                if source_date > self.as_of_date:
                    violation = ValidationViolation(
                        rule=ValidationRule.RULE_FUTURE_DATA_LEAK,
                        severity=ValidationSeverity.FATAL,
                        message=f"Future data detected: source_date {source_date_str} > as_of_date {self.as_of_date}",
                        context={
                            "event_id": event.get("event_id", "unknown"),
                            "source_date": source_date_str,
                            "as_of_date": self.as_of_date.isoformat(),
                        },
                        timestamp=self.as_of_date.isoformat(),
                    )
                    result.add_violation(violation)
            except (ValueError, TypeError):
                pass

        return result

    def validate_critical_fields(
        self,
        events: List[Dict[str, Any]],
    ) -> ValidationResult:
        """
        Validate critical fields are present.

        Required fields: ticker, nct_id, event_type
        """
        result = ValidationResult(is_valid=True)
        critical_fields = ["ticker", "nct_id", "event_type"]

        for event in events:
            for field in critical_fields:
                if not event.get(field):
                    violation = ValidationViolation(
                        rule=ValidationRule.RULE_CRITICAL_FIELD_MISSING,
                        severity=ValidationSeverity.FATAL,
                        message=f"Critical field missing: {field}",
                        context={
                            "event": {k: v for k, v in event.items() if k in ["ticker", "nct_id", "event_id"]},
                            "missing_field": field,
                        },
                        timestamp=self.as_of_date.isoformat(),
                    )
                    result.add_violation(violation)

        return result

    def validate_data_staleness(
        self,
        source_dates: List[str],
    ) -> ValidationResult:
        """
        Validate data is not too stale.

        WARNING at 90 days, ERROR at 180 days, FATAL at 365 days
        """
        result = ValidationResult(is_valid=True)

        for source_date_str in source_dates:
            try:
                source_date = date.fromisoformat(source_date_str)
                days_old = (self.as_of_date - source_date).days

                if days_old > STALENESS_REJECT_DAYS:
                    violation = ValidationViolation(
                        rule=ValidationRule.RULE_DATA_STALE,
                        severity=ValidationSeverity.FATAL,
                        message=f"Data critically stale: {days_old} days old (limit: {STALENESS_REJECT_DAYS})",
                        context={"source_date": source_date_str, "days_old": days_old},
                        timestamp=self.as_of_date.isoformat(),
                    )
                    result.add_violation(violation)
                elif days_old > STALENESS_CRITICAL_DAYS:
                    violation = ValidationViolation(
                        rule=ValidationRule.RULE_DATA_STALE,
                        severity=ValidationSeverity.ERROR,
                        message=f"Data stale: {days_old} days old (threshold: {STALENESS_CRITICAL_DAYS})",
                        context={"source_date": source_date_str, "days_old": days_old},
                        timestamp=self.as_of_date.isoformat(),
                    )
                    result.add_violation(violation)
                elif days_old > STALENESS_WARNING_DAYS:
                    violation = ValidationViolation(
                        rule=ValidationRule.RULE_DATA_STALE,
                        severity=ValidationSeverity.WARNING,
                        message=f"Data aging: {days_old} days old (warning: {STALENESS_WARNING_DAYS})",
                        context={"source_date": source_date_str, "days_old": days_old},
                        timestamp=self.as_of_date.isoformat(),
                    )
                    result.add_violation(violation)
            except (ValueError, TypeError):
                pass

        return result

    def validate_coverage(
        self,
        actual_trials: int,
        expected_trials: int,
    ) -> ValidationResult:
        """
        Validate trial coverage meets threshold.
        """
        result = ValidationResult(is_valid=True)

        if expected_trials <= 0:
            return result

        coverage = Decimal(actual_trials) / Decimal(expected_trials)

        if coverage < MIN_TRIAL_COVERAGE_PCT:
            violation = ValidationViolation(
                rule=ValidationRule.RULE_COVERAGE_LOW,
                severity=ValidationSeverity.ERROR,
                message=f"Trial coverage low: {coverage:.1%} (threshold: {MIN_TRIAL_COVERAGE_PCT:.0%})",
                context={
                    "actual_trials": actual_trials,
                    "expected_trials": expected_trials,
                    "coverage_pct": str(coverage),
                },
                timestamp=self.as_of_date.isoformat(),
            )
            result.add_violation(violation)

        return result

    def validate_duplicates(
        self,
        event_ids: List[str],
    ) -> ValidationResult:
        """
        Validate no duplicate event IDs.
        """
        result = ValidationResult(is_valid=True)
        seen: Set[str] = set()
        duplicates: Set[str] = set()

        for event_id in event_ids:
            if event_id in seen:
                duplicates.add(event_id)
            seen.add(event_id)

        if duplicates:
            violation = ValidationViolation(
                rule=ValidationRule.RULE_DUPLICATE_DETECTED,
                severity=ValidationSeverity.ERROR,
                message=f"Duplicate event IDs detected: {len(duplicates)} duplicates",
                context={"duplicate_ids": list(duplicates)[:10]},  # First 10
                timestamp=self.as_of_date.isoformat(),
            )
            result.add_violation(violation)

        return result

    def detect_black_swans(
        self,
        events: List[Dict[str, Any]],
    ) -> List[BlackSwanEvent]:
        """
        Detect black swan events from catalyst data.

        Looks for:
        - Clinical holds
        - Trial terminations
        - CRLs
        - Safety signals
        """
        black_swans = []

        # Keywords indicating black swan events
        hold_keywords = ["clinical hold", "fda hold", "partial hold", "full hold"]
        terminate_keywords = ["terminated", "termination", "discontinued"]
        crl_keywords = ["complete response letter", "crl", "rejection"]
        safety_keywords = ["safety signal", "adverse event", "death", "serious adverse"]

        for event in events:
            event_type = str(event.get("event_type", "")).lower()
            new_value = str(event.get("new_value", "")).lower()
            combined = f"{event_type} {new_value}"

            ticker = event.get("ticker", "UNKNOWN")
            event_date = event.get("event_date") or event.get("disclosed_at") or self.as_of_date.isoformat()

            # Check for clinical hold
            if any(kw in combined for kw in hold_keywords):
                bs = BlackSwanEvent(
                    ticker=ticker,
                    event_type=BlackSwanType.CLINICAL_HOLD,
                    event_date=event_date,
                    description=f"Clinical hold detected: {event.get('new_value', '')}",
                    severity_score=BLACK_SWAN_SEVERITY[BlackSwanType.CLINICAL_HOLD],
                    source="CTGOV",
                )
                black_swans.append(bs)

            # Check for termination
            elif any(kw in combined for kw in terminate_keywords):
                bs = BlackSwanEvent(
                    ticker=ticker,
                    event_type=BlackSwanType.TRIAL_TERMINATED,
                    event_date=event_date,
                    description=f"Trial terminated: {event.get('nct_id', '')}",
                    severity_score=BLACK_SWAN_SEVERITY[BlackSwanType.TRIAL_TERMINATED],
                    source="CTGOV",
                )
                black_swans.append(bs)

            # Check for CRL
            elif any(kw in combined for kw in crl_keywords):
                bs = BlackSwanEvent(
                    ticker=ticker,
                    event_type=BlackSwanType.CRL_RECEIVED,
                    event_date=event_date,
                    description=f"CRL received: {event.get('new_value', '')}",
                    severity_score=BLACK_SWAN_SEVERITY[BlackSwanType.CRL_RECEIVED],
                    source="INFERRED",
                )
                black_swans.append(bs)

            # Check for safety signals
            elif any(kw in combined for kw in safety_keywords):
                bs = BlackSwanEvent(
                    ticker=ticker,
                    event_type=BlackSwanType.SAFETY_SIGNAL,
                    event_date=event_date,
                    description=f"Safety signal: {event.get('new_value', '')}",
                    severity_score=BLACK_SWAN_SEVERITY[BlackSwanType.SAFETY_SIGNAL],
                    source="INFERRED",
                )
                black_swans.append(bs)

        self.black_swans.extend(black_swans)
        return black_swans

    def run_all_validations(
        self,
        events: List[Dict[str, Any]],
        expected_trial_count: int = 0,
    ) -> ValidationResult:
        """
        Run all validation rules.

        Returns combined validation result.
        """
        combined = ValidationResult(is_valid=True)

        # Rule 1 & 2: PIT compliance
        pit_result = self.validate_pit_compliance(events)
        for v in pit_result.violations:
            combined.add_violation(v)

        # Rule 3: Critical fields
        fields_result = self.validate_critical_fields(events)
        for v in fields_result.violations:
            combined.add_violation(v)

        # Rule 4: Staleness
        source_dates = [
            e.get("source_date") or e.get("disclosed_at")
            for e in events
            if e.get("source_date") or e.get("disclosed_at")
        ]
        staleness_result = self.validate_data_staleness(source_dates)
        for v in staleness_result.violations:
            combined.add_violation(v)

        # Rule 5: Coverage
        if expected_trial_count > 0:
            coverage_result = self.validate_coverage(len(events), expected_trial_count)
            for v in coverage_result.violations:
                combined.add_violation(v)

        # Rule 6: Duplicates
        event_ids = [e.get("event_id", "") for e in events if e.get("event_id")]
        dup_result = self.validate_duplicates(event_ids)
        for v in dup_result.violations:
            combined.add_violation(v)

        # Black swan detection
        black_swans = self.detect_black_swans(events)
        if black_swans:
            violation = ValidationViolation(
                rule=ValidationRule.RULE_BLACK_SWAN_DETECTED,
                severity=ValidationSeverity.ERROR,
                message=f"Black swan events detected: {len(black_swans)}",
                context={"black_swans": [bs.to_dict() for bs in black_swans]},
                timestamp=self.as_of_date.isoformat(),
            )
            combined.add_violation(violation)

        combined.validation_hash = combined.compute_hash()
        return combined

    def compute_governance_score(
        self,
        validation_result: ValidationResult,
        base_score: Decimal,
    ) -> Tuple[Decimal, Dict[str, Any]]:
        """
        Apply governance overlay to base score.

        Penalizes scores based on:
        - Number of errors/warnings
        - Black swan events
        - Data quality issues
        """
        adjustments = []
        adjusted_score = base_score

        # Fatal violations = score capped at 20
        if validation_result.n_fatal > 0:
            adjusted_score = min(adjusted_score, Decimal("20"))
            adjustments.append({
                "reason": "FATAL_VIOLATIONS",
                "penalty": str(base_score - adjusted_score),
                "n_violations": validation_result.n_fatal,
            })

        # Error violations = -5 per error (max -20)
        if validation_result.n_error > 0:
            error_penalty = min(Decimal("20"), Decimal(validation_result.n_error) * Decimal("5"))
            adjusted_score -= error_penalty
            adjustments.append({
                "reason": "ERROR_VIOLATIONS",
                "penalty": str(error_penalty),
                "n_violations": validation_result.n_error,
            })

        # Black swan events = -15 per event (max -45)
        if self.black_swans:
            bs_penalty = min(Decimal("45"), Decimal(len(self.black_swans)) * Decimal("15"))
            adjusted_score -= bs_penalty
            adjustments.append({
                "reason": "BLACK_SWAN_EVENTS",
                "penalty": str(bs_penalty),
                "n_events": len(self.black_swans),
            })

        # Warning violations = -1 per warning (max -5)
        if validation_result.n_warning > 0:
            warning_penalty = min(Decimal("5"), Decimal(validation_result.n_warning) * Decimal("1"))
            adjusted_score -= warning_penalty
            adjustments.append({
                "reason": "WARNING_VIOLATIONS",
                "penalty": str(warning_penalty),
                "n_violations": validation_result.n_warning,
            })

        # Clamp to [0, 100]
        adjusted_score = max(Decimal("0"), min(Decimal("100"), adjusted_score))

        return (
            adjusted_score.quantize(Decimal("0.01")),
            {
                "base_score": str(base_score),
                "adjusted_score": str(adjusted_score),
                "total_penalty": str(base_score - adjusted_score),
                "adjustments": adjustments,
            },
        )

    def generate_audit_record(
        self,
        ticker: str,
        validation_result: ValidationResult,
        governance_adjustment: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate complete audit record."""
        record = {
            "audit_id": hashlib.sha256(
                f"{ticker}|{self.as_of_date}".encode()
            ).hexdigest()[:16],
            "ticker": ticker,
            "as_of_date": self.as_of_date.isoformat(),
            "validation": validation_result.to_dict(),
            "governance": governance_adjustment,
            "black_swans": [bs.to_dict() for bs in self.black_swans if bs.ticker == ticker],
        }

        self.audit_log.append(record)
        return record


# =============================================================================
# FAIL-CLOSED WRAPPER
# =============================================================================

class FailClosedError(Exception):
    """Raised when fail-closed validation fails."""

    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        violations = [v.message for v in validation_result.violations if v.severity == ValidationSeverity.FATAL]
        super().__init__(f"Fail-closed validation failed with {validation_result.n_fatal} fatal violations: {violations}")


def fail_closed_validate(
    events: List[Dict[str, Any]],
    as_of_date: date,
    expected_trial_count: int = 0,
    raise_on_fatal: bool = True,
) -> ValidationResult:
    """
    Run fail-closed validation.

    If raise_on_fatal is True, raises FailClosedError on fatal violations.
    """
    engine = GovernanceEngine(as_of_date)
    result = engine.run_all_validations(events, expected_trial_count)

    if raise_on_fatal and result.n_fatal > 0:
        raise FailClosedError(result)

    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def canonical_json_dumps(obj: Any) -> str:
    """Serialize to canonical JSON."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
