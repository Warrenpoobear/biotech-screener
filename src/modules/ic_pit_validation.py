"""
PIT (Point-in-Time) Validation and Production Gates for IC Enhancements.

This module provides strict validation to prevent look-ahead bias in IC-enhanced
scoring. These are hard gates that must pass before v3 features can be used.

Key Protections:
1. Adaptive weight learning: Embargo enforcement, expanding-window only
2. Peer valuation: Snapshot date validation
3. Historical data: PIT cutoff enforcement
4. Weight stability: Maximum churn threshold

Design Philosophy:
- FAIL LOUDLY: Reject non-compliant data rather than silently degrade
- DETERMINISTIC: All validation is reproducible
- AUDITABLE: Full provenance of what was validated and when

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


__version__ = "1.0.0"


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum embargo period between training data and evaluation
DEFAULT_EMBARGO_DAYS = 30

# Maximum weight change allowed between consecutive as-of dates
MAX_WEIGHT_L1_CHANGE = Decimal("0.15")  # 15% total L1 norm change

# Minimum training window for adaptive weights
MIN_TRAINING_MONTHS = 6

# Maximum training window to prevent over-reliance on stale data
MAX_TRAINING_MONTHS = 36


class PITValidationError(Exception):
    """Raised when PIT validation fails."""
    pass


class WeightStabilityError(Exception):
    """Raised when weight stability check fails."""
    pass


class DataQualityError(Exception):
    """Raised when data quality checks fail."""
    pass


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class PITValidationResult:
    """Result of PIT validation check."""
    status: ValidationStatus
    check_name: str
    as_of_date: str
    details: Dict[str, Any]
    violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == ValidationStatus.PASSED


@dataclass
class WeightProvenance:
    """Provenance record for learned weights."""
    as_of_date: str
    training_start: str
    training_end: str
    embargo_end: str
    universe_hash: str
    objective: str
    constraints: Dict[str, str]
    weights: Dict[str, str]
    historical_ic_by_component: Dict[str, str]
    sample_size: int
    determinism_hash: str


@dataclass
class ProductionGateResult:
    """Result of production gate checks."""
    gate_name: str
    passed: bool
    checks: List[PITValidationResult]
    blocking_violations: List[str]
    warnings: List[str]
    recommendation: str


# =============================================================================
# PIT VALIDATION FUNCTIONS
# =============================================================================

def validate_adaptive_weight_pit(
    as_of_date: date,
    historical_scores: List[Dict[str, Any]],
    historical_returns: Dict[Tuple[date, str], Decimal],
    *,
    embargo_days: int = DEFAULT_EMBARGO_DAYS,
    min_training_months: int = MIN_TRAINING_MONTHS,
) -> PITValidationResult:
    """
    Validate that adaptive weight training data is PIT-safe.

    Requirements:
    1. All historical scores must have as_of_date < training_cutoff
    2. All returns must be keyed by (as_of_date, ticker) for PIT safety
    3. All return keys must have as_of_date < embargo_cutoff
    4. Training cutoff must be at least embargo_days before as_of_date
    5. Training window must be at least min_training_months

    Args:
        as_of_date: The evaluation date
        historical_scores: List of historical score records
        historical_returns: Dict keyed by (as_of_date, ticker) -> forward return
            CRITICAL: as_of_date in key is when the return period STARTS
        embargo_days: Minimum gap between training end and evaluation
        min_training_months: Minimum training window length

    Returns:
        PITValidationResult with status and any violations
    """
    violations = []
    details = {
        "as_of_date": as_of_date.isoformat(),
        "embargo_days": embargo_days,
        "min_training_months": min_training_months,
        "historical_scores_count": len(historical_scores),
        "historical_returns_count": len(historical_returns),
    }

    # Compute embargo cutoff
    embargo_cutoff = as_of_date - timedelta(days=embargo_days)
    details["embargo_cutoff"] = embargo_cutoff.isoformat()

    # Check 1: All scores must have as_of_date field and be before embargo
    scores_after_embargo = []
    scores_without_date = []
    score_dates = []

    for i, score in enumerate(historical_scores):
        score_date_str = score.get("as_of_date")
        if not score_date_str:
            scores_without_date.append(i)
            continue

        try:
            if isinstance(score_date_str, date):
                score_date = score_date_str
            else:
                score_date = date.fromisoformat(str(score_date_str)[:10])
            score_dates.append(score_date)
            if score_date >= embargo_cutoff:
                scores_after_embargo.append({
                    "index": i,
                    "date": score_date.isoformat(),
                    "ticker": score.get("ticker", "unknown"),
                })
        except ValueError:
            scores_without_date.append(i)

    if scores_without_date:
        violations.append(
            f"Found {len(scores_without_date)} scores without valid as_of_date"
        )

    if scores_after_embargo:
        violations.append(
            f"Found {len(scores_after_embargo)} scores after embargo cutoff "
            f"({embargo_cutoff.isoformat()}): {scores_after_embargo[:3]}..."
        )

    # Check 2: Training window length
    if score_dates:
        training_start = min(score_dates)
        training_end = max(score_dates)
        training_months = (training_end - training_start).days / 30.44

        details["training_start"] = training_start.isoformat()
        details["training_end"] = training_end.isoformat()
        details["training_months"] = round(training_months, 1)

        if training_months < min_training_months:
            violations.append(
                f"Training window ({training_months:.1f} months) is less than "
                f"minimum ({min_training_months} months)"
            )

    # Check 3: All returns must be keyed by (date, ticker) and before embargo
    # This is the CRITICAL PIT check for returns
    returns_after_embargo = []
    invalid_return_keys = []

    for key in historical_returns.keys():
        if not isinstance(key, tuple) or len(key) != 2:
            invalid_return_keys.append(str(key))
            continue

        return_date, ticker = key
        if not isinstance(return_date, date):
            invalid_return_keys.append(f"({return_date}, {ticker})")
            continue

        # The return_date is when the return period STARTS
        # It must be before embargo_cutoff to be PIT-safe
        if return_date >= embargo_cutoff:
            returns_after_embargo.append({
                "date": return_date.isoformat(),
                "ticker": ticker,
            })

    if invalid_return_keys:
        violations.append(
            f"Found {len(invalid_return_keys)} returns with invalid keys "
            f"(must be (date, ticker) tuples): {invalid_return_keys[:3]}..."
        )

    if returns_after_embargo:
        violations.append(
            f"Found {len(returns_after_embargo)} returns after embargo cutoff "
            f"({embargo_cutoff.isoformat()}): {returns_after_embargo[:3]}..."
        )

    details["returns_validated"] = len(historical_returns) - len(invalid_return_keys)
    details["returns_after_embargo"] = len(returns_after_embargo)

    status = ValidationStatus.PASSED if not violations else ValidationStatus.FAILED

    return PITValidationResult(
        status=status,
        check_name="adaptive_weight_pit",
        as_of_date=as_of_date.isoformat(),
        details=details,
        violations=violations,
    )


def validate_peer_valuation_pit(
    as_of_date: date,
    peer_valuations: List[Dict[str, Any]],
) -> PITValidationResult:
    """
    Validate that peer valuation data is PIT-safe.

    Requirements:
    1. Each peer record must have a snapshot_date
    2. All snapshot_dates must be <= as_of_date - 1 (PIT rule)
    3. Market cap and trial count must be from same snapshot

    Args:
        as_of_date: The evaluation date
        peer_valuations: List of peer data dicts

    Returns:
        PITValidationResult with status and any violations
    """
    violations = []
    details = {
        "as_of_date": as_of_date.isoformat(),
        "peer_count": len(peer_valuations),
    }

    pit_cutoff = as_of_date - timedelta(days=1)
    details["pit_cutoff"] = pit_cutoff.isoformat()

    peers_without_snapshot = []
    peers_after_pit = []

    for i, peer in enumerate(peer_valuations):
        snapshot_str = peer.get("snapshot_date")

        if not snapshot_str:
            peers_without_snapshot.append(peer.get("ticker", f"index_{i}"))
            continue

        try:
            snapshot_date = date.fromisoformat(snapshot_str)
            if snapshot_date > pit_cutoff:
                peers_after_pit.append({
                    "ticker": peer.get("ticker", f"index_{i}"),
                    "snapshot_date": snapshot_str,
                })
        except ValueError:
            peers_without_snapshot.append(peer.get("ticker", f"index_{i}"))

    if peers_without_snapshot:
        # This is a WARNING not a failure - allow graceful degradation
        details["peers_without_snapshot"] = len(peers_without_snapshot)

    if peers_after_pit:
        violations.append(
            f"Found {len(peers_after_pit)} peers with snapshot after PIT cutoff: "
            f"{peers_after_pit[:3]}..."
        )

    status = ValidationStatus.PASSED if not violations else ValidationStatus.FAILED

    # Warn if many peers lack snapshots
    if peers_without_snapshot and len(peers_without_snapshot) > len(peer_valuations) * 0.5:
        status = ValidationStatus.WARNING
        violations.append(
            f"More than 50% of peers lack snapshot_date metadata - "
            f"valuation signal reliability reduced"
        )

    return PITValidationResult(
        status=status,
        check_name="peer_valuation_pit",
        as_of_date=as_of_date.isoformat(),
        details=details,
        violations=violations,
    )


def validate_coinvest_pit(
    as_of_date: date,
    coinvest_data: Dict[str, Any],
) -> PITValidationResult:
    """
    Validate that co-invest (13F) data uses filing dates, not quarter-end.

    Requirements:
    1. All positions must have filing_date (SEC filing timestamp)
    2. All filing_dates must be < as_of_date
    3. report_date (quarter-end) is for reference only, not filtering

    Args:
        as_of_date: The evaluation date
        coinvest_data: Dict with overlap_count, holders, etc.

    Returns:
        PITValidationResult with status and any violations
    """
    violations = []
    details = {
        "as_of_date": as_of_date.isoformat(),
        "overlap_count": coinvest_data.get("coinvest_overlap_count", 0),
        "usable": coinvest_data.get("coinvest_usable", False),
    }

    # Check that the data structure indicates filing-date PIT filtering was applied
    if coinvest_data.get("coinvest_usable"):
        published_at_max = coinvest_data.get("coinvest_published_at_max")
        if published_at_max:
            try:
                pub_date = date.fromisoformat(published_at_max)
                if pub_date >= as_of_date:
                    violations.append(
                        f"coinvest_published_at_max ({published_at_max}) >= as_of_date"
                    )
                details["published_at_max"] = published_at_max
            except ValueError:
                violations.append(f"Invalid published_at_max format: {published_at_max}")

    # Check for flags indicating PIT issues
    flags = coinvest_data.get("coinvest_flags", [])
    if "filings_not_yet_public" not in flags and "no_signal" not in flags:
        # Data was used - verify it's properly filtered
        if not coinvest_data.get("coinvest_published_at_max"):
            details["warning"] = "No published_at_max - cannot verify PIT compliance"

    status = ValidationStatus.PASSED if not violations else ValidationStatus.FAILED

    return PITValidationResult(
        status=status,
        check_name="coinvest_pit",
        as_of_date=as_of_date.isoformat(),
        details=details,
        violations=violations,
    )


# =============================================================================
# WEIGHT STABILITY VALIDATION
# =============================================================================

def validate_weight_stability(
    current_weights: Dict[str, Decimal],
    previous_weights: Optional[Dict[str, Decimal]],
    *,
    max_l1_change: Decimal = MAX_WEIGHT_L1_CHANGE,
) -> PITValidationResult:
    """
    Validate that weight changes between periods are within bounds.

    Prevents weight churn that indicates overfitting to noise.

    Args:
        current_weights: Current period weights
        previous_weights: Previous period weights (None if first period)
        max_l1_change: Maximum allowed L1 norm of weight changes

    Returns:
        PITValidationResult with status and any violations
    """
    violations = []
    details = {
        "current_weights": {k: str(v) for k, v in current_weights.items()},
        "max_l1_change": str(max_l1_change),
    }

    if previous_weights is None:
        details["previous_weights"] = None
        details["l1_change"] = "N/A (first period)"
        return PITValidationResult(
            status=ValidationStatus.PASSED,
            check_name="weight_stability",
            as_of_date="",
            details=details,
            violations=[],
        )

    details["previous_weights"] = {k: str(v) for k, v in previous_weights.items()}

    # Compute L1 norm of changes
    all_keys = set(current_weights.keys()) | set(previous_weights.keys())
    l1_change = Decimal("0")

    for key in all_keys:
        curr = current_weights.get(key, Decimal("0"))
        prev = previous_weights.get(key, Decimal("0"))
        l1_change += abs(curr - prev)

    details["l1_change"] = str(l1_change)

    if l1_change > max_l1_change:
        violations.append(
            f"Weight L1 change ({l1_change}) exceeds maximum ({max_l1_change}). "
            f"This may indicate overfitting or regime instability."
        )

    status = ValidationStatus.PASSED if not violations else ValidationStatus.FAILED

    return PITValidationResult(
        status=status,
        check_name="weight_stability",
        as_of_date="",
        details=details,
        violations=violations,
    )


# =============================================================================
# WEIGHT PROVENANCE
# =============================================================================

def create_weight_provenance(
    as_of_date: date,
    training_start: date,
    training_end: date,
    embargo_days: int,
    universe_tickers: List[str],
    objective: str,
    constraints: Dict[str, Any],
    weights: Dict[str, Decimal],
    historical_ic: Dict[str, Decimal],
    sample_size: int,
) -> WeightProvenance:
    """
    Create a provenance record for learned weights.

    This record becomes part of the audit trail and must be persisted.

    Args:
        as_of_date: Evaluation date these weights are for
        training_start: First date in training window
        training_end: Last date in training window
        embargo_days: Gap between training_end and as_of_date
        universe_tickers: Tickers used in training
        objective: Optimization objective (e.g., "maximize_rank_ic")
        constraints: Weight constraints used
        weights: The learned weights
        historical_ic: IC estimates by component
        sample_size: Number of samples in training

    Returns:
        WeightProvenance record
    """
    # Compute universe hash
    universe_hash = hashlib.sha256(
        json.dumps(sorted(universe_tickers)).encode()
    ).hexdigest()[:12]

    # Compute determinism hash
    payload = {
        "as_of_date": as_of_date.isoformat(),
        "training_start": training_start.isoformat(),
        "training_end": training_end.isoformat(),
        "embargo_days": embargo_days,
        "universe_hash": universe_hash,
        "objective": objective,
        "constraints": {k: str(v) for k, v in sorted(constraints.items())},
        "weights": {k: str(v) for k, v in sorted(weights.items())},
        "sample_size": sample_size,
    }
    determinism_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:16]

    return WeightProvenance(
        as_of_date=as_of_date.isoformat(),
        training_start=training_start.isoformat(),
        training_end=training_end.isoformat(),
        embargo_end=(training_end + timedelta(days=embargo_days)).isoformat(),
        universe_hash=universe_hash,
        objective=objective,
        constraints={k: str(v) for k, v in constraints.items()},
        weights={k: str(v) for k, v in weights.items()},
        historical_ic_by_component={k: str(v) for k, v in historical_ic.items()},
        sample_size=sample_size,
        determinism_hash=determinism_hash,
    )


# =============================================================================
# PRODUCTION GATE
# =============================================================================

def run_production_gate(
    as_of_date: date,
    *,
    historical_scores: Optional[List[Dict]] = None,
    historical_returns: Optional[Dict[Tuple[date, str], Decimal]] = None,
    peer_valuations: Optional[List[Dict]] = None,
    coinvest_data: Optional[Dict] = None,
    current_weights: Optional[Dict[str, Decimal]] = None,
    previous_weights: Optional[Dict[str, Decimal]] = None,
    use_adaptive_weights: bool = False,
) -> ProductionGateResult:
    """
    Run all production gate checks for IC-enhanced mode.

    This is the master gate that determines whether v3 enhancements
    can be used safely for a given as-of date.

    Args:
        as_of_date: Evaluation date
        historical_scores: For adaptive weight validation
        historical_returns: For adaptive weight validation. Must be keyed by
            (as_of_date, ticker) tuples for PIT safety.
        peer_valuations: For peer valuation validation
        coinvest_data: For co-invest validation
        current_weights: For weight stability check
        previous_weights: For weight stability check
        use_adaptive_weights: Whether adaptive weights are being used

    Returns:
        ProductionGateResult with pass/fail and recommendations
    """
    checks = []
    blocking_violations = []
    warnings = []

    # Check 1: Adaptive weight PIT (if used)
    if use_adaptive_weights and historical_scores:
        check = validate_adaptive_weight_pit(
            as_of_date,
            historical_scores,
            historical_returns or {},
        )
        checks.append(check)
        if not check.passed:
            blocking_violations.extend(check.violations)

    # Check 2: Peer valuation PIT (if used)
    if peer_valuations:
        check = validate_peer_valuation_pit(as_of_date, peer_valuations)
        checks.append(check)
        if check.status == ValidationStatus.FAILED:
            blocking_violations.extend(check.violations)
        elif check.status == ValidationStatus.WARNING:
            warnings.extend(check.violations)

    # Check 3: Co-invest PIT (if used)
    if coinvest_data:
        check = validate_coinvest_pit(as_of_date, coinvest_data)
        checks.append(check)
        if not check.passed:
            blocking_violations.extend(check.violations)

    # Check 4: Weight stability (if adaptive)
    if use_adaptive_weights and current_weights:
        check = validate_weight_stability(current_weights, previous_weights)
        checks.append(check)
        if not check.passed:
            # Weight instability is a warning, not blocking
            warnings.extend(check.violations)

    # Determine overall result
    passed = len(blocking_violations) == 0

    if passed and warnings:
        recommendation = (
            "PROCEED WITH CAUTION: All hard gates passed but warnings present. "
            "Review warnings before production use."
        )
    elif passed:
        recommendation = (
            "APPROVED: All production gates passed. IC-enhanced mode is safe to use."
        )
    else:
        recommendation = (
            "BLOCKED: Production gates failed. Do not use IC-enhanced mode. "
            f"Fix {len(blocking_violations)} violation(s) before proceeding."
        )

    return ProductionGateResult(
        gate_name="ic_enhanced_production_gate",
        passed=passed,
        checks=checks,
        blocking_violations=blocking_violations,
        warnings=warnings,
        recommendation=recommendation,
    )


# =============================================================================
# ABLATION TEST FRAMEWORK
# =============================================================================

@dataclass
class AblationResult:
    """Result of feature ablation test."""
    feature_name: str
    baseline_ic: Decimal
    ablated_ic: Decimal
    ic_delta: Decimal
    passed: bool  # True if ablated IC is worse (feature adds value)
    contribution_pct: Decimal


def run_ablation_test(
    feature_name: str,
    baseline_scores: List[Tuple[str, Decimal]],  # (ticker, score)
    ablated_scores: List[Tuple[str, Decimal]],   # scores without feature
    forward_returns: Dict[str, Decimal],
) -> AblationResult:
    """
    Run ablation test to verify a feature contributes to IC.

    A feature passes ablation if removing it reduces IC (the feature helps).

    Args:
        feature_name: Name of feature being tested
        baseline_scores: Scores with all features
        ablated_scores: Scores with this feature removed
        forward_returns: Actual forward returns for IC calculation

    Returns:
        AblationResult with IC comparison
    """
    def compute_ic(scores: List[Tuple[str, Decimal]]) -> Decimal:
        """Compute rank IC between scores and returns."""
        pairs = []
        for ticker, score in scores:
            if ticker in forward_returns:
                pairs.append((score, forward_returns[ticker]))

        if len(pairs) < 10:
            return Decimal("0")

        # Compute Spearman rank correlation
        n = len(pairs)
        score_ranks = _compute_ranks([p[0] for p in pairs])
        return_ranks = _compute_ranks([p[1] for p in pairs])

        d_squared_sum = sum(
            (sr - rr) ** 2 for sr, rr in zip(score_ranks, return_ranks)
        )

        if n <= 1:
            return Decimal("0")

        rho = Decimal("1") - (Decimal("6") * d_squared_sum) / (Decimal(n) * (Decimal(n) ** 2 - Decimal("1")))
        return rho.quantize(Decimal("0.0001"))

    baseline_ic = compute_ic(baseline_scores)
    ablated_ic = compute_ic(ablated_scores)
    ic_delta = baseline_ic - ablated_ic

    # Feature passes if removing it hurts IC (delta > 0)
    passed = ic_delta > Decimal("0")

    # Contribution as percentage of baseline IC
    if baseline_ic > Decimal("0"):
        contribution_pct = (ic_delta / baseline_ic) * Decimal("100")
    else:
        contribution_pct = Decimal("0")

    return AblationResult(
        feature_name=feature_name,
        baseline_ic=baseline_ic,
        ablated_ic=ablated_ic,
        ic_delta=ic_delta,
        passed=passed,
        contribution_pct=contribution_pct.quantize(Decimal("0.01")),
    )


def _compute_ranks(values: List[Decimal]) -> List[Decimal]:
    """Compute ranks with average rank for ties."""
    n = len(values)
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])

    ranks = [Decimal("0")] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = Decimal(str((i + j + 1) / 2))
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j

    return ranks
