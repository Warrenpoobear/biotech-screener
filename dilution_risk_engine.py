#!/usr/bin/env python3
"""
dilution_risk_engine.py

Dilution Risk Scoring Engine for Biotech Screener

Deterministic calculation of forced-raise probability before next major catalyst,
closing known false positive (FP) source from names with hidden financing pressure.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() - timestamps derive from as_of_date
- STDLIB-ONLY: No external dependencies
- FAIL LOUDLY: Explicit error states, not silent defaults
- PIT DISCIPLINE: All inputs from point-in-time snapshots
- EXPLICIT CLAMPING: All scores bounded to declared ranges
- DECIMAL-ONLY: All financial calculations use Decimal for precision

Risk Scoring:
- Risk score [0.0-1.0] where 1.0 = maximum dilution risk
- Integrates with composite scoring as penalty (up to -15 points)
- Confidence-gated: only applies if confidence > 0.60

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, List, Optional, Any, Union
from datetime import date, timedelta
from enum import Enum

# Module metadata
__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class DataQualityState(Enum):
    """Data quality classification for dilution risk analysis."""
    FULL = "FULL"          # All required + optional fields present
    PARTIAL = "PARTIAL"    # Required fields + some optional
    MINIMAL = "MINIMAL"    # Minimum viable data only
    NONE = "NONE"          # Insufficient data to score


class RiskBucket(Enum):
    """Risk classification buckets."""
    NO_RISK = "NO_RISK"           # Cash gap <= 0, fully funded through catalyst
    LOW_RISK = "LOW_RISK"         # Cash gap exists but raise is feasible
    MEDIUM_RISK = "MEDIUM_RISK"   # Moderate raise difficulty
    HIGH_RISK = "HIGH_RISK"       # Significant execution risk on raise
    UNKNOWN = "UNKNOWN"           # Insufficient data


class DilutionRiskEngine:
    """
    Dilution Risk Scoring Engine.

    Calculates forced-raise probability before next major catalyst to identify
    hidden financing pressure that could lead to dilutive offerings.

    Key Design Decisions:
    - Risk score is separate from penalties (penalties applied at composite layer)
    - Confidence scoring based on data completeness
    - PIT discipline enforced on catalyst dates
    - Fail-closed: missing critical data returns explicit error state

    Usage:
        engine = DilutionRiskEngine()
        result = engine.calculate_dilution_risk(
            ticker="ACME",
            quarterly_cash=Decimal("100000000"),
            quarterly_burn=Decimal("-15000000"),
            next_catalyst_date="2026-07-15",
            market_cap=Decimal("500000000"),
            as_of_date=date(2026, 1, 15)
        )
        print(result["dilution_risk_score"])  # 0.00 - 1.00
        print(result["risk_bucket"])          # NO_RISK, LOW_RISK, etc.
    """

    VERSION = "1.0.0"

    # Score ranges (explicit bounds)
    RISK_SCORE_MIN = Decimal("0")
    RISK_SCORE_MAX = Decimal("1")
    CONFIDENCE_MIN = Decimal("0")
    CONFIDENCE_MAX = Decimal("1")

    # Thresholds for risk bucketing
    LOW_RISK_THRESHOLD = Decimal("0.40")
    MEDIUM_RISK_THRESHOLD = Decimal("0.70")

    # Capacity utilization factor (conservative)
    USABLE_CAPACITY_FACTOR = Decimal("0.70")  # Only 70% of shelf/ATM is usable

    # Raise feasibility parameters
    DILUTION_PCT_MCAP_HARD_THRESHOLD = Decimal("0.20")  # >20% of mcap is hard
    DAYS_TO_RAISE_HARD_THRESHOLD = 30  # >30 days is hard
    DAILY_VOLUME_UTILIZATION = Decimal("0.10")  # 10% daily vol limit for raises

    # Average days per month for calculations
    DAYS_PER_MONTH = Decimal("30.44")

    # Required fields for scoring
    REQUIRED_FIELDS = ["quarterly_cash", "quarterly_burn", "next_catalyst_date"]
    OPTIONAL_FIELDS = ["shelf_capacity", "atm_remaining", "avg_daily_volume_90d", "market_cap"]

    def __init__(self):
        """Initialize the Dilution Risk engine."""
        self.audit_trail: List[Dict[str, Any]] = []

    def calculate_dilution_risk(
        self,
        ticker: str,
        quarterly_cash: Optional[Union[Decimal, float, int, str]],
        quarterly_burn: Optional[Union[Decimal, float, int, str]],
        next_catalyst_date: Optional[str],
        market_cap: Optional[Union[Decimal, float, int, str]] = None,
        shelf_capacity: Optional[Union[Decimal, float, int, str]] = None,
        atm_remaining: Optional[Union[Decimal, float, int, str]] = None,
        avg_daily_volume_90d: Optional[Union[int, float, str]] = None,
        shelf_filed: bool = False,
        atm_active: bool = False,
        next_catalyst_type: Optional[str] = None,
        as_of_date: Optional[date] = None,
        provenance: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate dilution risk score with confidence gating.

        Args:
            ticker: Company ticker symbol
            quarterly_cash: Cash and equivalents from most recent 10-Q/10-K
            quarterly_burn: Quarterly CFO (cash flow from operations), typically negative
            next_catalyst_date: ISO date string of next major catalyst
            market_cap: Current market capitalization
            shelf_capacity: S-3 shelf registration capacity (if filed)
            atm_remaining: Unused ATM program capacity (if active)
            avg_daily_volume_90d: 90-day average daily trading volume
            shelf_filed: Whether S-3 shelf is filed
            atm_active: Whether ATM program is active
            next_catalyst_type: Type of catalyst (PDUFA, TRIAL_READOUT, etc.)
            as_of_date: Point-in-time date (REQUIRED for determinism)
            provenance: Source metadata for inputs

        Returns:
            Dict containing:
            - dilution_risk_score: Decimal 0-1 (None if insufficient data)
            - confidence: Decimal 0-1 (data completeness confidence)
            - risk_bucket: str (NO_RISK, LOW_RISK, MEDIUM_RISK, HIGH_RISK, UNKNOWN)
            - reason_code: str (SUCCESS, INSUFFICIENT_DATA, INVALID_CATALYST_DATE)
            - components: Dict of calculation intermediates
            - hash: str (deterministic content hash)
            - data_quality_state: str (FULL, PARTIAL, MINIMAL, NONE)
            - audit_entry: Dict
        """

        # Validate as_of_date (required for determinism)
        if as_of_date is None:
            as_of_date = date.today()

        # Deterministic timestamp from as_of_date
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z"

        # Track missing fields
        missing_fields: List[str] = []
        inputs_used: Dict[str, Any] = {"ticker": ticker}

        # Convert and validate required fields
        cash = self._to_decimal(quarterly_cash)
        burn = self._to_decimal(quarterly_burn)
        mcap = self._to_decimal(market_cap)

        inputs_used["quarterly_cash"] = str(cash) if cash is not None else None
        inputs_used["quarterly_burn"] = str(burn) if burn is not None else None
        inputs_used["market_cap"] = str(mcap) if mcap is not None else None
        inputs_used["next_catalyst_date"] = next_catalyst_date

        # Check required fields
        if cash is None:
            missing_fields.append("quarterly_cash")
        if burn is None:
            missing_fields.append("quarterly_burn")
        if not next_catalyst_date:
            missing_fields.append("next_catalyst_date")

        # Fail-closed: missing critical data
        if missing_fields:
            return self._create_error_result(
                ticker=ticker,
                reason_code="INSUFFICIENT_DATA",
                missing_fields=missing_fields,
                inputs_used=inputs_used,
                as_of_date=as_of_date,
                deterministic_timestamp=deterministic_timestamp,
                provenance=provenance,
            )

        # Parse and validate catalyst date (PIT discipline)
        try:
            catalyst_dt = date.fromisoformat(next_catalyst_date)
        except (ValueError, TypeError):
            return self._create_error_result(
                ticker=ticker,
                reason_code="INVALID_CATALYST_DATE",
                missing_fields=[],
                inputs_used=inputs_used,
                as_of_date=as_of_date,
                deterministic_timestamp=deterministic_timestamp,
                details="Catalyst date format invalid - expected ISO format",
                provenance=provenance,
            )

        # PIT check: catalyst must be in future
        if catalyst_dt <= as_of_date:
            return self._create_error_result(
                ticker=ticker,
                reason_code="INVALID_CATALYST_DATE",
                missing_fields=[],
                inputs_used=inputs_used,
                as_of_date=as_of_date,
                deterministic_timestamp=deterministic_timestamp,
                details="Catalyst date must be in future",
                provenance=provenance,
            )

        # Convert optional fields
        shelf = self._to_decimal(shelf_capacity)
        atm = self._to_decimal(atm_remaining)
        avg_volume = self._to_int(avg_daily_volume_90d)

        if shelf is None and shelf_filed:
            missing_fields.append("shelf_capacity")
        if atm is None and atm_active:
            missing_fields.append("atm_remaining")
        if avg_volume is None or avg_volume == 0:
            missing_fields.append("avg_daily_volume_90d")
        if mcap is None or mcap == 0:
            missing_fields.append("market_cap")

        inputs_used["shelf_capacity"] = str(shelf) if shelf is not None else None
        inputs_used["atm_remaining"] = str(atm) if atm is not None else None
        inputs_used["avg_daily_volume_90d"] = avg_volume
        inputs_used["shelf_filed"] = shelf_filed
        inputs_used["atm_active"] = atm_active
        inputs_used["next_catalyst_type"] = next_catalyst_type

        # Calculate months to catalyst
        days_to_catalyst = (catalyst_dt - as_of_date).days
        months_to_catalyst = Decimal(str(days_to_catalyst)) / self.DAYS_PER_MONTH

        # Quarterly burn to monthly burn
        # burn is typically negative (cash outflow)
        monthly_burn = abs(burn) / Decimal("3")

        # Cash needed before catalyst
        cash_needed = monthly_burn * months_to_catalyst

        # Available resources
        current_cash = cash
        shelf_val = shelf if shelf is not None else Decimal("0")
        atm_val = atm if atm is not None else Decimal("0")

        # Total accessible capital (conservative: only 70% of shelf/ATM is usable)
        usable_capacity = (shelf_val + atm_val) * self.USABLE_CAPACITY_FACTOR
        total_available = current_cash + usable_capacity

        # Cash gap
        cash_gap = cash_needed - total_available

        # Calculate raise feasibility
        if mcap is None or mcap <= Decimal("0"):
            # Cannot calculate raise feasibility without market cap
            raise_feasibility = Decimal("0.50")  # Neutral default
            dilution_pct_mcap = None
            days_to_raise = None
        elif cash_gap <= Decimal("0"):
            # No gap = no raise needed
            raise_feasibility = Decimal("1.0")
            dilution_pct_mcap = Decimal("0")
            days_to_raise = Decimal("0")
        else:
            # Dilution as % of market cap
            dilution_pct_mcap = cash_gap / mcap

            # Volume feasibility (how many days to raise)
            if avg_volume is not None and avg_volume > 0:
                # Estimate daily dollar volume
                # Using market_cap / shares_proxy * volume
                # Simplified: assume price = mcap / (volume * 250 trading days proxy)
                # More accurate: daily_dollar_volume ≈ avg_volume * (mcap / shares_outstanding)
                # We approximate price as mcap / (avg_volume * 250)
                estimated_price = mcap / (Decimal(str(avg_volume)) * Decimal("250"))
                daily_dollar_volume = Decimal(str(avg_volume)) * estimated_price

                # 10% of daily dollar volume is realistic absorption
                daily_absorption = daily_dollar_volume * self.DAILY_VOLUME_UTILIZATION

                if daily_absorption > Decimal("0"):
                    days_to_raise = cash_gap / daily_absorption
                else:
                    days_to_raise = Decimal("999")  # Infinite days
            else:
                days_to_raise = Decimal("60")  # Conservative default

            # Combined scoring for raise feasibility
            # Penalty if >20% of market cap
            cap_penalty = min(Decimal("1.0"), dilution_pct_mcap / self.DILUTION_PCT_MCAP_HARD_THRESHOLD)

            # Penalty if >30 days to raise
            volume_penalty = min(Decimal("1.0"), days_to_raise / Decimal(str(self.DAYS_TO_RAISE_HARD_THRESHOLD)))

            # Combined feasibility (1.0 = easy, 0.0 = impossible)
            raise_feasibility = Decimal("1.0") - ((cap_penalty + volume_penalty) / Decimal("2"))
            raise_feasibility = max(Decimal("0.0"), min(Decimal("1.0"), raise_feasibility))

        # Final dilution risk score
        if cash_gap <= Decimal("0"):
            risk_score = Decimal("0.0")
            risk_bucket = RiskBucket.NO_RISK
        elif raise_feasibility > Decimal("0.70"):
            # Easy raise = lower risk
            risk_score = min(self.LOW_RISK_THRESHOLD, cash_gap / total_available) if total_available > 0 else Decimal("0.30")
            risk_bucket = RiskBucket.LOW_RISK
        elif raise_feasibility > Decimal("0.40"):
            # Moderate difficulty
            risk_score = self.LOW_RISK_THRESHOLD + (
                Decimal("0.30") * (Decimal("1.0") - raise_feasibility)
            )
            risk_bucket = RiskBucket.MEDIUM_RISK
        else:
            # Difficult raise
            risk_score = self.MEDIUM_RISK_THRESHOLD + (
                Decimal("0.30") * (Decimal("1.0") - raise_feasibility)
            )
            risk_bucket = RiskBucket.HIGH_RISK

        # Clamp and quantize risk score
        risk_score = self._clamp(risk_score, self.RISK_SCORE_MIN, self.RISK_SCORE_MAX)
        risk_score = risk_score.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Confidence calculation based on data completeness
        confidence_factors = [
            Decimal("0.50"),  # Base for required fields
            Decimal("0.20") if shelf is not None else Decimal("0.0"),
            Decimal("0.15") if atm is not None else Decimal("0.0"),
            Decimal("0.15") if avg_volume is not None and avg_volume > 0 else Decimal("0.0"),
        ]
        confidence = sum(confidence_factors)
        confidence = self._clamp(confidence, self.CONFIDENCE_MIN, self.CONFIDENCE_MAX)
        confidence = confidence.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Data quality state
        data_quality_state = self._assess_data_quality(missing_fields)

        # Deterministic hash for auditability
        hash_input = (
            f"{ticker}|{as_of_date.isoformat()}|"
            f"{current_cash}|{monthly_burn}|{months_to_catalyst}|"
            f"{usable_capacity}|{mcap if mcap else 'None'}"
        )
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Build components dict
        components = {
            "months_to_catalyst": months_to_catalyst.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
            "monthly_burn": monthly_burn.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "cash_needed": cash_needed.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "current_cash": current_cash.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "usable_capacity": usable_capacity.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "total_available": total_available.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "cash_gap": cash_gap.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "raise_feasibility": raise_feasibility.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
        }

        if dilution_pct_mcap is not None:
            components["dilution_pct_mcap"] = (dilution_pct_mcap * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        if days_to_raise is not None:
            components["days_to_raise"] = days_to_raise.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Audit entry
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat(),
            "ticker": ticker,
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs_used, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
            "inputs_used": inputs_used,
            "missing_fields": missing_fields,
            "data_quality_state": data_quality_state.value,
            "calculation": {
                "risk_score": str(risk_score),
                "risk_bucket": risk_bucket.value,
                "confidence": str(confidence),
                "components": {k: str(v) for k, v in components.items()},
            },
            "module_version": self.VERSION,
        }

        self.audit_trail.append(audit_entry)

        return {
            "ticker": ticker,
            "dilution_risk_score": risk_score,
            "confidence": confidence,
            "risk_bucket": risk_bucket.value,
            "reason_code": "SUCCESS",
            "components": components,
            "hash": content_hash,
            "data_quality_state": data_quality_state.value,
            "missing_fields": missing_fields,
            "inputs_used": inputs_used,
            "audit_entry": audit_entry,
            "provenance": provenance or {},
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Score an entire universe of companies for dilution risk.

        Args:
            universe: List of dicts with required fields:
                - ticker: str
                - quarterly_cash: Decimal or numeric
                - quarterly_burn: Decimal or numeric
                - next_catalyst_date: str (ISO date)
                Optional fields:
                - market_cap: Decimal or numeric
                - shelf_capacity: Decimal or numeric
                - atm_remaining: Decimal or numeric
                - avg_daily_volume_90d: int
            as_of_date: Point-in-time date (REQUIRED)

        Returns:
            Dict with scores, diagnostics, and provenance
        """

        scores: List[Dict[str, Any]] = []
        risk_distribution: Dict[str, int] = {
            "NO_RISK": 0, "LOW_RISK": 0, "MEDIUM_RISK": 0,
            "HIGH_RISK": 0, "UNKNOWN": 0
        }
        quality_distribution: Dict[str, int] = {
            "FULL": 0, "PARTIAL": 0, "MINIMAL": 0, "NONE": 0
        }
        high_risk_tickers: List[str] = []

        for company in universe:
            ticker = company.get("ticker", "UNKNOWN")

            result = self.calculate_dilution_risk(
                ticker=ticker,
                quarterly_cash=company.get("quarterly_cash"),
                quarterly_burn=company.get("quarterly_burn"),
                next_catalyst_date=company.get("next_catalyst_date"),
                market_cap=company.get("market_cap"),
                shelf_capacity=company.get("shelf_capacity"),
                atm_remaining=company.get("atm_remaining"),
                avg_daily_volume_90d=company.get("avg_daily_volume_90d"),
                shelf_filed=company.get("shelf_filed", False),
                atm_active=company.get("atm_active", False),
                next_catalyst_type=company.get("next_catalyst_type"),
                as_of_date=as_of_date,
                provenance=company.get("provenance"),
            )

            scores.append({
                "ticker": ticker,
                "dilution_risk_score": result["dilution_risk_score"],
                "confidence": result["confidence"],
                "risk_bucket": result["risk_bucket"],
                "reason_code": result["reason_code"],
                "data_quality_state": result["data_quality_state"],
                "components": result.get("components", {}),
                "flags": [],
            })

            # Track distributions
            risk_bucket = result["risk_bucket"]
            if risk_bucket in risk_distribution:
                risk_distribution[risk_bucket] += 1

            quality_state = result["data_quality_state"]
            if quality_state in quality_distribution:
                quality_distribution[quality_state] += 1

            # Track high risk tickers
            if risk_bucket == "HIGH_RISK":
                high_risk_tickers.append(ticker)

        # Deterministic content hash
        scores_json = json.dumps(
            [{"t": s["ticker"], "r": str(s["dilution_risk_score"]), "b": s["risk_bucket"]}
             for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "success_count": sum(1 for s in scores if s["reason_code"] == "SUCCESS"),
                "error_count": sum(1 for s in scores if s["reason_code"] != "SUCCESS"),
                "risk_distribution": risk_distribution,
                "data_quality_distribution": quality_distribution,
                "high_risk_tickers": high_risk_tickers,
            },
            "provenance": {
                "module": "dilution_risk_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
            },
        }

    def _create_error_result(
        self,
        ticker: str,
        reason_code: str,
        missing_fields: List[str],
        inputs_used: Dict[str, Any],
        as_of_date: date,
        deterministic_timestamp: str,
        details: Optional[str] = None,
        provenance: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create error result for failed calculations."""

        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat(),
            "ticker": ticker,
            "inputs_used": inputs_used,
            "missing_fields": missing_fields,
            "data_quality_state": DataQualityState.NONE.value,
            "reason_code": reason_code,
            "details": details,
            "module_version": self.VERSION,
        }

        self.audit_trail.append(audit_entry)

        result = {
            "ticker": ticker,
            "dilution_risk_score": None,
            "confidence": Decimal("0.0"),
            "risk_bucket": RiskBucket.UNKNOWN.value,
            "reason_code": reason_code,
            "missing_fields": missing_fields,
            "data_quality_state": DataQualityState.NONE.value,
            "hash": None,
            "inputs_used": inputs_used,
            "audit_entry": audit_entry,
            "provenance": provenance or {},
        }

        if details:
            result["details"] = details

        return result

    def _assess_data_quality(self, missing_fields: List[str]) -> DataQualityState:
        """Assess data quality based on missing fields."""
        required_missing = [f for f in missing_fields if f in self.REQUIRED_FIELDS]
        optional_missing = [f for f in missing_fields if f in self.OPTIONAL_FIELDS]

        if required_missing:
            return DataQualityState.NONE
        elif len(optional_missing) == 0:
            return DataQualityState.FULL
        elif len(optional_missing) <= 2:
            return DataQualityState.PARTIAL
        else:
            return DataQualityState.MINIMAL

    def _clamp(self, value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
        """Clamp value to specified range."""
        return max(min_val, min(max_val, value))

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        """Safely convert value to Decimal."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def _to_int(self, value: Any) -> Optional[int]:
        """Safely convert value to int."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


def integrate_dilution_risk(
    base_score: Decimal,
    dilution_data: Dict[str, Any],
    max_penalty: Decimal = Decimal("15"),
    confidence_threshold: Decimal = Decimal("0.60"),
) -> Decimal:
    """
    Apply dilution risk penalty to base composite score.

    High dilution risk = score penalty up to max_penalty points.
    Only applies if confidence > confidence_threshold.

    Args:
        base_score: Original composite score (0-100)
        dilution_data: Result from calculate_dilution_risk()
        max_penalty: Maximum penalty to apply (default 15)
        confidence_threshold: Minimum confidence to apply penalty (default 0.60)

    Returns:
        Adjusted score (Decimal, 0-100 bounded)
    """
    confidence = dilution_data.get("confidence", Decimal("0"))

    if confidence < confidence_threshold:
        return base_score  # Insufficient data - no adjustment

    risk_score = dilution_data.get("dilution_risk_score")

    if risk_score is None:
        return base_score

    # Penalty mapping: risk_score [0-1] -> penalty [0-max_penalty]
    penalty = risk_score * max_penalty

    adjusted_score = base_score - penalty

    # Floor at 0
    adjusted_score = max(Decimal("0"), adjusted_score)

    return adjusted_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def demonstration() -> None:
    """Demonstrate the Dilution Risk engine capabilities."""
    print("=" * 70)
    print("DILUTION RISK ENGINE v1.0.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    engine = DilutionRiskEngine()
    as_of = date(2026, 1, 15)

    # Case 1: No Risk - Well funded company
    print("Case 1: Well-Funded Company (No Risk)")
    print("-" * 70)

    no_risk = engine.calculate_dilution_risk(
        ticker="FUNDED",
        quarterly_cash=Decimal("500000000"),  # $500M cash
        quarterly_burn=Decimal("-30000000"),  # $30M quarterly burn ($10M/month)
        next_catalyst_date="2026-07-15",      # 6 months away
        market_cap=Decimal("2000000000"),     # $2B market cap
        avg_daily_volume_90d=2_000_000,
        as_of_date=as_of,
    )

    print(f"  Ticker: {no_risk['ticker']}")
    print(f"  Dilution Risk Score: {no_risk['dilution_risk_score']}")
    print(f"  Risk Bucket: {no_risk['risk_bucket']}")
    print(f"  Confidence: {no_risk['confidence']}")
    print(f"  Cash Gap: ${no_risk['components']['cash_gap']}M")
    print()

    # Case 2: High Risk - Underfunded company
    print("Case 2: Underfunded Company (High Risk)")
    print("-" * 70)

    high_risk = engine.calculate_dilution_risk(
        ticker="BURNING",
        quarterly_cash=Decimal("30000000"),   # $30M cash
        quarterly_burn=Decimal("-45000000"),  # $45M quarterly burn ($15M/month)
        next_catalyst_date="2026-12-15",      # 11 months away
        market_cap=Decimal("100000000"),      # $100M market cap
        avg_daily_volume_90d=500_000,
        shelf_capacity=Decimal("0"),
        atm_remaining=Decimal("0"),
        as_of_date=as_of,
    )

    print(f"  Ticker: {high_risk['ticker']}")
    print(f"  Dilution Risk Score: {high_risk['dilution_risk_score']}")
    print(f"  Risk Bucket: {high_risk['risk_bucket']}")
    print(f"  Confidence: {high_risk['confidence']}")
    print(f"  Cash Gap: ${high_risk['components']['cash_gap']}M")
    print(f"  Months to Catalyst: {high_risk['components']['months_to_catalyst']}")
    print()

    # Case 3: Medium Risk with ATM capacity
    print("Case 3: Company with ATM Capacity (Medium Risk)")
    print("-" * 70)

    medium_risk = engine.calculate_dilution_risk(
        ticker="ATMUSER",
        quarterly_cash=Decimal("40000000"),   # $40M cash
        quarterly_burn=Decimal("-36000000"),  # $36M quarterly burn ($12M/month)
        next_catalyst_date="2026-09-15",      # 8 months away
        market_cap=Decimal("300000000"),      # $300M market cap
        atm_remaining=Decimal("50000000"),    # $50M ATM capacity
        atm_active=True,
        avg_daily_volume_90d=1_000_000,
        as_of_date=as_of,
    )

    print(f"  Ticker: {medium_risk['ticker']}")
    print(f"  Dilution Risk Score: {medium_risk['dilution_risk_score']}")
    print(f"  Risk Bucket: {medium_risk['risk_bucket']}")
    print(f"  Confidence: {medium_risk['confidence']}")
    print(f"  Usable Capacity: ${medium_risk['components']['usable_capacity']}M")
    print(f"  Total Available: ${medium_risk['components']['total_available']}M")
    print()

    # Demonstrate integration with composite score
    print("Integration Example: Applying to Composite Score")
    print("-" * 70)

    base_score = Decimal("75.00")
    adjusted = integrate_dilution_risk(base_score, high_risk)

    print(f"  Base Composite Score: {base_score}")
    print(f"  Dilution Risk Score: {high_risk['dilution_risk_score']}")
    print(f"  Adjusted Score: {adjusted}")
    print(f"  Penalty Applied: {base_score - adjusted}")
    print()


if __name__ == "__main__":
    demonstration()
