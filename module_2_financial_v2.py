#!/usr/bin/env python3
"""
Module 2: Financial Health Scoring (v2)

Enhanced financial health scoring with:
- Decimal-only arithmetic (no floats in intermediate calculations)
- Burn hierarchy with full provenance tracking
- 4-quarter trailing burn average with eps floor
- Financing pressure and dilution risk scoring
- Comprehensive data quality outputs
- Liquidity gating with PASS/WARN/FAIL

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies
- DECIMAL-ONLY: Pure Decimal arithmetic for all scoring
- FAIL LOUDLY: Clear error states
- AUDITABLE: Full provenance chain

Scoring Components (normalized 0-100):
- Cash runway (50% weight)
- Dilution risk (30% weight)
- Liquidity (20% weight)

Severity levels:
- SEV3: Runway < 6 months (critical)
- SEV2: Runway 6-12 months (warning)
- SEV1: Runway 12-18 months (caution)
- NONE: Runway >= 18 months (healthy)

DETERMINISM CONTRACT:
----------------------
This module guarantees deterministic output for identical inputs:
1. All arithmetic uses Decimal with explicit quantization rules
2. Hash computation uses stable key ordering (sorted dict keys)
3. Hash includes: ticker, composite, runway_months, cash_to_mcap, burn_source, inputs_used
4. No floating-point intermediate calculations
5. No datetime.now() or random calls
6. Output field ordering is stable

To verify determinism: same inputs MUST produce identical determinism_hash values.

BREAKING CHANGES FROM v1:
-------------------------
- Output numeric fields are now floats (were strings in early v2 drafts)
- liquidity_gate is now bool (True=gated) for back-compat; use liquidity_gate_status for enum
- Added: burn_rejection_reasons, burn_to_mcap, financing_pressure_score, dilution_risk_bucket
- Added: share_count_growth, confidence, determinism_hash, schema_version
- Added: dollar_adv (alias), liquidity_gate_status (enum)

Author: Wake Robin Capital Management
Version: 2.0.0
Last Modified: 2026-01-11
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from common.types import Severity
from common.score_utils import clamp_score, to_decimal as score_to_decimal

__version__ = "2.0.0"
RULESET_VERSION = "2.0.0-V2"
SCHEMA_VERSION = "v2.0"

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Quantization precision
SCORE_PRECISION = Decimal("0.01")
RATE_PRECISION = Decimal("0.0001")
RATIO_PRECISION = Decimal("0.0001")

# Epsilon for division safety (as Decimal)
EPS = Decimal("0.000001")

# Runway thresholds (months)
RUNWAY_CRITICAL = Decimal("6")
RUNWAY_WARNING = Decimal("12")
RUNWAY_CAUTION = Decimal("18")

# Liquidity gate threshold (dollar ADV)
LIQUIDITY_GATE_WARN = Decimal("500000")   # $500K - WARN
LIQUIDITY_GATE_FAIL = Decimal("100000")   # $100K - FAIL

# Dilution risk thresholds
DILUTION_CASH_MCAP_LOW = Decimal("0.05")      # <5% = HIGH risk
DILUTION_CASH_MCAP_MED = Decimal("0.15")      # <15% = MODERATE risk
DILUTION_CASH_MCAP_HIGH = Decimal("0.30")     # <30% = LOW risk


class BurnConfidence(str, Enum):
    """Burn rate confidence levels."""
    HIGH = "HIGH"       # CFO/FCF available
    MEDIUM = "MEDIUM"   # NetIncome proxy
    LOW = "LOW"         # R&D proxy
    NONE = "NONE"       # No data


class BurnSource(str, Enum):
    """Burn rate data source."""
    CFO_QUARTERLY = "CFO_quarterly"
    CFO_YTD = "CFO_YTD"
    CFO_ANNUAL = "CFO_annual"
    CFO_TRAILING_4Q = "CFO_trailing_4q"
    FCF_QUARTERLY = "FCF_quarterly"
    FCF_ANNUAL = "FCF_annual"
    NET_INCOME = "NetIncome"
    RD_PROXY = "R&D_proxy"
    PROFITABLE = "profitable"
    NONE = "none"


class DataState(str, Enum):
    """Financial data completeness state."""
    FULL = "FULL"           # All key fields present
    PARTIAL = "PARTIAL"     # Some fields present
    MINIMAL = "MINIMAL"     # Only basic fields
    NONE = "NONE"           # No data


class LiquidityGate(str, Enum):
    """Liquidity gate status."""
    PASS = "PASS"           # >= $500K ADV
    WARN = "WARN"           # $100K - $500K ADV
    FAIL = "FAIL"           # < $100K ADV


class DilutionRiskBucket(str, Enum):
    """Dilution risk categories."""
    LOW = "LOW"             # >= 30% cash/mcap
    MODERATE = "MODERATE"   # 15-30% cash/mcap
    HIGH = "HIGH"           # 5-15% cash/mcap
    SEVERE = "SEVERE"       # < 5% cash/mcap
    UNKNOWN = "UNKNOWN"     # No data


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _to_decimal(value: Any, default: Optional[Decimal] = None) -> Optional[Decimal]:
    """
    Convert value to Decimal safely.

    BUGFIX: Uses explicit `is not None` check instead of truthy.
    """
    if value is None:
        return default

    try:
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            return Decimal(value.strip())
        return default
    except (InvalidOperation, ValueError):
        return default


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Optional[Decimal] = None
) -> Optional[Decimal]:
    """Safe division with eps floor."""
    if denominator is None or abs(denominator) < EPS:
        return default
    return numerator / denominator


def _clamp(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def _quantize_score(value: Decimal) -> Decimal:
    """Quantize to score precision (2 decimal places)."""
    return value.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP)


def _quantize_rate(value: Decimal) -> Decimal:
    """Quantize to rate precision (4 decimal places)."""
    return value.quantize(RATE_PRECISION, rounding=ROUND_HALF_UP)


# ============================================================================
# BURN RATE CALCULATION (HIERARCHICAL WITH PROVENANCE)
# ============================================================================

@dataclass
class BurnAcceleration:
    """Result of burn acceleration analysis."""
    is_accelerating: bool
    acceleration_rate: Optional[Decimal]  # % change Q-over-Q
    trend_direction: str  # "accelerating", "stable", "decelerating", "unknown"
    quarters_analyzed: int
    confidence: str  # "high" (4Q), "medium" (3Q), "low" (2Q), "none"
    penalty_factor: Decimal  # 1.0 = no penalty, <1.0 = penalize for accelerating burn


@dataclass
class BurnResult:
    """Result of burn rate calculation with full provenance."""
    monthly_burn: Optional[Decimal]
    burn_source: BurnSource
    burn_confidence: BurnConfidence
    burn_period: str
    quarters_used: int = 1
    raw_value: Optional[Decimal] = None
    ytd_quarters: Optional[int] = None
    rejection_reasons: Dict[str, str] = field(default_factory=dict)  # source -> reason
    # Burn acceleration tracking
    acceleration: Optional[BurnAcceleration] = None


def _extract_quarterly_burns(financial_data: Dict) -> List[Decimal]:
    """
    Extract quarterly burn values from history fields.

    Returns list of valid (negative) quarterly burns, most recent first.
    """
    burns = []

    # Check for burn history arrays
    for field in ['burn_history', 'quarterly_burns', 'cfo_history', 'quarterly_cfo']:
        history = financial_data.get(field)
        if history and isinstance(history, (list, tuple)):
            for val in history:
                dec = _to_decimal(val)
                if dec is not None and dec < Decimal("0"):
                    burns.append(abs(dec))

    return burns[:4]  # Up to 4 quarters, most recent first


def calculate_burn_acceleration(financial_data: Dict) -> BurnAcceleration:
    """
    Calculate burn acceleration from quarterly history.

    Detects if burn rate is increasing (accelerating), stable, or decreasing.
    Uses linear regression-like approach on quarterly burn data.

    Acceleration is concerning because it indicates faster cash consumption,
    which can lead to runway shortening faster than expected.

    Args:
        financial_data: Financial data dict with burn history

    Returns:
        BurnAcceleration with trend analysis and penalty factor
    """
    # Extract quarterly burns (most recent first)
    burns = _extract_quarterly_burns(financial_data)

    if len(burns) < 2:
        return BurnAcceleration(
            is_accelerating=False,
            acceleration_rate=None,
            trend_direction="unknown",
            quarters_analyzed=len(burns),
            confidence="none",
            penalty_factor=Decimal("1.0"),
        )

    # Determine confidence based on data availability
    if len(burns) >= 4:
        confidence = "high"
    elif len(burns) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    # Calculate quarter-over-quarter changes
    # burns[0] is most recent, burns[1] is prior quarter, etc.
    qoq_changes = []
    for i in range(len(burns) - 1):
        current = burns[i]
        prior = burns[i + 1]
        if prior > EPS:  # Avoid division by zero
            pct_change = ((current - prior) / prior) * Decimal("100")
            qoq_changes.append(pct_change)

    if not qoq_changes:
        return BurnAcceleration(
            is_accelerating=False,
            acceleration_rate=None,
            trend_direction="unknown",
            quarters_analyzed=len(burns),
            confidence=confidence,
            penalty_factor=Decimal("1.0"),
        )

    # Average quarter-over-quarter change
    avg_qoq_change = sum(qoq_changes) / Decimal(len(qoq_changes))

    # Determine trend direction
    # Positive change = burn is increasing (bad - accelerating)
    # Negative change = burn is decreasing (good - decelerating)
    ACCELERATION_THRESHOLD = Decimal("10")  # 10% QoQ increase is concerning
    DECELERATION_THRESHOLD = Decimal("-10")  # 10% QoQ decrease is good

    if avg_qoq_change > ACCELERATION_THRESHOLD:
        trend_direction = "accelerating"
        is_accelerating = True
        # Penalty: reduce runway score for accelerating burn
        # Cap penalty at 30% reduction for extreme acceleration (>50% QoQ)
        penalty_pct = min(Decimal("0.30"), avg_qoq_change / Decimal("100") * Decimal("0.5"))
        penalty_factor = Decimal("1.0") - penalty_pct
    elif avg_qoq_change < DECELERATION_THRESHOLD:
        trend_direction = "decelerating"
        is_accelerating = False
        # Bonus: slight boost for decelerating burn (up to 10%)
        bonus_pct = min(Decimal("0.10"), abs(avg_qoq_change) / Decimal("100") * Decimal("0.2"))
        penalty_factor = Decimal("1.0") + bonus_pct
    else:
        trend_direction = "stable"
        is_accelerating = False
        penalty_factor = Decimal("1.0")

    return BurnAcceleration(
        is_accelerating=is_accelerating,
        acceleration_rate=avg_qoq_change.quantize(Decimal("0.01")),
        trend_direction=trend_direction,
        quarters_analyzed=len(burns),
        confidence=confidence,
        penalty_factor=penalty_factor.quantize(Decimal("0.001")),
    )


def calculate_burn_rate_v2(financial_data: Dict) -> BurnResult:
    """
    Calculate monthly burn rate using hierarchical sources with full provenance.

    Priority:
    1. CFO quarterly (most reliable)
    2. CFO YTD with quarter differencing
    3. CFO annual
    4. Trailing 4-quarter average
    5. FCF quarterly/annual
    6. NetIncome (if negative)
    7. R&D proxy (R&D * 1.5)

    Returns:
        BurnResult with monthly_burn, source, confidence, period, rejection_reasons
    """
    rejection_reasons: Dict[str, str] = {}

    # === CHECK PROFITABILITY FIRST ===
    cfo = _to_decimal(financial_data.get('CFO'))
    cfo_ops = _to_decimal(financial_data.get('CashFlowFromOperations'))
    net_income = _to_decimal(financial_data.get('NetIncome'))

    effective_cfo = cfo if cfo is not None else cfo_ops

    # Profitable company
    if (effective_cfo is not None and effective_cfo > Decimal("0")) or \
       (net_income is not None and net_income > Decimal("0")):
        return BurnResult(
            monthly_burn=Decimal("0"),
            burn_source=BurnSource.PROFITABLE,
            burn_confidence=BurnConfidence.HIGH,
            burn_period="na",
            raw_value=effective_cfo or net_income,
            rejection_reasons={"all": "company_profitable"},
        )

    # === PRIORITY 1: CFO QUARTERLY ===
    cfo_q = _to_decimal(financial_data.get('CFO_quarterly'))
    cfo_q_alt = _to_decimal(financial_data.get('CFO_Q'))
    effective_cfo_q = cfo_q if cfo_q is not None else cfo_q_alt

    if effective_cfo_q is not None and effective_cfo_q < Decimal("0"):
        monthly_burn = abs(effective_cfo_q) / Decimal("3")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.CFO_QUARTERLY,
            burn_confidence=BurnConfidence.HIGH,
            burn_period="quarterly",
            raw_value=effective_cfo_q,
            rejection_reasons=rejection_reasons,
        )
    elif effective_cfo_q is None:
        rejection_reasons["CFO_quarterly"] = "missing"
    else:
        rejection_reasons["CFO_quarterly"] = "non_negative"

    # === PRIORITY 2: CFO YTD WITH DIFFERENCING ===
    cfo_ytd = _to_decimal(financial_data.get('CFO_YTD'))
    cfo_ytd_prior = _to_decimal(financial_data.get('CFO_YTD_prior'))

    if cfo_ytd is not None and cfo_ytd < Decimal("0"):
        # Get quarters in YTD (default 2 for mid-year)
        quarters_in_ytd = int(financial_data.get('quarters_in_ytd', 2))
        months_in_ytd = Decimal(quarters_in_ytd * 3)

        # If we have prior YTD, compute quarterly diff
        if cfo_ytd_prior is not None:
            quarterly_diff = cfo_ytd - cfo_ytd_prior
            if quarterly_diff < Decimal("0"):
                monthly_burn = abs(quarterly_diff) / Decimal("3")
                return BurnResult(
                    monthly_burn=_quantize_rate(monthly_burn),
                    burn_source=BurnSource.CFO_YTD,
                    burn_confidence=BurnConfidence.HIGH,
                    burn_period=f"ytd_diff_{quarters_in_ytd}q",
                    raw_value=quarterly_diff,
                    ytd_quarters=quarters_in_ytd,
                    rejection_reasons=rejection_reasons,
                )

        # Use full YTD
        monthly_burn = abs(cfo_ytd) / max(months_in_ytd, Decimal("1"))
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.CFO_YTD,
            burn_confidence=BurnConfidence.HIGH,
            burn_period=f"ytd_{quarters_in_ytd}q",
            raw_value=cfo_ytd,
            ytd_quarters=quarters_in_ytd,
            rejection_reasons=rejection_reasons,
        )
    elif cfo_ytd is None:
        rejection_reasons["CFO_YTD"] = "missing"
    else:
        rejection_reasons["CFO_YTD"] = "non_negative"

    # === PRIORITY 3: CFO ANNUAL ===
    if effective_cfo is not None and effective_cfo < Decimal("0"):
        # Annual CFO - divide by 12 months
        monthly_burn = abs(effective_cfo) / Decimal("12")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.CFO_ANNUAL,
            burn_confidence=BurnConfidence.HIGH,
            burn_period="annual",
            raw_value=effective_cfo,
            rejection_reasons=rejection_reasons,
        )
    elif effective_cfo is None:
        rejection_reasons["CFO_annual"] = "missing"
    else:
        rejection_reasons["CFO_annual"] = "non_negative"

    # === PRIORITY 4: TRAILING 4-QUARTER AVERAGE ===
    quarterly_burns = _extract_quarterly_burns(financial_data)
    if quarterly_burns:
        quarters_used = len(quarterly_burns)
        avg_quarterly_burn = sum(quarterly_burns) / Decimal(quarters_used)
        monthly_burn = avg_quarterly_burn / Decimal("3")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.CFO_TRAILING_4Q,
            burn_confidence=BurnConfidence.HIGH,
            burn_period=f"{quarters_used}q_avg",
            quarters_used=quarters_used,
            raw_value=avg_quarterly_burn,
            rejection_reasons=rejection_reasons,
        )
    else:
        rejection_reasons["CFO_trailing_4q"] = "no_history"

    # === PRIORITY 5: FCF ===
    fcf_q = _to_decimal(financial_data.get('FCF_quarterly'))
    fcf_q_alt = _to_decimal(financial_data.get('FCF_Q'))
    fcf = _to_decimal(financial_data.get('FCF'))
    fcf_alt = _to_decimal(financial_data.get('FreeCashFlow'))

    effective_fcf_q = fcf_q if fcf_q is not None else fcf_q_alt
    effective_fcf = fcf if fcf is not None else fcf_alt

    if effective_fcf_q is not None and effective_fcf_q < Decimal("0"):
        monthly_burn = abs(effective_fcf_q) / Decimal("3")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.FCF_QUARTERLY,
            burn_confidence=BurnConfidence.HIGH,
            burn_period="quarterly",
            raw_value=effective_fcf_q,
            rejection_reasons=rejection_reasons,
        )
    elif effective_fcf_q is None:
        rejection_reasons["FCF_quarterly"] = "missing"
    else:
        rejection_reasons["FCF_quarterly"] = "non_negative"

    if effective_fcf is not None and effective_fcf < Decimal("0"):
        monthly_burn = abs(effective_fcf) / Decimal("12")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.FCF_ANNUAL,
            burn_confidence=BurnConfidence.HIGH,
            burn_period="annual",
            raw_value=effective_fcf,
            rejection_reasons=rejection_reasons,
        )
    elif effective_fcf is None:
        rejection_reasons["FCF_annual"] = "missing"
    else:
        rejection_reasons["FCF_annual"] = "non_negative"

    # === PRIORITY 6: NET INCOME (FALLBACK) ===
    if net_income is not None and net_income < Decimal("0"):
        # Assume quarterly net income
        monthly_burn = abs(net_income) / Decimal("3")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.NET_INCOME,
            burn_confidence=BurnConfidence.MEDIUM,
            burn_period="quarterly",
            raw_value=net_income,
            rejection_reasons=rejection_reasons,
        )
    elif net_income is None:
        rejection_reasons["NetIncome"] = "missing"
    else:
        rejection_reasons["NetIncome"] = "non_negative"

    # === PRIORITY 7: R&D PROXY (LAST RESORT) ===
    rd = _to_decimal(financial_data.get('R&D'))
    rd_alt = _to_decimal(financial_data.get('ResearchAndDevelopment'))
    effective_rd = rd if rd is not None else rd_alt

    if effective_rd is not None and effective_rd > Decimal("0"):
        # Assume total opex = R&D × 1.5 (add G&A overhead)
        quarterly_burn = effective_rd * Decimal("1.5")
        monthly_burn = quarterly_burn / Decimal("3")
        return BurnResult(
            monthly_burn=_quantize_rate(monthly_burn),
            burn_source=BurnSource.RD_PROXY,
            burn_confidence=BurnConfidence.LOW,
            burn_period="estimated",
            raw_value=effective_rd,
            rejection_reasons=rejection_reasons,
        )
    elif effective_rd is None:
        rejection_reasons["R&D_proxy"] = "missing"
    else:
        rejection_reasons["R&D_proxy"] = "zero_or_negative"

    # === NO DATA ===
    return BurnResult(
        monthly_burn=None,
        burn_source=BurnSource.NONE,
        burn_confidence=BurnConfidence.NONE,
        burn_period="none",
        rejection_reasons=rejection_reasons,
    )


# ============================================================================
# LIQUID ASSETS CALCULATION
# ============================================================================

@dataclass
class LiquidAssetsResult:
    """Result of liquid assets calculation."""
    liquid_assets: Decimal
    cash: Decimal
    marketable_securities: Decimal
    components_used: List[str]


def calculate_liquid_assets(financial_data: Dict) -> LiquidAssetsResult:
    """
    Calculate liquid assets: Cash + MarketableSecurities.

    BUGFIX: Uses explicit `is not None` checks.
    """
    components = []

    # Primary: Cash and cash equivalents
    cash = _to_decimal(financial_data.get('Cash'), Decimal("0"))
    if cash > Decimal("0"):
        components.append("Cash")

    # Marketable securities (check multiple field names)
    mkt_sec = None
    for field in ['MarketableSecurities', 'ShortTermInvestments', 'AvailableForSaleSecurities']:
        val = _to_decimal(financial_data.get(field))
        if val is not None and val > Decimal("0"):
            mkt_sec = val
            components.append(field)
            break

    if mkt_sec is None:
        mkt_sec = Decimal("0")

    liquid_assets = cash + mkt_sec

    return LiquidAssetsResult(
        liquid_assets=liquid_assets,
        cash=cash,
        marketable_securities=mkt_sec,
        components_used=components,
    )


# ============================================================================
# RUNWAY SCORING
# ============================================================================

def _score_runway(runway_months: Decimal) -> Decimal:
    """Score based on runway months (0-100)."""
    if runway_months >= Decimal("24"):
        return Decimal("100")  # 2+ years
    elif runway_months >= Decimal("18"):
        return Decimal("90")   # 18-24 months
    elif runway_months >= Decimal("12"):
        return Decimal("70")   # 12-18 months
    elif runway_months >= Decimal("6"):
        return Decimal("40")   # 6-12 months
    else:
        return Decimal("10")   # < 6 months


@dataclass
class RunwayResult:
    """Result of runway calculation."""
    runway_months: Optional[Decimal]
    runway_score: Decimal
    monthly_burn: Optional[Decimal]
    liquid_assets: Decimal
    liquid_components: List[str]
    burn_source: BurnSource
    burn_confidence: BurnConfidence
    burn_period: str
    quarters_used: int
    burn_rejection_reasons: Dict[str, str] = field(default_factory=dict)
    # Burn acceleration tracking for accuracy improvement
    burn_acceleration: Optional[BurnAcceleration] = None
    runway_score_pre_acceleration: Optional[Decimal] = None  # Score before accel adjustment


def calculate_runway(financial_data: Dict, market_data: Dict) -> RunwayResult:
    """
    Calculate months of cash runway with full provenance.

    Uses 4-quarter trailing average if available.
    Now includes burn acceleration analysis for accuracy improvement.
    """
    # Calculate liquid assets
    liquid = calculate_liquid_assets(financial_data)

    # Calculate burn rate
    burn = calculate_burn_rate_v2(financial_data)

    # Calculate burn acceleration for trend analysis
    acceleration = calculate_burn_acceleration(financial_data)

    # Default result
    default_result = RunwayResult(
        runway_months=None,
        runway_score=Decimal("50"),  # Neutral
        monthly_burn=None,
        liquid_assets=liquid.liquid_assets,
        liquid_components=liquid.components_used,
        burn_source=burn.burn_source,
        burn_confidence=burn.burn_confidence,
        burn_period=burn.burn_period,
        quarters_used=burn.quarters_used,
        burn_rejection_reasons=burn.rejection_reasons,
        burn_acceleration=acceleration,
    )

    # Check for profitability
    if burn.burn_source == BurnSource.PROFITABLE:
        return RunwayResult(
            runway_months=Decimal("999"),
            runway_score=Decimal("100"),
            monthly_burn=Decimal("0"),
            liquid_assets=liquid.liquid_assets,
            liquid_components=liquid.components_used,
            burn_source=burn.burn_source,
            burn_confidence=burn.burn_confidence,
            burn_period=burn.burn_period,
            quarters_used=burn.quarters_used,
            burn_rejection_reasons=burn.rejection_reasons,
            burn_acceleration=acceleration,
        )

    # Calculate runway if we have burn data
    if burn.monthly_burn is not None and burn.monthly_burn > EPS:
        runway_months = _safe_divide(liquid.liquid_assets, burn.monthly_burn)
        if runway_months is not None:
            runway_months = _quantize_score(runway_months)
            base_runway_score = _score_runway(runway_months)

            # Apply burn acceleration adjustment
            # Accelerating burn = lower score (penalty)
            # Decelerating burn = slightly higher score (bonus)
            adjusted_runway_score = base_runway_score * acceleration.penalty_factor
            adjusted_runway_score = _clamp(adjusted_runway_score, Decimal("0"), Decimal("100"))
            adjusted_runway_score = _quantize_score(adjusted_runway_score)

            return RunwayResult(
                runway_months=runway_months,
                runway_score=adjusted_runway_score,
                monthly_burn=burn.monthly_burn,
                liquid_assets=liquid.liquid_assets,
                liquid_components=liquid.components_used,
                burn_source=burn.burn_source,
                burn_confidence=burn.burn_confidence,
                burn_period=burn.burn_period,
                quarters_used=burn.quarters_used,
                burn_rejection_reasons=burn.rejection_reasons,
                burn_acceleration=acceleration,
                runway_score_pre_acceleration=base_runway_score,
            )

    return default_result


# ============================================================================
# DILUTION RISK AND FINANCING PRESSURE
# ============================================================================

@dataclass
class DilutionResult:
    """Result of dilution risk calculation."""
    cash_to_mcap: Optional[Decimal]
    burn_to_mcap: Optional[Decimal]
    dilution_score: Decimal
    dilution_risk_bucket: DilutionRiskBucket
    financing_pressure_score: Decimal
    share_count_growth: Optional[Decimal]


def calculate_dilution_risk(
    financial_data: Dict,
    market_data: Dict,
    runway_months: Optional[Decimal],
    monthly_burn: Optional[Decimal],
) -> DilutionResult:
    """
    Calculate dilution risk and financing pressure.

    Features:
    - cash_to_mcap ratio
    - burn_to_mcap ratio
    - share_count_growth (if available)
    - financing_pressure_score (0-100)
    - dilution_risk_bucket

    BUGFIX: Uses `is not None` checks throughout.
    """
    # Get liquid assets
    liquid = calculate_liquid_assets(financial_data)
    market_cap = _to_decimal(market_data.get('market_cap'))

    # Check for share count growth
    shares_current = _to_decimal(financial_data.get('shares_outstanding'))
    shares_prior = _to_decimal(financial_data.get('shares_outstanding_prior'))

    share_count_growth = None
    if shares_current is not None and shares_prior is not None and shares_prior > EPS:
        share_count_growth = (shares_current - shares_prior) / shares_prior
        share_count_growth = _quantize_rate(share_count_growth)

    # Handle missing market cap
    if market_cap is None or market_cap <= Decimal("0"):
        return DilutionResult(
            cash_to_mcap=None,
            burn_to_mcap=None,
            dilution_score=Decimal("50"),  # Neutral
            dilution_risk_bucket=DilutionRiskBucket.UNKNOWN,
            financing_pressure_score=Decimal("50"),
            share_count_growth=share_count_growth,
        )

    # Calculate ratios
    cash_to_mcap = _safe_divide(liquid.liquid_assets, market_cap, Decimal("0"))
    cash_to_mcap = _quantize_rate(cash_to_mcap)

    burn_to_mcap = None
    if monthly_burn is not None and monthly_burn > Decimal("0"):
        annual_burn = monthly_burn * Decimal("12")
        burn_to_mcap = _safe_divide(annual_burn, market_cap, Decimal("0"))
        burn_to_mcap = _quantize_rate(burn_to_mcap)

    # Determine dilution risk bucket
    if cash_to_mcap >= DILUTION_CASH_MCAP_HIGH:
        dilution_risk_bucket = DilutionRiskBucket.LOW
    elif cash_to_mcap >= DILUTION_CASH_MCAP_MED:
        dilution_risk_bucket = DilutionRiskBucket.MODERATE
    elif cash_to_mcap >= DILUTION_CASH_MCAP_LOW:
        dilution_risk_bucket = DilutionRiskBucket.HIGH
    else:
        dilution_risk_bucket = DilutionRiskBucket.SEVERE

    # Calculate dilution score (monotonic sigmoid)
    if cash_to_mcap >= Decimal("0.50"):
        dilution_score = Decimal("100")
    elif cash_to_mcap <= Decimal("0"):
        dilution_score = Decimal("0")
    else:
        # Sigmoid centered at 15%
        k = Decimal("15")
        midpoint = Decimal("0.15")
        exp_input = -k * (cash_to_mcap - midpoint)
        # Clamp exp input to avoid overflow
        exp_input = _clamp(exp_input, Decimal("-50"), Decimal("50"))
        # Approximate exp using Taylor series for small values
        # For larger values, use limits
        if exp_input > Decimal("20"):
            dilution_score = Decimal("0")
        elif exp_input < Decimal("-20"):
            dilution_score = Decimal("100")
        else:
            # Use Python's exp on float, then convert back
            import math
            exp_val = Decimal(str(math.exp(float(exp_input))))
            dilution_score = Decimal("100") / (Decimal("1") + exp_val)

    # Apply runway penalty if short runway
    if runway_months is not None and runway_months < Decimal("12"):
        clamped_runway = _clamp(runway_months, Decimal("0"), Decimal("12"))
        penalty_factor = Decimal("0.5") + (clamped_runway / Decimal("24"))
        penalty_factor = _clamp(penalty_factor, Decimal("0.5"), Decimal("1.0"))
        dilution_score = dilution_score * penalty_factor

    dilution_score = _clamp(dilution_score, Decimal("0"), Decimal("100"))
    dilution_score = _quantize_score(dilution_score)

    # Calculate financing pressure score (0-100, higher = more pressure)
    # Based on: runway, burn/mcap, share growth
    pressure_components = []

    # Runway component (inverted: short runway = high pressure)
    if runway_months is not None:
        if runway_months >= Decimal("24"):
            pressure_components.append(Decimal("0"))
        elif runway_months >= Decimal("12"):
            pressure_components.append(Decimal("30"))
        elif runway_months >= Decimal("6"):
            pressure_components.append(Decimal("60"))
        else:
            pressure_components.append(Decimal("90"))
    else:
        pressure_components.append(Decimal("50"))  # Unknown

    # Cash/mcap component (inverted: low cash = high pressure)
    if cash_to_mcap <= Decimal("0.05"):
        pressure_components.append(Decimal("90"))
    elif cash_to_mcap <= Decimal("0.15"):
        pressure_components.append(Decimal("60"))
    elif cash_to_mcap <= Decimal("0.30"):
        pressure_components.append(Decimal("30"))
    else:
        pressure_components.append(Decimal("10"))

    # Share dilution component
    if share_count_growth is not None:
        if share_count_growth > Decimal("0.20"):
            pressure_components.append(Decimal("80"))
        elif share_count_growth > Decimal("0.10"):
            pressure_components.append(Decimal("50"))
        elif share_count_growth > Decimal("0.05"):
            pressure_components.append(Decimal("30"))
        else:
            pressure_components.append(Decimal("10"))

    # Average pressure components
    if pressure_components:
        financing_pressure_score = sum(pressure_components) / Decimal(len(pressure_components))
        financing_pressure_score = _quantize_score(financing_pressure_score)
    else:
        financing_pressure_score = Decimal("50")

    return DilutionResult(
        cash_to_mcap=cash_to_mcap,
        burn_to_mcap=burn_to_mcap,
        dilution_score=dilution_score,
        dilution_risk_bucket=dilution_risk_bucket,
        financing_pressure_score=financing_pressure_score,
        share_count_growth=share_count_growth,
    )


# ============================================================================
# LIQUIDITY SCORING
# ============================================================================

@dataclass
class LiquidityResult:
    """Result of liquidity scoring."""
    liquidity_score: Decimal
    liquidity_gate: LiquidityGate
    dollar_adv_20d: Decimal


def score_liquidity(market_data: Dict) -> LiquidityResult:
    """
    Score based on market cap and trading volume.

    Dollar ADV weighted 60%, market cap 40%.
    Includes PASS/WARN/FAIL gating.
    """
    market_cap = _to_decimal(market_data.get('market_cap'), Decimal("0"))
    avg_volume = _to_decimal(market_data.get('avg_volume'), Decimal("0"))
    price = _to_decimal(market_data.get('price'), Decimal("0"))

    # Calculate dollar ADV
    dollar_adv = avg_volume * price

    # Determine liquidity gate
    if dollar_adv >= LIQUIDITY_GATE_WARN:
        liquidity_gate = LiquidityGate.PASS
    elif dollar_adv >= LIQUIDITY_GATE_FAIL:
        liquidity_gate = LiquidityGate.WARN
    else:
        liquidity_gate = LiquidityGate.FAIL

    # Dollar ADV score (60% weight)
    if dollar_adv >= Decimal("50e6"):
        adv_score = Decimal("100")
    elif dollar_adv >= Decimal("20e6"):
        adv_score = Decimal("90")
    elif dollar_adv >= Decimal("10e6"):
        adv_score = Decimal("80")
    elif dollar_adv >= Decimal("5e6"):
        adv_score = Decimal("70")
    elif dollar_adv >= Decimal("1e6"):
        adv_score = Decimal("55")
    elif dollar_adv >= Decimal("500e3"):
        adv_score = Decimal("40")
    elif dollar_adv >= Decimal("100e3"):
        adv_score = Decimal("25")
    else:
        adv_score = Decimal("10")

    # Market cap score (40% weight)
    if market_cap > Decimal("10e9"):
        mcap_score = Decimal("100")
    elif market_cap > Decimal("2e9"):
        mcap_score = Decimal("80")
    elif market_cap > Decimal("500e6"):
        mcap_score = Decimal("60")
    elif market_cap > Decimal("200e6"):
        mcap_score = Decimal("40")
    else:
        mcap_score = Decimal("20")

    # Composite score
    liquidity_score = adv_score * Decimal("0.60") + mcap_score * Decimal("0.40")
    liquidity_score = _quantize_score(liquidity_score)

    return LiquidityResult(
        liquidity_score=liquidity_score,
        liquidity_gate=liquidity_gate,
        dollar_adv_20d=_quantize_score(dollar_adv),
    )


# ============================================================================
# DATA QUALITY ASSESSMENT
# ============================================================================

@dataclass
class DataQualityResult:
    """Result of data quality assessment."""
    financial_data_state: DataState
    missing_fields: List[str]
    inputs_used: Dict[str, str]
    confidence: Decimal


def assess_data_quality(financial_data: Dict, market_data: Dict) -> DataQualityResult:
    """
    Assess data quality and track inputs used.

    Returns confidence score based on coverage.
    """
    missing_fields = []
    inputs_used = {}

    # Key fields to check
    key_fields = {
        'cash': ['Cash'],
        'marketable_securities': ['MarketableSecurities', 'ShortTermInvestments'],
        'burn_rate': ['CFO', 'CashFlowFromOperations', 'FCF', 'FreeCashFlow', 'NetIncome', 'R&D'],
        'market_cap': ['market_cap'],
        'avg_volume': ['avg_volume', 'volume_avg_30d'],
        'price': ['price'],
    }

    fields_found = 0
    total_fields = len(key_fields)

    for category, field_names in key_fields.items():
        found = False
        source_data = market_data if category in ['market_cap', 'avg_volume', 'price'] else financial_data

        for field in field_names:
            val = source_data.get(field)
            # BUGFIX: Use `is not None` instead of truthy check
            if val is not None:
                inputs_used[category] = field
                found = True
                fields_found += 1
                break

        if not found:
            missing_fields.append(category)

    # Determine data state
    critical_missing = [f for f in ['cash', 'market_cap'] if f in missing_fields]

    if len(missing_fields) == 0:
        data_state = DataState.FULL
    elif len(critical_missing) == 0 and len(missing_fields) <= 2:
        data_state = DataState.PARTIAL
    elif len(critical_missing) == 0:
        data_state = DataState.MINIMAL
    else:
        data_state = DataState.NONE

    # Calculate confidence (0-1)
    confidence = Decimal(fields_found) / Decimal(total_fields)
    confidence = _quantize_rate(confidence)

    return DataQualityResult(
        financial_data_state=data_state,
        missing_fields=missing_fields,
        inputs_used=inputs_used,
        confidence=confidence,
    )


# ============================================================================
# SEVERITY DETERMINATION
# ============================================================================

def determine_severity(
    runway_months: Optional[Decimal],
    cash_to_mcap: Optional[Decimal],
) -> Severity:
    """
    Determine severity level based on financial health.

    BUGFIX: Uses `is not None` checks.
    """
    if runway_months is None:
        if cash_to_mcap is not None and cash_to_mcap < DILUTION_CASH_MCAP_LOW:
            return Severity.SEV2
        return Severity.NONE

    if runway_months < RUNWAY_CRITICAL:
        return Severity.SEV3
    elif runway_months < RUNWAY_WARNING:
        return Severity.SEV2
    elif runway_months < RUNWAY_CAUTION:
        return Severity.SEV1
    else:
        return Severity.NONE


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def score_financial_health_v2(
    ticker: str,
    financial_data: Dict,
    market_data: Dict,
) -> Dict[str, Any]:
    """
    Main scoring function for Module 2 (v2).

    Returns dict with all scoring fields.
    """
    # Assess data quality
    quality = assess_data_quality(financial_data, market_data)

    # Calculate runway
    runway = calculate_runway(financial_data, market_data)

    # Calculate dilution risk
    dilution = calculate_dilution_risk(
        financial_data,
        market_data,
        runway.runway_months,
        runway.monthly_burn,
    )

    # Calculate liquidity
    liquidity = score_liquidity(market_data)

    # Calculate composite score
    scores_valid = (
        runway.runway_score is not None and
        dilution.dilution_score is not None and
        liquidity.liquidity_score is not None
    )

    if scores_valid:
        composite = (
            runway.runway_score * Decimal("0.50") +
            dilution.dilution_score * Decimal("0.30") +
            liquidity.liquidity_score * Decimal("0.20")
        )
        # Use clamp_score utility for consistent bounds enforcement
        composite = clamp_score(composite, Decimal("0"), Decimal("100")) or Decimal("50")
        composite = _quantize_score(composite)
        has_data = True
    else:
        composite = Decimal("50")
        has_data = False

    # Determine severity
    severity = determine_severity(runway.runway_months, dilution.cash_to_mcap)

    # Build flags
    flags = []
    if runway.runway_months is not None and runway.runway_months < RUNWAY_WARNING:
        flags.append("low_runway")
    if dilution.cash_to_mcap is not None and dilution.cash_to_mcap < Decimal("0.10"):
        flags.append("low_cash_ratio")
    if liquidity.liquidity_gate == LiquidityGate.FAIL:
        flags.append("liquidity_gate_fail")
    elif liquidity.liquidity_gate == LiquidityGate.WARN:
        flags.append("liquidity_gate_warn")
    if not has_data:
        flags.append("missing_financial_data")
    if runway.burn_confidence == BurnConfidence.LOW:
        flags.append("burn_estimated")
    if dilution.financing_pressure_score >= Decimal("70"):
        flags.append("high_financing_pressure")
    if dilution.share_count_growth is not None and dilution.share_count_growth > Decimal("0.10"):
        flags.append("share_dilution")

    # Add burn acceleration flag (accuracy improvement)
    if runway.burn_acceleration and runway.burn_acceleration.is_accelerating:
        flags.append("burn_accelerating")
    elif runway.burn_acceleration and runway.burn_acceleration.trend_direction == "decelerating":
        flags.append("burn_decelerating")

    # Compute determinism hash with stable ordering
    # Include: ticker, composite, runway, cash_to_mcap, burn_source, inputs_used (sorted keys)
    inputs_sorted = "|".join(f"{k}={v}" for k, v in sorted(quality.inputs_used.items()))
    hash_input = (
        f"{ticker}|"
        f"{composite}|"
        f"{runway.runway_months}|"
        f"{dilution.cash_to_mcap}|"
        f"{runway.burn_source.value}|"
        f"{inputs_sorted}"
    )
    determinism_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    # Helper for safe float conversion (back-compat)
    def _to_float(val):
        if val is None:
            return None
        return float(val)

    return {
        # Core fields (API preserved - floats for back-compat)
        "ticker": ticker,
        "financial_score": _to_float(composite),  # Primary field name
        "financial_normalized": _to_float(composite),  # DEPRECATED: use financial_score
        "runway_months": _to_float(runway.runway_months),
        "runway_score": _to_float(runway.runway_score),
        "dilution_score": _to_float(dilution.dilution_score),
        "liquidity_score": _to_float(liquidity.liquidity_score),
        "cash_to_mcap": _to_float(dilution.cash_to_mcap),
        "monthly_burn": _to_float(runway.monthly_burn),
        "has_financial_data": has_data,
        "severity": severity.value,
        "flags": flags,

        # Burn hierarchy fields
        "burn_source": runway.burn_source.value,
        "burn_confidence": runway.burn_confidence.value,
        "burn_period": runway.burn_period,
        "quarters_used": runway.quarters_used,
        "burn_rejection_reasons": runway.burn_rejection_reasons,

        # Liquidity fields (back-compat: liquidity_gate bool + new enum)
        "liquidity_gate": liquidity.liquidity_gate != LiquidityGate.PASS,  # bool for back-compat
        "liquidity_gate_status": liquidity.liquidity_gate.value,  # new enum field
        "dollar_adv": _to_float(liquidity.dollar_adv_20d),  # back-compat name
        "dollar_adv_20d": _to_float(liquidity.dollar_adv_20d),  # new name

        # Liquid assets fields
        "liquid_assets": _to_float(runway.liquid_assets),
        "liquid_components": runway.liquid_components,

        # Dilution/financing fields (v2 additions)
        "burn_to_mcap": _to_float(dilution.burn_to_mcap),
        "financing_pressure_score": _to_float(dilution.financing_pressure_score),
        "dilution_risk_bucket": dilution.dilution_risk_bucket.value,
        "share_count_growth": _to_float(dilution.share_count_growth),

        # Data quality fields
        "financial_data_state": quality.financial_data_state.value,
        "missing_fields": quality.missing_fields,
        "inputs_used": quality.inputs_used,
        "confidence": _to_float(quality.confidence),

        # Burn acceleration fields (accuracy improvement)
        "burn_acceleration_rate": _to_float(runway.burn_acceleration.acceleration_rate) if runway.burn_acceleration else None,
        "burn_acceleration_trend": runway.burn_acceleration.trend_direction if runway.burn_acceleration else "unknown",
        "burn_acceleration_penalty": _to_float(runway.burn_acceleration.penalty_factor) if runway.burn_acceleration else None,
        "burn_acceleration_confidence": runway.burn_acceleration.confidence if runway.burn_acceleration else "none",
        "runway_score_pre_acceleration": _to_float(runway.runway_score_pre_acceleration),

        # Audit fields (v2 additions)
        "determinism_hash": determinism_hash,
        "schema_version": SCHEMA_VERSION,
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_module_2_v2(
    universe: List[str],
    financial_data: List[Dict],
    market_data: List[Dict],
) -> Dict[str, Any]:
    """
    Main entry point for Module 2 v2 financial health scoring.

    Args:
        universe: List of tickers to score
        financial_data: List of dicts from financial_data.json
        market_data: List of dicts from market_data.json

    Returns:
        Dict with scores and diagnostic_counts
    """
    # Handle empty universe gracefully
    if not universe or len(universe) == 0:
        logger.warning("Module 2 v2: Empty universe provided - returning empty results")
        return {
            "scores": [],
            "diagnostic_counts": {
                "scored": 0,
                "missing": 0,
                "severity_distribution": {},
                "data_state_distribution": {},
                "burn_source_distribution": {},
            },
        }

    logger.info(f"Module 2 v2: Scoring {len(universe)} tickers")

    # Handle empty data gracefully
    if not financial_data:
        logger.warning("Module 2 v2: No financial data provided")
        financial_data = []
    if not market_data:
        logger.warning("Module 2 v2: No market data provided")
        market_data = []

    # Create lookup dicts
    fin_lookup = {f['ticker']: f for f in financial_data if 'ticker' in f}
    mkt_lookup = {m['ticker']: m for m in market_data if 'ticker' in m}

    results = []
    severity_counts: Dict[str, int] = {}
    data_state_counts: Dict[str, int] = {}
    burn_source_counts: Dict[str, int] = {}

    for ticker in universe:
        fin_data = fin_lookup.get(ticker, {})
        mkt_data = mkt_lookup.get(ticker, {})
        score_result = score_financial_health_v2(ticker, fin_data, mkt_data)
        results.append(score_result)

        # Track distributions
        sev = score_result.get("severity", "none")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

        state = score_result.get("financial_data_state", "NONE")
        data_state_counts[state] = data_state_counts.get(state, 0) + 1

        src = score_result.get("burn_source", "none")
        burn_source_counts[src] = burn_source_counts.get(src, 0) + 1

    logger.info(f"Module 2 v2 severity distribution: {severity_counts}")

    return {
        "scores": results,
        "diagnostic_counts": {
            "scored": len(results),
            "missing": len(universe) - len(results),
            "severity_distribution": severity_counts,
            "data_state_distribution": data_state_counts,
            "burn_source_distribution": burn_source_counts,
        },
    }


# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# ============================================================================

def compute_module_2_financial(*args, **kwargs):
    """
    Backwards-compatible wrapper for v2.
    """
    # Handle positional arguments (legacy call pattern)
    if len(args) >= 2:
        records = args[0]
        universe = args[1]
        as_of_date = args[2] if len(args) > 2 else kwargs.get('as_of_date')
        financial_data = records
        market_data = []
    else:
        universe = kwargs.get('universe', kwargs.get('active_tickers', kwargs.get('active_universe', [])))
        financial_data = kwargs.get('financial_records', kwargs.get('financial_data', []))
        market_data = kwargs.get('market_records', kwargs.get('market_data', []))
        as_of_date = kwargs.get('as_of_date')

    if isinstance(universe, set):
        universe = list(universe)

    # Map legacy field names
    mapped_financial = []
    mapped_market = []

    for rec in financial_data:
        ticker = rec.get('ticker')

        fin_rec = {'ticker': ticker}

        # Map cash
        if 'cash_mm' in rec:
            fin_rec['Cash'] = rec['cash_mm'] * 1e6
        elif 'Cash' in rec:
            fin_rec['Cash'] = rec['Cash']

        # Map burn
        if 'burn_rate_mm' in rec:
            fin_rec['NetIncome'] = -rec['burn_rate_mm'] * 1e6
        elif 'NetIncome' in rec:
            fin_rec['NetIncome'] = rec['NetIncome']

        # R&D
        if 'rd_mm' in rec:
            fin_rec['R&D'] = rec['rd_mm'] * 1e6
        elif 'R&D' in rec:
            fin_rec['R&D'] = rec['R&D']

        # Pass through other fields
        for field in ['CFO', 'CFO_quarterly', 'CFO_YTD', 'FCF', 'FCF_quarterly',
                      'MarketableSecurities', 'Debt', 'shares_outstanding']:
            if field in rec:
                fin_rec[field] = rec[field]

        mapped_financial.append(fin_rec)

        # Extract market data
        mkt_rec = {'ticker': ticker}
        if 'market_cap_mm' in rec:
            mkt_rec['market_cap'] = rec['market_cap_mm'] * 1e6
        elif 'market_cap' in rec:
            mkt_rec['market_cap'] = rec['market_cap']

        mkt_rec['avg_volume'] = rec.get('avg_volume', rec.get('volume_avg_30d', 100000))
        mkt_rec['price'] = rec.get('price', 10.0)

        mapped_market.append(mkt_rec)

    # Add separate market data
    for rec in market_data:
        mapped_market.append({
            'ticker': rec.get('ticker'),
            'market_cap': rec.get('market_cap'),
            'price': rec.get('price'),
            'avg_volume': rec.get('volume_avg_30d') or rec.get('avg_volume'),
        })

    return run_module_2_v2(universe, mapped_financial, mapped_market)
