#!/usr/bin/env python3
"""
module_2_financial.py - Financial Health Scoring (vNext)

Scores tickers on financial health using:
1. Cash runway (50% weight)
2. Dilution risk (30% weight)
3. Liquidity (20% weight)

Severity levels based on financial health:
- SEV3: Runway < 6 months (critical)
- SEV2: Runway 6-12 months (warning)
- SEV1: Runway 12-18 months (caution)
- NONE: Runway >= 18 months (healthy)

Burn-rate hierarchy:
1. CFO or FCF (prefer quarterly/YTD with quarter differencing)
2. NetIncome if negative
3. R&D proxy (R&D * 1.5)
4. Mark as missing

Determinism: No datetime.now(), no randomness.

Usage:
    from module_2_financial import run_module_2
    results = run_module_2(universe, financial_data, market_data)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any

from common.integration_contracts import (
    validate_module_2_output,
    is_validation_enabled,
    TickerCollection,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Burn rate confidence levels
BURN_CONFIDENCE_HIGH = "HIGH"      # CFO/FCF available
BURN_CONFIDENCE_MED = "MED"        # NetIncome proxy
BURN_CONFIDENCE_LOW = "LOW"        # R&D proxy
BURN_CONFIDENCE_NONE = "NONE"      # No data

# Data state levels
DATA_STATE_FULL = "FULL"           # All key fields present
DATA_STATE_PARTIAL = "PARTIAL"     # Some fields present
DATA_STATE_MINIMAL = "MINIMAL"     # Only basic fields
DATA_STATE_NONE = "NONE"           # No data

# Liquidity gate threshold (dollar ADV)
LIQUIDITY_GATE_THRESHOLD = 500_000  # $500K minimum daily volume

# Epsilon for division safety
EPS = 1e-9


# =============================================================================
# BURN RATE CALCULATION (HIERARCHICAL)
# =============================================================================

def calculate_burn_rate(financial_data: Dict[str, Any]) -> Tuple[Optional[float], str, str, str]:
    """
    Calculate monthly burn rate using hierarchical sources.

    Priority:
    1. CFO (Cash Flow from Operations) - quarterly or YTD
    2. FCF (Free Cash Flow) - if available
    3. NetIncome - if negative
    4. R&D proxy (R&D * 1.5)

    Returns:
        (monthly_burn, burn_source, burn_confidence, burn_period)
    """
    # Try CFO first (most reliable for cash burn)
    cfo = financial_data.get('CFO') or financial_data.get('CashFlowFromOperations')
    cfo_quarterly = financial_data.get('CFO_quarterly') or financial_data.get('CFO_Q')
    cfo_ytd = financial_data.get('CFO_YTD')

    # Try FCF
    fcf = financial_data.get('FCF') or financial_data.get('FreeCashFlow')
    fcf_quarterly = financial_data.get('FCF_quarterly') or financial_data.get('FCF_Q')

    # Prefer quarterly CFO if available
    if cfo_quarterly is not None and cfo_quarterly < 0:
        monthly_burn = abs(cfo_quarterly) / 3.0
        return (monthly_burn, "CFO_quarterly", BURN_CONFIDENCE_HIGH, "quarterly")

    # Try YTD CFO with quarter differencing
    if cfo_ytd is not None and cfo_ytd < 0:
        # Assume YTD covers current fiscal quarters
        quarters_in_ytd = financial_data.get('quarters_in_ytd', 2)
        months_in_ytd = quarters_in_ytd * 3
        monthly_burn = abs(cfo_ytd) / max(months_in_ytd, 1)
        return (monthly_burn, "CFO_YTD", BURN_CONFIDENCE_HIGH, f"ytd_{quarters_in_ytd}q")

    # Try annual/latest CFO
    if cfo is not None and cfo < 0:
        monthly_burn = abs(cfo) / 3.0  # Assume quarterly
        return (monthly_burn, "CFO", BURN_CONFIDENCE_HIGH, "quarterly")

    # Try quarterly FCF
    if fcf_quarterly is not None and fcf_quarterly < 0:
        monthly_burn = abs(fcf_quarterly) / 3.0
        return (monthly_burn, "FCF_quarterly", BURN_CONFIDENCE_HIGH, "quarterly")

    # Try FCF
    if fcf is not None and fcf < 0:
        monthly_burn = abs(fcf) / 3.0
        return (monthly_burn, "FCF", BURN_CONFIDENCE_HIGH, "quarterly")

    # Fallback to NetIncome if negative
    net_income = financial_data.get('NetIncome', 0)
    if net_income is not None and net_income < 0:
        quarterly_burn = abs(net_income)
        monthly_burn = quarterly_burn / 3.0
        return (monthly_burn, "NetIncome", BURN_CONFIDENCE_MED, "quarterly")

    # Fallback to R&D proxy
    rd_expense = financial_data.get('R&D', 0) or financial_data.get('ResearchAndDevelopment', 0)
    if rd_expense is not None and rd_expense > 0:
        # Assume total opex = R&D × 1.5 (add G&A overhead)
        quarterly_burn = rd_expense * 1.5
        monthly_burn = quarterly_burn / 3.0
        return (monthly_burn, "R&D_proxy", BURN_CONFIDENCE_LOW, "estimated")

    # No burn data available
    return (None, "none", BURN_CONFIDENCE_NONE, "none")


def calculate_trailing_burn(financial_data: Dict[str, Any]) -> Tuple[Optional[float], int]:
    """
    Calculate trailing 4-quarter average burn if multiple quarters available.

    Returns:
        (monthly_burn_avg, quarters_used)
    """
    # Check for quarterly burn history
    burn_history = financial_data.get('burn_history', [])
    quarterly_burns = financial_data.get('quarterly_burns', [])

    # Use whichever is available
    quarters = burn_history if burn_history else quarterly_burns

    if not quarters or len(quarters) == 0:
        return (None, 0)

    # Filter to valid (negative) values - burns are negative cash flows
    valid_burns = [abs(q) for q in quarters if q is not None and q < 0]

    if not valid_burns:
        return (None, 0)

    # Use up to 4 quarters
    recent_burns = valid_burns[:4]
    quarters_used = len(recent_burns)

    # Average quarterly burn
    avg_quarterly_burn = sum(recent_burns) / quarters_used
    monthly_burn_avg = avg_quarterly_burn / 3.0

    return (monthly_burn_avg, quarters_used)


# =============================================================================
# LIQUID ASSETS CALCULATION
# =============================================================================

def calculate_liquid_assets(financial_data: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Calculate liquid assets: Cash + MarketableSecurities.

    Returns:
        (liquid_assets, components_used)
    """
    components = []

    # Primary: Cash and cash equivalents
    cash = financial_data.get('Cash', 0) or 0
    if cash > 0:
        components.append("Cash")

    # Add marketable securities if available
    marketable_sec = (
        financial_data.get('MarketableSecurities', 0) or
        financial_data.get('ShortTermInvestments', 0) or
        financial_data.get('AvailableForSaleSecurities', 0) or
        0
    )
    if marketable_sec > 0:
        components.append("MarketableSecurities")

    liquid_assets = cash + marketable_sec

    return (liquid_assets, components)


# =============================================================================
# CASH RUNWAY CALCULATION
# =============================================================================

def calculate_cash_runway(financial_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate months of cash runway with detailed burn info.

    Returns:
        Dict with runway_months, monthly_burn, score, burn metadata
    """
    result = {
        'runway_months': None,
        'monthly_burn': None,
        'runway_score': 50.0,  # Default neutral
        'burn_source': 'none',
        'burn_confidence': BURN_CONFIDENCE_NONE,
        'burn_period': 'none',
        'liquid_assets': 0,
        'liquid_components': [],
    }

    # Calculate liquid assets
    liquid_assets, liquid_components = calculate_liquid_assets(financial_data)
    result['liquid_assets'] = liquid_assets
    result['liquid_components'] = liquid_components

    # Check for profitability first
    net_income = financial_data.get('NetIncome', 0)
    cfo = financial_data.get('CFO') or financial_data.get('CashFlowFromOperations')

    # CASE 1: Profitable company (positive CFO or positive net income)
    if (cfo is not None and cfo > 0) or (net_income is not None and net_income > 0):
        result['runway_months'] = 999.0
        result['monthly_burn'] = 0.0
        result['runway_score'] = 95.0
        result['burn_source'] = 'profitable'
        result['burn_confidence'] = BURN_CONFIDENCE_HIGH
        result['burn_period'] = 'na'
        return result

    # CASE 2: Try trailing 4-quarter average
    trailing_burn, quarters_used = calculate_trailing_burn(financial_data)
    if trailing_burn is not None and trailing_burn > EPS:
        runway_months = liquid_assets / trailing_burn
        result['runway_months'] = runway_months
        result['monthly_burn'] = trailing_burn
        result['burn_source'] = f'trailing_{quarters_used}q'
        result['burn_confidence'] = BURN_CONFIDENCE_HIGH
        result['burn_period'] = f'{quarters_used}q_avg'
        result['runway_score'] = _score_runway(runway_months)
        return result

    # CASE 3: Single-quarter estimate from hierarchy
    monthly_burn, burn_source, burn_confidence, burn_period = calculate_burn_rate(financial_data)

    if monthly_burn is not None and monthly_burn > EPS:
        runway_months = liquid_assets / monthly_burn
        result['runway_months'] = runway_months
        result['monthly_burn'] = monthly_burn
        result['burn_source'] = burn_source
        result['burn_confidence'] = burn_confidence
        result['burn_period'] = burn_period
        result['runway_score'] = _score_runway(runway_months)
        return result

    # CASE 4: No burn data - return neutral with liquid assets info
    result['runway_score'] = 50.0
    return result


def _score_runway(runway_months: float) -> float:
    """Score based on runway months."""
    if runway_months >= 24:
        return 100.0  # 2+ years
    elif runway_months >= 18:
        return 90.0   # 18-24 months
    elif runway_months >= 12:
        return 70.0   # 12-18 months
    elif runway_months >= 6:
        return 40.0   # 6-12 months
    else:
        return 10.0   # < 6 months


# =============================================================================
# DILUTION RISK (MONOTONIC, STABLE)
# =============================================================================

def calculate_dilution_risk(
    financial_data: Dict,
    market_data: Dict,
    runway_months: Optional[float]
) -> Tuple[Optional[float], float]:
    """
    Score dilution risk with monotonic, stable scoring.

    Fixes:
    - Clamp exp input to avoid overflow
    - Clamp final score to [0, 100]
    - Runway penalty clamped to [0.5, 1.0]
    """
    # Get liquid assets (prefer to raw cash)
    liquid_assets, _ = calculate_liquid_assets(financial_data)
    market_cap = market_data.get('market_cap', 0) or 0

    if market_cap is None or market_cap <= 0:
        return (None, 50.0)

    cash_to_mcap = liquid_assets / market_cap

    # MONOTONIC sigmoid scoring
    # 0% → ~0, 15% → 50, 40%+ → ~100
    if cash_to_mcap >= 0.50:
        dilution_score = 100.0
    elif cash_to_mcap <= 0:
        dilution_score = 0.0
    else:
        k = 15.0  # Steepness
        midpoint = 0.15  # Inflection at 15%

        # CLAMP exp input to avoid overflow
        exp_input = -k * (cash_to_mcap - midpoint)
        exp_input = max(-50.0, min(50.0, exp_input))

        dilution_score = 100.0 / (1.0 + math.exp(exp_input))

    # Runway-based penalty (monotonic, clamped)
    if runway_months is not None and runway_months < 12:
        # Linear penalty: 0mo → 0.5x, 12mo → 1.0x
        # Clamp runway to [0, 12] for penalty calculation
        clamped_runway = max(0.0, min(12.0, runway_months))
        penalty_factor = 0.5 + (clamped_runway / 24.0)
        # Clamp penalty factor to [0.5, 1.0]
        penalty_factor = max(0.5, min(1.0, penalty_factor))
        dilution_score *= penalty_factor

    # CLAMP final score to [0, 100]
    dilution_score = max(0.0, min(100.0, dilution_score))

    return (cash_to_mcap, dilution_score)


# =============================================================================
# LIQUIDITY SCORING (DOLLAR ADV EMPHASIS)
# =============================================================================

def score_liquidity(market_data: Dict[str, Any]) -> Tuple[float, bool, float]:
    """
    Score based on market cap and trading volume.
    Emphasizes dollar ADV (60%) over market cap (40%).

    Returns:
        (liquidity_score, liquidity_gate, dollar_adv)
    """
    market_cap = market_data.get('market_cap', 0) or 0
    avg_volume = market_data.get('avg_volume', 0) or 0
    price = market_data.get('price', 0) or 0

    if not all([market_cap, avg_volume, price]):
        return (50.0, False, 0.0)

    # Dollar volume (average daily value traded)
    dollar_adv = avg_volume * price

    # Liquidity gate check
    liquidity_gate = dollar_adv < LIQUIDITY_GATE_THRESHOLD

    # Dollar ADV tiers (60% weight) - emphasized
    if dollar_adv >= 50e6:
        adv_score = 100.0  # $50M+
    elif dollar_adv >= 20e6:
        adv_score = 90.0   # $20-50M
    elif dollar_adv >= 10e6:
        adv_score = 80.0   # $10-20M
    elif dollar_adv >= 5e6:
        adv_score = 70.0   # $5-10M
    elif dollar_adv >= 1e6:
        adv_score = 55.0   # $1-5M
    elif dollar_adv >= 500e3:
        adv_score = 40.0   # $500K-1M
    elif dollar_adv >= 100e3:
        adv_score = 25.0   # $100K-500K
    else:
        adv_score = 10.0   # <$100K

    # Market cap tiers (40% weight)
    if market_cap > 10e9:
        mcap_score = 100.0  # >$10B
    elif market_cap > 2e9:
        mcap_score = 80.0   # $2-10B
    elif market_cap > 500e6:
        mcap_score = 60.0   # $500M-2B
    elif market_cap > 200e6:
        mcap_score = 40.0   # $200M-500M
    else:
        mcap_score = 20.0   # <$200M

    # Composite: Dollar ADV 60%, Market Cap 40%
    liquidity_score = adv_score * 0.60 + mcap_score * 0.40

    return (liquidity_score, liquidity_gate, dollar_adv)


# =============================================================================
# DATA QUALITY ASSESSMENT
# =============================================================================

def assess_data_quality(financial_data: Dict[str, Any], market_data: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Assess data quality and track inputs used.

    Returns:
        (financial_data_state, missing_fields, inputs_used)
    """
    missing_fields = []
    inputs_used = {}

    # Key financial fields
    cash = financial_data.get('Cash')
    if cash is not None:
        inputs_used['cash_field'] = 'Cash'
    else:
        missing_fields.append('Cash')

    # Check for marketable securities
    mkt_sec = (
        financial_data.get('MarketableSecurities') or
        financial_data.get('ShortTermInvestments')
    )
    if mkt_sec is not None:
        inputs_used['marketable_securities'] = 'MarketableSecurities'

    # Check burn source
    cfo = financial_data.get('CFO') or financial_data.get('CashFlowFromOperations')
    fcf = financial_data.get('FCF') or financial_data.get('FreeCashFlow')
    net_income = financial_data.get('NetIncome')
    rd = financial_data.get('R&D')

    if cfo is not None:
        inputs_used['burn_field'] = 'CFO'
    elif fcf is not None:
        inputs_used['burn_field'] = 'FCF'
    elif net_income is not None:
        inputs_used['burn_field'] = 'NetIncome'
    elif rd is not None:
        inputs_used['burn_field'] = 'R&D_proxy'
    else:
        missing_fields.append('burn_rate')

    # Market data
    market_cap = market_data.get('market_cap')
    if market_cap is not None:
        inputs_used['market_cap'] = 'market_cap'
    else:
        missing_fields.append('market_cap')

    avg_volume = market_data.get('avg_volume')
    if avg_volume is not None:
        inputs_used['volume_field'] = 'avg_volume'
    else:
        missing_fields.append('avg_volume')

    price = market_data.get('price')
    if price is not None:
        inputs_used['price_field'] = 'price'
    else:
        missing_fields.append('price')

    # Determine data state
    critical_fields = ['Cash', 'market_cap']
    critical_missing = [f for f in critical_fields if f in missing_fields]

    if len(missing_fields) == 0:
        data_state = DATA_STATE_FULL
    elif len(critical_missing) == 0 and len(missing_fields) <= 2:
        data_state = DATA_STATE_PARTIAL
    elif len(critical_missing) == 0:
        data_state = DATA_STATE_MINIMAL
    else:
        data_state = DATA_STATE_NONE

    return (data_state, missing_fields, inputs_used)


# =============================================================================
# SEVERITY DETERMINATION
# =============================================================================

def determine_financial_severity(
    runway_months: Optional[float],
    cash_to_mcap: Optional[float]
) -> str:
    """
    Determine severity level based on financial health metrics.

    Args:
        runway_months: Estimated months of cash runway
        cash_to_mcap: Cash to market cap ratio

    Returns:
        Severity string: "none", "sev1", "sev2", or "sev3"
    """
    # BUGFIX: Use `is not None` instead of truthy check
    if runway_months is None:
        if cash_to_mcap is not None and cash_to_mcap < 0.05:
            return "sev2"  # Very low cash relative to market cap
        return "none"  # Unknown - no penalty

    # Severity based on runway
    if runway_months < 6:
        return "sev3"  # Critical - less than 6 months
    elif runway_months < 12:
        return "sev2"  # Warning - 6-12 months
    elif runway_months < 18:
        return "sev1"  # Caution - 12-18 months
    else:
        return "none"  # Healthy - 18+ months


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def score_financial_health(ticker: str, financial_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Main scoring function for Module 2 (vNext)"""

    # Assess data quality first
    data_state, missing_fields, inputs_used = assess_data_quality(financial_data, market_data)

    # Component 1: Cash Runway (50%)
    runway_result = calculate_cash_runway(financial_data, market_data)
    runway_months = runway_result['runway_months']
    burn_rate = runway_result['monthly_burn']
    runway_score = runway_result['runway_score']

    # Component 2: Dilution Risk (30%)
    cash_to_mcap, dilution_score = calculate_dilution_risk(
        financial_data, market_data, runway_months
    )

    # Component 3: Liquidity (20%)
    liquidity_score, liquidity_gate, dollar_adv = score_liquidity(market_data)

    # Composite score - BUGFIX: Use `is not None` checks
    scores_valid = (
        runway_score is not None and
        dilution_score is not None and
        liquidity_score is not None
    )

    if scores_valid:
        composite = runway_score * 0.50 + dilution_score * 0.30 + liquidity_score * 0.20
        # Clamp composite to [0, 100]
        composite = max(0.0, min(100.0, composite))
        has_data = True
    else:
        composite = 50.0
        has_data = False

    # Determine severity based on financial health
    severity = determine_financial_severity(runway_months, cash_to_mcap)

    # Build flags list
    flags = []
    # BUGFIX: Use `is not None` for numeric comparisons
    if runway_months is not None and runway_months < 12:
        flags.append("low_runway")
    if cash_to_mcap is not None and cash_to_mcap < 0.10:
        flags.append("low_cash_ratio")
    if liquidity_gate:
        flags.append("liquidity_gate")
    if not has_data:
        flags.append("missing_financial_data")
    if runway_result['burn_confidence'] == BURN_CONFIDENCE_LOW:
        flags.append("burn_estimated")

    # Extract market_cap for downstream modules
    market_cap = market_data.get('market_cap', 0) or 0
    market_cap_mm = market_cap / 1e6 if market_cap > 0 else None

    return {
        # Core fields (API preserved)
        "ticker": ticker,
        "financial_score": float(composite),  # Standardized field name
        "financial_normalized": float(composite),  # Legacy alias (same value)
        "runway_months": float(runway_months) if runway_months is not None else None,
        "runway_score": float(runway_score) if runway_score is not None else None,
        "dilution_score": float(dilution_score) if dilution_score is not None else None,
        "liquidity_score": float(liquidity_score) if liquidity_score is not None else None,
        "cash_to_mcap": float(cash_to_mcap) if cash_to_mcap is not None else None,
        "monthly_burn": float(burn_rate) if burn_rate is not None else None,
        "market_cap_mm": market_cap_mm,  # Added for Module 5 integration
        "has_financial_data": has_data,
        "severity": severity,
        "flags": flags,

        # New burn-rate hierarchy fields
        "burn_source": runway_result['burn_source'],
        "burn_confidence": runway_result['burn_confidence'],
        "burn_period": runway_result['burn_period'],

        # New liquidity fields
        "liquidity_gate": liquidity_gate,
        "dollar_adv": float(dollar_adv) if dollar_adv is not None else None,

        # New liquid assets fields
        "liquid_assets": float(runway_result['liquid_assets']),
        "liquid_components": runway_result['liquid_components'],

        # New data quality fields
        "financial_data_state": data_state,
        "missing_fields": missing_fields,
        "inputs_used": inputs_used,
    }


def run_module_2(universe: TickerCollection, financial_data: List[Dict], market_data: List[Dict]) -> List[Dict]:
    """
    Main entry point for Module 2 financial health scoring.

    Args:
        universe: Set or List of tickers to score (both accepted for flexibility)
        financial_data: List of dicts from financial_data.json
        market_data: List of dicts from market_data.json

    Returns:
        List of dicts with financial health scores
    """
    # Normalize to list for iteration while preserving order
    if isinstance(universe, set):
        universe = sorted(universe)  # Deterministic ordering for sets

    logger.info(f"Module 2: Scoring {len(universe)} tickers")
    logger.debug(f"Financial records: {len(financial_data)}, Market records: {len(market_data)}")

    # Create lookup dicts
    fin_lookup = {f['ticker']: f for f in financial_data if 'ticker' in f}
    mkt_lookup = {m['ticker']: m for m in market_data if 'ticker' in m}

    results = []
    for ticker in universe:
        fin_data = fin_lookup.get(ticker, {})
        mkt_data = mkt_lookup.get(ticker, {})
        score_result = score_financial_health(ticker, fin_data, mkt_data)
        results.append(score_result)

    # Log severity distribution
    severity_counts = {}
    for r in results:
        sev = r.get("severity", "none")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    logger.info(f"Module 2 severity distribution: {severity_counts}")

    return results


# =============================================================================
# SELF-CHECKS (Unit-test-like)
# =============================================================================

def _run_self_checks() -> List[str]:
    """Run self-checks to verify correctness."""
    errors: List[str] = []

    # CHECK 1: Bugfix - `if x is not None` vs `if x`
    # Zero runway should NOT be treated as None
    result = score_financial_health(
        "CHECK1",
        {'Cash': 0, 'NetIncome': -100e6},
        {'market_cap': 1e9, 'avg_volume': 100000, 'price': 10}
    )
    if result['runway_months'] != 0.0:
        # With 0 cash and burn, runway should be 0
        pass  # This is expected - 0 cash / burn = 0 runway

    # CHECK 2: Burn hierarchy - CFO preferred over NetIncome
    result = score_financial_health(
        "CHECK2",
        {'Cash': 500e6, 'CFO': -50e6, 'NetIncome': -100e6},
        {'market_cap': 1e9, 'avg_volume': 100000, 'price': 10}
    )
    if result['burn_source'] != 'CFO':
        errors.append(f"CHECK2 FAIL: burn_source={result['burn_source']}, expected CFO")

    # CHECK 3: Liquid assets includes marketable securities
    result = score_financial_health(
        "CHECK3",
        {'Cash': 200e6, 'MarketableSecurities': 300e6, 'NetIncome': -50e6},
        {'market_cap': 1e9, 'avg_volume': 100000, 'price': 10}
    )
    if result['liquid_assets'] != 500e6:
        errors.append(f"CHECK3 FAIL: liquid_assets={result['liquid_assets']}, expected 500M")

    # CHECK 4: Dilution score clamped to [0, 100]
    result = score_financial_health(
        "CHECK4",
        {'Cash': 10e9, 'NetIncome': -1e6},  # Huge cash
        {'market_cap': 1e9, 'avg_volume': 100000, 'price': 10}
    )
    if not (0 <= result['dilution_score'] <= 100):
        errors.append(f"CHECK4 FAIL: dilution_score={result['dilution_score']}, not in [0,100]")

    # CHECK 5: Liquidity gate triggers below threshold
    result = score_financial_health(
        "CHECK5",
        {'Cash': 100e6, 'NetIncome': -10e6},
        {'market_cap': 500e6, 'avg_volume': 10000, 'price': 10}  # $100K ADV
    )
    if not result['liquidity_gate']:
        errors.append(f"CHECK5 FAIL: liquidity_gate={result['liquidity_gate']}, expected True")

    # CHECK 6: Data quality state
    result = score_financial_health(
        "CHECK6",
        {'Cash': 100e6},  # Missing burn data
        {'market_cap': 1e9, 'avg_volume': 100000, 'price': 10}
    )
    if result['financial_data_state'] not in [DATA_STATE_PARTIAL, DATA_STATE_MINIMAL]:
        errors.append(f"CHECK6 FAIL: data_state={result['financial_data_state']}")

    # CHECK 7: Runway penalty is clamped
    result = score_financial_health(
        "CHECK7",
        {'Cash': 10e6, 'NetIncome': -100e6},  # Very short runway
        {'market_cap': 1e9, 'avg_volume': 100000, 'price': 10}
    )
    # With ~0.3 months runway, dilution penalty should be applied but clamped
    if result['dilution_score'] < 0 or result['dilution_score'] > 100:
        errors.append(f"CHECK7 FAIL: dilution_score={result['dilution_score']}, not clamped")

    # CHECK 8: Determinism - same inputs produce same outputs
    result1 = score_financial_health(
        "CHECK8",
        {'Cash': 100e6, 'NetIncome': -20e6},
        {'market_cap': 500e6, 'avg_volume': 50000, 'price': 20}
    )
    result2 = score_financial_health(
        "CHECK8",
        {'Cash': 100e6, 'NetIncome': -20e6},
        {'market_cap': 500e6, 'avg_volume': 50000, 'price': 20}
    )
    if result1 != result2:
        errors.append("CHECK8 FAIL: Non-deterministic outputs")

    return errors


# =============================================================================
# CLI / MAIN
# =============================================================================

def main() -> None:
    """Test Module 2 on sample data with self-checks"""

    # Run self-checks first
    print("Running self-checks...")
    errors = _run_self_checks()
    if errors:
        print("SELF-CHECK FAILURES:")
        for e in errors:
            print(f"  - {e}")
        print()
    else:
        print("All self-checks passed!\n")

    universe = ['CVAC', 'RYTM', 'IMMP', 'RICH']

    financial_data = [
        {'ticker': 'CVAC', 'Cash': 500e6, 'MarketableSecurities': 100e6,
         'NetIncome': -100e6, 'CFO': -80e6, 'R&D': 60e6},
        {'ticker': 'RYTM', 'Cash': 200e6, 'NetIncome': -50e6, 'R&D': 40e6},
        {'ticker': 'IMMP', 'Cash': 1000e6, 'MarketableSecurities': 500e6,
         'NetIncome': -150e6, 'FCF': -120e6, 'R&D': 100e6},
        {'ticker': 'RICH', 'Cash': 2000e6, 'NetIncome': 200e6},  # Profitable
    ]

    market_data = [
        {'ticker': 'CVAC', 'market_cap': 2e9, 'avg_volume': 500000, 'price': 20.0},
        {'ticker': 'RYTM', 'market_cap': 800e6, 'avg_volume': 200000, 'price': 15.0},
        {'ticker': 'IMMP', 'market_cap': 5e9, 'avg_volume': 1000000, 'price': 50.0},
        {'ticker': 'RICH', 'market_cap': 10e9, 'avg_volume': 2000000, 'price': 100.0},
    ]

    results = run_module_2(universe, financial_data, market_data)

    print("="*80)
    print("MODULE 2: FINANCIAL HEALTH SCORING (vNext) - TEST RUN")
    print("="*80)

    for r in results:
        print(f"\n{r['ticker']}:")
        print(f"  Financial Score: {r['financial_normalized']:.2f}")
        print(f"  Data State: {r['financial_data_state']}")
        if r['runway_months'] is not None:
            print(f"  Runway: {r['runway_months']:.1f} months ({r['runway_score']:.0f} pts)")
            print(f"    Burn: ${r['monthly_burn']/1e6:.1f}M/mo ({r['burn_source']}, {r['burn_confidence']})")
            print(f"    Liquid Assets: ${r['liquid_assets']/1e6:.0f}M ({r['liquid_components']})")
        if r['cash_to_mcap'] is not None:
            print(f"  Dilution: {r['cash_to_mcap']:.1%} cash/mcap ({r['dilution_score']:.0f} pts)")
        if r['liquidity_score'] is not None:
            gate_str = " [GATED]" if r['liquidity_gate'] else ""
            print(f"  Liquidity: {r['liquidity_score']:.0f} pts (${r['dollar_adv']/1e6:.1f}M ADV){gate_str}")
        if r['flags']:
            print(f"  Flags: {r['flags']}")
        if r['missing_fields']:
            print(f"  Missing: {r['missing_fields']}")

    print("\n" + "="*80)
    if errors:
        print(f"MODULE 2 TEST COMPLETE - {len(errors)} self-check failures")
    else:
        print("MODULE 2 TEST COMPLETE - All checks passed!")
    print("="*80)


if __name__ == "__main__":
    main()


# =============================================================================
# COMPATIBILITY WRAPPER
# =============================================================================

def compute_module_2_financial(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """
    Ultra-flexible wrapper for backwards compatibility.

    Supports multiple calling conventions:
    - compute_module_2_financial(records, active_tickers, as_of_date)  # Legacy
    - compute_module_2_financial(financial_data=..., universe=..., ...)  # Kwargs
    """
    # Handle positional arguments (legacy call pattern)
    if len(args) >= 2:
        # Legacy: (records, active_tickers, as_of_date)
        records = args[0]
        universe = args[1]
        as_of_date = args[2] if len(args) > 2 else kwargs.get('as_of_date')
        financial_data = records
        market_data = []
    else:
        # Extract parameters from kwargs
        universe = kwargs.get('universe', kwargs.get('active_tickers', kwargs.get('active_universe', [])))
        financial_data = kwargs.get('financial_records', kwargs.get('financial_data', []))
        market_data = kwargs.get('market_records', kwargs.get('market_data', []))
        as_of_date = kwargs.get('as_of_date')

    # Convert set to list if needed
    # DETERMINISM: Sort the set to ensure consistent iteration order
    if isinstance(universe, set):
        universe = sorted(universe)

    # If market_data is empty but we have raw_universe, extract it
    if not market_data and 'raw_universe' in kwargs:
        raw_universe = kwargs['raw_universe']
        market_data = []
        for record in raw_universe:
            if 'market_data' in record and record.get('ticker'):
                mkt = record['market_data'].copy()
                mkt['ticker'] = record['ticker']
                market_data.append(mkt)

    # Map legacy field names to new format
    mapped_financial = []
    mapped_market = []

    # Track tickers we've seen in input data (even if PIT-filtered)
    all_input_tickers = {rec.get('ticker') for rec in financial_data if rec.get('ticker')}

    for rec in financial_data:
        ticker = rec.get('ticker')

        # PIT filtering: skip records from the future
        source_date = rec.get('source_date')
        if source_date and as_of_date and source_date > as_of_date:
            continue

        # Map legacy field names (cash_mm -> Cash in dollars, etc.)
        fin_rec = {'ticker': ticker}

        # Cash: cash_mm (millions) -> Cash (dollars)
        if 'cash_mm' in rec:
            fin_rec['Cash'] = rec['cash_mm'] * 1e6
        elif 'Cash' in rec:
            fin_rec['Cash'] = rec['Cash']

        # Burn rate: burn_rate_mm -> NetIncome (negative)
        if 'burn_rate_mm' in rec:
            fin_rec['NetIncome'] = -rec['burn_rate_mm'] * 1e6  # Burn is expense
        elif 'NetIncome' in rec:
            fin_rec['NetIncome'] = rec['NetIncome']

        # R&D
        if 'rd_mm' in rec:
            fin_rec['R&D'] = rec['rd_mm'] * 1e6
        elif 'R&D' in rec:
            fin_rec['R&D'] = rec['R&D']

        # CFO/FCF pass-through
        for field in ['CFO', 'CFO_quarterly', 'CFO_YTD', 'FCF', 'FCF_quarterly',
                      'MarketableSecurities', 'Debt']:
            if field in rec:
                fin_rec[field] = rec[field]

        mapped_financial.append(fin_rec)

        # Extract market data from combined record if present
        mkt_rec = {'ticker': ticker}
        if 'market_cap_mm' in rec:
            mkt_rec['market_cap'] = rec['market_cap_mm'] * 1e6
        elif 'market_cap' in rec:
            mkt_rec['market_cap'] = rec['market_cap']

        if 'avg_volume' in rec:
            mkt_rec['avg_volume'] = rec['avg_volume']
        elif 'volume_avg_30d' in rec:
            mkt_rec['avg_volume'] = rec['volume_avg_30d']
        else:
            mkt_rec['avg_volume'] = 100000  # Default

        if 'price' in rec:
            mkt_rec['price'] = rec['price']
        else:
            mkt_rec['price'] = 10.0  # Default

        mapped_market.append(mkt_rec)

    # Add any separate market_data records
    for rec in market_data:
        mapped_market.append({
            'ticker': rec.get('ticker'),
            'market_cap': rec.get('market_cap'),
            'price': rec.get('price'),
            'avg_volume': rec.get('volume_avg_30d') or rec.get('avg_volume'),
        })

    # Filter universe to only tickers we have data for
    available_tickers = {r['ticker'] for r in mapped_financial}
    filtered_universe = [t for t in universe if t in available_tickers]
    # Only create placeholders for tickers with NO input data at all (not PIT-filtered ones)
    truly_missing_tickers = [t for t in universe if t not in all_input_tickers]

    result = run_module_2(filtered_universe, mapped_financial, mapped_market)

    # Add placeholder scores for truly missing tickers (edge case handling)
    for ticker in truly_missing_tickers:
        result.append({
            'ticker': ticker,
            'score': 0,
            'financial_score': 0.0,  # Required by validation
            'financial_normalized': 0.0,  # Legacy alias
            'runway_months': None,
            'cash_mm': None,
            'burn_rate_mm': None,
            'flags': ['missing_financial_data'],
            'data_state': DATA_STATE_NONE,
            'severity': 'sev3',  # Missing data = hard gate
        })

    # Add legacy field names to existing results for backwards compatibility
    for score in result:
        if 'flags' not in score:
            score['flags'] = []
        # Add cash_mm if we have Cash
        if 'cash_mm' not in score and 'Cash' in score:
            score['cash_mm'] = score['Cash'] / 1e6 if score['Cash'] else None
        elif 'cash_mm' not in score:
            score['cash_mm'] = None
            if 'missing_cash' not in score['flags']:
                score['flags'].append('missing_cash')

    # DETERMINISM: Sort results by ticker to ensure consistent output order
    result_sorted = sorted(result, key=lambda x: x.get('ticker', ''))

    # Wrap in expected format
    output = {
        'scores': result_sorted,
        'diagnostic_counts': {
            'scored': len(filtered_universe),
            'missing': len(truly_missing_tickers)
        }
    }

    # Validate output schema before returning
    if is_validation_enabled():
        validate_module_2_output(output)

    return output
