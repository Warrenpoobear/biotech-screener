#!/usr/bin/env python3
"""
financial_module_2_survivability.py - Financial Survivability & Capital Discipline

Bounded scoring module that evaluates:
A) Effective Runway Score (max +2, min -6)
B) Burn Efficiency / Discipline Score (max +2, min -2)
C) Burn Acceleration / Cash Concentration Score (max 0, min -2)
D) Debt Fragility & Maturity Mismatch Score (max +1, min -3)

Final score bounded to [-10.0, +5.0]

Constraints:
- Deterministic, point-in-time safe (PIT)
- Stdlib-only, uses Decimal for all numeric work
- No market cap dependence
- Graceful degradation for missing data

Usage:
    from financial_module_2_survivability import compute_survivability_score
    result = compute_survivability_score(financial_data, debt_data, catalyst_data, as_of_date)
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Tuple
from datetime import date, datetime
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# Score bounds
FINAL_SCORE_MIN = Decimal("-10.0")
FINAL_SCORE_MAX = Decimal("+5.0")

SCORE_A_MIN = Decimal("-6.0")
SCORE_A_MAX = Decimal("+2.0")

SCORE_B_MIN = Decimal("-2.0")
SCORE_B_MAX = Decimal("+2.0")

SCORE_C_MIN = Decimal("-2.0")
SCORE_C_MAX = Decimal("0.0")

SCORE_D_MIN = Decimal("-3.0")
SCORE_D_MAX = Decimal("+1.0")

# Runway thresholds (months)
RUNWAY_EXCELLENT = Decimal("24")
RUNWAY_GOOD = Decimal("18")
RUNWAY_ADEQUATE = Decimal("12")
RUNWAY_CAUTION = Decimal("9")
RUNWAY_WARNING = Decimal("6")

# R&D ratio thresholds
RD_RATIO_EXCELLENT = Decimal("0.55")
RD_RATIO_GOOD = Decimal("0.45")
RD_RATIO_ADEQUATE = Decimal("0.30")
RD_RATIO_POOR = Decimal("0.20")

# R&D to burn ratio thresholds (fallback)
RD_BURN_GOOD = Decimal("0.60")
RD_BURN_ADEQUATE = Decimal("0.40")

# Burn acceleration thresholds
BURN_ACCEL_HIGH = Decimal("0.50")
BURN_ACCEL_MODERATE = Decimal("0.25")

# Cash concentration thresholds (quarterly burn / cash)
CASH_CONC_HIGH = Decimal("0.25")
CASH_CONC_MODERATE = Decimal("0.15")

# Debt to cash thresholds
DEBT_CASH_LOW = Decimal("0.10")
DEBT_CASH_MODERATE = Decimal("0.40")
DEBT_CASH_HIGH = Decimal("0.80")

# Epsilon for safe division
EPS = Decimal("1")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def to_decimal(value: Any, default: Decimal = Decimal("0")) -> Decimal:
    """Safely convert value to Decimal."""
    if value is None:
        return default
    try:
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))
    except Exception:
        return default


def clamp(value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def parse_date(date_val: Any) -> Optional[date]:
    """Parse date from various formats."""
    if date_val is None:
        return None
    if isinstance(date_val, date):
        return date_val
    if isinstance(date_val, datetime):
        return date_val.date()
    if isinstance(date_val, str):
        try:
            return datetime.fromisoformat(date_val.split('T')[0]).date()
        except Exception:
            return None
    return None


# =============================================================================
# SUB-SCORE CALCULATIONS
# =============================================================================

def compute_effective_runway(
    cash_total: Decimal,
    burn_ttm: Decimal,
    interest_cash: Decimal,
    near_term_debt: Decimal
) -> Tuple[Decimal, str]:
    """
    Compute effective runway in months.

    Returns:
        (effective_runway_months, calculation_method)
    """
    if burn_ttm <= 0:
        # Non-burner or missing data
        if cash_total > 0:
            return Decimal("999"), "non_burner"
        return Decimal("0"), "no_cash"

    # Effective cash = cash - near-term debt obligations
    effective_cash = max(Decimal("0"), cash_total - near_term_debt)

    # Total monthly outflow = (annual burn + annual interest) / 12
    annual_outflow = burn_ttm + interest_cash

    if annual_outflow <= 0:
        return Decimal("999"), "no_outflow"

    runway_months = (Decimal("12") * effective_cash) / annual_outflow

    return runway_months, "calculated"


def score_runway(
    effective_runway_months: Decimal,
    burn_ttm: Decimal,
    cash_total: Decimal
) -> Tuple[Decimal, str]:
    """
    Score A: Effective Runway Score (bounded [+2, -6])

    Returns:
        (score, bucket_description)
    """
    # Non-burner with cash gets max score
    if burn_ttm <= 0 and cash_total > 0:
        return SCORE_A_MAX, "non_burner_with_cash"

    # Bucket by runway months
    if effective_runway_months >= RUNWAY_EXCELLENT:
        return Decimal("+2.0"), "runway_>=24mo"
    elif effective_runway_months >= RUNWAY_GOOD:
        return Decimal("+1.0"), "runway_18-24mo"
    elif effective_runway_months >= RUNWAY_ADEQUATE:
        return Decimal("0.0"), "runway_12-18mo"
    elif effective_runway_months >= RUNWAY_CAUTION:
        return Decimal("-1.0"), "runway_9-12mo"
    elif effective_runway_months >= RUNWAY_WARNING:
        return Decimal("-3.0"), "runway_6-9mo"
    else:
        return Decimal("-6.0"), "runway_<6mo"


def score_burn_discipline(
    r_and_d_expense_ttm: Optional[Decimal],
    total_operating_expense_ttm: Optional[Decimal],
    burn_ttm: Decimal
) -> Tuple[Decimal, str, bool]:
    """
    Score B: Burn Efficiency / Discipline Score (bounded [+2, -2])

    Returns:
        (score, bucket_description, is_missing)
    """
    # Primary method: R&D ratio to total operating expense
    if (r_and_d_expense_ttm is not None and
        total_operating_expense_ttm is not None and
        total_operating_expense_ttm > 0):

        rd_ratio = r_and_d_expense_ttm / total_operating_expense_ttm

        if rd_ratio >= RD_RATIO_EXCELLENT:
            return Decimal("+2.0"), f"rd_ratio_{rd_ratio:.2f}_>=0.55", False
        elif rd_ratio >= RD_RATIO_GOOD:
            return Decimal("+1.0"), f"rd_ratio_{rd_ratio:.2f}_0.45-0.55", False
        elif rd_ratio >= RD_RATIO_ADEQUATE:
            return Decimal("0.0"), f"rd_ratio_{rd_ratio:.2f}_0.30-0.45", False
        elif rd_ratio >= RD_RATIO_POOR:
            return Decimal("-1.0"), f"rd_ratio_{rd_ratio:.2f}_0.20-0.30", False
        else:
            return Decimal("-2.0"), f"rd_ratio_{rd_ratio:.2f}_<0.20", False

    # Fallback method: R&D to burn ratio
    if r_and_d_expense_ttm is not None and burn_ttm > 0:
        rd_to_burn = r_and_d_expense_ttm / burn_ttm

        if rd_to_burn >= RD_BURN_GOOD:
            return Decimal("+1.0"), f"rd_to_burn_{rd_to_burn:.2f}_>=0.60", False
        elif rd_to_burn >= RD_BURN_ADEQUATE:
            return Decimal("0.0"), f"rd_to_burn_{rd_to_burn:.2f}_0.40-0.60", False
        else:
            return Decimal("-1.0"), f"rd_to_burn_{rd_to_burn:.2f}_<0.40", False

    # Missing data
    return Decimal("0.0"), "discipline_data_missing", True


def score_burn_acceleration(
    burn_last_quarter: Optional[Decimal],
    burn_prev_quarter: Optional[Decimal],
    burn_ttm: Decimal,
    cash_total: Decimal
) -> Tuple[Decimal, str, Optional[Decimal]]:
    """
    Score C: Burn Acceleration / Cash Concentration Score (bounded [0, -2])

    Returns:
        (score, bucket_description, metric_value)
    """
    # Primary method: Quarter-over-quarter acceleration
    if burn_last_quarter is not None and burn_prev_quarter is not None:
        prev_safe = max(EPS, burn_prev_quarter)
        burn_accel = (burn_last_quarter - burn_prev_quarter) / prev_safe

        if burn_accel >= BURN_ACCEL_HIGH:
            return Decimal("-2.0"), f"burn_accel_{burn_accel:.2f}_>=0.50", burn_accel
        elif burn_accel >= BURN_ACCEL_MODERATE:
            return Decimal("-1.0"), f"burn_accel_{burn_accel:.2f}_0.25-0.50", burn_accel
        else:
            return Decimal("0.0"), f"burn_accel_{burn_accel:.2f}_<0.25", burn_accel

    # Fallback: Cash concentration (quarterly burn / cash)
    if burn_ttm > 0 and cash_total > 0:
        quarterly_burn = burn_ttm / Decimal("4")
        cash_concentration = quarterly_burn / max(EPS, cash_total)

        if cash_concentration >= CASH_CONC_HIGH:
            return Decimal("-2.0"), f"cash_conc_{cash_concentration:.2f}_>=0.25", cash_concentration
        elif cash_concentration >= CASH_CONC_MODERATE:
            return Decimal("-1.0"), f"cash_conc_{cash_concentration:.2f}_0.15-0.25", cash_concentration
        else:
            return Decimal("0.0"), f"cash_conc_{cash_concentration:.2f}_<0.15", cash_concentration

    # No burn or no cash - neutral
    return Decimal("0.0"), "no_burn_or_cash", None


def score_debt_fragility(
    debt_total: Decimal,
    cash_total: Decimal,
    burn_ttm: Decimal,
    nearest_maturity_date: Optional[date],
    next_major_catalyst_date: Optional[date],
    amount_due_12m: Optional[Decimal],
    effective_runway_months: Decimal
) -> Tuple[Decimal, str, Optional[Decimal]]:
    """
    Score D: Debt Fragility & Maturity Mismatch Score (bounded [+1, -3])

    Returns:
        (score, bucket_description, debt_to_cash_ratio)
    """
    notes = []

    # Calculate debt-to-cash ratio
    debt_to_cash = debt_total / max(EPS, cash_total) if cash_total > 0 else Decimal("0")

    # Base score from debt-to-cash ratio
    is_burner = burn_ttm > 0

    if debt_to_cash < DEBT_CASH_LOW:
        base_score = Decimal("+1.0")
        notes.append(f"debt_to_cash_{debt_to_cash:.2f}_<0.10")
    elif debt_to_cash < DEBT_CASH_MODERATE:
        base_score = Decimal("0.0")
        notes.append(f"debt_to_cash_{debt_to_cash:.2f}_0.10-0.40")
    elif debt_to_cash < DEBT_CASH_HIGH:
        base_score = Decimal("-1.0")
        notes.append(f"debt_to_cash_{debt_to_cash:.2f}_0.40-0.80")
    else:
        # >= 0.80
        if is_burner:
            base_score = Decimal("-2.0")
            notes.append(f"debt_to_cash_{debt_to_cash:.2f}_>=0.80_burner")
        else:
            base_score = Decimal("-1.0")
            notes.append(f"debt_to_cash_{debt_to_cash:.2f}_>=0.80_non_burner")

    # Maturity mismatch penalty
    maturity_penalty = Decimal("0")

    # Check if debt matures before catalyst
    if nearest_maturity_date is not None and next_major_catalyst_date is not None:
        if nearest_maturity_date <= next_major_catalyst_date:
            maturity_penalty = Decimal("-1.0")
            notes.append("maturity_before_catalyst")
    # Fallback: Check if significant debt due within 12m with short runway
    elif amount_due_12m is not None and amount_due_12m > 0:
        if effective_runway_months < Decimal("12"):
            maturity_penalty = Decimal("-1.0")
            notes.append("debt_due_12m_short_runway")

    # Combine and clamp
    raw_score = base_score + maturity_penalty
    final_score = clamp(raw_score, SCORE_D_MIN, SCORE_D_MAX)

    return final_score, "|".join(notes), debt_to_cash


# =============================================================================
# MAIN SCORING FUNCTION
# =============================================================================

def compute_survivability_score(
    financial_data: Dict[str, Any],
    debt_data: Optional[Dict[str, Any]] = None,
    catalyst_data: Optional[Dict[str, Any]] = None,
    as_of_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Compute Financial Module 2: Survivability & Capital Discipline score.

    Args:
        financial_data: Dict with cash, burn, R&D, operating expense data
        debt_data: Optional dict with debt maturities
        catalyst_data: Optional dict with next catalyst date
        as_of_date: Point-in-time date for PIT safety

    Returns:
        Dict with score, subscores, metrics, coverage flags, and notes
    """
    debt_data = debt_data or {}
    catalyst_data = catalyst_data or {}

    coverage_flags: List[str] = []
    notes: List[str] = []

    # ==========================================================================
    # EXTRACT AND COMPUTE SHARED METRICS
    # ==========================================================================

    # Cash total = cash_and_equivalents + short_term_investments
    cash_and_equiv = to_decimal(financial_data.get('cash_and_equivalents') or
                                 financial_data.get('Cash'))
    short_term_inv = to_decimal(financial_data.get('short_term_investments') or
                                 financial_data.get('ShortTermInvestments') or
                                 financial_data.get('MarketableSecurities'))

    cash_total = cash_and_equiv + short_term_inv

    if cash_and_equiv <= 0:
        coverage_flags.append("missing_cash")
    if short_term_inv <= 0:
        coverage_flags.append("no_short_term_investments")

    # Burn TTM = max(0, -operating_cash_flow_ttm)
    ocf_ttm = to_decimal(financial_data.get('operating_cash_flow_ttm') or
                          financial_data.get('CFO') or
                          financial_data.get('NetIncome'))

    burn_ttm = Decimal("0")
    burn_method = "none"

    if ocf_ttm is not None and ocf_ttm != 0:
        if ocf_ttm < 0:
            burn_ttm = abs(ocf_ttm)
            burn_method = "ocf"
    elif financial_data.get('total_operating_expense_ttm') is not None:
        # Fallback: approximate burn from opex - revenue
        opex = to_decimal(financial_data.get('total_operating_expense_ttm'))
        revenue = to_decimal(financial_data.get('revenue_ttm') or
                              financial_data.get('Revenue'))
        approx_burn = opex - revenue
        if approx_burn > 0:
            burn_ttm = approx_burn
            burn_method = "approximated"
            coverage_flags.append("burn_approximated")

    if burn_method == "none":
        coverage_flags.append("missing_burn_data")

    # Interest expense (check both normalized and raw SEC field names)
    interest_cash = to_decimal(financial_data.get('interest_expense_ttm') or
                                financial_data.get('InterestExpense'))
    if interest_cash < 0:
        interest_cash = abs(interest_cash)
    if financial_data.get('interest_expense_ttm') is None and financial_data.get('InterestExpense') is None:
        coverage_flags.append("missing_interest_expense")

    # Near-term debt
    amount_due_12m = to_decimal(debt_data.get('amount_due_12m'))
    current_debt = to_decimal(financial_data.get('current_debt') or
                               financial_data.get('LongTermDebtCurrent'))

    if amount_due_12m > 0:
        near_term_debt = amount_due_12m
    elif current_debt > 0:
        near_term_debt = current_debt
    else:
        near_term_debt = Decimal("0")
        coverage_flags.append("missing_near_term_debt")

    # Total debt
    total_debt = to_decimal(financial_data.get('total_debt'))
    if total_debt <= 0:
        # Try to compute from components
        long_term_debt = to_decimal(financial_data.get('long_term_debt') or
                                     financial_data.get('LongTermDebt'))
        if current_debt > 0 or long_term_debt > 0:
            total_debt = current_debt + long_term_debt
        else:
            coverage_flags.append("missing_total_debt")

    # R&D and operating expenses
    r_and_d_expense = to_decimal(financial_data.get('r_and_d_expense_ttm') or
                                  financial_data.get('R&D'))
    if r_and_d_expense <= 0:
        r_and_d_expense = None
        coverage_flags.append("missing_rd_expense")

    total_opex = to_decimal(financial_data.get('total_operating_expense_ttm') or
                             financial_data.get('OperatingExpenses'))
    if total_opex <= 0:
        total_opex = None
        coverage_flags.append("missing_total_opex")

    # Quarterly burn data (if available)
    burn_last_quarter = None
    burn_prev_quarter = None
    quarterly_burns = financial_data.get('quarterly_burns') or financial_data.get('burn_history')
    if quarterly_burns and len(quarterly_burns) >= 2:
        try:
            burn_last_quarter = to_decimal(quarterly_burns[0])
            burn_prev_quarter = to_decimal(quarterly_burns[1])
            if burn_last_quarter < 0:
                burn_last_quarter = abs(burn_last_quarter)
            if burn_prev_quarter < 0:
                burn_prev_quarter = abs(burn_prev_quarter)
        except Exception:
            pass

    if burn_last_quarter is None:
        coverage_flags.append("missing_quarterly_burn")

    # Dates
    nearest_maturity_date = parse_date(debt_data.get('nearest_maturity_date'))
    next_major_catalyst_date = parse_date(catalyst_data.get('next_major_catalyst_date') or
                                           catalyst_data.get('next_catalyst_date'))

    # ==========================================================================
    # COMPUTE EFFECTIVE RUNWAY
    # ==========================================================================

    effective_runway_months, runway_method = compute_effective_runway(
        cash_total, burn_ttm, interest_cash, near_term_debt
    )

    notes.append(f"runway_calc:{runway_method}")

    # Runway data confidence
    if cash_total > 0 and burn_method == "ocf":
        runway_confidence = "HIGH"
    elif cash_total > 0 and burn_method == "approximated":
        runway_confidence = "MED"
    else:
        runway_confidence = "LOW"

    # ==========================================================================
    # COMPUTE SUB-SCORES
    # ==========================================================================

    # A) Runway Score
    score_a, note_a = score_runway(effective_runway_months, burn_ttm, cash_total)
    notes.append(f"A:{note_a}")

    # B) Burn Discipline Score
    score_b, note_b, discipline_missing = score_burn_discipline(
        r_and_d_expense, total_opex, burn_ttm
    )
    notes.append(f"B:{note_b}")
    if discipline_missing:
        coverage_flags.append("discipline_missing")

    # C) Burn Acceleration / Cash Concentration Score
    score_c, note_c, accel_metric = score_burn_acceleration(
        burn_last_quarter, burn_prev_quarter, burn_ttm, cash_total
    )
    notes.append(f"C:{note_c}")

    # D) Debt Fragility Score
    score_d, note_d, debt_to_cash = score_debt_fragility(
        total_debt, cash_total, burn_ttm,
        nearest_maturity_date, next_major_catalyst_date,
        amount_due_12m, effective_runway_months
    )
    notes.append(f"D:{note_d}")

    # ==========================================================================
    # COMPUTE FINAL SCORE
    # ==========================================================================

    raw_score = score_a + score_b + score_c + score_d
    final_score = clamp(raw_score, FINAL_SCORE_MIN, FINAL_SCORE_MAX)

    notes.append(f"raw={raw_score},clamped={final_score}")

    # ==========================================================================
    # BUILD RESULT
    # ==========================================================================

    # Build metrics dict
    metrics = {
        "cash_total": float(cash_total),
        "burn_ttm": float(burn_ttm),
        "interest_cash": float(interest_cash),
        "near_term_debt": float(near_term_debt),
        "effective_runway_months": float(effective_runway_months) if effective_runway_months < 999 else None,
        "runway_confidence": runway_confidence,
    }

    # Add optional metrics
    if r_and_d_expense is not None and total_opex is not None and total_opex > 0:
        metrics["rd_ratio"] = float(r_and_d_expense / total_opex)

    if accel_metric is not None:
        if burn_last_quarter is not None and burn_prev_quarter is not None:
            metrics["burn_acceleration"] = float(accel_metric)
        else:
            metrics["cash_concentration"] = float(accel_metric)

    if debt_to_cash is not None:
        metrics["debt_to_cash"] = float(debt_to_cash)

    return {
        "module": "financial_module_2_survivability",
        "score": float(final_score),
        "subscores": {
            "runway": float(score_a),
            "discipline": float(score_b),
            "accel": float(score_c),
            "debt": float(score_d),
        },
        "metrics": metrics,
        "coverage": coverage_flags,
        "notes": notes,
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def compute_survivability_scores_batch(
    tickers: List[str],
    financial_lookup: Dict[str, Dict[str, Any]],
    debt_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    catalyst_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
    as_of_date: Optional[date] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Compute survivability scores for a batch of tickers.

    Returns:
        Dict mapping ticker -> result dict
    """
    debt_lookup = debt_lookup or {}
    catalyst_lookup = catalyst_lookup or {}

    results = {}
    for ticker in tickers:
        financial_data = financial_lookup.get(ticker, {})
        debt_data = debt_lookup.get(ticker, {})
        catalyst_data = catalyst_lookup.get(ticker, {})

        try:
            result = compute_survivability_score(
                financial_data, debt_data, catalyst_data, as_of_date
            )
            result["ticker"] = ticker
            results[ticker] = result
        except Exception as e:
            logger.warning(f"Error computing survivability for {ticker}: {e}")
            results[ticker] = {
                "module": "financial_module_2_survivability",
                "ticker": ticker,
                "score": 0.0,
                "subscores": {"runway": 0.0, "discipline": 0.0, "accel": 0.0, "debt": 0.0},
                "metrics": {},
                "coverage": ["computation_error"],
                "notes": [f"error:{str(e)}"],
            }

    return results


# =============================================================================
# CLI / MAIN
# =============================================================================

def main():
    """Test with sample data."""
    print("=" * 80)
    print("FINANCIAL MODULE 2: SURVIVABILITY & CAPITAL DISCIPLINE")
    print("=" * 80)

    # Test case 1: Healthy company
    healthy = compute_survivability_score({
        'Cash': 500e6,
        'ShortTermInvestments': 200e6,
        'CFO': -80e6,  # $80M annual burn
        'R&D': 60e6,
        'total_operating_expense_ttm': 100e6,
        'LongTermDebt': 50e6,
    })
    print(f"\nHealthy Company:")
    print(f"  Score: {healthy['score']:.1f}")
    print(f"  Subscores: {healthy['subscores']}")
    print(f"  Runway: {healthy['metrics'].get('effective_runway_months', 'N/A')} months")
    print(f"  Coverage: {healthy['coverage']}")

    # Test case 2: Distressed company
    distressed = compute_survivability_score({
        'Cash': 30e6,
        'CFO': -100e6,  # $100M annual burn
        'R&D': 20e6,
        'total_operating_expense_ttm': 120e6,
        'LongTermDebt': 80e6,
        'current_debt': 20e6,
    })
    print(f"\nDistressed Company:")
    print(f"  Score: {distressed['score']:.1f}")
    print(f"  Subscores: {distressed['subscores']}")
    print(f"  Runway: {distressed['metrics'].get('effective_runway_months', 'N/A')} months")
    print(f"  Coverage: {distressed['coverage']}")

    # Test case 3: Non-burner (profitable)
    profitable = compute_survivability_score({
        'Cash': 1000e6,
        'CFO': 200e6,  # Positive cash flow
        'R&D': 150e6,
        'total_operating_expense_ttm': 200e6,
    })
    print(f"\nProfitable Company:")
    print(f"  Score: {profitable['score']:.1f}")
    print(f"  Subscores: {profitable['subscores']}")
    print(f"  Coverage: {profitable['coverage']}")

    # Test case 4: Missing data
    missing = compute_survivability_score({
        'Cash': 100e6,
    })
    print(f"\nMissing Data Company:")
    print(f"  Score: {missing['score']:.1f}")
    print(f"  Subscores: {missing['subscores']}")
    print(f"  Coverage: {missing['coverage']}")

    print("\n" + "=" * 80)
    print("MODULE 2 TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
