"""
score_to_er.py - Score to Expected Return Conversion

Institutional-grade conversion from composite scores to expected excess returns.

Methodology:
1. score → rank → percentile (Blom plotting position)
2. percentile → z-score (Acklam inverse normal approximation)
3. z-score → expected excess return (z × λ)

Design:
- DETERMINISTIC: Stable sorts, no randomness
- STDLIB-ONLY: No scipy/numpy required
- PIT-SAFE: Only uses data available at as_of_date
- AUDIT-FRIENDLY: Full provenance metadata

Default λ = 0.08 (8% per 1σ per year):
- Conservative biotech-appropriate starting point
- Meaningfully above generic equity factors (3-6%/σ/yr)
- Not so high as to become "fantasy optimizer bait"
- Can be calibrated per-regime once backtest harness exists

Author: Wake Robin Capital Management
Version: 1.0.0
"""
from __future__ import annotations

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Provisional biotech-appropriate lambda (excess return per 1σ per year)
# Conservative starting point - can be calibrated with historical backtest
DEFAULT_LAMBDA_ANNUAL = Decimal("0.08")  # 8% per 1σ per year

# ER model identifier for provenance
ER_MODEL_ID = "zscore_linear_lambda"
ER_MODEL_VERSION = "1.0.0"


# =============================================================================
# INVERSE NORMAL CDF (Acklam Approximation)
# =============================================================================

def _norm_ppf(p: float) -> float:
    """
    Compute inverse standard normal CDF (probit function).

    Uses Acklam's approximation - accurate to ~1.15e-9 for p in (0,1).
    Deterministic, stdlib-only, no scipy required.

    Args:
        p: Probability in (0, 1)

    Returns:
        z-score such that Φ(z) = p

    Reference:
        Peter J. Acklam's rational approximation algorithm
        https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
    """
    # Guard against edge cases
    if p <= 0.0:
        return -10.0  # Practical lower bound
    if p >= 1.0:
        return 10.0   # Practical upper bound

    # Coefficients for rational approximation
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ]

    # Split points for approximation regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        # Lower tail approximation
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den

    if p > p_high:
        # Upper tail approximation
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
        return num / den

    # Central region approximation
    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    return num / den


# =============================================================================
# SCORE → PERCENTILE → Z-SCORE
# =============================================================================

def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def attach_rank_and_z(
    rows: List[Dict[str, Any]],
    score_key: str = "composite_score",
    ticker_key: str = "ticker",
    max_invalid_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Compute rank percentile and z-score for each row.

    Mutates rows in-place. Adds:
        - score_rank_pct: Percentile in (0, 1), best rank → highest percentile
        - score_z: Standard normal z-score, best rank → positive z

    Uses continuity-corrected percentile: p = (N - rank + 0.5) / N
    This maps best rank (1) to high percentile → positive z → positive ER.

    Deterministic: uses stable sort by (score DESC, ticker ASC).

    Args:
        rows: List of security dicts with score_key field
        score_key: Field name for composite score
        ticker_key: Field name for ticker (used for tie-breaking)
        max_invalid_pct: Max fraction of invalid scores before degraded mode (default 10%)

    Returns:
        Dict with validation metadata: {valid_count, invalid_count, degraded, warnings}
    """
    n = len(rows)
    result = {
        "valid_count": 0,
        "invalid_count": 0,
        "degraded": False,
        "warnings": [],
    }

    if n == 0:
        return result

    # Validate scores before sorting
    valid_count = 0
    invalid_tickers = []

    for r in rows:
        score_raw = r.get(score_key)
        ticker = r.get(ticker_key, "???")

        if score_raw is None:
            invalid_tickers.append(f"{ticker}:missing")
        else:
            try:
                float(score_raw)
                valid_count += 1
            except (ValueError, TypeError):
                invalid_tickers.append(f"{ticker}:non-numeric")

    invalid_count = n - valid_count
    invalid_pct = invalid_count / n if n > 0 else 0

    result["valid_count"] = valid_count
    result["invalid_count"] = invalid_count

    # Check degradation threshold
    if invalid_pct > max_invalid_pct:
        result["degraded"] = True
        result["warnings"].append(
            f"ER DEGRADED: {invalid_count}/{n} ({invalid_pct:.1%}) rows have invalid '{score_key}'. "
            f"Threshold: {max_invalid_pct:.0%}. First 5: {invalid_tickers[:5]}"
        )
        # In degraded mode, set z=0 for all rows (neutral ER)
        for r in rows:
            r["score_rank_pct"] = 0.5
            r["score_z"] = 0.0
        return result

    # Log warning if any invalid (but below threshold)
    if invalid_count > 0:
        result["warnings"].append(
            f"ER WARNING: {invalid_count}/{n} ({invalid_pct:.1%}) rows have invalid '{score_key}'. "
            f"Proceeding with valid rows. Invalid: {invalid_tickers[:5]}"
        )

    # Stable sort: highest score first, then alphabetically by ticker for ties
    # Use _safe_float to handle any edge cases gracefully
    ordered = sorted(
        rows,
        key=lambda r: (-_safe_float(r.get(score_key), 0.0), str(r.get(ticker_key, ""))),
    )

    # Continuity-corrected percentile: p = (N - rank + 0.5) / N
    # Best rank (1) → high percentile → positive z → positive expected return
    for i, r in enumerate(ordered, start=1):
        # rank = i (1-indexed), so p = (n - i + 0.5) / n
        p = (n - i + 0.5) / n

        # Clamp to avoid edge issues with inverse CDF
        p = max(0.001, min(0.999, p))

        z = _norm_ppf(p)

        # Store as floats for JSON serialization
        r["score_rank_pct"] = round(p, 6)
        r["score_z"] = round(z, 4)

    return result


# =============================================================================
# Z-SCORE → EXPECTED EXCESS RETURN
# =============================================================================

def attach_expected_return(
    rows: List[Dict[str, Any]],
    lambda_annual: Decimal = DEFAULT_LAMBDA_ANNUAL,
    include_daily: bool = True,
) -> None:
    """
    Compute expected excess return from z-score.

    ER = z × λ (excess return vs universe/benchmark)

    Mutates rows in-place. Adds:
        - expected_excess_return_annual: Annualized expected alpha
        - expected_excess_return_daily: Daily expected alpha (optional)

    Args:
        rows: List of security dicts with score_z field
        lambda_annual: Excess return per 1σ per year (default 0.08)
        include_daily: Also compute daily ER (default True)
    """
    trading_days = Decimal("252")

    for r in rows:
        z = r.get("score_z")
        if z is None:
            continue

        # Convert z to Decimal for consistency
        z_dec = Decimal(str(z))

        # Expected annual excess return
        er_annual = z_dec * lambda_annual
        r["expected_excess_return_annual"] = float(
            er_annual.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        )

        # Expected daily excess return (optional)
        if include_daily:
            er_daily = er_annual / trading_days
            r["expected_excess_return_daily"] = float(
                er_daily.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
            )


# =============================================================================
# COMBINED CONVENIENCE FUNCTION
# =============================================================================

def compute_expected_returns(
    ranked_securities: List[Dict[str, Any]],
    lambda_annual: Decimal = DEFAULT_LAMBDA_ANNUAL,
    score_key: str = "composite_score",
    include_daily: bool = True,
) -> Dict[str, Any]:
    """
    Full pipeline: score → percentile → z → expected return.

    Mutates ranked_securities in-place. Adds:
        - score_rank_pct: Blom plotting position
        - score_z: Standard normal z-score
        - expected_excess_return_annual: Annualized expected alpha
        - expected_excess_return_daily: Daily expected alpha (optional)

    Returns provenance metadata for audit trail.

    Args:
        ranked_securities: List of security dicts (already ranked by score)
        lambda_annual: Excess return per 1σ per year
        score_key: Field name for composite score
        include_daily: Also compute daily ER

    Returns:
        Provenance dict with model metadata
    """
    # Step 1: Rank → percentile → z-score (with validation)
    validation = attach_rank_and_z(ranked_securities, score_key=score_key)

    # Step 2: z-score → expected return (skip if degraded)
    if not validation.get("degraded", False):
        attach_expected_return(
            ranked_securities,
            lambda_annual=lambda_annual,
            include_daily=include_daily,
        )
    else:
        # In degraded mode, set ER=0 for all rows
        for r in ranked_securities:
            r["expected_excess_return_annual"] = 0.0
            if include_daily:
                r["expected_excess_return_daily"] = 0.0

    # Return provenance metadata with validation status
    provenance = {
        "er_model": ER_MODEL_ID,
        "er_model_version": ER_MODEL_VERSION,
        "lambda_annual": str(lambda_annual),
        "lambda_interpretation": "annualized_excess_return_per_sigma",
        "z_method": "acklam_inverse_normal",
        "percentile_method": "continuity_corrected",
        "validation": {
            "valid_count": validation["valid_count"],
            "invalid_count": validation["invalid_count"],
            "degraded": validation["degraded"],
        },
    }

    # Add warnings to provenance if any
    if validation["warnings"]:
        provenance["warnings"] = validation["warnings"]

    return provenance


# =============================================================================
# VALIDATION
# =============================================================================

def validate_er_output(rows: List[Dict[str, Any]]) -> List[str]:
    """
    Validate expected return output.

    Returns list of warnings (empty if all OK).
    """
    warnings = []

    if not rows:
        warnings.append("No rows to validate")
        return warnings

    # Check all rows have required fields
    missing_z = sum(1 for r in rows if "score_z" not in r)
    missing_er = sum(1 for r in rows if "expected_excess_return_annual" not in r)

    if missing_z > 0:
        warnings.append(f"{missing_z} rows missing score_z")
    if missing_er > 0:
        warnings.append(f"{missing_er} rows missing expected_excess_return_annual")

    # Check z-score distribution is reasonable
    z_values = [r["score_z"] for r in rows if "score_z" in r]
    if z_values:
        z_min = min(z_values)
        z_max = max(z_values)

        # Z-scores should be roughly symmetric around 0
        if abs(z_min + z_max) > 1.0:
            warnings.append(f"Z-scores not symmetric: min={z_min:.2f}, max={z_max:.2f}")

        # Z-scores should span reasonable range for the sample size
        n = len(z_values)
        expected_range = 2.0 * _norm_ppf(1 - 0.5/n) if n > 1 else 0
        actual_range = z_max - z_min
        if actual_range < expected_range * 0.5:
            warnings.append(f"Z-score range too narrow: {actual_range:.2f} vs expected {expected_range:.2f}")

    return warnings


if __name__ == "__main__":
    # Test with sample data
    print("Testing score_to_er.py...")
    print(f"Default λ: {DEFAULT_LAMBDA_ANNUAL} (8% per 1σ per year)")
    print()

    # Sample ranked securities
    test_rows = [
        {"ticker": "BEST", "composite_score": "95.50"},
        {"ticker": "GOOD", "composite_score": "82.30"},
        {"ticker": "OKAY", "composite_score": "65.10"},
        {"ticker": "POOR", "composite_score": "45.80"},
        {"ticker": "WORST", "composite_score": "22.15"},
    ]

    # Compute expected returns
    provenance = compute_expected_returns(test_rows)

    print("Results:")
    print(f"{'Ticker':<8} {'Score':<10} {'Rank%':<10} {'Z':<10} {'ER Ann':<12} {'ER Daily'}")
    print("-" * 70)
    for r in test_rows:
        print(
            f"{r['ticker']:<8} "
            f"{r['composite_score']:<10} "
            f"{r['score_rank_pct']:<10.4f} "
            f"{r['score_z']:<10.4f} "
            f"{r['expected_excess_return_annual']:<12.4f} "
            f"{r.get('expected_excess_return_daily', 'N/A')}"
        )

    print()
    print("Provenance:")
    for k, v in provenance.items():
        print(f"  {k}: {v}")

    # Validate
    warnings = validate_er_output(test_rows)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[OK] Validation passed")
