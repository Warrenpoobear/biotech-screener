"""
defensive_overlay_adapter.py - PROPERLY FIXED VERSION

Key fixes:
1. Corrected field names (corr_xbi, drawdown_60d)
2. DYNAMIC position floor that scales with universe size
3. More aggressive inverse-volatility weighting

For 44 stocks: min=1.0%, works well
For 100 stocks: min=0.5%, allows proper differentiation
For 200 stocks: min=0.3%, maximum diversification
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, List, Optional, Tuple

WQ = Decimal("0.0001")  # Weight quantization

def _q(x: Decimal) -> Decimal:
    """Quantize weight to 4 decimal places."""
    return x.quantize(WQ, rounding=ROUND_HALF_UP)

def sanitize_corr(defensive_features: Dict[str, str]) -> Tuple[Optional[Decimal], List[str]]:
    """
    Sanitize correlation data, treating placeholders and invalid values as missing.
    
    Returns: (correlation_value or None, list_of_flags)
    
    Common issues:
    - Placeholder 0.50 when correlation calculation failed
    - Missing field entirely
    - NaN or Infinity values
    - Out of valid range [-1, 1]
    """
    flags: List[str] = []
    PLACEHOLDER_CORR = Decimal("0.50")
    
    # Try both field names
    corr_s = defensive_features.get("corr_xbi") or defensive_features.get("corr_xbi_120d")
    
    if not corr_s:
        flags.append("def_corr_missing")
        return None, flags
    
    try:
        corr = Decimal(str(corr_s))
    except Exception:
        flags.append("def_corr_parse_fail")
        return None, flags
    
    # CRITICAL: Check if Decimal is NaN or Inf BEFORE doing comparisons
    # This prevents InvalidOperation errors
    if not corr.is_finite():
        flags.append("def_corr_not_finite")
        return None, flags
    
    # Treat placeholder as missing
    if corr == PLACEHOLDER_CORR:
        flags.append("def_corr_placeholder_0.50")
        return None, flags
    
    # Validate range (safe now that we know it's finite)
    if corr < Decimal("-1") or corr > Decimal("1"):
        flags.append("def_corr_out_of_range")
        return None, flags
    
    return corr, flags


def defensive_multiplier(defensive_features: Dict[str, str]) -> Tuple[Decimal, List[str]]:
    """
    Calculate defensive multiplier (0.95-1.20 range).
    Rewards only ELITE diversifiers with verified correlation data.
    
    ENHANCED: 
    - Handles correlation placeholders (treats 0.50 as missing)
    - Bigger bonus (1.20x) for truly elite diversifiers
    - Only awards bonus when correlation is verified real
    """
    m = Decimal("1.00")
    notes: List[str] = []

    # Get volatility first (needed for correlation logic)
    vol_s = defensive_features.get("vol_60d")
    vol = None
    if vol_s:
        try:
            vol = Decimal(vol_s)
        except (ValueError, InvalidOperation):
            notes.append("def_vol_parse_fail")

    # Sanitize correlation (handle placeholders)
    corr, corr_flags = sanitize_corr(defensive_features or {})
    notes.extend(corr_flags)
    
    # Correlation multiplier (only if correlation is real)
    if corr is not None:
        # High correlation penalty (always applies)
        if corr > Decimal("0.80"):
            m *= Decimal("0.95")
            notes.append("def_mult_high_corr_0.95")
        
        # Elite diversifier bonus (VERY selective)
        # Requires: corr < 0.30 AND vol < 0.40 AND real correlation data
        elif corr < Decimal("0.30"):
            if vol and vol < Decimal("0.40"):
                m *= Decimal("1.40")  # Maximum bonus to overcome normalization
                notes.append("def_mult_elite_1.40")
            else:
                notes.append("def_skip_not_elite_vol")
        
        # Good diversifier bonus (less selective)
        elif corr < Decimal("0.40"):
            if vol and vol < Decimal("0.50"):
                m *= Decimal("1.10")
                notes.append("def_mult_good_diversifier_1.10")
            else:
                notes.append("def_skip_vol_too_high")

    # Drawdown warning
    dd_s = defensive_features.get("drawdown_60d") or defensive_features.get("drawdown_current")
    if dd_s:
        try:
            if Decimal(dd_s) < Decimal("-0.30"):
                notes.append("def_warn_drawdown_gt_30pct")
        except (ValueError, InvalidOperation):
            notes.append("def_drawdown_parse_fail")

    return m, notes


def raw_inv_vol_weight(defensive_features: Dict[str, str], power: Decimal = Decimal("2.0")) -> Optional[Decimal]:
    """
    Calculate raw inverse-volatility weight with exponential scaling.
    
    ENHANCED: Uses vol^2.0 (default) for maximum differentiation.
    This creates very strong spread between low-vol and high-vol stocks.
    
    Examples (vol^2.0):
    - 20% vol: 1/(0.20^2.0) = 25.0  (very large weight)
    - 25% vol: 1/(0.25^2.0) = 16.0  (large weight)
    - 40% vol: 1/(0.40^2.0) = 6.25  (medium weight)
    - 100% vol: 1/(1.00^2.0) = 1.0  (small weight)
    - 200% vol: 1/(2.00^2.0) = 0.25 (tiny weight)
    
    Args:
        power: Exponent for volatility (2.0 = maximum, 1.8 = aggressive, 1.5 = moderate)
    """
    vol_s = defensive_features.get("vol_60d")
    if not vol_s:
        return None
    try:
        vol = Decimal(vol_s)
        if vol <= 0:
            return None
        return Decimal("1") / (vol ** power)
    except (ValueError, InvalidOperation, ZeroDivisionError):
        return None


def calculate_dynamic_floor(n_securities: int) -> Decimal:
    """
    Calculate dynamic position floor based on universe size.
    
    Logic:
    - 20-50 stocks: 1.0% floor (traditional focused portfolio)
    - 51-100 stocks: 0.5% floor (allows proper differentiation)
    - 101-200 stocks: 0.3% floor (maximum diversification)
    - 201+ stocks: 0.2% floor (ultra-diversified)
    
    This ensures the floor is always well below the average weight,
    allowing inverse-volatility weighting to work properly.
    """
    if n_securities <= 50:
        return Decimal("0.01")  # 1.0%
    elif n_securities <= 100:
        return Decimal("0.005")  # 0.5%
    elif n_securities <= 200:
        return Decimal("0.003")  # 0.3%
    else:
        return Decimal("0.002")  # 0.2%


def apply_caps_and_renormalize(
    records: List[Dict],
    cash_target: Decimal = Decimal("0.10"),
    max_pos: Decimal = Decimal("0.07"),  # 7% max
    min_pos: Optional[Decimal] = None,  # Dynamic - calculated based on universe
    top_n: Optional[int] = None,  # NEW: If set, only invest in top N includable names
) -> None:
    """
    Apply position caps and renormalize weights.
    Mutates records in-place, setting record["position_weight"].
    
    ENHANCED: 
    - Automatically calculates appropriate floor based on universe size
    - Max position reduced to 7% for better risk management at scale
    - NEW: Top-N selection for conviction-based portfolios
    
    Args:
        top_n: If provided, only size the top N includable names post-ranking.
               All others get zero weight and "NOT_IN_TOP_N" flag.
               This increases max weights naturally without changing math.
               Recommended: 60 for balanced, 40 for high conviction.
    """
    investable = Decimal("1.0") - cash_target

    # Get all potentially includable securities (before top-N cut)
    included_all = [r for r in records if r.get("rankable", True)]
    
    # Apply top-N selection if specified
    if top_n is not None and len(included_all) > top_n:
        # Records should already be sorted by composite_rank
        # (or if not, sort them now by rank ascending)
        included_all_sorted = sorted(included_all, key=lambda x: x.get("composite_rank", 999))
        
        # Select top N
        included = included_all_sorted[:top_n]
        excluded_by_topn = included_all_sorted[top_n:]
        
        # Mark excluded securities
        for r in excluded_by_topn:
            r["rankable"] = False  # Mark as non-rankable
            position_flags = r.get("position_flags", [])
            if isinstance(position_flags, list):
                position_flags.append("NOT_IN_TOP_N")
            else:
                position_flags = ["NOT_IN_TOP_N"]
            r["position_flags"] = position_flags
            r["position_weight"] = "0.0000"
    else:
        included = included_all
    
    # Excluded get zero weight
    for r in records:
        if not r.get("rankable", True):
            r["position_weight"] = "0.0000"

    if not included:
        return

    # Calculate dynamic floor if not provided
    if min_pos is None:
        min_pos = calculate_dynamic_floor(len(included))

    # Collect raw weights
    raw = []
    for r in included:
        w = r.get("_position_weight_raw")
        raw.append(w if isinstance(w, Decimal) else Decimal("0"))

    total_raw = sum(raw)
    
    # Fallback to equal-weight if no valid weights
    if total_raw <= 0:
        n = Decimal(str(max(1, len(included))))
        ew = investable / n
        for r in included:
            r["position_weight"] = str(_q(max(min_pos, min(max_pos, ew))))
        return

    # Initial normalize to investable capital
    weights = [(w / total_raw) * investable for w in raw]

    # Apply caps and floors
    capped = [max(min_pos, min(max_pos, w)) for w in weights]

    # Renormalize after capping, then re-enforce floor
    # (renormalization can push weights below floor)
    total_capped = sum(capped)
    if total_capped > 0:
        scale = investable / total_capped
        capped = [w * scale for w in capped]

        # Re-enforce floor after scaling (iterative approach)
        for _ in range(3):  # Max 3 iterations to converge
            floor_violations = sum(1 for w in capped if w < min_pos)
            if floor_violations == 0:
                break
            # Bring floor violations up to floor, reduce others proportionally
            floored = []
            excess_needed = Decimal("0")
            for w in capped:
                if w < min_pos:
                    excess_needed += min_pos - w
                    floored.append(min_pos)
                else:
                    floored.append(w)
            # Reduce weights above floor proportionally
            above_floor = [(i, w) for i, w in enumerate(floored) if w > min_pos]
            if above_floor and excess_needed > 0:
                total_above = sum(w - min_pos for i, w in above_floor)
                if total_above > 0:
                    for i, w in above_floor:
                        reduction = (w - min_pos) / total_above * excess_needed
                        floored[i] = max(min_pos, w - reduction)
            capped = floored

    # Quantize all weights first
    quantized = [_q(w) for w in capped]

    # Calculate residual and allocate to anchor (highest weight) for exact sum
    quantized_sum = sum(quantized)
    residual = investable - quantized_sum

    if residual != Decimal("0") and quantized:
        # Find anchor: highest weight position (first by index if tied)
        anchor_idx = max(range(len(quantized)), key=lambda i: (quantized[i], -i))
        # Add residual to anchor and re-quantize
        quantized[anchor_idx] = _q(quantized[anchor_idx] + residual)

    # Assign final weights
    for r, w in zip(included, quantized):
        r["position_weight"] = str(w)


def enrich_with_defensive_overlays(
    output: Dict,
    scores_by_ticker: Dict[str, Dict],
    apply_multiplier: bool = True,
    apply_position_sizing: bool = True,
    top_n: Optional[int] = None,  # NEW: Top-N selection for conviction portfolios
) -> Dict:
    """
    Enrich Module 5 output with defensive overlays.
    
    This function:
    1. Applies defensive multiplier to existing composite scores
    2. Calculates position weights using inverse-volatility
    3. Adds defensive_notes and position_weight fields to each security
    4. NEW: Optionally applies top-N selection for conviction portfolios
    
    Args:
        output: Output dict from rank_securities()
        scores_by_ticker: Dict with defensive_features per ticker
        apply_multiplier: If True, apply correlation-based score multiplier
        apply_position_sizing: If True, calculate position weights
        top_n: If provided, only invest in top N names (e.g., 60 for balanced, 40 for conviction)
    
    Returns:
        Modified output dict (mutated in-place, also returned for convenience)
    """
    ranked = output.get("ranked_securities", [])
    
    if not ranked:
        return output
    
    # Step 1: Calculate raw weights for position sizing (needed for Step 3)
    # This must run BEFORE multiplier application if position sizing is enabled
    if apply_position_sizing:
        for rec in ranked:
            ticker = rec["ticker"]
            ticker_data = scores_by_ticker.get(ticker, {})
            defensive_features = ticker_data.get("defensive_features", {})
            rec["_position_weight_raw"] = raw_inv_vol_weight(defensive_features or {})

    # Step 2: Apply defensive multiplier to composite scores (optional)
    if apply_multiplier:
        for rec in ranked:
            ticker = rec["ticker"]
            ticker_data = scores_by_ticker.get(ticker, {})
            defensive_features = ticker_data.get("defensive_features", {})

            # Get current composite score
            current_score = Decimal(rec["composite_score"])

            # Apply multiplier
            mult, notes = defensive_multiplier(defensive_features or {})
            adjusted_score = current_score * mult

            # Cap at 100
            adjusted_score = min(Decimal("100"), max(Decimal("0"), adjusted_score))

            # Update score
            rec["composite_score_before_defensive"] = str(current_score)
            rec["composite_score"] = str(adjusted_score.quantize(Decimal("0.01")))
            rec["defensive_multiplier"] = str(mult)
            rec["defensive_notes"] = notes

    # Step 3: Re-rank after multiplier application
    if apply_multiplier:
        # Re-sort by adjusted composite score
        ranked.sort(key=lambda x: (-Decimal(x["composite_score"]), x["ticker"]))
        # Re-assign ranks
        for i, rec in enumerate(ranked):
            rec["composite_rank"] = i + 1

    # Step 4: Apply position sizing with dynamic floor and optional top-N selection
    if apply_position_sizing:
        apply_caps_and_renormalize(ranked, top_n=top_n)  # Pass top_n parameter
        
        # Add position sizing diagnostics
        total_weight = sum(Decimal(r["position_weight"]) for r in ranked)
        nonzero = sum(1 for r in ranked if Decimal(r["position_weight"]) > 0)
        
        if "diagnostic_counts" not in output:
            output["diagnostic_counts"] = {}
        
        output["diagnostic_counts"]["with_nonzero_weight"] = nonzero
        output["diagnostic_counts"]["total_allocated_weight"] = str(total_weight.quantize(Decimal("0.0001")))
        
        # Add top-N diagnostic if applied
        if top_n is not None:
            output["diagnostic_counts"]["top_n_cutoff"] = top_n
            excluded_by_topn = sum(1 for r in ranked if "NOT_IN_TOP_N" in r.get("position_flags", []))
            output["diagnostic_counts"]["excluded_by_top_n"] = excluded_by_topn

    # Convert any remaining Decimal objects to strings for JSON serialization
    for rec in ranked:
        # Remove internal _position_weight_raw field (not needed in output)
        if "_position_weight_raw" in rec:
            del rec["_position_weight_raw"]
    
    return output


def validate_defensive_integration(output: Dict) -> None:
    """
    Validate defensive overlay integration.
    Prints validation results to console.
    """
    ranked = output.get("ranked_securities", [])
    
    print("\n" + "="*60)
    print("DEFENSIVE OVERLAY VALIDATION")
    print("="*60)
    
    # 1. Check weights sum
    total_weight = sum(Decimal(r.get("position_weight", "0")) for r in ranked)
    expected = Decimal("0.9000")
    tolerance = Decimal("0.0001")
    
    if abs(total_weight - expected) >= tolerance:
        print(f"[WARN] Weights sum: {total_weight} (expected {expected}, diff: {abs(total_weight - expected)})")
    else:
        print(f"[OK] Weights sum: {total_weight} (target: {expected})")
    
    # 2. Check excluded have zero weight
    excluded_with_weight = [
        r["ticker"] for r in ranked 
        if not r.get("rankable", True) and Decimal(r.get("position_weight", "0")) != 0
    ]
    if excluded_with_weight:
        print(f"[WARN] {len(excluded_with_weight)} excluded securities have non-zero weight: {excluded_with_weight}")
    else:
        print("[OK] All excluded securities have zero weight")
    
    # 3. Count securities with defensive adjustments
    with_def_notes = sum(1 for r in ranked if r.get("defensive_notes"))
    print(f"[OK] {with_def_notes}/{len(ranked)} securities have defensive adjustments")
    
    # 4. Weight distribution
    nonzero_weights = [r for r in ranked if Decimal(r.get("position_weight", "0")) > 0]
    if nonzero_weights:
        weights = [Decimal(r["position_weight"]) for r in nonzero_weights]
        print("\nPosition sizing:")
        print(f"  - {len(nonzero_weights)} positions")
        print(f"  - Max weight: {max(weights):.4f} ({max(weights)*100:.2f}%)")
        print(f"  - Min weight: {min(weights):.4f} ({min(weights)*100:.2f}%)")
        print(f"  - Avg weight: {sum(weights)/len(weights):.4f}")
        print(f"  - Range: {max(weights)/min(weights):.1f}:1")

        # Calculate and show dynamic floor used
        n = len(nonzero_weights)
        floor = calculate_dynamic_floor(n)
        print(f"  - Dynamic floor: {floor:.4f} ({floor*100:.2f}%) for {n} securities")
    
    # 5. Top 10 with weights
    print("\nTop 10 holdings:")
    print(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Def Notes'}")
    print("-" * 60)
    for r in ranked[:10]:
        notes_str = ", ".join(r.get("defensive_notes", [])) if r.get("defensive_notes") else "-"
        print(f"{r['composite_rank']:<6}{r['ticker']:<8}{r['composite_score']:<10}{r.get('position_weight', '0.0000'):<10}{notes_str}")
    
    print("="*60)


if __name__ == "__main__":
    print("Testing defensive_overlay_adapter with dynamic floor...")
    
    # Test dynamic floor calculation
    print("\nDynamic floor calculation:")
    for n in [20, 44, 50, 80, 100, 150, 200, 300]:
        floor = calculate_dynamic_floor(n)
        avg = Decimal("0.90") / Decimal(str(n))
        ratio = floor / avg
        print(f"  {n:3} securities: floor={floor:.4f} ({floor*100:.2f}%), avg={avg:.4f}, floor/avg={ratio:.2f}x")
    
    print("\n[OK] Test complete!")
