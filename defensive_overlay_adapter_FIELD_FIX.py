"""
defensive_overlay_adapter.py - FIXED VERSION

Drop-in adapter for adding defensive overlays to existing Module 5.
Minimal changes to your module_5_composite.py code.

FIXES:
- Corrected field name: corr_xbi_120d → corr_xbi
- Corrected field name: drawdown_current → drawdown_60d
- Added fallback for both field names for compatibility

Usage in module_5_composite.py:
    from defensive_overlay_adapter import enrich_with_defensive_overlays
    
    # After your existing rank_securities() returns the output:
    output = rank_securities(...)
    enrich_with_defensive_overlays(output, scores_by_ticker)
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, List, Optional, Tuple

WQ = Decimal("0.0001")  # Weight quantization

def _q(x: Decimal) -> Decimal:
    """Quantize weight to 4 decimal places."""
    return x.quantize(WQ, rounding=ROUND_HALF_UP)

def defensive_multiplier(defensive_features: Dict[str, str]) -> Tuple[Decimal, List[str]]:
    """
    Calculate defensive multiplier (0.95-1.05 range).
    Small multiplicative adjustment for correlation-based diversification.
    """
    m = Decimal("1.00")
    notes: List[str] = []

    # Correlation multiplier (reward diversification)
    # FIXED: Try both field names for compatibility
    corr_s = defensive_features.get("corr_xbi") or defensive_features.get("corr_xbi_120d")
    if corr_s:
        try:
            corr = Decimal(corr_s)
            if corr > Decimal("0.80"):
                m *= Decimal("0.95")
                notes.append("def_mult_high_corr_0.95")
            elif corr < Decimal("0.40"):
                m *= Decimal("1.05")
                notes.append("def_mult_low_corr_1.05")
        except (ValueError, TypeError, InvalidOperation):
            pass

    # Drawdown warning
    # FIXED: Try both field names for compatibility
    dd_s = defensive_features.get("drawdown_60d") or defensive_features.get("drawdown_current")
    if dd_s:
        try:
            if Decimal(dd_s) < Decimal("-0.30"):
                notes.append("def_warn_drawdown_gt_30pct")
        except (ValueError, TypeError, InvalidOperation):
            pass

    return m, notes


def raw_inv_vol_weight(defensive_features: Dict[str, str]) -> Optional[Decimal]:
    """Calculate raw inverse-volatility weight."""
    vol_s = defensive_features.get("vol_60d")
    if not vol_s:
        return None
    try:
        vol = Decimal(vol_s)
        if vol <= 0:
            return None
        return Decimal("1") / vol
    except (ValueError, TypeError, InvalidOperation):
        return None


def apply_caps_and_renormalize(
    records: List[Dict],
    cash_target: Decimal = Decimal("0.10"),
    max_pos: Decimal = Decimal("0.08"),
    min_pos: Decimal = Decimal("0.01"),
) -> None:
    """
    Apply position caps and renormalize weights.
    Mutates records in-place, setting record["position_weight"].
    """
    investable = Decimal("1.0") - cash_target

    # Only include rankable securities
    included = [r for r in records if r.get("rankable", True)]
    
    # Excluded get zero weight
    for r in records:
        if not r.get("rankable", True):
            r["position_weight"] = "0.0000"

    if not included:
        return

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

    # Renormalize after capping
    total_capped = sum(capped)
    if total_capped > 0:
        scale = investable / total_capped
        capped = [w * scale for w in capped]

    # Assign final weights
    for r, w in zip(included, capped):
        r["position_weight"] = str(_q(w))


def enrich_with_defensive_overlays(
    output: Dict,
    scores_by_ticker: Dict[str, Dict],
    apply_multiplier: bool = True,
    apply_position_sizing: bool = True,
) -> Dict:
    """
    Enrich Module 5 output with defensive overlays.
    
    This function:
    1. Applies defensive multiplier to existing composite scores
    2. Calculates position weights using inverse-volatility
    3. Adds defensive_notes and position_weight fields to each security
    
    Args:
        output: Output dict from rank_securities()
        scores_by_ticker: Dict with defensive_features per ticker
        apply_multiplier: If True, apply correlation-based score multiplier
        apply_position_sizing: If True, calculate position weights
    
    Returns:
        Modified output dict (mutated in-place, also returned for convenience)
    
    Usage:
        output = rank_securities(scores_by_ticker, ...)
        enrich_with_defensive_overlays(output, scores_by_ticker)
    """
    ranked = output.get("ranked_securities", [])
    
    if not ranked:
        return output
    
    # Step 1: Apply defensive multiplier to composite scores
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
            
            # Store raw weight for position sizing
            rec["_position_weight_raw"] = raw_inv_vol_weight(defensive_features or {})
    
    # Step 2: Re-rank after multiplier application
    if apply_multiplier:
        # Re-sort by adjusted composite score
        ranked.sort(key=lambda x: (-Decimal(x["composite_score"]), x["ticker"]))
        # Re-assign ranks
        for i, rec in enumerate(ranked):
            rec["composite_rank"] = i + 1
    
    # Step 3: Apply position sizing
    if apply_position_sizing:
        apply_caps_and_renormalize(ranked)
        
        # Add position sizing diagnostics
        total_weight = sum(Decimal(r["position_weight"]) for r in ranked)
        nonzero = sum(1 for r in ranked if Decimal(r["position_weight"]) > 0)
        
        if "diagnostic_counts" not in output:
            output["diagnostic_counts"] = {}
        
        output["diagnostic_counts"]["with_nonzero_weight"] = nonzero
        output["diagnostic_counts"]["total_allocated_weight"] = str(total_weight.quantize(Decimal("0.0001")))

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
        print(f"⚠️  Weights sum: {total_weight} (expected {expected}, diff: {abs(total_weight - expected)})")
    else:
        print(f"✓ Weights sum: {total_weight} (target: {expected})")
    
    # 2. Check excluded have zero weight
    excluded_with_weight = [
        r["ticker"] for r in ranked 
        if not r.get("rankable", True) and Decimal(r.get("position_weight", "0")) != 0
    ]
    if excluded_with_weight:
        print(f"⚠️  {len(excluded_with_weight)} excluded securities have non-zero weight: {excluded_with_weight}")
    else:
        print(f"✓ All excluded securities have zero weight")
    
    # 3. Count securities with defensive adjustments
    with_def_notes = sum(1 for r in ranked if r.get("defensive_notes"))
    print(f"✓ {with_def_notes}/{len(ranked)} securities have defensive adjustments")
    
    # 4. Weight distribution
    nonzero_weights = [r for r in ranked if Decimal(r.get("position_weight", "0")) > 0]
    if nonzero_weights:
        weights = [Decimal(r["position_weight"]) for r in nonzero_weights]
        print(f"\nPosition sizing:")
        print(f"  • {len(nonzero_weights)} positions")
        print(f"  • Max weight: {max(weights):.4f} ({max(weights)*100:.2f}%)")
        print(f"  • Min weight: {min(weights):.4f} ({min(weights)*100:.2f}%)")
        print(f"  • Avg weight: {sum(weights)/len(weights):.4f}")
    
    # 5. Top 10 with weights
    print(f"\nTop 10 holdings:")
    print(f"{'Rank':<6}{'Ticker':<8}{'Score':<10}{'Weight':<10}{'Def Notes'}")
    print("-" * 60)
    for r in ranked[:10]:
        notes_str = ", ".join(r.get("defensive_notes", [])) if r.get("defensive_notes") else "-"
        print(f"{r['composite_rank']:<6}{r['ticker']:<8}{r['composite_score']:<10}{r.get('position_weight', '0.0000'):<10}{notes_str}")
    
    print("="*60)


if __name__ == "__main__":
    # Test with sample data
    print("Testing defensive_overlay_adapter...")
    
    sample_output = {
        "ranked_securities": [
            {
                "ticker": "VRTX",
                "composite_score": "85.50",
                "composite_rank": 1,
                "rankable": True,
            },
            {
                "ticker": "GOSS",
                "composite_score": "45.00",
                "composite_rank": 2,
                "rankable": False,  # Excluded by SEV3
            }
        ],
        "diagnostic_counts": {}
    }
    
    sample_scores = {
        "VRTX": {
            "defensive_features": {
                "vol_60d": "0.25",
                "corr_xbi": "0.35",  # FIXED: using corr_xbi
                "drawdown_60d": "-0.10",  # FIXED: using drawdown_60d
            }
        },
        "GOSS": {
            "defensive_features": {
                "vol_60d": "0.60",
                "corr_xbi": "0.85",  # FIXED: using corr_xbi
                "drawdown_60d": "-0.45",  # FIXED: using drawdown_60d
            }
        }
    }
    
    enrich_with_defensive_overlays(sample_output, sample_scores)
    validate_defensive_integration(sample_output)
    
    print("\n✓ Test complete!")
