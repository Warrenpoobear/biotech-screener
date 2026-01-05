"""
Module 5: Composite Ranker

Combines scores from Modules 2-4 into final composite ranking.
Features:
- Cohort normalization (rank within stage/mcap bucket)
- Uncertainty penalty (missing subfactors)
- Severity gates (sev3 excluded, sev2/sev1 penalties)

Weights: Clinical 40%, Financial 35%, Catalyst 25%
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from statistics import mean, stdev
from typing import Any, Dict, List, Optional, Tuple

from common.provenance import create_provenance
from common.types import Severity

RULESET_VERSION = "1.1.0"

# Default weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "clinical_dev": Decimal("0.40"),
    "financial": Decimal("0.35"),
    "catalyst": Decimal("0.25"),
}

# Severity multipliers
SEVERITY_MULTIPLIERS = {
    Severity.NONE: Decimal("1.0"),
    Severity.SEV1: Decimal("0.90"),  # 10% penalty
    Severity.SEV2: Decimal("0.50"),  # 50% penalty
    Severity.SEV3: Decimal("0.0"),   # Excluded
}

# Minimum cohort size for reliable normalization
MIN_COHORT_SIZE = 5

# Max uncertainty penalty
MAX_UNCERTAINTY_PENALTY = Decimal("0.30")

# Cohort normalization modes
COHORT_MODE_STAGE_MCAP = "stage_mcap"  # Normalize by stage x mcap (granular)
COHORT_MODE_STAGE_ONLY = "stage_only"  # Normalize by stage only (recommended)

# Cohort fallback policies (used when cohort_mode=stage_mcap)
COHORT_FALLBACK_STAGE_ONLY = "stage_only"  # Merge mcap buckets within stage
COHORT_FALLBACK_GLOBAL = "global"           # Use global ranking
COHORT_FALLBACK_NONE = "none"               # No normalization (raw scores)


def _market_cap_bucket(market_cap_mm: Optional[Any]) -> str:
    """
    Classify market cap into bucket.
    
    Handles string, int, float, Decimal inputs.
    """
    if market_cap_mm is None:
        return "unknown"
    
    try:
        # Handle various input types
        if isinstance(market_cap_mm, Decimal):
            mcap = market_cap_mm
        elif isinstance(market_cap_mm, (int, float)):
            mcap = Decimal(str(market_cap_mm))
        else:
            mcap = Decimal(str(market_cap_mm))
    except:
        return "unknown"
    
    if mcap >= 10000:
        return "large"
    elif mcap >= 2000:
        return "mid"
    elif mcap >= 300:
        return "small"
    else:
        return "micro"


def _stage_bucket(lead_phase: Optional[str]) -> str:
    """Classify development stage into bucket."""
    if not lead_phase:
        return "early"
    
    phase = lead_phase.lower()
    if "3" in phase or "approved" in phase:
        return "late"
    elif "2" in phase:
        return "mid"
    else:
        return "early"


def _rank_normalize(values: List[Decimal]) -> List[Decimal]:
    """
    Percentile rank normalization.
    Returns values 0-100 representing percentile within group.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [Decimal("50")]
    
    # Create (value, index) pairs, sort by value
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])
    
    # Assign ranks (handle ties with average rank)
    ranks = [Decimal("0")] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = Decimal(str((i + j - 1) / 2))  # 0-indexed
        for k in range(i, j):
            ranks[indexed[k][1]] = avg_rank
        i = j
    
    # Convert to percentile (0-100)
    result = []
    for r in ranks:
        pct = (r / Decimal(str(n - 1))) * Decimal("100") if n > 1 else Decimal("50")
        result.append(pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    
    return result


def _get_worst_severity(severities: List[str]) -> Severity:
    """Get worst severity from list."""
    priority = {Severity.SEV3: 3, Severity.SEV2: 2, Severity.SEV1: 1, Severity.NONE: 0}
    worst = Severity.NONE
    
    for s in severities:
        try:
            sev = Severity(s)
            if priority[sev] > priority[worst]:
                worst = sev
        except:
            continue
    
    return worst


def _apply_cohort_normalization(members: List[Dict]) -> None:
    """
    Apply rank normalization to cohort members in-place.
    """
    if not members:
        return
    
    # Extract raw scores
    clin_scores = [m["clinical_dev_raw"] or Decimal("0") for m in members]
    fin_scores = [m["financial_raw"] or Decimal("0") for m in members]
    cat_scores = [m["catalyst_raw"] or Decimal("0") for m in members]
    
    # Normalize
    clin_norm = _rank_normalize(clin_scores)
    fin_norm = _rank_normalize(fin_scores)
    cat_norm = _rank_normalize(cat_scores)
    
    for i, m in enumerate(members):
        m["clinical_dev_normalized"] = clin_norm[i]
        m["financial_normalized"] = fin_norm[i]
        m["catalyst_normalized"] = cat_norm[i]
        m["normalization_applied"] = "cohort"


def compute_module_5_composite(
    universe_result: Dict[str, Any],
    financial_result: Dict[str, Any],
    catalyst_result: Dict[str, Any],
    clinical_result: Dict[str, Any],
    as_of_date: str,
    weights: Optional[Dict[str, Decimal]] = None,
    normalization: str = "rank",
    cohort_mode: str = COHORT_MODE_STAGE_ONLY,  # Default to stage-only
) -> Dict[str, Any]:
    """
    Compute composite scores with cohort normalization.
    
    Args:
        universe_result: Module 1 output
        financial_result: Module 2 output
        catalyst_result: Module 3 output
        clinical_result: Module 4 output
        as_of_date: Analysis date
        weights: Override default weights
        normalization: "rank" (default) or "zscore"
        cohort_mode: "stage_only" (recommended) or "stage_mcap" (granular)
    
    Returns:
        {
            "as_of_date": str,
            "normalization_method": str,
            "cohort_mode": str,
            "weights_used": {...},
            "ranked_securities": [{ticker, composite_score, composite_rank, ...}],
            "excluded_securities": [{ticker, reason, flags}],
            "cohort_stats": {...},
            "diagnostic_counts": {...},
            "provenance": {...}
        }
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    
    # Index module outputs by ticker
    active_tickers = {s["ticker"] for s in universe_result.get("active_securities", [])}
    
    financial_by_ticker = {s["ticker"]: s for s in financial_result.get("scores", [])}
    catalyst_by_ticker = {s["ticker"]: s for s in catalyst_result.get("scores", [])}
    clinical_by_ticker = {s["ticker"]: s for s in clinical_result.get("scores", [])}
    
    # Build combined records
    combined = []
    excluded = []
    
    for ticker in active_tickers:
        fin = financial_by_ticker.get(ticker, {})
        cat = catalyst_by_ticker.get(ticker, {})
        clin = clinical_by_ticker.get(ticker, {})
        
        # Get raw scores
        fin_score = Decimal(fin.get("financial_score", "0")) if fin.get("financial_score") else None
        cat_score = Decimal(cat.get("catalyst_score", "0")) if cat.get("catalyst_score") else None
        clin_score = Decimal(clin.get("clinical_score", "0")) if clin.get("clinical_score") else None
        
        # Get severities
        severities = [
            fin.get("severity", "none"),
            cat.get("severity", "none"),
            clin.get("severity", "none"),
        ]
        worst_severity = _get_worst_severity(severities)
        
        # Collect flags
        flags = []
        flags.extend(fin.get("flags", []))
        flags.extend(cat.get("flags", []))
        flags.extend(clin.get("flags", []))
        
        # Check for sev3 exclusion
        if worst_severity == Severity.SEV3:
            excluded.append({
                "ticker": ticker,
                "reason": "sev3_gate",
                "flags": sorted(set(flags)),
            })
            continue
        
        # Get cohort info
        market_cap_mm = fin.get("market_cap_mm")
        lead_phase = clin.get("lead_phase")
        
        combined.append({
            "ticker": ticker,
            "clinical_dev_raw": clin_score,
            "financial_raw": fin_score,
            "catalyst_raw": cat_score,
            "market_cap_bucket": _market_cap_bucket(market_cap_mm),
            "stage_bucket": _stage_bucket(lead_phase),
            "severity": worst_severity,
            "flags": flags,
        })
    
    # Group by cohort for normalization
    # cohort_mode determines the grouping key
    cohorts: Dict[str, List[Dict]] = {}
    
    for rec in combined:
        if cohort_mode == COHORT_MODE_STAGE_ONLY:
            # Normalize by stage only (recommended - avoids small cohort fallbacks)
            key = rec['stage_bucket']
        else:
            # Normalize by stage x mcap (granular - may trigger fallbacks)
            key = f"{rec['stage_bucket']}_{rec['market_cap_bucket']}"
        
        rec["cohort_key"] = key  # Track which cohort for reporting
        
        if key not in cohorts:
            cohorts[key] = []
        cohorts[key].append(rec)
    
    # Identify small cohorts and apply fallback (only relevant for stage_mcap mode)
    cohort_fallbacks: Dict[str, str] = {}
    stage_pools: Dict[str, List[Dict]] = {}  # For stage-only fallback
    
    for cohort_key, members in cohorts.items():
        if len(members) >= MIN_COHORT_SIZE:
            # Normal cohort normalization
            cohort_fallbacks[cohort_key] = "normal"
        else:
            if cohort_mode == COHORT_MODE_STAGE_ONLY:
                # In stage-only mode, small cohorts still get flagged but normalized globally
                cohort_fallbacks[cohort_key] = "small_stage"
                # Fall back to global pool
                if "global" not in stage_pools:
                    stage_pools["global"] = []
                stage_pools["global"].extend(members)
            else:
                # In stage_mcap mode, fall back to stage-only normalization
                stage = cohort_key.split("_")[0]
                cohort_fallbacks[cohort_key] = COHORT_FALLBACK_STAGE_ONLY
                
                if stage not in stage_pools:
                    stage_pools[stage] = []
                stage_pools[stage].extend(members)
    
    # Normalize within normal cohorts
    for cohort_key, members in cohorts.items():
        if cohort_fallbacks[cohort_key] == "normal":
            _apply_cohort_normalization(members)
    
    # Normalize within stage pools (for small cohorts)
    for stage, members in stage_pools.items():
        if len(members) >= MIN_COHORT_SIZE:
            _apply_cohort_normalization(members)
            # Mark members as stage-normalized
            for m in members:
                m["normalization_applied"] = f"stage_{stage}"
        elif len(members) > 0:
            # Even stage pool is too small - fall back to no normalization
            for m in members:
                m["clinical_dev_normalized"] = m["clinical_dev_raw"] or Decimal("0")
                m["financial_normalized"] = m["financial_raw"] or Decimal("0")
                m["catalyst_normalized"] = m["catalyst_raw"] or Decimal("0")
                m["normalization_applied"] = "none"
                m["flags"].append("cohort_too_small_no_normalization")
    
    # Compute composite scores
    for rec in combined:
        clin_n = rec["clinical_dev_normalized"] or Decimal("0")
        fin_n = rec["financial_normalized"] or Decimal("0")
        cat_n = rec["catalyst_normalized"] or Decimal("0")
        
        # Weighted sum
        composite = (
            clin_n * weights["clinical_dev"] +
            fin_n * weights["financial"] +
            cat_n * weights["catalyst"]
        )
        
        # Count missing subfactors
        missing = sum(1 for x in [rec["clinical_dev_raw"], rec["financial_raw"], rec["catalyst_raw"]] if x is None)
        missing_pct = Decimal(str(missing / 3))
        
        # Uncertainty penalty
        uncertainty_penalty = min(MAX_UNCERTAINTY_PENALTY, missing_pct)
        composite = composite * (Decimal("1") - uncertainty_penalty)
        
        # Severity multiplier
        multiplier = SEVERITY_MULTIPLIERS[rec["severity"]]
        composite = composite * multiplier
        
        # Cap at 100
        composite = min(Decimal("100"), max(Decimal("0"), composite))
        
        rec["composite_score"] = composite.quantize(Decimal("0.01"))
        rec["uncertainty_penalty"] = uncertainty_penalty.quantize(Decimal("0.01"))
        rec["missing_subfactor_pct"] = missing_pct.quantize(Decimal("0.01"))
        
        # Add penalty flags
        if uncertainty_penalty > 0:
            rec["flags"].append("uncertainty_penalty_applied")
        if rec["severity"] == Severity.SEV2:
            rec["flags"].append("sev2_penalty_applied")
        elif rec["severity"] == Severity.SEV1:
            rec["flags"].append("sev1_penalty_applied")
    
    # Sort and rank
    combined.sort(key=lambda x: (-x["composite_score"], x["ticker"]))
    for i, rec in enumerate(combined):
        rec["composite_rank"] = i + 1
    
    # Cohort stats with fallback info
    cohort_stats = {}
    for cohort_key, members in cohorts.items():
        scores = [m["composite_score"] for m in members]
        if scores:
            cohort_stats[cohort_key] = {
                "count": len(scores),
                "mean": str(Decimal(str(mean([float(s) for s in scores]))).quantize(Decimal("0.01"))),
                "min": str(min(scores)),
                "max": str(max(scores)),
                "normalization_fallback": cohort_fallbacks.get(cohort_key, "unknown"),
            }
    
    # Format output
    ranked_securities = []
    for rec in combined:
        ranked_securities.append({
            "ticker": rec["ticker"],
            "composite_score": str(rec["composite_score"]),
            "composite_rank": rec["composite_rank"],
            "clinical_dev_normalized": str(rec["clinical_dev_normalized"]),
            "financial_normalized": str(rec["financial_normalized"]),
            "catalyst_normalized": str(rec["catalyst_normalized"]),
            "clinical_dev_raw": str(rec["clinical_dev_raw"]) if rec["clinical_dev_raw"] else None,
            "financial_raw": str(rec["financial_raw"]) if rec["financial_raw"] else None,
            "catalyst_raw": str(rec["catalyst_raw"]) if rec["catalyst_raw"] else None,
            "uncertainty_penalty": str(rec["uncertainty_penalty"]),
            "missing_subfactor_pct": str(rec["missing_subfactor_pct"]),
            "market_cap_bucket": rec["market_cap_bucket"],
            "stage_bucket": rec["stage_bucket"],
            "cohort_key": rec.get("cohort_key", "unknown"),
            "normalization_applied": rec.get("normalization_applied", "cohort"),
            "severity": rec["severity"].value,
            "flags": sorted(set(rec["flags"])),
            "rankable": True,
        })
    
    return {
        "as_of_date": as_of_date,
        "normalization_method": normalization,
        "cohort_mode": cohort_mode,
        "weights_used": {k: str(v) for k, v in weights.items()},
        "ranked_securities": ranked_securities,
        "excluded_securities": excluded,
        "cohort_stats": cohort_stats,
        "diagnostic_counts": {
            "total_input": len(active_tickers),
            "rankable": len(ranked_securities),
            "excluded": len(excluded),
            "cohort_count": len(cohorts),
        },
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": list(active_tickers), "weights": {k: str(v) for k, v in weights.items()}},
            as_of_date,
        ),
    }
