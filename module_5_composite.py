"""
Module 5: Composite Ranker

Combines scores from Modules 2-4 into final composite ranking.
Features:
- Cohort normalization (rank within stage/mcap bucket)
- Uncertainty penalty (missing subfactors)
- Severity gates (sev3 excluded, sev2/sev1 penalties)
- Co-invest overlay (tie-breaker, PIT-safe)

Weights: Clinical 40%, Financial 35%, Catalyst 25%
"""
from __future__ import annotations
from datetime import datetime, date

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple

from common.provenance import create_provenance
from common.types import Severity

RULESET_VERSION = "1.2.1"  # Bumped for determinism + exception fixes


def _decimal_mean(values: List[Decimal]) -> Decimal:
    """Compute mean using Decimal arithmetic only (no floats)."""
    if not values:
        return Decimal("0")
    return sum(values) / Decimal(len(values))

# Default weights (sum to 1.0)
# SCHEME A: Financial = pure risk control via severity multipliers only
# Rationale: Financial health is downside asymmetry/dilution pressure, not alpha.
# The severity gates (SEV3 excluded, SEV2/SEV1 penalized) handle risk control.
# This is regime-robust: avoids 2021 "easy funding" vs 2022 "funding shut" flip.
DEFAULT_WEIGHTS = {
    "clinical_dev": Decimal("0.60"),  # Primary driver (was 0.40)
    "financial": Decimal("0.00"),     # Risk control via severity only (was 0.35)
    "catalyst": Decimal("0.40"),      # Secondary driver (was 0.25)
}

# Enhanced weights when PoS scoring is available (sum to 1.0)
# Same philosophy: financial = 0, risk control via severity gates
ENHANCED_WEIGHTS = {
    "clinical_dev": Decimal("0.45"),  # Primary driver
    "financial": Decimal("0.00"),     # Risk control via severity only
    "catalyst": Decimal("0.30"),      # Secondary driver
    "pos": Decimal("0.25"),           # PoS from BIO benchmarks
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


# =============================================================================
# CO-INVEST OVERLAY HELPERS
# =============================================================================

def _quarter_from_date(d: date) -> str:
    """Convert date to quarter string (e.g., '2025Q3')."""
    q = (d.month - 1) // 3 + 1
    return f"{d.year}Q{q}"


def _enrich_with_coinvest(
    ticker: str,
    coinvest_signals: dict,
    as_of_date: date,
) -> dict:
    """
    Look up co-invest signal for a ticker and return overlay fields.
    
    PIT Rule: Only include filings where filing_date < as_of_date.
    This ensures we don't "know" about positions until they're actually filed.
    """
    signal = coinvest_signals.get(ticker)
    
    if not signal:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_quarter": None,
            "coinvest_published_at_max": None,
            "coinvest_usable": False,
            "coinvest_flags": ["no_signal"],
        }
    
    # PIT filter: only count positions from filings before as_of_date
    pit_positions = [
        p for p in signal.positions
        if p.filing_date < as_of_date
    ]
    
    if not pit_positions:
        return {
            "coinvest_overlap_count": 0,
            "coinvest_holders": [],
            "coinvest_quarter": _quarter_from_date(signal.positions[0].report_date) if signal.positions else None,
            "coinvest_published_at_max": None,
            "coinvest_usable": False,
            "coinvest_flags": ["filings_not_yet_public"],
        }
    
    # Compute PIT-safe metrics
    holders = sorted(set(p.manager_name for p in pit_positions))
    max_filing_date = max(p.filing_date for p in pit_positions)
    report_quarter = _quarter_from_date(pit_positions[0].report_date)
    
    flags = []
    if len(pit_positions) < len(signal.positions):
        flags.append("partial_manager_coverage")
    if signal.ticker is None:
        flags.append("cusip_unmapped")
    
    return {
        "coinvest_overlap_count": len(holders),
        "coinvest_holders": holders,
        "coinvest_quarter": report_quarter,
        "coinvest_published_at_max": max_filing_date.isoformat(),
        "coinvest_usable": True,
        "coinvest_flags": sorted(flags),
    }


# =============================================================================
# COHORT HELPERS
# =============================================================================

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
    except (ValueError, TypeError, InvalidOperation):
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
        except (ValueError, KeyError):
            continue
    
    return worst


def _apply_cohort_normalization(members: List[Dict], include_pos: bool = False) -> None:
    """
    Apply rank normalization to cohort members in-place.

    Args:
        members: List of member records to normalize
        include_pos: Whether to normalize PoS scores (if available)
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

    # Optionally normalize PoS scores
    pos_norm = None
    if include_pos:
        pos_scores = [m.get("pos_raw") or Decimal("0") for m in members]
        # Only normalize if any non-zero scores
        if any(p > 0 for p in pos_scores):
            pos_norm = _rank_normalize(pos_scores)

    for i, m in enumerate(members):
        m["clinical_dev_normalized"] = clin_norm[i]
        m["financial_normalized"] = fin_norm[i]
        m["catalyst_normalized"] = cat_norm[i]
        if pos_norm:
            m["pos_normalized"] = pos_norm[i]
        else:
            m["pos_normalized"] = m.get("pos_raw") or Decimal("0")
        m["normalization_applied"] = "cohort"


# =============================================================================
# MAIN COMPOSITE FUNCTION
# =============================================================================

def compute_module_5_composite(
    universe_result: Dict[str, Any],
    financial_result: Dict[str, Any],
    catalyst_result: Dict[str, Any],
    clinical_result: Dict[str, Any],
    as_of_date: str,
    weights: Optional[Dict[str, Decimal]] = None,
    normalization: str = "rank",
    coinvest_signals: Optional[Dict] = None,
    cohort_mode: str = COHORT_MODE_STAGE_ONLY,
    enhancement_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute composite scores with cohort normalization and co-invest overlay.

    Args:
        universe_result: Module 1 output
        financial_result: Module 2 output
        catalyst_result: Module 3 output
        clinical_result: Module 4 output
        as_of_date: Analysis date (YYYY-MM-DD)
        weights: Override default weights
        normalization: "rank" (default) or "zscore"
        coinvest_signals: Optional dict of ticker -> AggregatedSignal from 13F aggregator
        cohort_mode: "stage_only" (recommended) or "stage_mcap" (granular)
        enhancement_result: Optional enhancement module results (PoS, regime, SI)

    Returns:
        {
            "as_of_date": str,
            "normalization_method": str,
            "cohort_mode": str,
            "weights_used": {...},
            "ranked_securities": [{ticker, composite_score, composite_rank, coinvest_*, ...}],
            "excluded_securities": [{ticker, reason, flags}],
            "cohort_stats": {...},
            "diagnostic_counts": {...},
            "coinvest_coverage": {...} or None,
            "enhancement_applied": bool,
            "provenance": {...}
        }
    """
    # Extract enhancement data if provided
    enhancement_applied = enhancement_result is not None
    pos_by_ticker = {}
    si_by_ticker = {}
    regime_adjustments = {}
    regime_name = "UNKNOWN"

    if enhancement_result:
        # Extract PoS scores
        pos_scores = enhancement_result.get("pos_scores", {}).get("scores", [])
        for ps in pos_scores:
            ticker = ps.get("ticker")
            if ticker:
                pos_by_ticker[ticker.upper()] = ps

        # Extract SI signals (handle None values)
        si_data = enhancement_result.get("short_interest_scores") or {}
        si_scores = si_data.get("scores", [])
        for si in si_scores:
            ticker = si.get("ticker")
            if ticker:
                si_by_ticker[ticker.upper()] = si

        # Extract regime adjustments
        regime_data = enhancement_result.get("regime", {})
        regime_name = regime_data.get("regime", "UNKNOWN")
        regime_adjustments = regime_data.get("signal_adjustments", {})

    # Use enhanced weights if PoS data is available
    if weights is None:
        if pos_by_ticker:
            weights = ENHANCED_WEIGHTS.copy()
        else:
            weights = DEFAULT_WEIGHTS.copy()

    # Apply regime-based weight adjustments
    if regime_adjustments and pos_by_ticker:
        # Regime adjusts relative importance of signals
        # quality_adj > 1 in BEAR = favor quality/financial
        # momentum_adj < 1 in BEAR = dampen catalyst/momentum signals
        quality_adj = regime_adjustments.get("quality", Decimal("1.0"))
        momentum_adj = regime_adjustments.get("momentum", Decimal("1.0"))
        catalyst_adj = regime_adjustments.get("catalyst", Decimal("1.0"))

        # Convert to Decimal if needed
        if not isinstance(quality_adj, Decimal):
            quality_adj = Decimal(str(quality_adj))
        if not isinstance(momentum_adj, Decimal):
            momentum_adj = Decimal(str(momentum_adj))
        if not isinstance(catalyst_adj, Decimal):
            catalyst_adj = Decimal(str(catalyst_adj))

        # Adjust weights based on regime
        adjusted_weights = {
            "clinical_dev": weights.get("clinical_dev", Decimal("0.30")),
            "financial": weights.get("financial", Decimal("0.30")) * quality_adj,
            "catalyst": weights.get("catalyst", Decimal("0.20")) * catalyst_adj * momentum_adj,
            "pos": weights.get("pos", Decimal("0.20")),
        }

        # Renormalize to sum to 1.0
        total = sum(adjusted_weights.values())
        if total > 0:
            weights = {k: (v / total).quantize(Decimal("0.0001")) for k, v in adjusted_weights.items()}

    # Index module outputs by ticker
    active_tickers = {s["ticker"] for s in universe_result.get("active_securities", [])}

    financial_by_ticker = {s["ticker"]: s for s in financial_result.get("scores", [])}
    catalyst_by_ticker = catalyst_result.get("summaries", {})  # Module 3 returns summaries dict
    clinical_by_ticker = {s["ticker"]: s for s in clinical_result.get("scores", [])}
    
    # Build combined records
    combined = []
    excluded = []
    
    for ticker in active_tickers:
        fin = financial_by_ticker.get(ticker, {})
        cat = catalyst_by_ticker.get(ticker, {})
        clin = clinical_by_ticker.get(ticker, {})

        # Handle TickerCatalystSummaryV2 objects (dataclass) or dicts
        # vNext uses score_override/score_blended, legacy uses catalyst_score_net
        if hasattr(cat, 'score_blended'):
            # It's a TickerCatalystSummaryV2 dataclass
            cat_score_val = float(cat.score_blended)
            cat_proximity = float(cat.catalyst_proximity_score) if hasattr(cat, 'catalyst_proximity_score') else 0
            cat_delta = float(cat.catalyst_delta_score) if hasattr(cat, 'catalyst_delta_score') else 0
            cat_severity = 'sev1' if cat.severe_negative_flag else 'none'
            cat_flags = []  # vNext dataclass uses flags dict, not list
            cat_severe_neg = cat.severe_negative_flag
        elif isinstance(cat, dict):
            # Check for vNext dict format first (nested scores structure)
            scores = cat.get("scores", {})
            if scores and "score_blended" in scores:
                cat_score_val = float(scores.get("score_blended", 50))
                cat_proximity = float(scores.get("catalyst_proximity_score", 0))
                cat_delta = float(scores.get("catalyst_delta_score", 0))
            elif "score_blended" in cat:
                # Flat vNext format
                cat_score_val = float(cat.get("score_blended", 50))
                cat_proximity = float(cat.get("catalyst_proximity_score", 0))
                cat_delta = float(cat.get("catalyst_delta_score", 0))
            else:
                # Legacy format
                cat_score_val = cat.get("catalyst_score_net")
                cat_proximity = 0
                cat_delta = 0
            flags_dict = cat.get("flags", {})
            cat_severity = 'sev1' if flags_dict.get("severe_negative_flag", False) else cat.get("severity", "none")
            cat_flags = []
            cat_severe_neg = flags_dict.get("severe_negative_flag", False) if isinstance(flags_dict, dict) else False
        elif hasattr(cat, 'catalyst_score_net'):
            # Legacy TickerCatalystSummary dataclass
            cat_score_val = cat.catalyst_score_net
            cat_proximity = 0
            cat_delta = 0
            cat_severity = getattr(cat, 'severity', 'none') if hasattr(cat, 'severity') else 'none'
            cat_flags = getattr(cat, 'flags', []) if hasattr(cat, 'flags') else []
            cat_severe_neg = getattr(cat, 'severe_negative_flag', False)
        else:
            cat_score_val = None
            cat_proximity = 0
            cat_delta = 0
            cat_severity = "none"
            cat_flags = []
            cat_severe_neg = False

        # Get raw scores
        fin_score = Decimal(fin.get("financial_normalized", "0")) if fin.get("financial_normalized") else None
        cat_score = Decimal(str(cat_score_val)) if cat_score_val is not None else None
        clin_score = Decimal(clin.get("clinical_score", "0")) if clin.get("clinical_score") else None

        # Get severities
        severities = [
            fin.get("severity", "none"),
            cat_severity,
            clin.get("severity", "none"),
        ]
        worst_severity = _get_worst_severity(severities)

        # Collect flags
        flags = []
        flags.extend(fin.get("flags", []))
        flags.extend(cat_flags)
        flags.extend(clin.get("flags", []))

        # Extract enhancement data for this ticker
        pos_data = pos_by_ticker.get(ticker.upper(), {})
        si_data = si_by_ticker.get(ticker.upper(), {})

        # Get PoS score (0-100 scale)
        pos_score = None
        if pos_data:
            pos_score_raw = pos_data.get("pos_score")
            if pos_score_raw is not None:
                pos_score = Decimal(str(pos_score_raw))

        # Get SI signals and add flags
        si_squeeze_potential = None
        si_signal_direction = None
        if si_data:
            si_squeeze_potential = si_data.get("squeeze_potential")
            si_signal_direction = si_data.get("signal_direction")
            si_flags = si_data.get("flags", [])
            if si_squeeze_potential in ("EXTREME", "HIGH"):
                flags.append(f"si_squeeze_{si_squeeze_potential.lower()}")
            if si_data.get("crowding_risk") in ("HIGH", "EXTREME"):
                flags.append("si_crowding_risk")
            if "COVERING_TREND" in si_flags:
                flags.append("si_covering_trend")

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
            "pos_raw": pos_score,
            "catalyst_proximity": Decimal(str(cat_proximity)) if cat_proximity else Decimal("0"),
            "catalyst_delta": Decimal(str(cat_delta)) if cat_delta else Decimal("0"),
            "market_cap_bucket": _market_cap_bucket(market_cap_mm),
            "stage_bucket": _stage_bucket(lead_phase),
            "severity": worst_severity,
            "flags": flags,
            "si_squeeze_potential": si_squeeze_potential,
            "si_signal_direction": si_signal_direction,
        })
    
    # Group by cohort for normalization
    cohorts: Dict[str, List[Dict]] = {}
    
    for rec in combined:
        if cohort_mode == COHORT_MODE_STAGE_ONLY:
            key = rec['stage_bucket']
        else:
            key = f"{rec['stage_bucket']}_{rec['market_cap_bucket']}"
        
        rec["cohort_key"] = key
        
        if key not in cohorts:
            cohorts[key] = []
        cohorts[key].append(rec)
    
    # Identify small cohorts and apply fallback
    cohort_fallbacks: Dict[str, str] = {}
    stage_pools: Dict[str, List[Dict]] = {}
    
    for cohort_key, members in cohorts.items():
        if len(members) >= MIN_COHORT_SIZE:
            cohort_fallbacks[cohort_key] = "normal"
        else:
            if cohort_mode == COHORT_MODE_STAGE_ONLY:
                cohort_fallbacks[cohort_key] = "small_stage"
                if "global" not in stage_pools:
                    stage_pools["global"] = []
                stage_pools["global"].extend(members)
            else:
                stage = cohort_key.split("_")[0]
                cohort_fallbacks[cohort_key] = COHORT_FALLBACK_STAGE_ONLY
                
                if stage not in stage_pools:
                    stage_pools[stage] = []
                stage_pools[stage].extend(members)
    
    # Normalize within normal cohorts
    for cohort_key, members in cohorts.items():
        if cohort_fallbacks[cohort_key] == "normal":
            _apply_cohort_normalization(members, include_pos=enhancement_applied)

    # Normalize within stage pools (for small cohorts)
    for stage, members in stage_pools.items():
        if len(members) >= MIN_COHORT_SIZE:
            _apply_cohort_normalization(members, include_pos=enhancement_applied)
            for m in members:
                m["normalization_applied"] = f"stage_{stage}"
        elif len(members) > 0:
            for m in members:
                m["clinical_dev_normalized"] = m["clinical_dev_raw"] or Decimal("0")
                m["financial_normalized"] = m["financial_raw"] or Decimal("0")
                m["catalyst_normalized"] = m["catalyst_raw"] or Decimal("0")
                m["pos_normalized"] = m.get("pos_raw") or Decimal("0")
                m["normalization_applied"] = "none"
                m["flags"].append("cohort_too_small_no_normalization")
    
    # Compute composite scores
    for rec in combined:
        clin_n = rec["clinical_dev_normalized"] or Decimal("0")
        fin_n = rec["financial_normalized"] or Decimal("0")
        cat_n = rec["catalyst_normalized"] or Decimal("0")
        pos_n = rec.get("pos_normalized") or Decimal("0")

        # Catalyst proximity bonus (upcoming catalysts boost score)
        # Scale: 0-100 proximity -> 0-5 point bonus
        cat_proximity = rec.get("catalyst_proximity", Decimal("0"))
        proximity_bonus = (cat_proximity / Decimal("20")).quantize(Decimal("0.01"))  # Max +5 pts

        # Catalyst delta adjustment (recent changes)
        # Scale: -50 to +50 delta -> -2.5 to +2.5 point adjustment
        cat_delta = rec.get("catalyst_delta", Decimal("0"))
        delta_adjustment = (cat_delta / Decimal("20")).quantize(Decimal("0.01"))  # Max ±2.5 pts

        # ROBUSTNESS FIX: Renormalize weights based on available data
        # This prevents missing data from acting as a negative signal
        available_weights = {}
        if rec["clinical_dev_raw"] is not None:
            available_weights["clinical_dev"] = weights.get("clinical_dev", Decimal("0.60"))
        if rec["catalyst_raw"] is not None:
            available_weights["catalyst"] = weights.get("catalyst", Decimal("0.40"))
        # Financial weight is 0 in Scheme A, but include if non-zero
        if rec["financial_raw"] is not None and weights.get("financial", Decimal("0")) > 0:
            available_weights["financial"] = weights.get("financial", Decimal("0"))
        if pos_n > 0 and "pos" in weights:
            available_weights["pos"] = weights["pos"]

        # Renormalize to sum to 1.0
        weight_sum = sum(available_weights.values()) or Decimal("1")
        renorm_weights = {k: v / weight_sum for k, v in available_weights.items()}

        # Weighted sum with renormalized weights
        composite = Decimal("0")
        if "clinical_dev" in renorm_weights:
            composite += clin_n * renorm_weights["clinical_dev"]
        if "catalyst" in renorm_weights:
            composite += cat_n * renorm_weights["catalyst"]
        if "financial" in renorm_weights:
            composite += fin_n * renorm_weights["financial"]
        if "pos" in renorm_weights:
            composite += pos_n * renorm_weights["pos"]
            rec["flags"].append("pos_score_applied")

        # Add proximity and delta adjustments
        composite += proximity_bonus + delta_adjustment

        # Track missing data for diagnostics (but no penalty)
        subfactors = [rec["clinical_dev_raw"], rec["catalyst_raw"]]
        if enhancement_applied:
            subfactors.append(rec.get("pos_raw"))
        missing = sum(1 for x in subfactors if x is None)
        missing_pct = Decimal(str(missing / len(subfactors))) if subfactors else Decimal("0")

        # Flag if renormalization was applied
        if len(available_weights) < len([w for w in weights.values() if w > 0]):
            rec["flags"].append("weights_renormalized_missing_data")
        
        # Severity multiplier
        multiplier = SEVERITY_MULTIPLIERS[rec["severity"]]
        composite = composite * multiplier
        
        # Cap at 100
        composite = min(Decimal("100"), max(Decimal("0"), composite))
        
        rec["composite_score"] = composite.quantize(Decimal("0.01"))
        rec["missing_subfactor_pct"] = missing_pct.quantize(Decimal("0.01"))
        rec["weights_renormalized"] = len(available_weights) < len([w for w in weights.values() if w > 0])

        # Add severity penalty flags
        if rec["severity"] == Severity.SEV2:
            rec["flags"].append("sev2_penalty_applied")
        elif rec["severity"] == Severity.SEV1:
            rec["flags"].append("sev1_penalty_applied")
    
    # Enrich with co-invest overlay (if provided)
    for rec in combined:
        if coinvest_signals:
            as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            rec["coinvest"] = _enrich_with_coinvest(rec["ticker"], coinvest_signals, as_of_dt)
        else:
            rec["coinvest"] = None
    
    # Sort and rank (tie-breaker: coinvest_overlap_count DESC, then ticker ASC)
    def sort_key(x):
        coinvest_count = x["coinvest"]["coinvest_overlap_count"] if x["coinvest"] else 0
        market_cap_mm = x.get("market_cap_mm", 0) or 0  # Market cap for deterministic tiebreak
        return (-x["composite_score"], -market_cap_mm, -coinvest_count, x["ticker"])
    combined.sort(key=sort_key)
    for i, rec in enumerate(combined):
        rec["composite_rank"] = i + 1
    
    # Cohort stats with fallback info
    cohort_stats = {}
    for cohort_key, members in cohorts.items():
        scores = [m["composite_score"] for m in members]
        if scores:
            cohort_stats[cohort_key] = {
                "count": len(scores),
                "mean": str(_decimal_mean(scores).quantize(Decimal("0.01"))),
                "min": str(min(scores)),
                "max": str(max(scores)),
                "normalization_fallback": cohort_fallbacks.get(cohort_key, "unknown"),
            }
    
    # Format output
    ranked_securities = []
    for rec in combined:
        coinvest = rec.get("coinvest") or {}
        security_data = {
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
            # Co-invest overlay fields
            "coinvest_overlap_count": coinvest.get("coinvest_overlap_count", 0),
            "coinvest_holders": coinvest.get("coinvest_holders", []),
            "coinvest_quarter": coinvest.get("coinvest_quarter"),
            "coinvest_published_at_max": coinvest.get("coinvest_published_at_max"),
            "coinvest_usable": coinvest.get("coinvest_usable", False),
            "coinvest_flags": coinvest.get("coinvest_flags", []),
        }

        # Add enhancement fields if available
        if enhancement_applied:
            pos_normalized = rec.get("pos_normalized")
            security_data["pos_normalized"] = str(pos_normalized) if pos_normalized else None
            security_data["pos_raw"] = str(rec.get("pos_raw")) if rec.get("pos_raw") else None
            security_data["si_squeeze_potential"] = rec.get("si_squeeze_potential")
            security_data["si_signal_direction"] = rec.get("si_signal_direction")

        ranked_securities.append(security_data)
    
    # Compute co-invest coverage diagnostics
    coinvest_coverage = None
    if coinvest_signals:
        coinvest_coverage = {
            "rankable": len(ranked_securities),
            "with_overlap_ge_1": sum(1 for r in ranked_securities if r["coinvest_overlap_count"] >= 1),
            "with_overlap_ge_2": sum(1 for r in ranked_securities if r["coinvest_overlap_count"] >= 2),
            "unmapped_cusips_count": sum(1 for r in ranked_securities if "cusip_unmapped" in r.get("coinvest_flags", [])),
        }
    
    # Enhancement diagnostics
    enhancement_diagnostics = None
    if enhancement_applied:
        pos_applied_count = sum(1 for r in ranked_securities if "pos_score_applied" in r.get("flags", []))
        si_squeeze_count = sum(1 for r in ranked_securities if r.get("si_squeeze_potential") in ("EXTREME", "HIGH"))
        enhancement_diagnostics = {
            "regime": regime_name,
            "regime_adjustments": {k: str(v) for k, v in regime_adjustments.items()} if regime_adjustments else {},
            "pos_scores_applied": pos_applied_count,
            "si_squeeze_signals": si_squeeze_count,
        }

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
        "coinvest_coverage": coinvest_coverage,
        "enhancement_applied": enhancement_applied,
        "enhancement_diagnostics": enhancement_diagnostics,
        "provenance": create_provenance(
            RULESET_VERSION,
            {"tickers": list(active_tickers), "weights": {k: str(v) for k, v in weights.items()}},
            as_of_date,
        ),
    }
