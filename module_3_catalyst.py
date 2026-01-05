"""
Module 3: Catalyst Analysis

Identifies upcoming catalysts (trial readouts) and scores by proximity.
Registry-anchored: Uses primary_completion_date from ClinicalTrials.gov.

Input: Trial records with NCT IDs, phases, completion dates
Output: Catalyst scores per security (nearest readout timing)
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity

RULESET_VERSION = "1.0.0"

# Phase weights for catalyst importance
PHASE_WEIGHTS = {
    "phase 3": Decimal("40"),
    "phase 2/3": Decimal("35"),
    "phase 2": Decimal("25"),
    "phase 1/2": Decimal("15"),
    "phase 1": Decimal("10"),
}

# Proximity scoring (days until catalyst)
PROXIMITY_BRACKETS = [
    (30, Decimal("50")),   # < 30 days: imminent
    (90, Decimal("40")),   # 30-90 days: near-term
    (180, Decimal("30")),  # 90-180 days: medium-term
    (365, Decimal("20")),  # 180-365 days: forward
    (730, Decimal("10")),  # 1-2 years: distant
]


def _parse_phase(phase_str: Optional[str]) -> str:
    """Normalize phase string."""
    if not phase_str:
        return "unknown"
    phase = phase_str.lower().strip()
    
    # Normalize common variations
    if "3" in phase:
        if "2" in phase:
            return "phase 2/3"
        return "phase 3"
    elif "2" in phase:
        if "1" in phase:
            return "phase 1/2"
        return "phase 2"
    elif "1" in phase:
        return "phase 1"
    
    return phase


def _days_until(target_date: str, as_of: str) -> Optional[int]:
    """Calculate days until target date."""
    try:
        target = date.fromisoformat(target_date[:10])
        current = date.fromisoformat(as_of)
        return (target - current).days
    except (ValueError, TypeError, AttributeError):
        return None


def _compute_proximity_score(days: Optional[int]) -> Decimal:
    """Score based on days until catalyst."""
    if days is None or days < 0:
        return Decimal("0")
    
    for threshold, score in PROXIMITY_BRACKETS:
        if days <= threshold:
            return score
    
    return Decimal("5")  # > 2 years


def compute_module_3_catalyst(
    trial_records: List[Dict[str, Any]],
    active_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Compute catalyst scores based on upcoming trial readouts.
    
    Args:
        trial_records: List with ticker, nct_id, phase, primary_completion_date, status
        active_tickers: Tickers from Module 1
        as_of_date: Analysis date
    
    Returns:
        {
            "as_of_date": str,
            "scores": [{ticker, catalyst_score, nearest_catalyst, days_to_catalyst, phase, flags}],
            "no_catalyst_tickers": [str],
            "diagnostic_counts": {...},
            "provenance": {...}
        }
    """
    pit_cutoff = compute_pit_cutoff(as_of_date)
    
    # Track PIT-filtered trials for diagnostics
    pit_filtered_count = 0
    
    # Group trials by ticker
    ticker_trials: Dict[str, List[Dict]] = {}
    for trial in trial_records:
        ticker = trial.get("ticker", "").upper()
        if ticker not in active_tickers:
            continue
        
        # PIT FILTER: Only include trials with data available before cutoff
        # This prevents lookahead bias from future trial announcements
        source_date = trial.get("last_update_posted") or trial.get("source_date")
        if source_date and not is_pit_admissible(source_date, pit_cutoff):
            pit_filtered_count += 1
            continue  # Skip future data
        
        # Must have future primary completion date
        pcd = trial.get("primary_completion_date")
        if not pcd:
            continue
        
        days = _days_until(pcd, as_of_date)
        if days is None or days < 0:
            continue  # Past or invalid
        
        # Status filter (only active-ish trials)
        status = trial.get("status", "").lower()
        if status in ("withdrawn", "terminated", "suspended"):
            continue
        
        if ticker not in ticker_trials:
            ticker_trials[ticker] = []
        
        ticker_trials[ticker].append({
            "nct_id": trial.get("nct_id"),
            "phase": _parse_phase(trial.get("phase")),
            "pcd": pcd,
            "days": days,
            "status": status,
        })
    
    scores = []
    no_catalyst = []
    
    for ticker in active_tickers:
        if ticker not in ticker_trials or not ticker_trials[ticker]:
            no_catalyst.append(ticker)
            scores.append({
                "ticker": ticker,
                "catalyst_score": "0",
                "nearest_catalyst": None,
                "days_to_catalyst": None,
                "phase": None,
                "flags": ["no_upcoming_catalyst"],
                "severity": Severity.SEV1.value,
            })
            continue
        
        # Find best catalyst (highest phase weight * proximity)
        trials = ticker_trials[ticker]
        best = None
        best_score = Decimal("-1")
        
        for t in trials:
            phase_weight = PHASE_WEIGHTS.get(t["phase"], Decimal("5"))
            prox_score = _compute_proximity_score(t["days"])
            combined = (phase_weight + prox_score) / Decimal("2")
            
            if combined > best_score:
                best_score = combined
                best = t
        
        flags = []
        severity = Severity.NONE
        
        if best["days"] <= 30:
            flags.append("imminent_catalyst")
        
        scores.append({
            "ticker": ticker,
            "catalyst_score": str(best_score.quantize(Decimal("0.01"))),
            "nearest_catalyst": best["nct_id"],
            "days_to_catalyst": best["days"],
            "phase": best["phase"],
            "flags": flags,
            "severity": severity.value,
        })
    
    return {
        "as_of_date": as_of_date,
        "scores": sorted(scores, key=lambda x: x["ticker"]),
        "no_catalyst_tickers": sorted(no_catalyst),
        "diagnostic_counts": {
            "with_catalyst": len(scores) - len(no_catalyst),
            "no_catalyst": len(no_catalyst),
            "total_trials_evaluated": sum(len(v) for v in ticker_trials.values()),
            "pit_filtered": pit_filtered_count,
        },
        "provenance": create_provenance(RULESET_VERSION, {"tickers": active_tickers}, pit_cutoff),
    }
