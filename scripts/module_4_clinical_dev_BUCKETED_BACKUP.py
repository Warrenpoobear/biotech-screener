"""
Module 4: Clinical Development Quality

Scores clinical programs on:
- Phase advancement (most advanced phase)
- Trial design quality (endpoints, controls)
- Execution track record
- Endpoint strength

Input: Trial records with design details
Output: Clinical development scores per security
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity

RULESET_VERSION = "1.0.0"

# Phase scores (0-30)
PHASE_SCORES = {
    "approved": Decimal("30"),
    "phase 3": Decimal("25"),
    "phase 2/3": Decimal("22"),
    "phase 2": Decimal("18"),
    "phase 1/2": Decimal("12"),
    "phase 1": Decimal("8"),
    "preclinical": Decimal("3"),
}

# Design quality indicators
STRONG_ENDPOINTS = frozenset([
    "overall survival", "os", "progression-free survival", "pfs",
    "complete response", "cr", "objective response rate", "orr",
])

WEAK_ENDPOINTS = frozenset([
    "biomarker", "pharmacokinetic", "pk", "safety",
])


def _parse_phase(phase_str: Optional[str]) -> str:
    """Normalize phase string."""
    if not phase_str:
        return "unknown"
    phase = phase_str.lower().strip()
    
    if "approved" in phase or "4" in phase:
        return "approved"
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
    elif "preclinical" in phase:
        return "preclinical"
    
    return "unknown"


def _score_design(trial: Dict[str, Any]) -> Decimal:
    """Score trial design quality (0-25)."""
    score = Decimal("12")  # Base
    
    # Randomized bonus
    if trial.get("randomized"):
        score += Decimal("5")
    
    # Double-blind bonus
    if trial.get("blinded", "").lower() in ("double", "double-blind"):
        score += Decimal("4")
    
    # Endpoint strength
    primary_endpoint = (trial.get("primary_endpoint") or "").lower()
    
    for strong in STRONG_ENDPOINTS:
        if strong in primary_endpoint:
            score += Decimal("4")
            break
    else:
        for weak in WEAK_ENDPOINTS:
            if weak in primary_endpoint:
                score -= Decimal("3")
                break
    
    return max(Decimal("0"), min(Decimal("25"), score))


def _score_execution(trials: List[Dict[str, Any]]) -> Decimal:
    """Score execution track record (0-25)."""
    if not trials:
        return Decimal("12")
    
    completed = sum(1 for t in trials if t.get("status", "").lower() == "completed")
    terminated = sum(1 for t in trials if t.get("status", "").lower() in ("terminated", "withdrawn"))
    total = len(trials)
    
    if total == 0:
        return Decimal("12")
    
    completion_rate = Decimal(str(completed / total))
    termination_rate = Decimal(str(terminated / total))
    
    # Base on completion rate
    score = Decimal("12") + (completion_rate * Decimal("10")) - (termination_rate * Decimal("8"))
    
    return max(Decimal("0"), min(Decimal("25"), score))


def _score_endpoints(trials: List[Dict[str, Any]]) -> Decimal:
    """Score endpoint portfolio strength (0-20)."""
    if not trials:
        return Decimal("10")
    
    strong_count = 0
    weak_count = 0
    
    for t in trials:
        endpoint = (t.get("primary_endpoint") or "").lower()
        for strong in STRONG_ENDPOINTS:
            if strong in endpoint:
                strong_count += 1
                break
        for weak in WEAK_ENDPOINTS:
            if weak in endpoint:
                weak_count += 1
                break
    
    score = Decimal("10")
    score += Decimal(str(strong_count)) * Decimal("2")
    score -= Decimal(str(weak_count)) * Decimal("1")
    
    return max(Decimal("0"), min(Decimal("20"), score))


def compute_module_4_clinical_dev(
    trial_records: List[Dict[str, Any]],
    active_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Compute clinical development quality scores.
    
    Args:
        trial_records: List with ticker, nct_id, phase, status, randomized, blinded, primary_endpoint
        active_tickers: Tickers from Module 1
        as_of_date: Analysis date
    
    Returns:
        {
            "as_of_date": str,
            "scores": [{ticker, clinical_score, phase_score, design_score, execution_score, 
                       endpoint_score, lead_phase, flags, severity}],
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
        # This prevents lookahead bias from future trial updates
        source_date = trial.get("last_update_posted") or trial.get("source_date")
        if source_date and not is_pit_admissible(source_date, pit_cutoff):
            pit_filtered_count += 1
            continue  # Skip future data
        
        if ticker not in ticker_trials:
            ticker_trials[ticker] = []
        
        ticker_trials[ticker].append({
            "nct_id": trial.get("nct_id"),
            "phase": _parse_phase(trial.get("phase")),
            "status": trial.get("status", ""),
            "randomized": trial.get("randomized", False),
            "blinded": trial.get("blinded", ""),
            "primary_endpoint": trial.get("primary_endpoint", ""),
        })
    
    scores = []
    
    for ticker in active_tickers:
        trials = ticker_trials.get(ticker, [])
        
        # Find lead phase (most advanced)
        lead_phase = "preclinical"
        lead_phase_score = Decimal("3")
        
        for t in trials:
            phase = t["phase"]
            phase_score = PHASE_SCORES.get(phase, Decimal("0"))
            if phase_score > lead_phase_score:
                lead_phase = phase
                lead_phase_score = phase_score
        
        # Component scores
        phase_score = lead_phase_score
        
        # Design score from best trial
        design_score = Decimal("12")
        if trials:
            design_scores = [_score_design(t) for t in trials]
            design_score = max(design_scores)
        
        execution_score = _score_execution(trials)
        endpoint_score = _score_endpoints(trials)
        
        # Total (0-100)
        total = phase_score + design_score + execution_score + endpoint_score
        
        # Flags and severity
        flags = []
        severity = Severity.NONE
        
        if lead_phase in ("preclinical", "phase 1"):
            flags.append("early_stage")
        
        if not trials:
            flags.append("no_trials")
            severity = Severity.SEV1
        
        scores.append({
            "ticker": ticker,
            "clinical_score": str(total.quantize(Decimal("0.01"))),
            "phase_score": str(phase_score),
            "design_score": str(design_score.quantize(Decimal("0.01"))),
            "execution_score": str(execution_score.quantize(Decimal("0.01"))),
            "endpoint_score": str(endpoint_score.quantize(Decimal("0.01"))),
            "lead_phase": lead_phase,
            "trial_count": len(trials),
            "flags": flags,
            "severity": severity.value,
        })
    
    return {
        "as_of_date": as_of_date,
        "scores": sorted(scores, key=lambda x: x["ticker"]),
        "diagnostic_counts": {
            "scored": len(scores),
            "total_trials": sum(len(v) for v in ticker_trials.values()),
            "pit_filtered": pit_filtered_count,
        },
        "provenance": create_provenance(RULESET_VERSION, {"tickers": active_tickers}, pit_cutoff),
    }
