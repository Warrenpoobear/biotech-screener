"""
Module 4: Clinical Development Quality (CONTINUOUS SCORING VERSION)

CHANGES FROM ORIGINAL:
- Added phase_progress_bonus (0-5 pts) based on trial completion
- Added enrollment_bonus (0-5 pts) based on trial size
- These break up the discrete phase buckets
- Normalization adjusted to new max score (110 pts)

Expected improvement: Clinical uniqueness from 36% to 70%+
CRITICAL: Eliminates the 42.53 bucket that contains 50% of universe!
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity

RULESET_VERSION = "1.1.0"  # Incremented for continuous scoring

# Phase scores (0-30) - unchanged
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


def _calculate_phase_progress_bonus(trials: List[Dict[str, Any]], lead_phase: str) -> Decimal:
    """
    NEW: Calculate bonus for progress within a phase (0-5 points).
    This breaks up the discrete phase buckets!
    """
    if not trials:
        return Decimal("0")
    
    # Count trials by status in lead phase
    lead_trials = [t for t in trials if t.get("phase") == lead_phase]
    if not lead_trials:
        return Decimal("0")
    
    completed = sum(1 for t in lead_trials if t.get("status", "").lower() == "completed")
    active = sum(1 for t in lead_trials if t.get("status", "").lower() in ("recruiting", "active", "not yet recruiting"))
    total = len(lead_trials)
    
    # Progress bonus based on completion and activity
    if completed > 0:
        # Has completed trials in this phase
        completion_rate = completed / total
        bonus = Decimal(str(completion_rate)) * Decimal("5")
    elif active > 0:
        # Has active but no completed trials
        activity_rate = active / total
        bonus = Decimal(str(activity_rate)) * Decimal("2.5")
    else:
        # No active or completed trials
        bonus = Decimal("0")
    
    return bonus


def _calculate_enrollment_bonus(trials: List[Dict[str, Any]]) -> Decimal:
    """
    NEW: Calculate bonus for enrollment size (0-5 points).
    Adds continuous variation based on trial scale.
    """
    if not trials:
        return Decimal("2")  # Neutral for missing data
    
    # Get enrollments
    enrollments = [t.get("enrollment", 0) for t in trials if t.get("enrollment", 0) > 0]
    
    if not enrollments:
        return Decimal("2")  # Neutral
    
    # Use maximum enrollment as quality indicator
    max_enrollment = max(enrollments)
    
    # Continuous scoring based on size
    if max_enrollment >= 1000:
        bonus = Decimal("5.0")
    elif max_enrollment >= 500:
        # 500-1000: 4.0-5.0
        bonus = Decimal("4.0") + Decimal(str((max_enrollment - 500) / 500)) * Decimal("1.0")
    elif max_enrollment >= 200:
        # 200-500: 3.0-4.0
        bonus = Decimal("3.0") + Decimal(str((max_enrollment - 200) / 300)) * Decimal("1.0")
    elif max_enrollment >= 100:
        # 100-200: 2.0-3.0
        bonus = Decimal("2.0") + Decimal(str((max_enrollment - 100) / 100)) * Decimal("1.0")
    elif max_enrollment >= 50:
        # 50-100: 1.0-2.0
        bonus = Decimal("1.0") + Decimal(str((max_enrollment - 50) / 50)) * Decimal("1.0")
    else:
        # <50: 0.0-1.0
        bonus = Decimal(str(max_enrollment / 50)) * Decimal("1.0")
    
    return bonus


def compute_module_4_clinical_dev(
    trial_records: List[Dict[str, Any]],
    active_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Compute clinical development quality scores with CONTINUOUS sub-scoring.
    
    Max score now: 30 (phase) + 5 (progress) + 5 (enrollment) + 25 (design) + 25 (execution) + 20 (endpoint) = 110
    Normalized to 0-100 scale.
    """
    pit_cutoff = compute_pit_cutoff(as_of_date)
    pit_filtered_count = 0
    
    # Group trials by ticker
    ticker_trials: Dict[str, List[Dict]] = {}
    for trial in trial_records:
        ticker = trial.get("ticker", "").upper()
        if ticker not in active_tickers:
            continue
        
        # PIT filter
        source_date = trial.get("last_update_posted") or trial.get("source_date")
        if source_date and not is_pit_admissible(source_date, pit_cutoff):
            pit_filtered_count += 1
            continue
        
        if ticker not in ticker_trials:
            ticker_trials[ticker] = []
        
        ticker_trials[ticker].append({
            "nct_id": trial.get("nct_id"),
            "phase": _parse_phase(trial.get("phase")),
            "status": trial.get("status", ""),
            "randomized": trial.get("randomized", False),
            "blinded": trial.get("blinded", ""),
            "primary_endpoint": trial.get("primary_endpoint", ""),
            "enrollment": trial.get("enrollment", 0),  # NEW: track enrollment
        })
    
    scores = []
    
    for ticker in active_tickers:
        trials = ticker_trials.get(ticker, [])
        
        # Find lead phase
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
        
        # NEW: Continuous sub-scores to break up buckets
        phase_progress_bonus = _calculate_phase_progress_bonus(trials, lead_phase)
        enrollment_bonus = _calculate_enrollment_bonus(trials)
        
        # Existing scores
        design_score = Decimal("12")
        if trials:
            design_scores = [_score_design(t) for t in trials]
            design_score = max(design_scores)
        
        execution_score = _score_execution(trials)
        endpoint_score = _score_endpoints(trials)
        
        # Total raw score (0-110)
        total_raw = (phase_score + phase_progress_bonus + enrollment_bonus + 
                     design_score + execution_score + endpoint_score)
        
        # Normalize to 0-100 scale
        # Max possible: 30 + 5 + 5 + 25 + 25 + 20 = 110
        total = (total_raw / Decimal("110")) * Decimal("100")
        
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
            "phase_progress_bonus": str(phase_progress_bonus.quantize(Decimal("0.01"))),  # NEW
            "enrollment_bonus": str(enrollment_bonus.quantize(Decimal("0.01"))),  # NEW
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
