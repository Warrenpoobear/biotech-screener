"""
Module 4: Clinical Development Quality (CONTINUOUS SCORING VERSION v2)
HOTFIX: Fixed None handling in enrollment bonus
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity

RULESET_VERSION = "1.1.0"

PHASE_SCORES = {
    "approved": Decimal("30"),
    "phase 3": Decimal("25"),
    "phase 2/3": Decimal("22"),
    "phase 2": Decimal("18"),
    "phase 1/2": Decimal("12"),
    "phase 1": Decimal("8"),
    "preclinical": Decimal("3"),
}

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
    score = Decimal("12")
    
    if trial.get("randomized"):
        score += Decimal("5")
    
    if trial.get("blinded", "").lower() in ("double", "double-blind"):
        score += Decimal("4")
    
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
    """Calculate bonus for progress within a phase (0-5 points)."""
    if not trials:
        return Decimal("0")
    
    lead_trials = [t for t in trials if t.get("phase") == lead_phase]
    if not lead_trials:
        return Decimal("0")
    
    completed = sum(1 for t in lead_trials if t.get("status", "").lower() == "completed")
    active = sum(1 for t in lead_trials if t.get("status", "").lower() in ("recruiting", "active", "not yet recruiting"))
    total = len(lead_trials)
    
    if completed > 0:
        completion_rate = completed / total
        bonus = Decimal(str(completion_rate)) * Decimal("5")
    elif active > 0:
        activity_rate = active / total
        bonus = Decimal(str(activity_rate)) * Decimal("2.5")
    else:
        bonus = Decimal("0")
    
    return bonus


def _calculate_enrollment_bonus(trials: List[Dict[str, Any]]) -> Decimal:
    """
    Calculate bonus for enrollment size (0-5 points).
    HOTFIX: Properly handles None values in enrollment field.
    """
    if not trials:
        return Decimal("2")
    
    # Get enrollments - handle None explicitly
    enrollments = []
    for t in trials:
        enrollment = t.get("enrollment")
        # Only include if not None and > 0
        if enrollment is not None and enrollment > 0:
            enrollments.append(enrollment)
    
    if not enrollments:
        return Decimal("2")  # Neutral
    
    # Use maximum enrollment as quality indicator
    max_enrollment = max(enrollments)
    
    # Continuous scoring based on size
    if max_enrollment >= 1000:
        bonus = Decimal("5.0")
    elif max_enrollment >= 500:
        bonus = Decimal("4.0") + Decimal(str((max_enrollment - 500) / 500)) * Decimal("1.0")
    elif max_enrollment >= 200:
        bonus = Decimal("3.0") + Decimal(str((max_enrollment - 200) / 300)) * Decimal("1.0")
    elif max_enrollment >= 100:
        bonus = Decimal("2.0") + Decimal(str((max_enrollment - 100) / 100)) * Decimal("1.0")
    elif max_enrollment >= 50:
        bonus = Decimal("1.0") + Decimal(str((max_enrollment - 50) / 50)) * Decimal("1.0")
    else:
        bonus = Decimal(str(max_enrollment / 50)) * Decimal("1.0")
    
    return bonus


def compute_module_4_clinical_dev(
    trial_records: List[Dict[str, Any]],
    active_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Any]:
    """Compute clinical development quality scores with CONTINUOUS sub-scoring."""
    pit_cutoff = compute_pit_cutoff(as_of_date)
    pit_filtered_count = 0
    
    ticker_trials: Dict[str, List[Dict]] = {}
    for trial in trial_records:
        ticker = trial.get("ticker", "").upper()
        if ticker not in active_tickers:
            continue
        
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
            "enrollment": trial.get("enrollment"),  # Keep as-is, handle None in function
        })
    
    scores = []
    
    for ticker in active_tickers:
        trials = ticker_trials.get(ticker, [])
        
        lead_phase = "preclinical"
        lead_phase_score = Decimal("3")
        
        for t in trials:
            phase = t["phase"]
            phase_score = PHASE_SCORES.get(phase, Decimal("0"))
            if phase_score > lead_phase_score:
                lead_phase = phase
                lead_phase_score = phase_score
        
        phase_score = lead_phase_score
        phase_progress_bonus = _calculate_phase_progress_bonus(trials, lead_phase)
        enrollment_bonus = _calculate_enrollment_bonus(trials)
        
        design_score = Decimal("12")
        if trials:
            design_scores = [_score_design(t) for t in trials]
            design_score = max(design_scores)
        
        execution_score = _score_execution(trials)
        endpoint_score = _score_endpoints(trials)
        
        total_raw = (phase_score + phase_progress_bonus + enrollment_bonus + 
                     design_score + execution_score + endpoint_score)
        
        # Normalize to 0-100 scale (max: 110)
        total = (total_raw / Decimal("110")) * Decimal("100")
        
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
            "phase_progress_bonus": str(phase_progress_bonus.quantize(Decimal("0.01"))),
            "enrollment_bonus": str(enrollment_bonus.quantize(Decimal("0.01"))),
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
