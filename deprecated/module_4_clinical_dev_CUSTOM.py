"""
Module 4: Clinical Development Quality (CUSTOM VERSION with Continuous Scoring)

CHANGES FROM ORIGINAL:
- Added Trial Count Bonus (0-5 pts) - diversified pipeline
- Added Indication Diversity Bonus (0-5 pts) - less binary risk
- Added Recency Bonus (0-5 pts) - active programs
- Removed Enrollment Bonus (0% field coverage)

These bonuses use fields with 100% coverage to break up score bucketing.

Scores clinical programs on:
- Phase advancement (most advanced phase) - 30 pts
- Phase progress bonus - 5 pts
- Trial count bonus - 5 pts (NEW)
- Indication diversity bonus - 5 pts (NEW)
- Recency bonus - 5 pts (NEW)
- Trial design quality - 25 pts
- Execution track record - 25 pts
- Endpoint strength - 20 pts
TOTAL: 120 pts (normalized to 0-100)
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional
from datetime import datetime, date

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity

RULESET_VERSION = "1.0.1-CUSTOM"

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


def _score_trial_count(num_trials: int) -> Decimal:
    """
    Score based on number of trials (0-5 pts).
    More trials = more diversified pipeline.
    """
    if num_trials == 0:
        return Decimal("0")
    elif num_trials == 1:
        return Decimal("0.5")
    elif num_trials == 2:
        return Decimal("1.0")
    elif num_trials <= 5:
        return Decimal("2.0")
    elif num_trials <= 10:
        return Decimal("3.5")
    elif num_trials <= 20:
        return Decimal("4.5")
    else:
        return Decimal("5.0")


def _score_indication_diversity(conditions: List[str]) -> Decimal:
    """
    Score based on number of unique indications (0-5 pts).
    Multiple indications = less binary risk.
    """
    unique_conditions = len(set(conditions))
    
    if unique_conditions == 0:
        return Decimal("0")
    elif unique_conditions == 1:
        return Decimal("0.7")
    elif unique_conditions == 2:
        return Decimal("1.5")
    elif unique_conditions <= 5:
        return Decimal("3.0")
    elif unique_conditions <= 10:
        return Decimal("4.0")
    else:
        return Decimal("5.0")


def _score_recency(last_update: Optional[str], as_of_date: str) -> Decimal:
    """
    Score based on most recent trial update (0-5 pts).
    Recent activity = active program.
    
    Args:
        last_update: ISO date string of most recent update
        as_of_date: Current analysis date
    """
    if not last_update:
        return Decimal("1.0")  # Unknown
    
    try:
        # Parse dates
        update_date = datetime.fromisoformat(last_update.replace('Z', '+00:00')).date()
        analysis_date = datetime.fromisoformat(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
        
        days_since_update = (analysis_date - update_date).days
        
        # Scoring based on recency
        if days_since_update < 0:
            return Decimal("1.0")  # Future date (data quality issue)
        elif days_since_update < 30:
            return Decimal("5.0")  # Very active
        elif days_since_update < 90:
            return Decimal("4.5")  # Active
        elif days_since_update < 180:
            return Decimal("4.0")  # Recent
        elif days_since_update < 365:
            return Decimal("3.0")  # Moderate
        elif days_since_update < 730:
            return Decimal("2.0")  # Older
        else:
            return Decimal("1.0")  # Stale
    except (ValueError, TypeError):
        return Decimal("1.0")  # Parse error


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
    Compute clinical development quality scores with continuous bonuses.
    
    Args:
        trial_records: List with ticker, nct_id, phase, status, conditions, last_update_posted, etc.
        active_tickers: Tickers from Module 1
        as_of_date: Analysis date
    
    Returns:
        {
            "as_of_date": str,
            "scores": [{ticker, clinical_score, phase_score, trial_count_bonus, diversity_bonus,
                       recency_bonus, design_score, execution_score, endpoint_score, 
                       lead_phase, flags, severity}],
            "diagnostic_counts": {...},
            "provenance": {...}
        }
    """
    pit_cutoff = compute_pit_cutoff(as_of_date)
    
    # Track PIT-filtered trials
    pit_filtered_count = 0
    
    # Group trials by ticker
    ticker_trials: Dict[str, List[Dict]] = {}
    for trial in trial_records:
        ticker = trial.get("ticker", "").upper()
        if ticker not in active_tickers:
            continue
        
        # PIT FILTER
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
            "conditions": trial.get("conditions", ""),
            "last_update_posted": trial.get("last_update_posted"),
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
        
        # NEW BONUSES using 100% coverage fields
        trial_count_bonus = _score_trial_count(len(trials))
        
        # Extract conditions from all trials
        all_conditions = []
        for t in trials:
            cond = t.get("conditions", "")
            if cond:
                all_conditions.append(cond)
        diversity_bonus = _score_indication_diversity(all_conditions)
        
        # Get most recent update across all trials
        most_recent = None
        for t in trials:
            update = t.get("last_update_posted")
            if update:
                if not most_recent or update > most_recent:
                    most_recent = update
        recency_bonus = _score_recency(most_recent, as_of_date)
        
        # Phase progress bonus (placeholder, 0-5 pts based on phase)
        if lead_phase == "approved":
            phase_progress = Decimal("5")
        elif lead_phase == "phase 3":
            phase_progress = Decimal("4")
        elif lead_phase == "phase 2/3":
            phase_progress = Decimal("3.5")
        elif lead_phase == "phase 2":
            phase_progress = Decimal("3")
        elif lead_phase == "phase 1/2":
            phase_progress = Decimal("2")
        elif lead_phase == "phase 1":
            phase_progress = Decimal("1")
        else:
            phase_progress = Decimal("0")
        
        # Original scores
        design_score = Decimal("12")
        if trials:
            design_scores = [_score_design(t) for t in trials]
            design_score = max(design_scores)
        
        execution_score = _score_execution(trials)
        endpoint_score = _score_endpoints(trials)
        
        # Total (0-120, normalized to 0-100)
        total = (phase_score + phase_progress + trial_count_bonus + 
                diversity_bonus + recency_bonus + design_score + 
                execution_score + endpoint_score)
        
        # Normalize to 0-100 scale (max possible is 120)
        normalized_score = (total / Decimal("120")) * Decimal("100")
        
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
            "clinical_score": str(normalized_score.quantize(Decimal("0.01"))),
            "phase_score": str(phase_score),
            "phase_progress": str(phase_progress),
            "trial_count_bonus": str(trial_count_bonus),
            "diversity_bonus": str(diversity_bonus.quantize(Decimal("0.01"))),
            "recency_bonus": str(recency_bonus.quantize(Decimal("0.01"))),
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
