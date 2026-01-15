"""
Module 4: Clinical Development Quality (vNext)

Scores clinical programs on:
- Phase advancement (most advanced phase) - 30 pts
- Phase progress bonus - 5 pts
- Trial count bonus - 5 pts
- Indication diversity bonus - 5 pts
- Recency bonus - 5 pts
- Trial design quality - 25 pts
- Execution track record - 25 pts
- Endpoint strength - 20 pts
TOTAL: 120 pts (normalized to 0-100)

vNext changes:
- Deterministic: no datetime.now(), proper date parsing
- Deduplication by nct_id per ticker
- PIT field priority: first_posted > last_update_posted > source_date
- Condition tokenization for diversity scoring
- Mutual exclusivity for endpoint scoring (strong wins)
- Recency flags: recency_unknown, recency_stale
"""
from __future__ import annotations

from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict
from datetime import datetime, date

from common.provenance import create_provenance
from common.pit_enforcement import compute_pit_cutoff, is_pit_admissible
from common.types import Severity, TrialRecord, ClinicalResult, Ticker


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class ClinicalScoreRecord(TypedDict, total=False):
    """Output record for clinical scoring."""
    ticker: str
    clinical_score: str
    phase_score: str
    phase_progress: str
    trial_count_bonus: str
    diversity_bonus: str
    recency_bonus: str
    design_score: str
    execution_score: str
    endpoint_score: str
    lead_phase: str
    trial_count: int
    flags: List[str]
    severity: str
    n_trials_unique: int
    n_trials_raw: int
    pit_filtered_count_ticker: int
    lead_trial_nct_id: Optional[str]
    recency_days: Optional[int]


class Module4DiagnosticCounts(TypedDict):
    """Diagnostic counts for module 4."""
    scored: int
    total_trials_raw: int
    total_trials_unique: int
    total_trials: int
    pit_filtered: int
    pit_fields_used: Dict[str, int]


class Module4Result(TypedDict):
    """Complete result from Module 4 clinical development scoring."""
    as_of_date: str
    scores: List[ClinicalScoreRecord]
    diagnostic_counts: Module4DiagnosticCounts
    provenance: Dict[str, object]


# Internal types
TrialDict = Dict[str, Union[str, int, float, bool, List[str], None]]
NormalizedTrialDict = Dict[str, Union[str, int, bool, List[str], None]]

RULESET_VERSION = "1.0.2-VNEXT"

# Recency thresholds (days)
RECENCY_STALE_THRESHOLD = 730  # 2 years
RECENCY_UNKNOWN_PENALTY = Decimal("2.5")  # Neutral score for unknown

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


def _normalize_conditions(conditions_raw: Union[str, List[str], List[List[str]], None]) -> List[str]:
    """
    Normalize conditions to List[str] (lower, strip).
    Handles string, list of strings, or nested structures.
    """
    result = []
    if isinstance(conditions_raw, str):
        # Single string - split on common delimiters
        for part in conditions_raw.replace(';', ',').replace('|', ',').split(','):
            cleaned = part.lower().strip()
            if cleaned:
                result.append(cleaned)
    elif isinstance(conditions_raw, list):
        for item in conditions_raw:
            if isinstance(item, str):
                cleaned = item.lower().strip()
                if cleaned:
                    result.append(cleaned)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, str):
                        cleaned = sub.lower().strip()
                        if cleaned:
                            result.append(cleaned)
    return result


def _tokenize_conditions(conditions: List[str]) -> set[str]:
    """
    Tokenize conditions for diversity scoring.
    Returns unique tokens (words) across all conditions.
    """
    tokens = set()
    for cond in conditions:
        # Split on whitespace and common separators
        for word in cond.replace('-', ' ').replace('/', ' ').split():
            # Skip common stopwords
            if word and len(word) > 2 and word not in {'and', 'the', 'for', 'with', 'not'}:
                tokens.add(word)
    return tokens


def _score_indication_diversity(all_conditions: List[List[str]]) -> Decimal:
    """
    Score based on unique condition tokens across all trials (0-5 pts).
    Multiple indications = less binary risk.

    Args:
        all_conditions: List of condition lists per trial
    """
    # Union all conditions across trials
    union_conditions = set()
    for cond_list in all_conditions:
        union_conditions.update(cond_list)

    # Tokenize for diversity count
    unique_tokens = _tokenize_conditions(list(union_conditions))
    token_count = len(unique_tokens)

    if token_count == 0:
        return Decimal("0")
    elif token_count <= 2:
        return Decimal("0.7")
    elif token_count <= 5:
        return Decimal("1.5")
    elif token_count <= 10:
        return Decimal("3.0")
    elif token_count <= 20:
        return Decimal("4.0")
    else:
        return Decimal("5.0")

def _parse_date_safe(date_str: Optional[str]) -> Optional[date]:
    """Parse ISO date string to date object, return None on failure."""
    if not date_str:
        return None
    try:
        # Handle various ISO formats
        cleaned = date_str.replace('Z', '+00:00')
        if 'T' in cleaned:
            return datetime.fromisoformat(cleaned).date()
        else:
            return date.fromisoformat(cleaned[:10])
    except (ValueError, TypeError):
        return None


def _score_recency(
    last_update: Optional[str],
    as_of_date: str
) -> Tuple[Decimal, Optional[int], bool, bool]:
    """
    Score based on most recent trial update (0-5 pts).

    Args:
        last_update: ISO date string of most recent update
        as_of_date: Current analysis date

    Returns:
        (score, recency_days, recency_unknown, recency_stale)
    """
    if not last_update:
        # Unknown recency - return neutral score (not penalty)
        return (RECENCY_UNKNOWN_PENALTY, None, True, False)

    update_date = _parse_date_safe(last_update)
    analysis_date = _parse_date_safe(as_of_date)

    if not update_date or not analysis_date:
        return (RECENCY_UNKNOWN_PENALTY, None, True, False)

    days_since_update = (analysis_date - update_date).days

    # Future date (data quality issue) - treat as unknown
    if days_since_update < 0:
        return (RECENCY_UNKNOWN_PENALTY, days_since_update, True, False)

    # Check stale threshold
    is_stale = days_since_update >= RECENCY_STALE_THRESHOLD

    # Scoring based on recency
    if days_since_update < 30:
        score = Decimal("5.0")  # Very active
    elif days_since_update < 90:
        score = Decimal("4.5")  # Active
    elif days_since_update < 180:
        score = Decimal("4.0")  # Recent
    elif days_since_update < 365:
        score = Decimal("3.0")  # Moderate
    elif days_since_update < RECENCY_STALE_THRESHOLD:
        score = Decimal("2.0")  # Older
    else:
        score = Decimal("1.0")  # Stale

    return (score, days_since_update, False, is_stale)


def _score_design(trial: NormalizedTrialDict) -> Decimal:
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


def _score_execution(trials: List[NormalizedTrialDict]) -> Decimal:
    """Score execution track record (0-25)."""
    if not trials:
        return Decimal("12")

    completed = sum(1 for t in trials if t.get("status", "").lower() == "completed")
    terminated = sum(1 for t in trials if t.get("status", "").lower() in ("terminated", "withdrawn"))
    total = len(trials)

    if total == 0:
        return Decimal("12")

    # Use Decimal division to avoid float conversion
    completion_rate = Decimal(completed) / Decimal(total)
    termination_rate = Decimal(terminated) / Decimal(total)

    # Base on completion rate
    score = Decimal("12") + (completion_rate * Decimal("10")) - (termination_rate * Decimal("8"))

    return max(Decimal("0"), min(Decimal("25"), score))


def _classify_endpoint(endpoint_text: str) -> str:
    """
    Classify endpoint as 'strong', 'weak', or 'neutral'.
    Strong wins over weak (mutually exclusive).
    """
    endpoint = endpoint_text.lower()

    # Check strong first - if found, it wins
    for strong in STRONG_ENDPOINTS:
        if strong in endpoint:
            return "strong"

    # Only check weak if not strong
    for weak in WEAK_ENDPOINTS:
        if weak in endpoint:
            return "weak"

    return "neutral"


def _score_endpoints(trials: List[NormalizedTrialDict]) -> Decimal:
    """Score endpoint portfolio strength (0-20)."""
    if not trials:
        return Decimal("10")

    strong_count = 0
    weak_count = 0

    for t in trials:
        endpoint = t.get("primary_endpoint") or ""
        classification = _classify_endpoint(endpoint)
        if classification == "strong":
            strong_count += 1
        elif classification == "weak":
            weak_count += 1

    score = Decimal("10")
    score += Decimal(strong_count) * Decimal("2")
    score -= Decimal(weak_count) * Decimal("1")

    return max(Decimal("0"), min(Decimal("20"), score))


def _select_pit_date(trial: TrialDict) -> Tuple[Optional[str], str]:
    """
    Select PIT date field in priority order.

    Priority: first_posted > last_update_posted > source_date

    Returns:
        (date_value, field_name)
    """
    if trial.get("first_posted"):
        return (trial["first_posted"], "first_posted")
    if trial.get("last_update_posted"):
        return (trial["last_update_posted"], "last_update_posted")
    if trial.get("source_date"):
        return (trial["source_date"], "source_date")
    return (None, "none")


def compute_module_4_clinical_dev(
    trial_records: List[TrialDict],
    active_tickers: List[Ticker],
    as_of_date: str,
) -> Module4Result:
    """
    Compute clinical development quality scores (vNext).

    Args:
        trial_records: List with ticker, nct_id, phase, status, conditions, etc.
        active_tickers: Tickers from Module 1
        as_of_date: Analysis date

    Returns:
        {
            "as_of_date": str,
            "scores": [{ticker, clinical_score, phase_score, trial_count_bonus, diversity_bonus,
                       recency_bonus, design_score, execution_score, endpoint_score,
                       lead_phase, flags, severity, ...per-ticker diagnostics}],
            "diagnostic_counts": {...},
            "provenance": {...}
        }
    """
    pit_cutoff = compute_pit_cutoff(as_of_date)

    # Track counts
    total_raw = 0
    total_unique = 0
    total_pit_filtered = 0
    pit_fields_used: Dict[str, int] = {}

    # Per-ticker tracking
    ticker_raw_counts: Dict[str, int] = {}
    ticker_pit_filtered: Dict[str, int] = {}

    # Group trials by ticker with deduplication
    ticker_trials: Dict[str, Dict[str, NormalizedTrialDict]] = {}  # ticker -> nct_id -> trial

    for trial in trial_records:
        ticker = trial.get("ticker", "").upper()
        if ticker not in active_tickers:
            continue

        total_raw += 1
        ticker_raw_counts[ticker] = ticker_raw_counts.get(ticker, 0) + 1

        # PIT FILTER with priority field selection
        pit_date, pit_field = _select_pit_date(trial)
        pit_fields_used[pit_field] = pit_fields_used.get(pit_field, 0) + 1

        if pit_date and not is_pit_admissible(pit_date, pit_cutoff):
            total_pit_filtered += 1
            ticker_pit_filtered[ticker] = ticker_pit_filtered.get(ticker, 0) + 1
            continue

        nct_id = trial.get("nct_id", "")
        if not nct_id:
            continue

        if ticker not in ticker_trials:
            ticker_trials[ticker] = {}

        # Deduplicate by nct_id (keep latest by update date)
        existing = ticker_trials[ticker].get(nct_id)
        if existing:
            existing_date = _parse_date_safe(existing.get("last_update_posted"))
            new_date = _parse_date_safe(trial.get("last_update_posted"))
            if existing_date and new_date and new_date <= existing_date:
                continue  # Keep existing

        # Normalize conditions
        conditions_normalized = _normalize_conditions(trial.get("conditions", ""))

        ticker_trials[ticker][nct_id] = {
            "nct_id": nct_id,
            "phase": _parse_phase(trial.get("phase")),
            "status": trial.get("status", ""),
            "randomized": trial.get("randomized", False),
            "blinded": trial.get("blinded", ""),
            "primary_endpoint": trial.get("primary_endpoint", ""),
            "conditions": conditions_normalized,
            "last_update_posted": trial.get("last_update_posted"),
            "pit_date_field": pit_field,
        }

    # Count unique trials
    for ticker_dict in ticker_trials.values():
        total_unique += len(ticker_dict)

    scores: List[ClinicalScoreRecord] = []

    for ticker in active_tickers:
        trials_dict = ticker_trials.get(ticker, {})
        trials = list(trials_dict.values())
        n_trials_unique = len(trials)
        pit_filtered_ticker = ticker_pit_filtered.get(ticker, 0)
        raw_count_ticker = ticker_raw_counts.get(ticker, 0)

        # Find lead phase and lead trial
        lead_phase = "preclinical"
        lead_phase_score = Decimal("3")
        lead_trial_nct_id = None

        for t in trials:
            phase = t["phase"]
            ps = PHASE_SCORES.get(phase, Decimal("0"))
            if ps > lead_phase_score:
                lead_phase = phase
                lead_phase_score = ps
                lead_trial_nct_id = t["nct_id"]

        # Component scores
        phase_score = lead_phase_score

        # Trial count bonus
        trial_count_bonus = _score_trial_count(n_trials_unique)

        # Extract conditions from all trials (already normalized as List[str])
        all_conditions = [t.get("conditions", []) for t in trials]
        diversity_bonus = _score_indication_diversity(all_conditions)

        # Get most recent update using proper date comparison
        most_recent_str = None
        most_recent_date = None
        for t in trials:
            update_str = t.get("last_update_posted")
            if update_str:
                update_date = _parse_date_safe(update_str)
                if update_date:
                    if most_recent_date is None or update_date > most_recent_date:
                        most_recent_date = update_date
                        most_recent_str = update_str

        # Recency with flags
        recency_bonus, recency_days, recency_unknown, recency_stale = _score_recency(
            most_recent_str, as_of_date
        )

        # Phase progress bonus
        phase_progress_map = {
            "approved": Decimal("5"),
            "phase 3": Decimal("4"),
            "phase 2/3": Decimal("3.5"),
            "phase 2": Decimal("3"),
            "phase 1/2": Decimal("2"),
            "phase 1": Decimal("1"),
        }
        phase_progress = phase_progress_map.get(lead_phase, Decimal("0"))

        # Design score (best trial)
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

        normalized_score = (total / Decimal("120")) * Decimal("100")

        # Flags and severity
        flags = []
        severity = Severity.NONE

        if lead_phase in ("preclinical", "phase 1"):
            flags.append("early_stage")

        if not trials:
            flags.append("no_trials")
            severity = Severity.SEV1

        if recency_unknown:
            flags.append("recency_unknown")
        if recency_stale:
            flags.append("recency_stale")

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
            "trial_count": n_trials_unique,
            "flags": flags,
            "severity": severity.value,
            # Per-ticker diagnostics (new)
            "n_trials_unique": n_trials_unique,
            "n_trials_raw": raw_count_ticker,
            "pit_filtered_count_ticker": pit_filtered_ticker,
            "lead_trial_nct_id": lead_trial_nct_id,
            "recency_days": recency_days,
        })

    return {
        "as_of_date": as_of_date,
        "scores": sorted(scores, key=lambda x: x["ticker"]),
        "diagnostic_counts": {
            "scored": len(scores),
            "total_trials_raw": total_raw,
            "total_trials_unique": total_unique,
            "total_trials": total_unique,  # Backwards compat
            "pit_filtered": total_pit_filtered,
            "pit_fields_used": pit_fields_used,
        },
        "provenance": create_provenance(RULESET_VERSION, {"tickers": active_tickers}, pit_cutoff),
    }
