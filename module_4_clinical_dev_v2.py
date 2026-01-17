"""
Module 4: Clinical Development Quality (v2)

Enhanced clinical development scoring with:
- Full PIT auditability (per-trial and per-ticker tracking)
- Deterministic dedup by nct_id
- Regex-based endpoint parsing with word boundaries
- Pure Decimal arithmetic throughout
- Status quality scoring with termination_rate
- Lead program identification with deterministic tie-breaks

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now(), no randomness, no float intermediates
- STDLIB-ONLY: No external dependencies
- PIT DISCIPLINE: Every trial has pit_date_field_used, pit_reference_date, pit_admissible
- FAIL LOUDLY: Clear error states
- AUDITABLE: Full provenance chain

Scoring Components (0-120, normalized to 0-100):
- Phase advancement (most advanced phase) - 30 pts
- Phase progress bonus - 5 pts
- Trial count bonus - 5 pts
- Indication diversity bonus - 5 pts
- Recency bonus - 5 pts
- Trial design quality - 25 pts
- Execution track record - 25 pts
- Endpoint strength - 20 pts

Author: Wake Robin Capital Management
Version: 2.0.0
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Union

from common.integration_contracts import (
    validate_module_4_output,
    is_validation_enabled,
    normalize_date_string,
    normalize_ticker_set,
    TickerCollection,
)

from common.provenance import create_provenance
from common.types import Severity


__version__ = "2.0.0"
RULESET_VERSION = "2.0.0-V2"
SCHEMA_VERSION = "v2.0"


# ============================================================================
# CONSTANTS
# ============================================================================

# Recency thresholds (days)
RECENCY_STALE_THRESHOLD = 730  # 2 years
RECENCY_UNKNOWN_PENALTY = Decimal("2.5")  # Neutral score for unknown

# Quantization precision
SCORE_PRECISION = Decimal("0.01")
RATE_PRECISION = Decimal("0.0001")

# Phase scores (0-30) - ordered for deterministic tie-breaking
PHASE_PRIORITY = [
    ("approved", Decimal("30")),
    ("phase 3", Decimal("25")),
    ("phase 2/3", Decimal("22")),
    ("phase 2", Decimal("18")),
    ("phase 1/2", Decimal("12")),
    ("phase 1", Decimal("8")),
    ("preclinical", Decimal("3")),
    ("unknown", Decimal("0")),
]

PHASE_SCORES = {phase: score for phase, score in PHASE_PRIORITY}


class TrialStatus(str, Enum):
    """Standardized trial status categories."""
    COMPLETED = "completed"
    ACTIVE = "active"
    RECRUITING = "recruiting"
    NOT_YET_RECRUITING = "not_yet_recruiting"
    ENROLLING_BY_INVITATION = "enrolling_by_invitation"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"
    WITHDRAWN = "withdrawn"
    UNKNOWN = "unknown"


# Status category weights for quality scoring
STATUS_QUALITY_WEIGHTS = {
    TrialStatus.COMPLETED: Decimal("1.0"),
    TrialStatus.ACTIVE: Decimal("0.8"),
    TrialStatus.RECRUITING: Decimal("0.7"),
    TrialStatus.NOT_YET_RECRUITING: Decimal("0.6"),
    TrialStatus.ENROLLING_BY_INVITATION: Decimal("0.7"),
    TrialStatus.SUSPENDED: Decimal("0.2"),
    TrialStatus.TERMINATED: Decimal("0.0"),
    TrialStatus.WITHDRAWN: Decimal("0.0"),
    TrialStatus.UNKNOWN: Decimal("0.5"),
}

# Negative status categories
NEGATIVE_STATUSES = frozenset([
    TrialStatus.TERMINATED,
    TrialStatus.WITHDRAWN,
    TrialStatus.SUSPENDED,
])


# ============================================================================
# ENDPOINT PATTERNS (Regex with word boundaries)
# ============================================================================

# Strong endpoints - hard clinical outcomes
STRONG_ENDPOINT_PATTERNS = [
    (re.compile(r'\boverall\s+survival\b', re.IGNORECASE), "overall_survival"),
    (re.compile(r'\b(?:os)\b', re.IGNORECASE), "os"),  # Common abbreviation
    (re.compile(r'\bprogression[- ]?free\s+survival\b', re.IGNORECASE), "pfs"),
    (re.compile(r'\b(?:pfs)\b', re.IGNORECASE), "pfs_abbrev"),
    (re.compile(r'\bcomplete\s+response\b', re.IGNORECASE), "complete_response"),
    (re.compile(r'\b(?:cr)\b', re.IGNORECASE), "cr_abbrev"),
    (re.compile(r'\bobjective\s+response\s+rate\b', re.IGNORECASE), "orr"),
    (re.compile(r'\b(?:orr)\b', re.IGNORECASE), "orr_abbrev"),
    (re.compile(r'\bdisease[- ]?free\s+survival\b', re.IGNORECASE), "dfs"),
    (re.compile(r'\bevent[- ]?free\s+survival\b', re.IGNORECASE), "efs"),
    (re.compile(r'\bmajor\s+molecular\s+response\b', re.IGNORECASE), "mmr"),
]

# Weak endpoints - surrogate markers
WEAK_ENDPOINT_PATTERNS = [
    (re.compile(r'\bbiomarker\b', re.IGNORECASE), "biomarker"),
    (re.compile(r'\bpharmacokinetic[s]?\b', re.IGNORECASE), "pk"),
    (re.compile(r'\b(?:pk)\b', re.IGNORECASE), "pk_abbrev"),
    (re.compile(r'\bsafety\b', re.IGNORECASE), "safety"),
    (re.compile(r'\btolerab(?:ility|le)\b', re.IGNORECASE), "tolerability"),
    (re.compile(r'\bdose[- ]?finding\b', re.IGNORECASE), "dose_finding"),
    (re.compile(r'\bmaximum\s+tolerated\s+dose\b', re.IGNORECASE), "mtd"),
    (re.compile(r'\b(?:mtd)\b', re.IGNORECASE), "mtd_abbrev"),
]


# ============================================================================
# PIT DATE FIELD PRIORITY
# ============================================================================

PIT_DATE_FIELDS_PRIORITY = [
    "first_posted",
    "last_update_posted",
    "source_date",
    "collected_at",
]


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class TrialPITRecord:
    """
    Trial record with full PIT auditability.
    """
    nct_id: str
    ticker: str
    phase: str
    status: TrialStatus
    conditions: List[str]
    primary_endpoint: str
    randomized: bool
    blinded: str
    last_update_posted: Optional[str]

    # PIT audit fields
    pit_date_field_used: str
    pit_reference_date: Optional[str]
    pit_admissible: bool
    pit_reason: str

    # Scoring metadata
    endpoint_classification: str  # "strong", "weak", "neutral"
    endpoint_matched_pattern: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "nct_id": self.nct_id,
            "ticker": self.ticker,
            "phase": self.phase,
            "status": self.status.value,
            "conditions": self.conditions,
            "primary_endpoint": self.primary_endpoint,
            "randomized": self.randomized,
            "blinded": self.blinded,
            "last_update_posted": self.last_update_posted,
            "pit_date_field_used": self.pit_date_field_used,
            "pit_reference_date": self.pit_reference_date,
            "pit_admissible": self.pit_admissible,
            "pit_reason": self.pit_reason,
            "endpoint_classification": self.endpoint_classification,
            "endpoint_matched_pattern": self.endpoint_matched_pattern,
        }


@dataclass
class TickerClinicalSummaryV2:
    """
    Per-ticker clinical development summary with full diagnostics.
    """
    ticker: str
    as_of_date: str

    # Main score
    clinical_score: Decimal

    # Component scores
    phase_score: Decimal
    phase_progress: Decimal
    trial_count_bonus: Decimal
    diversity_bonus: Decimal
    recency_bonus: Decimal
    design_score: Decimal
    execution_score: Decimal
    endpoint_score: Decimal

    # Lead program identification
    lead_phase: str
    lead_trial_nct_id: Optional[str]
    lead_program_key: str  # Deterministic key for tie-breaking

    # PIT diagnostics
    n_trials_raw: int
    n_trials_unique: int
    n_trials_pit_admissible: int
    pit_filtered_count: int

    # Execution metrics
    completion_rate: Decimal
    termination_rate: Decimal
    status_quality_score: Decimal

    # Endpoint metrics
    n_strong_endpoints: int
    n_weak_endpoints: int
    n_neutral_endpoints: int

    # Recency
    recency_days: Optional[int]
    recency_unknown: bool
    recency_stale: bool

    # Flags and severity
    flags: List[str]
    severity: str

    # Schema metadata
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict with deterministic ordering."""
        return {
            "_schema": {"schema_version": self.schema_version},
            "ticker": self.ticker,
            "as_of_date": self.as_of_date,
            "clinical_score": str(self.clinical_score),
            "phase_score": str(self.phase_score),
            "phase_progress": str(self.phase_progress),
            "trial_count_bonus": str(self.trial_count_bonus),
            "diversity_bonus": str(self.diversity_bonus),
            "recency_bonus": str(self.recency_bonus),
            "design_score": str(self.design_score),
            "execution_score": str(self.execution_score),
            "endpoint_score": str(self.endpoint_score),
            "lead_phase": self.lead_phase,
            "lead_trial_nct_id": self.lead_trial_nct_id,
            "lead_program_key": self.lead_program_key,
            "n_trials_raw": self.n_trials_raw,
            "n_trials_unique": self.n_trials_unique,
            "n_trials_pit_admissible": self.n_trials_pit_admissible,
            "pit_filtered_count": self.pit_filtered_count,
            "completion_rate": str(self.completion_rate),
            "termination_rate": str(self.termination_rate),
            "status_quality_score": str(self.status_quality_score),
            "n_strong_endpoints": self.n_strong_endpoints,
            "n_weak_endpoints": self.n_weak_endpoints,
            "n_neutral_endpoints": self.n_neutral_endpoints,
            "recency_days": self.recency_days,
            "recency_unknown": self.recency_unknown,
            "recency_stale": self.recency_stale,
            "flags": self.flags,
            "severity": self.severity,
        }


@dataclass
class DiagnosticCountsV2:
    """Aggregate diagnostic counters."""
    tickers_scored: int = 0
    total_trials_raw: int = 0
    total_trials_unique: int = 0
    total_trials_pit_admissible: int = 0
    total_pit_filtered: int = 0
    pit_fields_used: Dict[str, int] = field(default_factory=dict)

    # Status distribution
    status_distribution: Dict[str, int] = field(default_factory=dict)

    # Endpoint distribution
    endpoint_distribution: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# DATE PARSING
# ============================================================================

def _parse_date_safe(date_str: Optional[str]) -> Optional[date]:
    """
    Parse ISO date string to date object.

    Handles:
    - Full ISO dates (YYYY-MM-DD)
    - Partial dates (YYYY-MM, YYYY)
    - Datetime strings (YYYY-MM-DDTHH:MM:SS)

    Returns None on failure.
    """
    if not date_str:
        return None

    try:
        # Clean the string
        cleaned = str(date_str).strip()

        # Handle datetime format
        if 'T' in cleaned:
            cleaned = cleaned.split('T')[0]

        # Handle timezone suffix
        if '+' in cleaned:
            cleaned = cleaned.split('+')[0]
        if 'Z' in cleaned:
            cleaned = cleaned.replace('Z', '')

        # Parse based on length
        if len(cleaned) >= 10:
            return date.fromisoformat(cleaned[:10])
        elif len(cleaned) == 7:  # YYYY-MM
            return date.fromisoformat(cleaned + "-01")
        elif len(cleaned) == 4:  # YYYY
            return date.fromisoformat(cleaned + "-01-01")

        return None
    except (ValueError, TypeError, AttributeError):
        return None


def _compute_pit_cutoff(as_of_date: str) -> str:
    """
    Compute PIT cutoff date.

    Convention: source_date <= as_of_date - 1
    """
    dt = date.fromisoformat(as_of_date)
    cutoff = dt - timedelta(days=1)
    return cutoff.isoformat()


# ============================================================================
# PIT ENFORCEMENT
# ============================================================================

def _select_pit_date(trial: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """
    Select PIT date field in priority order.

    Returns:
        (date_value, field_name)
    """
    for field_name in PIT_DATE_FIELDS_PRIORITY:
        value = trial.get(field_name)
        if value:
            return (value, field_name)

    return (None, "none")


def _is_pit_admissible(
    source_date: Optional[str],
    pit_cutoff: str
) -> Tuple[bool, str]:
    """
    Check if source_date is PIT-admissible.

    Returns:
        (is_admissible, reason)
    """
    if source_date is None:
        return (False, "missing_date")

    try:
        src = _parse_date_safe(source_date)
        cutoff = date.fromisoformat(pit_cutoff)

        if src is None:
            return (False, "unparseable_date")

        if src <= cutoff:
            return (True, "admissible")
        else:
            return (False, f"future_date:{src.isoformat()}")

    except (ValueError, TypeError):
        return (False, "invalid_date_format")


# ============================================================================
# PHASE PARSING
# ============================================================================

def _parse_phase(phase_str: Optional[str]) -> str:
    """
    Normalize phase string to canonical form.

    Handles CT.gov formats: PHASE1, PHASE2, PHASE3, etc.
    """
    if not phase_str:
        return "unknown"

    phase = str(phase_str).lower().strip()

    # Handle CT.gov enum format (PHASE1, PHASE2, etc.)
    phase = phase.replace("phase", "").replace("_", "").replace("-", "")

    if "approved" in phase or phase == "4":
        return "approved"

    # Check for combination phases first
    if "3" in phase and "2" in phase:
        return "phase 2/3"
    if "2" in phase and "1" in phase:
        return "phase 1/2"

    # Single phases
    if "3" in phase:
        return "phase 3"
    if "2" in phase:
        return "phase 2"
    if "1" in phase:
        return "phase 1"
    if "preclinical" in phase or "pre" in phase:
        return "preclinical"

    return "unknown"


# ============================================================================
# STATUS PARSING
# ============================================================================

def _parse_status(status_str: Optional[str]) -> TrialStatus:
    """
    Parse trial status to canonical enum.

    CT.gov statuses:
    - COMPLETED
    - ACTIVE_NOT_RECRUITING (trial running, no new patients)
    - RECRUITING
    - NOT_YET_RECRUITING
    - ENROLLING_BY_INVITATION
    - SUSPENDED
    - TERMINATED
    - WITHDRAWN
    """
    if not status_str:
        return TrialStatus.UNKNOWN

    status = str(status_str).lower().strip().replace("_", " ").replace("-", " ")

    if "completed" in status:
        return TrialStatus.COMPLETED

    # "active not recruiting" is a specific status (trial running, not enrolling)
    if "active" in status and "not" in status and "recruiting" in status:
        return TrialStatus.ACTIVE

    if "active" in status:
        return TrialStatus.ACTIVE

    # Check specific recruiting statuses
    if "not yet recruiting" in status:
        return TrialStatus.NOT_YET_RECRUITING
    if "enrolling" in status or "invitation" in status:
        return TrialStatus.ENROLLING_BY_INVITATION
    if "recruiting" in status:
        return TrialStatus.RECRUITING

    if "suspended" in status:
        return TrialStatus.SUSPENDED
    if "terminated" in status:
        return TrialStatus.TERMINATED
    if "withdrawn" in status:
        return TrialStatus.WITHDRAWN

    return TrialStatus.UNKNOWN


# ============================================================================
# ENDPOINT CLASSIFICATION (Regex-based)
# ============================================================================

def _classify_endpoint(endpoint_text: str) -> Tuple[str, Optional[str]]:
    """
    Classify endpoint using regex with word boundaries.

    Priority: strong > weak > neutral
    Uses first match within priority tier.

    Returns:
        (classification, matched_pattern_name)
    """
    if not endpoint_text:
        return ("neutral", None)

    # Check strong patterns first
    for pattern, name in STRONG_ENDPOINT_PATTERNS:
        if pattern.search(endpoint_text):
            return ("strong", name)

    # Check weak patterns
    for pattern, name in WEAK_ENDPOINT_PATTERNS:
        if pattern.search(endpoint_text):
            return ("weak", name)

    return ("neutral", None)


# ============================================================================
# CONDITIONS PARSING
# ============================================================================

def _normalize_conditions(conditions_raw: Any) -> List[str]:
    """
    Normalize conditions to List[str] (lower, strip).
    Handles string, list of strings, or nested structures.
    """
    result = []

    if isinstance(conditions_raw, str):
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

    # Deterministic ordering
    return sorted(set(result))


def _tokenize_conditions(conditions: List[str]) -> Set[str]:
    """
    Tokenize conditions for diversity scoring.
    Returns unique tokens (words) across all conditions.
    """
    tokens = set()
    stopwords = frozenset(['and', 'the', 'for', 'with', 'not', 'or', 'in', 'of', 'to'])

    for cond in conditions:
        for word in cond.replace('-', ' ').replace('/', ' ').split():
            if word and len(word) > 2 and word not in stopwords:
                tokens.add(word)

    return tokens


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

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


def _score_indication_diversity(all_conditions: List[List[str]]) -> Decimal:
    """
    Score based on unique condition tokens across all trials (0-5 pts).
    """
    union_conditions: Set[str] = set()
    for cond_list in all_conditions:
        union_conditions.update(cond_list)

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


def _score_recency(
    last_update: Optional[str],
    as_of_date: str
) -> Tuple[Decimal, Optional[int], bool, bool]:
    """
    Score based on most recent trial update (0-5 pts).

    Returns:
        (score, recency_days, recency_unknown, recency_stale)
    """
    if not last_update:
        return (RECENCY_UNKNOWN_PENALTY, None, True, False)

    update_date = _parse_date_safe(last_update)
    analysis_date = _parse_date_safe(as_of_date)

    if not update_date or not analysis_date:
        return (RECENCY_UNKNOWN_PENALTY, None, True, False)

    days_since_update = (analysis_date - update_date).days

    # Future date (data quality issue)
    if days_since_update < 0:
        return (RECENCY_UNKNOWN_PENALTY, days_since_update, True, False)

    is_stale = days_since_update >= RECENCY_STALE_THRESHOLD

    # Scoring ladder
    if days_since_update < 30:
        score = Decimal("5.0")
    elif days_since_update < 90:
        score = Decimal("4.5")
    elif days_since_update < 180:
        score = Decimal("4.0")
    elif days_since_update < 365:
        score = Decimal("3.0")
    elif days_since_update < RECENCY_STALE_THRESHOLD:
        score = Decimal("2.0")
    else:
        score = Decimal("1.0")

    return (score, days_since_update, False, is_stale)


def _score_design(trial: TrialPITRecord) -> Decimal:
    """Score trial design quality (0-25)."""
    score = Decimal("12")  # Base

    # Randomized bonus
    if trial.randomized:
        score += Decimal("5")

    # Double-blind bonus
    if trial.blinded.lower() in ("double", "double-blind", "double blind"):
        score += Decimal("4")

    # Endpoint strength (already classified)
    if trial.endpoint_classification == "strong":
        score += Decimal("4")
    elif trial.endpoint_classification == "weak":
        score -= Decimal("3")

    # Clamp
    return max(Decimal("0"), min(Decimal("25"), score))


def _score_execution(trials: List[TrialPITRecord]) -> Tuple[Decimal, Decimal, Decimal, Decimal]:
    """
    Score execution track record (0-25).

    Returns:
        (execution_score, completion_rate, termination_rate, status_quality_score)
    """
    if not trials:
        return (
            Decimal("12"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0.5"),
        )

    total = Decimal(len(trials))
    completed = Decimal(sum(1 for t in trials if t.status == TrialStatus.COMPLETED))
    terminated = Decimal(sum(
        1 for t in trials
        if t.status in (TrialStatus.TERMINATED, TrialStatus.WITHDRAWN)
    ))

    # Compute rates with Decimal only
    completion_rate = (completed / total).quantize(RATE_PRECISION, rounding=ROUND_HALF_UP)
    termination_rate = (terminated / total).quantize(RATE_PRECISION, rounding=ROUND_HALF_UP)

    # Status quality score (weighted average)
    quality_sum = sum(
        STATUS_QUALITY_WEIGHTS.get(t.status, Decimal("0.5"))
        for t in trials
    )
    status_quality_score = (quality_sum / total).quantize(RATE_PRECISION, rounding=ROUND_HALF_UP)

    # Execution score
    score = Decimal("12") + (completion_rate * Decimal("10")) - (termination_rate * Decimal("8"))
    score = max(Decimal("0"), min(Decimal("25"), score))

    return (score, completion_rate, termination_rate, status_quality_score)


def _score_endpoints(trials: List[TrialPITRecord]) -> Tuple[Decimal, int, int, int]:
    """
    Score endpoint portfolio strength (0-20).

    Returns:
        (endpoint_score, n_strong, n_weak, n_neutral)
    """
    if not trials:
        return (Decimal("10"), 0, 0, 0)

    n_strong = sum(1 for t in trials if t.endpoint_classification == "strong")
    n_weak = sum(1 for t in trials if t.endpoint_classification == "weak")
    n_neutral = len(trials) - n_strong - n_weak

    score = Decimal("10")
    score += Decimal(n_strong) * Decimal("2")
    score -= Decimal(n_weak) * Decimal("1")

    score = max(Decimal("0"), min(Decimal("20"), score))

    return (score, n_strong, n_weak, n_neutral)


# ============================================================================
# LEAD PROGRAM IDENTIFICATION
# ============================================================================

def _compute_lead_program_key(nct_id: str, phase: str) -> str:
    """
    Compute deterministic lead program key for tie-breaking.

    Key: SHA256(phase|nct_id) - ensures same result across runs.
    """
    canonical = f"{phase}|{nct_id}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _identify_lead_program(
    trials: List[TrialPITRecord]
) -> Tuple[str, Optional[str], str]:
    """
    Identify lead program with deterministic tie-breaking.

    Priority:
    1. Highest phase score
    2. If tied: deterministic key (SHA256 of phase|nct_id)

    Returns:
        (lead_phase, lead_trial_nct_id, lead_program_key)
    """
    if not trials:
        return ("unknown", None, "")

    # Find highest phase score
    best_phase = "unknown"
    best_score = Decimal("0")
    best_trial: Optional[TrialPITRecord] = None
    best_key = ""

    for trial in trials:
        phase_score = PHASE_SCORES.get(trial.phase, Decimal("0"))
        trial_key = _compute_lead_program_key(trial.nct_id, trial.phase)

        # Compare: higher score wins, or same score with lower key (deterministic)
        if phase_score > best_score or (phase_score == best_score and trial_key < best_key):
            best_phase = trial.phase
            best_score = phase_score
            best_trial = trial
            best_key = trial_key

    return (
        best_phase,
        best_trial.nct_id if best_trial else None,
        best_key,
    )


# ============================================================================
# DEDUPLICATION
# ============================================================================

def _dedup_trials_by_nct_id(
    trials: List[TrialPITRecord]
) -> List[TrialPITRecord]:
    """
    Deduplicate trials by nct_id.

    For duplicate nct_id, keep:
    1. PIT-admissible record if available
    2. Most recent by last_update_posted
    3. Deterministic tie-break by nct_id (lexicographic)
    """
    by_nct: Dict[str, List[TrialPITRecord]] = {}

    for trial in trials:
        if trial.nct_id not in by_nct:
            by_nct[trial.nct_id] = []
        by_nct[trial.nct_id].append(trial)

    result = []

    for nct_id in sorted(by_nct.keys()):  # Deterministic order
        candidates = by_nct[nct_id]

        if len(candidates) == 1:
            result.append(candidates[0])
            continue

        # Sort by: pit_admissible DESC, last_update_posted DESC, deterministic
        def sort_key(t: TrialPITRecord) -> tuple:
            pit_priority = 0 if t.pit_admissible else 1
            update_date = _parse_date_safe(t.last_update_posted)
            # Invert date for descending sort
            date_key = (date.max - update_date).days if update_date else 999999
            return (pit_priority, date_key, t.nct_id)

        candidates.sort(key=sort_key)
        result.append(candidates[0])

    return result


# ============================================================================
# MAIN SCORING FUNCTION
# ============================================================================

def compute_module_4_clinical_dev_v2(
    trial_records: List[Dict[str, Any]],
    active_tickers: TickerCollection,
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Compute clinical development quality scores (v2).

    Args:
        trial_records: List with ticker, nct_id, phase, status, conditions, etc.
        active_tickers: Set or List of tickers from Module 1 (both accepted)
        as_of_date: Analysis date (ISO format)

    Returns:
        {
            "as_of_date": str,
            "scores": [TickerClinicalSummaryV2.to_dict(), ...],
            "diagnostic_counts": DiagnosticCountsV2,
            "provenance": {...}
        }
    """
    # Normalize to sorted list for deterministic iteration
    if isinstance(active_tickers, set):
        active_tickers = sorted(active_tickers)

    pit_cutoff = _compute_pit_cutoff(as_of_date)

    # Diagnostics
    diagnostics = DiagnosticCountsV2()

    # Parse all trials with PIT audit
    all_trials: List[TrialPITRecord] = []
    ticker_raw_counts: Dict[str, int] = {}

    for trial in trial_records:
        ticker = str(trial.get("ticker", "")).upper()
        if ticker not in active_tickers:
            continue

        diagnostics.total_trials_raw += 1
        ticker_raw_counts[ticker] = ticker_raw_counts.get(ticker, 0) + 1

        # PIT audit
        pit_date, pit_field = _select_pit_date(trial)
        diagnostics.pit_fields_used[pit_field] = diagnostics.pit_fields_used.get(pit_field, 0) + 1

        pit_admissible, pit_reason = _is_pit_admissible(pit_date, pit_cutoff)

        if not pit_admissible:
            diagnostics.total_pit_filtered += 1

        nct_id = str(trial.get("nct_id", "")).strip()
        if not nct_id:
            continue

        # Parse fields
        phase = _parse_phase(trial.get("phase"))
        status = _parse_status(trial.get("status"))
        conditions = _normalize_conditions(trial.get("conditions", ""))
        primary_endpoint = str(trial.get("primary_endpoint", ""))
        endpoint_class, endpoint_pattern = _classify_endpoint(primary_endpoint)

        # Track status distribution
        status_key = status.value
        diagnostics.status_distribution[status_key] = diagnostics.status_distribution.get(status_key, 0) + 1

        # Track endpoint distribution
        diagnostics.endpoint_distribution[endpoint_class] = diagnostics.endpoint_distribution.get(endpoint_class, 0) + 1

        pit_record = TrialPITRecord(
            nct_id=nct_id,
            ticker=ticker,
            phase=phase,
            status=status,
            conditions=conditions,
            primary_endpoint=primary_endpoint,
            randomized=bool(trial.get("randomized", False)),
            blinded=str(trial.get("blinded", "")),
            last_update_posted=trial.get("last_update_posted"),
            pit_date_field_used=pit_field,
            pit_reference_date=pit_date,
            pit_admissible=pit_admissible,
            pit_reason=pit_reason,
            endpoint_classification=endpoint_class,
            endpoint_matched_pattern=endpoint_pattern,
        )

        all_trials.append(pit_record)

    # Deduplicate by nct_id
    deduped_trials = _dedup_trials_by_nct_id(all_trials)
    diagnostics.total_trials_unique = len(deduped_trials)

    # Count PIT-admissible after dedup
    pit_admissible_trials = [t for t in deduped_trials if t.pit_admissible]
    diagnostics.total_trials_pit_admissible = len(pit_admissible_trials)

    # Group by ticker (only PIT-admissible for scoring)
    ticker_trials: Dict[str, List[TrialPITRecord]] = {}
    for trial in pit_admissible_trials:
        if trial.ticker not in ticker_trials:
            ticker_trials[trial.ticker] = []
        ticker_trials[trial.ticker].append(trial)

    # Score each ticker
    scores: List[TickerClinicalSummaryV2] = []

    for ticker in sorted(active_tickers):
        trials = ticker_trials.get(ticker, [])
        n_trials_unique = len(trials)
        n_trials_raw = ticker_raw_counts.get(ticker, 0)

        # Count PIT-filtered for this ticker
        all_ticker_trials = [t for t in deduped_trials if t.ticker == ticker]
        pit_filtered_count = sum(1 for t in all_ticker_trials if not t.pit_admissible)

        # Lead program identification
        lead_phase, lead_trial_nct_id, lead_program_key = _identify_lead_program(trials)
        phase_score = PHASE_SCORES.get(lead_phase, Decimal("0"))

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

        # Component scores
        trial_count_bonus = _score_trial_count(n_trials_unique)

        all_conditions = [t.conditions for t in trials]
        diversity_bonus = _score_indication_diversity(all_conditions)

        # Recency (most recent update)
        most_recent_str = None
        most_recent_date = None
        for t in trials:
            update_str = t.last_update_posted
            if update_str:
                update_date = _parse_date_safe(update_str)
                if update_date:
                    if most_recent_date is None or update_date > most_recent_date:
                        most_recent_date = update_date
                        most_recent_str = update_str

        recency_bonus, recency_days, recency_unknown, recency_stale = _score_recency(
            most_recent_str, as_of_date
        )

        # Design score (best trial)
        design_score = Decimal("12")
        if trials:
            design_scores = [_score_design(t) for t in trials]
            design_score = max(design_scores)

        # Execution score
        execution_score, completion_rate, termination_rate, status_quality_score = _score_execution(trials)

        # Endpoint score
        endpoint_score, n_strong, n_weak, n_neutral = _score_endpoints(trials)

        # Total (0-120, normalized to 0-100)
        total = (
            phase_score + phase_progress + trial_count_bonus +
            diversity_bonus + recency_bonus + design_score +
            execution_score + endpoint_score
        )

        clinical_score = ((total / Decimal("120")) * Decimal("100")).quantize(
            SCORE_PRECISION, rounding=ROUND_HALF_UP
        )

        # Flags and severity
        flags = []
        severity = Severity.NONE

        if lead_phase in ("preclinical", "phase 1", "unknown"):
            flags.append("early_stage")

        if not trials:
            flags.append("no_trials")
            severity = Severity.SEV1

        if recency_unknown:
            flags.append("recency_unknown")
        if recency_stale:
            flags.append("recency_stale")

        if termination_rate > Decimal("0.5"):
            flags.append("high_termination_rate")

        summary = TickerClinicalSummaryV2(
            ticker=ticker,
            as_of_date=as_of_date,
            clinical_score=clinical_score,
            phase_score=phase_score.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            phase_progress=phase_progress.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            trial_count_bonus=trial_count_bonus.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            diversity_bonus=diversity_bonus.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            recency_bonus=recency_bonus.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            design_score=design_score.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            execution_score=execution_score.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            endpoint_score=endpoint_score.quantize(SCORE_PRECISION, rounding=ROUND_HALF_UP),
            lead_phase=lead_phase,
            lead_trial_nct_id=lead_trial_nct_id,
            lead_program_key=lead_program_key,
            n_trials_raw=n_trials_raw,
            n_trials_unique=n_trials_unique,
            n_trials_pit_admissible=n_trials_unique,  # Already filtered
            pit_filtered_count=pit_filtered_count,
            completion_rate=completion_rate,
            termination_rate=termination_rate,
            status_quality_score=status_quality_score,
            n_strong_endpoints=n_strong,
            n_weak_endpoints=n_weak,
            n_neutral_endpoints=n_neutral,
            recency_days=recency_days,
            recency_unknown=recency_unknown,
            recency_stale=recency_stale,
            flags=flags,
            severity=severity.value,
        )

        scores.append(summary)

    diagnostics.tickers_scored = len(scores)

    return {
        "as_of_date": as_of_date,
        "scores": [s.to_dict() for s in sorted(scores, key=lambda x: x.ticker)],
        "diagnostic_counts": {
            "tickers_scored": diagnostics.tickers_scored,
            "total_trials_raw": diagnostics.total_trials_raw,
            "total_trials_unique": diagnostics.total_trials_unique,
            "total_trials_pit_admissible": diagnostics.total_trials_pit_admissible,
            "total_pit_filtered": diagnostics.total_pit_filtered,
            "pit_fields_used": diagnostics.pit_fields_used,
            "status_distribution": diagnostics.status_distribution,
            "endpoint_distribution": diagnostics.endpoint_distribution,
        },
        "provenance": create_provenance(RULESET_VERSION, {"tickers": active_tickers}, pit_cutoff),
    }


# ============================================================================
# BACKWARDS COMPATIBILITY WRAPPER
# ============================================================================

def compute_module_4_clinical_dev(
    trial_records: List[Dict[str, Any]],
    active_tickers: TickerCollection,
    as_of_date: str,
) -> Dict[str, Any]:
    """
    Backwards-compatible wrapper for v2.

    Preserves original function signature and output keys.
    Accepts both Set[str] and List[str] for active_tickers.
    """
    result = compute_module_4_clinical_dev_v2(trial_records, active_tickers, as_of_date)

    # Convert v2 scores to v1 format
    v1_scores = []
    for score in result["scores"]:
        v1_scores.append({
            "ticker": score["ticker"],
            "clinical_score": score["clinical_score"],
            "phase_score": score["phase_score"],
            "phase_progress": score["phase_progress"],
            "trial_count_bonus": score["trial_count_bonus"],
            "diversity_bonus": score["diversity_bonus"],
            "recency_bonus": score["recency_bonus"],
            "design_score": score["design_score"],
            "execution_score": score["execution_score"],
            "endpoint_score": score["endpoint_score"],
            "lead_phase": score["lead_phase"],
            "trial_count": score["n_trials_unique"],
            "flags": score["flags"],
            "severity": score["severity"],
            # v1 diagnostics
            "n_trials_unique": score["n_trials_unique"],
            "n_trials_raw": score["n_trials_raw"],
            "pit_filtered_count_ticker": score["pit_filtered_count"],
            "lead_trial_nct_id": score["lead_trial_nct_id"],
            "recency_days": score["recency_days"],
            # v2 additions (preserved)
            "lead_program_key": score["lead_program_key"],
            "completion_rate": score["completion_rate"],
            "termination_rate": score["termination_rate"],
            "status_quality_score": score["status_quality_score"],
            "n_strong_endpoints": score["n_strong_endpoints"],
            "n_weak_endpoints": score["n_weak_endpoints"],
            "n_neutral_endpoints": score["n_neutral_endpoints"],
        })

    output = {
        "as_of_date": result["as_of_date"],
        "scores": v1_scores,
        "diagnostic_counts": {
            "scored": result["diagnostic_counts"]["tickers_scored"],
            "total_trials_raw": result["diagnostic_counts"]["total_trials_raw"],
            "total_trials_unique": result["diagnostic_counts"]["total_trials_unique"],
            "total_trials": result["diagnostic_counts"]["total_trials_unique"],  # Backwards compat
            "pit_filtered": result["diagnostic_counts"]["total_pit_filtered"],
            "pit_fields_used": result["diagnostic_counts"]["pit_fields_used"],
        },
        "provenance": result["provenance"],
    }

    # Validate output schema before returning
    if is_validation_enabled():
        validate_module_4_output(output)

    return output
