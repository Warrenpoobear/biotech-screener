"""
Clinical Quality Score — Spec 057: coarse clinical rigor layer.

Separate from the existing clinical structure/timing block (Module 4 / B6).
This module scores **clinical truth** — how rigorous and credible the science is —
not event timing or pipeline maturity.

Four sub-scores:
  1. endpoint_strength_tier   — how clinically meaningful the primary endpoint is
  2. design_rigor_tier        — how trustworthy the study design is
  3. prior_evidence_tier      — directional view of prior readout history
  4. mechanism_maturity_tier   — whether the biology is validated

Composite:
  clinical_quality_score = 0.30 * endpoint + 0.30 * design + 0.25 * prior_evidence
                         + 0.15 * mechanism_maturity
  clamped to [-1, +1].

Confidence:
  available_component_count / 4, haircut for heuristic/stale sources.
  Bucketed: high (>=0.75), medium (>=0.50), low (<0.50).

Design constraints:
  - Monitor-only: does NOT feed into ranking, selector, trap, or EES.
  - Coarse: policy mappings only, no fitted thresholds.
  - PIT-safe: strict < as_of_date.
  - Deterministic: pure functions, no side effects.
  - Fail-soft: missing data → neutral (0.0), never crash.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__all__ = [
    "ClinicalQualityResult",
    "compute_clinical_quality_scores",
    "compute_endpoint_strength_tier",
    "compute_design_rigor_tier",
    "compute_prior_evidence_tier",
    "compute_mechanism_maturity_tier",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(s: Any) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _pit_ok(trial: dict, as_of: date) -> bool:
    """PIT gate: trial publicly known BEFORE as_of (strict <)."""
    fp = _parse_date(trial.get("first_posted"))
    lup = _parse_date(trial.get("last_update_posted"))
    pit_date = fp or lup
    return pit_date is not None and pit_date < as_of


def _phase_num(phase_str: str) -> float:
    """Map phase string to numeric value."""
    p = phase_str.lower().strip()
    mapping = {
        "phase 3": 3.0,
        "phase3": 3.0,
        "phase 2/3": 2.5,
        "phase2/phase3": 2.5,
        "phase 2/phase 3": 2.5,
        "phase 2": 2.0,
        "phase2": 2.0,
        "phase 1/2": 1.5,
        "phase1/phase2": 1.5,
        "phase 1/phase 2": 1.5,
        "phase 1": 1.0,
        "phase1": 1.0,
        "early_phase1": 0.5,
    }
    return mapping.get(p, 0.0)


# ---------------------------------------------------------------------------
# 1. Endpoint Strength Tier
# ---------------------------------------------------------------------------

# Hard endpoints: definitive clinical outcomes
_HARD_ENDPOINT_PATTERNS = [
    re.compile(r"\boverall\s+survival\b", re.I),
    re.compile(r"\bmortality\b", re.I),
    re.compile(r"\bdeath\b", re.I),
    re.compile(r"\b(?:major\s+adverse\s+)?card(?:iac|iovascular)\s+event", re.I),
    re.compile(r"\bMACE\b"),
    re.compile(r"\bevent[- ]free\s+survival\b", re.I),
    re.compile(r"\bdisease[- ]free\s+survival\b", re.I),
]

# Semi-hard: validated clinical outcomes, not definitive
_SEMI_HARD_ENDPOINT_PATTERNS = [
    re.compile(r"\bprogression[- ]free\s+survival\b", re.I),
    re.compile(r"\bPFS\b"),
    re.compile(r"\bhospitali[sz]ation\b", re.I),
    re.compile(r"\brelapse[- ]free\s+survival\b", re.I),
    re.compile(r"\bfunctional\s+(?:improvement|outcome|score)\b", re.I),
    re.compile(r"\bcomplete\s+(?:response|remission)\b", re.I),
    re.compile(r"\bACR\s*\d+\b", re.I),  # rheumatology
    re.compile(r"\bPASI\s*\d+\b", re.I),  # dermatology
    re.compile(r"\bHbA1c\b", re.I),  # metabolic
    re.compile(r"\bEDSS\b", re.I),  # multiple sclerosis
]

# Surrogate: biomarker or early signal endpoints
_SURROGATE_ENDPOINT_PATTERNS = [
    re.compile(r"\bbiomarker\b", re.I),
    re.compile(r"\bpharmacokinetic\b", re.I),
    re.compile(r"\b(?:PK|PD)\b"),
    re.compile(r"\bsafety\b", re.I),
    re.compile(r"\btolerab", re.I),
    re.compile(r"\bdose[- ]?(?:finding|escalation|limiting)\b", re.I),
    re.compile(r"\badverse\s+event", re.I),
    re.compile(r"\bmaximum\s+tolerated\s+dose\b", re.I),
]


def compute_endpoint_strength_tier(
    primary_endpoints: List[str],
    title: str = "",
) -> Tuple[float, str, str]:
    """Score endpoint strength from primary endpoint text.

    Args:
        primary_endpoints: List of primary endpoint measure texts (from design_outcomes).
        title: Trial title as fallback.

    Returns:
        (score, tier_label, matched_signal)
        - score: +1.0 (hard), +0.4 (semi-hard), 0.0 (surrogate/unknown)
        - tier_label: "hard" / "semi_hard" / "surrogate" / "unknown"
        - matched_signal: first matched pattern text or ""
    """
    scan_text = " ".join(primary_endpoints) + " " + title

    if not scan_text.strip():
        return 0.0, "unknown", ""

    # Check hard endpoints first
    for pat in _HARD_ENDPOINT_PATTERNS:
        m = pat.search(scan_text)
        if m:
            return 1.0, "hard", m.group(0)

    # Check semi-hard
    for pat in _SEMI_HARD_ENDPOINT_PATTERNS:
        m = pat.search(scan_text)
        if m:
            return 0.4, "semi_hard", m.group(0)

    # Check surrogate
    for pat in _SURROGATE_ENDPOINT_PATTERNS:
        m = pat.search(scan_text)
        if m:
            return 0.0, "surrogate", m.group(0)

    # ORR / objective response rate — intermediate but not semi-hard
    if re.search(r"\b(?:objective|overall)\s+response\s+rate\b", scan_text, re.I) or re.search(r"\bORR\b", scan_text):
        return 0.4, "semi_hard", "ORR"

    return 0.0, "unknown", ""


# ---------------------------------------------------------------------------
# 2. Design Rigor Tier
# ---------------------------------------------------------------------------


def compute_design_rigor_tier(
    allocation: Optional[str],
    masking: Optional[str],
    intervention_model: Optional[str],
    interventions: Optional[List[str]] = None,
    study_type: str = "",
    title: str = "",
) -> Tuple[float, str, List[str]]:
    """Score trial design rigor from structured AACT fields.

    Uses structured fields (allocation, masking) when available, falls back
    to title parsing.

    Args:
        allocation: "RANDOMIZED" / "NON_RANDOMIZED" / "NA" / None
        masking: "DOUBLE" / "TRIPLE" / "QUADRUPLE" / "SINGLE" / "NONE" / None
        intervention_model: "PARALLEL" / "CROSSOVER" / "SINGLE_GROUP" / etc.
        interventions: list of intervention names (for placebo detection)
        study_type: "INTERVENTIONAL" / "OBSERVATIONAL"
        title: trial title for fallback parsing

    Returns:
        (score, tier_label, signals)
        - score: +1.0 (gold standard) → -0.5 (observational)
        - tier_label: "gold_standard" / "strong" / "moderate" / "weak" / "observational"
        - signals: list of detected design features
    """
    signals: List[str] = []

    # Observational studies
    if study_type.upper() == "OBSERVATIONAL":
        return -0.5, "observational", ["observational"]

    # Build feature detection from structured fields + title fallback
    is_randomized = False
    if allocation and allocation.upper() == "RANDOMIZED":
        is_randomized = True
        signals.append("randomized")
    elif re.search(r"\brandomized\b", title, re.I):
        is_randomized = True
        signals.append("randomized_title")

    blind_level = 0  # 0=none, 1=single, 2=double+
    masking_upper = (masking or "").upper()
    if masking_upper in ("DOUBLE", "TRIPLE", "QUADRUPLE"):
        blind_level = 2
        signals.append(f"masked_{masking_upper.lower()}")
    elif masking_upper == "SINGLE":
        blind_level = 1
        signals.append("masked_single")
    elif re.search(r"\bdouble[- ]?blind\b", title, re.I):
        blind_level = 2
        signals.append("double_blind_title")
    elif re.search(r"\bsingle[- ]?blind\b", title, re.I):
        blind_level = 1
        signals.append("single_blind_title")

    has_control = False
    intervs = interventions or []
    intervs_lower = [str(iv).lower() for iv in intervs]
    if any("placebo" in iv for iv in intervs_lower):
        has_control = True
        signals.append("placebo_controlled")
    elif re.search(r"\bplacebo[- ]?controlled\b", title, re.I):
        has_control = True
        signals.append("placebo_controlled_title")
    elif intervention_model and intervention_model.upper() in ("PARALLEL", "CROSSOVER", "FACTORIAL"):
        has_control = True
        signals.append(f"controlled_{intervention_model.lower()}")

    is_single_arm = (intervention_model or "").upper() == "SINGLE_GROUP"
    if is_single_arm:
        signals.append("single_arm")

    # Tier assignment (policy mapping, no fitting)
    if is_randomized and blind_level >= 2 and has_control:
        return 1.0, "gold_standard", signals
    elif is_randomized and blind_level >= 1:
        return 0.75, "strong", signals
    elif is_randomized and has_control:
        return 0.65, "strong", signals
    elif is_randomized:
        return 0.5, "moderate", signals
    elif is_single_arm:
        return 0.0, "weak", signals
    elif not signals:
        # No design info available
        return 0.0, "unknown", signals

    # Open-label with some control
    if has_control:
        return 0.25, "moderate", signals
    return 0.0, "weak", signals


# ---------------------------------------------------------------------------
# 3. Prior Evidence Tier
# ---------------------------------------------------------------------------


def compute_prior_evidence_tier(
    ticker_trials: List[dict],
    as_of: date,
) -> Tuple[float, str, str]:
    """Score prior evidence direction for a ticker.

    Uses completed trial outcomes as proxy: results_first_posted = evidence exists,
    status=TERMINATED/WITHDRAWN = negative signal.

    This is coarse by design. Does NOT model effect sizes.

    Args:
        ticker_trials: All trials for this ticker.
        as_of: Point-in-time date.

    Returns:
        (score, tier_label, notes)
        - score: +1.0 (positive), 0.0 (mixed/unknown), -1.0 (negative)
        - tier_label: "positive" / "mixed" / "negative" / "unknown"
        - notes: diagnostic string
    """
    completed_with_results = 0
    completed_no_results = 0
    terminated = 0
    withdrawn = 0
    total_completed_ish = 0

    for t in ticker_trials:
        if not _pit_ok(t, as_of):
            continue
        status = (t.get("status") or "").upper()

        # PIT-safe results check: results must have been posted BEFORE as_of
        rfp = _parse_date(t.get("results_first_posted"))
        has_results = rfp is not None and rfp < as_of

        # PIT-safe status: use last_update_posted as proxy for when the status
        # was observable. If the trial's last_update is after as_of, the current
        # status may not have been visible yet — treat as still running.
        lup = _parse_date(t.get("last_update_posted"))
        status_observable = lup is not None and lup < as_of

        if status == "COMPLETED" and status_observable:
            total_completed_ish += 1
            if has_results:
                completed_with_results += 1
            else:
                completed_no_results += 1
        elif status in ("TERMINATED", "SUSPENDED") and status_observable:
            terminated += 1
            total_completed_ish += 1
        elif status == "WITHDRAWN" and status_observable:
            withdrawn += 1
            total_completed_ish += 1

    if total_completed_ish == 0:
        return 0.0, "unknown", "no_completed_trials"

    # Compute directional signal
    positive = completed_with_results
    negative = terminated + withdrawn

    if positive == 0 and negative == 0:
        return 0.0, "unknown", f"completed={completed_no_results}_no_results"

    ratio = positive / (positive + negative) if (positive + negative) > 0 else 0.5
    notes = f"pos={positive},neg={negative},ratio={ratio:.2f}"

    # Policy mapping (coarse)
    if ratio >= 0.70 and positive >= 2:
        return 1.0, "positive", notes
    elif ratio >= 0.50 and positive >= 1:
        return 0.5, "leaning_positive", notes
    elif ratio <= 0.30 and negative >= 2:
        return -1.0, "negative", notes
    elif ratio <= 0.40:
        return -0.5, "leaning_negative", notes
    else:
        return 0.0, "mixed", notes


# ---------------------------------------------------------------------------
# 4. Mechanism Maturity Tier
# ---------------------------------------------------------------------------

# Therapeutic areas with strong validated-target history
# (approved drugs exist, regulatory path well-trodden)
_VALIDATED_TA = frozenset(
    {
        "oncology",  # PD-1, HER2, VEGF, CDK4/6, etc.
        "cardiovascular",  # statins, antihypertensives
        "infectious_disease",  # antivirals, antibiotics
        "metabolic",  # GLP-1, SGLT2, insulin
    }
)

# TAs with partial validation (some approved, many novel)
_PARTIAL_TA = frozenset(
    {
        "autoimmune",  # TNF, IL-17, JAK
        "dermatology",  # IL-4/13, PDE4
        "respiratory",  # biologics emerging
        "ophthalmology",  # anti-VEGF
    }
)

# TAs where novel mechanisms dominate
_NOVEL_TA = frozenset(
    {
        "cns",  # high failure rates, few validated targets
        "rare_disease",  # often first-in-class
    }
)

# Known validated mechanism keywords (in intervention names or titles)
_VALIDATED_MECHANISM_KEYWORDS = [
    re.compile(r"\banti[- ]?PD[- ]?[L1]?\b", re.I),
    re.compile(r"\bcheckpoint\s+inhibitor\b", re.I),
    re.compile(r"\bCAR[- ]?T\b", re.I),
    re.compile(r"\bbispecific\b", re.I),
    re.compile(r"\bADC\b"),  # antibody-drug conjugate
    re.compile(r"\bGLP[- ]?1\b", re.I),
    re.compile(r"\bSGLT[- ]?2\b", re.I),
    re.compile(r"\bJAK\s*inhibitor\b", re.I),
    re.compile(r"\bTNF\b", re.I),
    re.compile(r"\banti[- ]?VEGF\b", re.I),
    re.compile(r"\bPARP\s*inhibitor\b", re.I),
    re.compile(r"\bBTK\s*inhibitor\b", re.I),
    re.compile(r"\bBCL[- ]?2\b", re.I),
    re.compile(r"\bPI3K\b", re.I),
    re.compile(r"\bmTOR\b", re.I),
    re.compile(r"\bproteasome\s*inhibitor\b", re.I),
]

# Red flag patterns (mechanisms with history of class-wide failures)
_RED_FLAG_KEYWORDS = [
    re.compile(r"\bamyloid\s+beta\b", re.I),  # high failure, though some recent approvals
    re.compile(r"\banti[- ]?amyloid\b", re.I),
]


def compute_mechanism_maturity_tier(
    ticker_trials: List[dict],
    as_of: date,
) -> Tuple[float, str, str]:
    """Score mechanism maturity from therapeutic area + intervention signals.

    Args:
        ticker_trials: All PIT-eligible trials for this ticker.
        as_of: Point-in-time date.

    Returns:
        (score, tier_label, notes)
        - score: +1.0 (validated) → -0.5 (red flag)
        - tier_label: "validated" / "partially_validated" / "novel" / "red_flag" / "unknown"
        - notes: diagnostic string
    """
    from common.accuracy_improvements import classify_therapeutic_area

    pit_trials = [t for t in ticker_trials if _pit_ok(t, as_of)]
    if not pit_trials:
        return 0.0, "unknown", "no_pit_trials"

    # Collect conditions and interventions
    all_conditions: List[str] = []
    all_text: List[str] = []
    max_phase = 0.0
    for t in pit_trials:
        conds = t.get("conditions") or []
        if isinstance(conds, str):
            conds = [conds]
        all_conditions.extend(conds)
        intervs = t.get("interventions") or []
        if isinstance(intervs, str):
            intervs = [intervs]
        title = t.get("title") or ""
        all_text.extend(intervs)
        all_text.append(title)
        max_phase = max(max_phase, _phase_num(str(t.get("phase", ""))))

    ta = classify_therapeutic_area(all_conditions)
    ta_val = ta.value
    combined_text = " ".join(str(x) for x in all_text)

    # Check red flags first
    for pat in _RED_FLAG_KEYWORDS:
        m = pat.search(combined_text)
        if m:
            return -0.5, "red_flag", f"ta={ta_val},match={m.group(0)}"

    # Check validated mechanisms
    for pat in _VALIDATED_MECHANISM_KEYWORDS:
        m = pat.search(combined_text)
        if m:
            return 1.0, "validated", f"ta={ta_val},match={m.group(0)}"

    # Fall back to TA-based classification
    if ta_val in _VALIDATED_TA:
        # Late-stage in validated TA = more validated
        if max_phase >= 3.0:
            return 1.0, "validated", f"ta={ta_val},phase3+"
        elif max_phase >= 2.0:
            return 0.5, "partially_validated", f"ta={ta_val},phase2+"
        else:
            return 0.5, "partially_validated", f"ta={ta_val},early"

    if ta_val in _PARTIAL_TA:
        if max_phase >= 3.0:
            return 0.5, "partially_validated", f"ta={ta_val},phase3+"
        else:
            return 0.25, "partially_validated", f"ta={ta_val},early"

    if ta_val in _NOVEL_TA:
        return 0.0, "novel", f"ta={ta_val}"

    # Default: no strong signal either way
    if max_phase >= 3.0:
        return 0.25, "partially_validated", f"ta={ta_val},phase3+_default"
    return 0.0, "novel", f"ta={ta_val},default"


# ---------------------------------------------------------------------------
# Composite Score + Confidence
# ---------------------------------------------------------------------------

# Weights for composite (sum to 1.0)
_W_ENDPOINT = 0.30
_W_DESIGN = 0.30
_W_EVIDENCE = 0.25
_W_MECHANISM = 0.15


@dataclass(frozen=True)
class ClinicalQualityResult:
    """Output for a single ticker's clinical quality assessment."""

    ticker: str

    # Composite
    clinical_quality_score: float  # [-1, +1]
    clinical_quality_confidence: str  # "high" / "medium" / "low"
    confidence_raw: float  # [0, 1] before bucketing

    # Sub-scores
    endpoint_strength: float
    endpoint_strength_tier: str
    endpoint_signal: str

    design_rigor: float
    design_rigor_tier: str
    design_signals: List[str]

    prior_evidence: float
    prior_evidence_tier: str
    prior_evidence_notes: str

    mechanism_maturity: float
    mechanism_maturity_tier: str
    mechanism_notes: str

    # Diagnostics
    n_components_available: int
    notes: str


def _compute_confidence(
    n_components: int,
    endpoint_tier: str,
    design_tier: str,
    evidence_tier: str,
    mechanism_tier: str,
) -> Tuple[float, str]:
    """Compute confidence score and bucket.

    Base: available_component_count / 4.
    Haircut: 0.75 for heuristic/inferred-only sources.
    """
    raw = n_components / 4.0

    # Haircut for weak evidence
    n_unknowns = sum(1 for t in [endpoint_tier, design_tier, evidence_tier, mechanism_tier] if t == "unknown")
    if n_unknowns >= 2:
        raw *= 0.75

    raw = max(0.0, min(1.0, raw))

    if raw >= 0.75:
        bucket = "high"
    elif raw >= 0.50:
        bucket = "medium"
    else:
        bucket = "low"

    return round(raw, 4), bucket


def _design_quality_rank(trial: dict) -> tuple:
    """Rank a trial by design quality for tie-breaking within same phase.

    Returns a tuple suitable for max() — higher is better.
    Prefers: randomized > non-randomized, blinded > unblinded, controlled > uncontrolled.
    """
    alloc = (trial.get("allocation") or "").upper()
    mask = (trial.get("masking") or "").upper()
    model = (trial.get("intervention_model") or "").upper()
    title = (trial.get("title") or "").lower()

    is_rand = 1 if alloc == "RANDOMIZED" or "randomized" in title else 0
    blind = (
        2
        if mask in ("DOUBLE", "TRIPLE", "QUADRUPLE") or "double-blind" in title
        else (1 if mask == "SINGLE" or "single-blind" in title else 0)
    )
    is_controlled = 1 if model in ("PARALLEL", "CROSSOVER", "FACTORIAL") else 0
    has_endpoints = 1 if trial.get("primary_endpoints") else 0

    return (is_rand, blind, is_controlled, has_endpoints)


def _best_trial_for_scoring(ticker_trials: List[dict], as_of: date) -> dict:
    """Select the best interventional trial for quality scoring.

    Prefers most advanced phase, then best design quality within that phase.
    """
    candidates = []
    for t in ticker_trials:
        if not _pit_ok(t, as_of):
            continue
        if (t.get("study_type") or "").upper() != "INTERVENTIONAL":
            continue
        phase = _phase_num(str(t.get("phase", "")))
        if phase < 1.0:
            continue
        dq = _design_quality_rank(t)
        candidates.append((phase, dq, t.get("primary_completion_date") or "", t))

    if not candidates:
        return {}

    # Sort by phase DESC, then design quality DESC, then date DESC
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def compute_clinical_quality_scores(
    trial_records: List[dict],
    as_of_date: str,
) -> Dict[str, ClinicalQualityResult]:
    """Compute clinical quality scores for all tickers.

    PIT-safe: strict < as_of_date.

    Args:
        trial_records: Full trial records list (all tickers).
        as_of_date: YYYY-MM-DD string.

    Returns:
        {ticker: ClinicalQualityResult}
    """
    as_of = _parse_date(as_of_date)
    if as_of is None:
        logger.warning("clinical_quality_score: invalid as_of_date, returning empty")
        return {}

    # Group by ticker
    by_ticker: Dict[str, List[dict]] = {}
    for t in trial_records:
        tk = (t.get("ticker") or "").upper()
        if tk:
            by_ticker.setdefault(tk, []).append(t)

    results: Dict[str, ClinicalQualityResult] = {}

    for ticker, trials in by_ticker.items():
        # 1. Endpoint strength (from best trial's primary endpoints)
        best_ep_trial = _best_trial_for_scoring(trials, as_of)
        if best_ep_trial:
            ep_texts = best_ep_trial.get("primary_endpoints") or []
            ep_title = best_ep_trial.get("title", "")
            ep_score, ep_tier, ep_signal = compute_endpoint_strength_tier(ep_texts, ep_title)
            ep_available = True
        else:
            ep_score, ep_tier, ep_signal = 0.0, "unknown", ""
            ep_available = False

        # 2. Design rigor (from best trial's structured fields)
        best_design_trial = _best_trial_for_scoring(trials, as_of)
        if best_design_trial:
            dr_score, dr_tier, dr_signals = compute_design_rigor_tier(
                allocation=best_design_trial.get("allocation"),
                masking=best_design_trial.get("masking"),
                intervention_model=best_design_trial.get("intervention_model"),
                interventions=best_design_trial.get("interventions"),
                study_type=best_design_trial.get("study_type", ""),
                title=best_design_trial.get("title", ""),
            )
            dr_available = True
        else:
            dr_score, dr_tier, dr_signals = 0.0, "unknown", []
            dr_available = False

        # 3. Prior evidence direction (from all completed trials)
        pe_score, pe_tier, pe_notes = compute_prior_evidence_tier(trials, as_of)
        pe_available = pe_tier != "unknown"

        # 4. Mechanism maturity (from all trials)
        mm_score, mm_tier, mm_notes = compute_mechanism_maturity_tier(trials, as_of)
        mm_available = mm_tier != "unknown"

        # Count available components
        n_available = sum([ep_available, dr_available, pe_available, mm_available])

        # Composite score (if >=2 components available, else 0.0)
        if n_available >= 2:
            composite = _W_ENDPOINT * ep_score + _W_DESIGN * dr_score + _W_EVIDENCE * pe_score + _W_MECHANISM * mm_score
            composite = max(-1.0, min(1.0, round(composite, 4)))
        else:
            composite = 0.0

        # Confidence
        conf_raw, conf_bucket = _compute_confidence(n_available, ep_tier, dr_tier, pe_tier, mm_tier)

        # Build notes
        note_parts = []
        if ep_tier != "unknown":
            note_parts.append(f"{ep_tier} endpoint")
        if dr_tier not in ("unknown", "weak"):
            note_parts.append(dr_tier.replace("_", " ") + " design")
        if pe_tier not in ("unknown", "mixed"):
            note_parts.append(f"prior evidence {pe_tier.replace('_', ' ')}")
        if mm_tier != "unknown":
            note_parts.append(f"{mm_tier.replace('_', ' ')} mechanism")
        notes_str = "; ".join(note_parts) if note_parts else "insufficient data"

        results[ticker] = ClinicalQualityResult(
            ticker=ticker,
            clinical_quality_score=composite,
            clinical_quality_confidence=conf_bucket,
            confidence_raw=conf_raw,
            endpoint_strength=ep_score,
            endpoint_strength_tier=ep_tier,
            endpoint_signal=ep_signal,
            design_rigor=dr_score,
            design_rigor_tier=dr_tier,
            design_signals=dr_signals,
            prior_evidence=pe_score,
            prior_evidence_tier=pe_tier,
            prior_evidence_notes=pe_notes,
            mechanism_maturity=mm_score,
            mechanism_maturity_tier=mm_tier,
            mechanism_notes=mm_notes,
            n_components_available=n_available,
            notes=notes_str,
        )

    return results
