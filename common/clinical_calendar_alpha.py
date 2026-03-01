"""
Clinical Calendar Alpha v2 — Program-Aware Clinical Scoring

Post-processing layer that enriches Module 4 clinical scores with:
  1. Program clustering (intervention/condition grouping)
  2. Readout density (calendar-weighted event curves)
  3. Execution momentum (status progression, PCD shifts)
  4. Design quality (title-parsed signals)
  5. Endpoint strength (TA-based proxy)
  6. Competitive intensity (crowding + positioning)
  7. Score composition + sizing multiplier

All functions are pure, deterministic, and PIT-safe (strict < as_of_date).
No side effects, no network calls.

Author: Wake Robin Capital Management
Version: 2.0.0
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

__version__ = "2.0.0"
__all__ = [
    "CalendarAlphaConfig",
    "cluster_programs",
    "compute_program_features",
    "compute_readout_density",
    "compute_execution_momentum",
    "compute_design_quality",
    "compute_endpoint_strength",
    "compute_competitive_intensity",
    "compose_clinical_score_v2",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s) -> Optional[date]:
    """Parse YYYY-MM-DD string to date, or None."""
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _pit_ok(trial: dict, as_of: date) -> bool:
    """PIT gate: trial publicly known BEFORE as_of (strict <, no same-day leak)."""
    fp = _parse_date(trial.get("first_posted"))
    lup = _parse_date(trial.get("last_update_posted"))
    pit_date = fp or lup
    return pit_date is not None and pit_date < as_of


_CLINICAL_PHASES = frozenset({
    "PHASE1", "PHASE2", "PHASE3",
    "PHASE 1", "PHASE 2", "PHASE 3",
    "PHASE1/PHASE2", "PHASE 1/PHASE 2",
    "PHASE2/PHASE3", "PHASE 2/PHASE 3",
    "EARLY_PHASE1",
})


def _is_interventional_phase(trial: dict) -> bool:
    """True if INTERVENTIONAL and Phase 1-3."""
    if (trial.get("study_type") or "").upper() != "INTERVENTIONAL":
        return False
    phase = str(trial.get("phase", "")).upper()
    return phase in _CLINICAL_PHASES


_PHASE_NUM_MAP = {
    "phase 3": 3.0, "phase3": 3.0, "phase 2/3": 2.5, "phase2/phase3": 2.5,
    "phase 2": 2.0, "phase2": 2.0, "phase 1/2": 1.5, "phase1/phase2": 1.5,
    "phase 1": 1.0, "phase1": 1.0, "early_phase1": 0.5,
}


def _phase_num(phase_str: str) -> float:
    return _PHASE_NUM_MAP.get(phase_str.lower().strip(), 0.0)


# ---------------------------------------------------------------------------
# Section 1: Program Clustering
# ---------------------------------------------------------------------------

# Tokens to skip when normalizing interventions
_PLACEBO_PREFIXES = re.compile(
    r"^(placebo|sham|vehicle|saline|standard\s+of\s+care|no\s+intervention)", re.I
)

_DOSAGE_RE = re.compile(r"\d+\s*m[gl]\b|\d+\s*mg/m[l2]\b|\d+\s*iu\b|\d+\s*mcg\b", re.I)
_FORMULATION_RE = re.compile(
    r"\b(tablets?|injection|capsules?|infusion|solution|suspension|powder|spray|patch)\b", re.I
)

# Generic intervention tokens that shouldn't be used for clustering
_INTERVENTION_STOPLIST = frozenset({
    "antibody", "vaccine", "cell", "therapy", "treatment",
    "monoclonal", "recombinant", "human", "inhibitor", "agent",
    "combination", "regimen", "protocol", "standard", "care",
    "placebo", "control", "comparator", "active",
})

_TEXT_STOP_TOKENS = frozenset({
    "study", "phase", "randomized", "double", "blind", "single",
    "open", "label", "multicenter", "multi", "center", "trial",
    "clinical", "safety", "efficacy", "dose", "finding", "escalation",
    "the", "and", "for", "with", "from", "that", "this", "are",
    "was", "were", "been", "have", "has", "had", "will", "shall",
    "may", "can", "its", "also", "each", "not", "but", "other",
    "evaluate", "assess", "determine", "compare", "investigation",
    "investigational", "subjects", "patients", "participants",
    "healthy", "volunteers", "adults", "adult",
})


def normalize_intervention(raw: str) -> Optional[str]:
    """Normalize an intervention string to a canonical token, or None."""
    if not raw:
        return None
    text = raw.strip().lower()
    if _PLACEBO_PREFIXES.match(text):
        return None
    text = _DOSAGE_RE.sub("", text)
    text = _FORMULATION_RE.sub("", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text or len(text) <= 2:
        return None
    return text


def _tokenize_text(text: str) -> frozenset:
    """Tokenize text into a frozenset of meaningful tokens."""
    if not text:
        return frozenset()
    words = re.split(r"[\s\-/,;:()]+", text.lower())
    return frozenset(
        w for w in words
        if len(w) > 2 and w not in _TEXT_STOP_TOKENS
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def cluster_programs(
    trials: list,
    ticker: str,
    as_of: date,
) -> list:
    """Cluster trials into therapeutic programs for a single ticker.

    Returns list of program dicts sorted deterministically:
    phase_max DESC -> readout_days ASC (None last) -> trial_count DESC -> sha1(key).
    """
    pit_trials = [t for t in trials if _pit_ok(t, as_of)]
    if not pit_trials:
        return []

    # Extract per-trial features
    trial_features = []
    for t in pit_trials:
        # Intervention tokens: from interventions list, filtering generic stoplist
        interv_raw = t.get("interventions") or []
        if isinstance(interv_raw, str):
            interv_raw = [interv_raw]
        interv_tokens = set()
        for iv in interv_raw:
            normed = normalize_intervention(iv)
            if normed:
                for tok in _tokenize_text(normed):
                    if tok not in _INTERVENTION_STOPLIST:
                        interv_tokens.add(tok)

        # Fallback to title tokens if no intervention tokens
        if not interv_tokens:
            title = t.get("title", "")
            interv_tokens = set(
                tok for tok in _tokenize_text(title)
                if tok not in _INTERVENTION_STOPLIST
            )

        # Condition tokens
        conditions = t.get("conditions") or []
        if isinstance(conditions, str):
            conditions = [conditions]
        cond_tokens = set()
        for c in conditions:
            cond_tokens.update(_tokenize_text(c))

        # Top condition token (most specific — longest token, deterministic)
        _ultra_generic = {"disease", "disorder", "syndrome", "condition", "cancer"}
        specific_conds = sorted(
            [tok for tok in cond_tokens if tok not in _ultra_generic],
            key=lambda x: (-len(x), x),
        )
        top_cond = specific_conds[0] if specific_conds else ""

        # Readout date (future PCD/CD only, strict < as_of for past exclusion)
        pcd = _parse_date(t.get("primary_completion_date"))
        cd = _parse_date(t.get("completion_date"))
        readout_days = None
        if pcd is not None and pcd >= as_of:
            readout_days = (pcd - as_of).days
        elif cd is not None and cd >= as_of:
            readout_days = (cd - as_of).days
        # Past PCD/CD: excluded entirely (no fabricated days=0)

        phase = _phase_num(str(t.get("phase", "")))

        trial_features.append({
            "trial": t,
            "interv_tokens": frozenset(sorted(interv_tokens)[:8]),
            "cond_tokens": frozenset(cond_tokens),
            "top_cond": top_cond,
            "readout_days": readout_days,
            "phase": phase,
            "nct_id": t.get("nct_id", ""),
        })

    # Group into programs using Jaccard overlap ≥ 0.3 on intervention tokens
    # AND matching top condition token
    programs = []  # list of lists of trial_features
    for tf in trial_features:
        merged = False
        for prog in programs:
            # Check intervention token overlap with Jaccard threshold
            ref_tokens = prog[0]["interv_tokens"]
            overlap = _jaccard(tf["interv_tokens"], ref_tokens)
            same_cond = (tf["top_cond"] == prog[0]["top_cond"]) if tf["top_cond"] and prog[0]["top_cond"] else True

            if overlap >= 0.3 and same_cond and tf["interv_tokens"] and ref_tokens:
                prog.append(tf)
                merged = True
                break
        if not merged:
            programs.append([tf])

    # Build program summaries
    result = []
    for prog in programs:
        phases = [tf["phase"] for tf in prog]
        readout_days_list = [tf["readout_days"] for tf in prog if tf["readout_days"] is not None]
        late_count = sum(1 for p in phases if p >= 2.0)
        # Deterministic key for tiebreak
        key_str = "|".join(sorted(tf["nct_id"] for tf in prog))
        key_hash = hashlib.sha1(key_str.encode()).hexdigest()[:8]

        result.append({
            "phase_max": max(phases) if phases else 0.0,
            "upcoming_readout_days": min(readout_days_list) if readout_days_list else None,
            "trial_count": len(prog),
            "late_stage_count": late_count,
            "nct_ids": [tf["nct_id"] for tf in prog],
            "_key_hash": key_hash,
        })

    # Sort: phase_max DESC -> readout_days ASC (None last) -> trial_count DESC -> hash
    result.sort(key=lambda p: (
        -p["phase_max"],
        p["upcoming_readout_days"] if p["upcoming_readout_days"] is not None else 999999,
        -p["trial_count"],
        p["_key_hash"],
    ))
    return result


def compute_program_features(
    trial_records: list,
    as_of_date: str,
) -> Dict[str, dict]:
    """Compute program-level features per ticker.

    Returns {ticker: {lead_program_phase, lead_program_readout_days,
    lead_program_trial_count, top2_program_phase, program_count,
    program_diversification_score, program_concentration}}.
    """
    as_of = _parse_date(as_of_date)
    if as_of is None:
        return {}

    by_ticker: Dict[str, list] = {}
    for t in trial_records:
        tk = (t.get("ticker") or "").upper()
        if tk:
            by_ticker.setdefault(tk, []).append(t)

    result = {}
    for ticker, trials in by_ticker.items():
        programs = cluster_programs(trials, ticker, as_of)
        if not programs:
            continue

        lead = programs[0]
        top2 = programs[1] if len(programs) > 1 else None

        n_programs = len(programs)
        # Diversification: min(1.0, (n-1)/3) if top2 exists and phase ≥ 1
        if top2 and top2["phase_max"] >= 1.0:
            diversification = min(1.0, (n_programs - 1) / 3.0)
        else:
            diversification = 0.0

        # Concentration: lead trial fraction
        total_trials = sum(p["trial_count"] for p in programs)
        concentration = lead["trial_count"] / total_trials if total_trials > 0 else 1.0

        result[ticker] = {
            "lead_program_phase": lead["phase_max"],
            "lead_program_readout_days": lead["upcoming_readout_days"],
            "lead_program_trial_count": lead["trial_count"],
            "top2_program_phase": top2["phase_max"] if top2 else None,
            "program_count": n_programs,
            "program_diversification_score": round(diversification, 4),
            "program_concentration": round(concentration, 4),
        }

    return result


# ---------------------------------------------------------------------------
# Section 2: Readout Density
# ---------------------------------------------------------------------------

_READOUT_PHASE_WEIGHTS = {
    "phase 3": 1.5, "phase3": 1.5,
    "phase 2/3": 1.25, "phase2/phase3": 1.25,
    "phase 2": 1.0, "phase2": 1.0,
    "phase 1/2": 0.5, "phase1/phase2": 0.5,
    "phase 1": 0.25, "phase1": 0.25,
    "early_phase1": 0.1,
}


def compute_readout_density(
    trial_records: list,
    as_of_date: str,
    K: int = 8,
    tau: float = 120.0,
) -> Dict[str, dict]:
    """Compute readout density features per ticker.

    PIT-safe: strict < as_of_date. Only future PCD/CD events counted.
    Past PCD with no results_first_posted is EXCLUDED (not fabricated as days=0).

    Returns {ticker: {readout_curve_score, readout_density_30, readout_density_90,
    readout_density_180, late_stage_readouts_180}}.
    """
    as_of = _parse_date(as_of_date)
    if as_of is None:
        return {}

    by_ticker: Dict[str, list] = {}
    for t in trial_records:
        tk = (t.get("ticker") or "").upper()
        if tk:
            by_ticker.setdefault(tk, []).append(t)

    result = {}
    for ticker, trials in by_ticker.items():
        events = []  # (days, phase_weight)
        for t in trials:
            if not _pit_ok(t, as_of):
                continue
            if not _is_interventional_phase(t):
                continue

            phase_str = str(t.get("phase", "")).lower().strip()
            phase_w = _READOUT_PHASE_WEIGHTS.get(phase_str, 0.1)
            phase_n = _phase_num(phase_str)

            # Collect future event dates only
            pcd = _parse_date(t.get("primary_completion_date"))
            cd = _parse_date(t.get("completion_date"))

            candidate_date = None
            if pcd is not None and pcd >= as_of:
                candidate_date = pcd
            elif cd is not None and cd >= as_of:
                candidate_date = cd
            # Past dates: skip entirely (no fabricated imminent events)

            if candidate_date is None:
                continue

            days = (candidate_date - as_of).days
            events.append((days, phase_w, phase_n))

        if not events:
            result[ticker] = {
                "readout_curve_score": 0.0,
                "readout_density_30": 0,
                "readout_density_90": 0,
                "readout_density_180": 0,
                "late_stage_readouts_180": 0,
            }
            continue

        # Sort by days ascending, take top K
        events.sort(key=lambda e: (e[0], -e[1]))
        top_events = events[:K]

        curve_score = sum(pw * math.exp(-d / tau) for d, pw, _ in top_events)
        d30 = sum(1 for d, _, _ in events if d <= 30)
        d90 = sum(1 for d, _, _ in events if d <= 90)
        d180 = sum(1 for d, _, _ in events if d <= 180)
        late_180 = sum(1 for d, _, pn in events if pn >= 2.0 and d <= 180)

        result[ticker] = {
            "readout_curve_score": round(curve_score, 4),
            "readout_density_30": d30,
            "readout_density_90": d90,
            "readout_density_180": d180,
            "late_stage_readouts_180": late_180,
        }

    return result


# ---------------------------------------------------------------------------
# Section 3: Execution Momentum
# ---------------------------------------------------------------------------

_STATUS_ORDER = {
    "WITHDRAWN": 0, "TERMINATED": 1, "SUSPENDED": 2,
    "NOT_YET_RECRUITING": 3, "RECRUITING": 4,
    "ENROLLING_BY_INVITATION": 5, "ACTIVE_NOT_RECRUITING": 6,
    "ACTIVE": 6, "COMPLETED": 7,
}

_REGRESSION_STATUSES = frozenset({"TERMINATED", "WITHDRAWN", "SUSPENDED"})


def compute_execution_momentum(
    current_trials: list,
    prior_trials: Optional[list],
    as_of_date: str,
) -> Dict[str, dict]:
    """Compute execution momentum by comparing current vs prior trial snapshots.

    PIT-safe: strict < as_of_date.
    If prior_trials is None, returns zeros with coverage="no_prior".
    """
    as_of = _parse_date(as_of_date)
    if as_of is None:
        return {}

    by_ticker_current: Dict[str, Dict[str, dict]] = {}
    for t in current_trials:
        if not _pit_ok(t, as_of):
            continue
        tk = (t.get("ticker") or "").upper()
        nct = t.get("nct_id", "")
        if tk and nct:
            by_ticker_current.setdefault(tk, {})[nct] = t

    all_tickers = set(by_ticker_current.keys())

    if prior_trials is None:
        return {
            tk: {
                "execution_momentum_score": 0.0,
                "execution_advancements": 0,
                "execution_slippages": 0,
                "execution_stale_count": 0,
                "execution_delta_coverage": "no_prior",
            }
            for tk in all_tickers
        }

    by_ticker_prior: Dict[str, Dict[str, dict]] = {}
    for t in prior_trials:
        tk = (t.get("ticker") or "").upper()
        nct = t.get("nct_id", "")
        if tk and nct:
            by_ticker_prior.setdefault(tk, {})[nct] = t

    result = {}
    for ticker in all_tickers:
        cur_map = by_ticker_current.get(ticker, {})
        prior_map = by_ticker_prior.get(ticker, {})
        common_ncts = set(cur_map.keys()) & set(prior_map.keys())

        advancements = 0
        regressions = 0
        slippages = 0
        stale_count = 0
        matched = len(common_ncts)

        for nct in common_ncts:
            cur = cur_map[nct]
            pri = prior_map[nct]

            # Status delta
            cur_status = str(cur.get("status") or cur.get("overall_status") or "").upper()
            pri_status = str(pri.get("status") or pri.get("overall_status") or "").upper()
            cur_ord = _STATUS_ORDER.get(cur_status, 3)
            pri_ord = _STATUS_ORDER.get(pri_status, 3)
            delta = cur_ord - pri_ord
            if delta > 0:
                advancements += 1
            elif delta < 0 and cur_status in _REGRESSION_STATUSES:
                regressions += 1

            # PCD shift detection
            cur_pcd = _parse_date(cur.get("primary_completion_date"))
            pri_pcd = _parse_date(pri.get("primary_completion_date"))
            if cur_pcd and pri_pcd:
                shift = (cur_pcd - pri_pcd).days
                if shift >= 14:
                    slippages += 1  # pushed out
                # Pull-in (shift <= -14) counted as advancement signal

            # Staleness: RECRUITING but not updated > 365 days
            lup = _parse_date(cur.get("last_update_posted"))
            if cur_status in ("RECRUITING", "ENROLLING_BY_INVITATION") and lup:
                if (as_of - lup).days > 365:
                    stale_count += 1

        # Score: clamped ratio
        denom = max(1, matched)
        raw = (advancements - regressions) / denom
        momentum_score = max(-1.0, min(1.0, raw))

        result[ticker] = {
            "execution_momentum_score": round(momentum_score, 4),
            "execution_advancements": advancements,
            "execution_slippages": slippages,
            "execution_stale_count": stale_count,
            "execution_delta_coverage": f"{matched}/{len(cur_map)}",
        }

    return result


# ---------------------------------------------------------------------------
# Section 4: Design Quality (from title parsing)
# ---------------------------------------------------------------------------

_RANDOMIZED_RE = re.compile(r"\brandom(?:ized|ised|ization)\b", re.I)
_DOUBLE_BLIND_RE = re.compile(r"\bdouble[- ]?blind", re.I)
_SINGLE_BLIND_RE = re.compile(r"\bsingle[- ]?blind", re.I)
_OPEN_LABEL_RE = re.compile(r"\bopen[- ]?label", re.I)
_PARALLEL_RE = re.compile(r"\bparallel\b", re.I)
_CROSSOVER_RE = re.compile(r"\bcrossover\b", re.I)
_PLACEBO_CTRL_RE = re.compile(r"\bplacebo[- ]?controlled\b", re.I)
_PLACEBO_RE = re.compile(r"\bplacebo\b", re.I)


def _score_trial_design_from_title(trial: dict) -> Tuple[float, List[str]]:
    """Score trial design quality from title + interventions.

    Returns (score 0-1, list of signal tags).
    Missing/unclear signals stay at base (not penalized).
    """
    title = str(trial.get("title") or "")
    # Also scan interventions for placebo mentions
    intervs = trial.get("interventions") or []
    if isinstance(intervs, str):
        intervs = [intervs]
    search_text = title + " " + " ".join(str(iv) for iv in intervs)

    score = 0.3  # base for any trial
    signals = []

    if _RANDOMIZED_RE.search(search_text):
        score += 0.2
        signals.append("randomized")

    if _DOUBLE_BLIND_RE.search(search_text):
        score += 0.2
        signals.append("double_blind")
    elif _SINGLE_BLIND_RE.search(search_text):
        score += 0.1
        signals.append("single_blind")

    if _PARALLEL_RE.search(search_text):
        score += 0.1
        signals.append("parallel")

    if _PLACEBO_CTRL_RE.search(search_text) or _PLACEBO_RE.search(search_text):
        score += 0.1
        signals.append("placebo")

    # Open-label without blinding doesn't add to score
    if _OPEN_LABEL_RE.search(search_text) and not signals:
        signals.append("open_label")

    return min(1.0, score), signals


def compute_design_quality(
    trial_records: list,
    as_of_date: str,
) -> Dict[str, dict]:
    """Compute design quality features per ticker.

    PIT-safe: strict < as_of_date.
    Returns {ticker: {design_quality_score, design_signals}}.
    """
    as_of = _parse_date(as_of_date)
    if as_of is None:
        return {}

    by_ticker: Dict[str, list] = {}
    for t in trial_records:
        tk = (t.get("ticker") or "").upper()
        if tk:
            by_ticker.setdefault(tk, []).append(t)

    result = {}
    for ticker, trials in by_ticker.items():
        best_score = 0.0
        best_signals: List[str] = []
        for t in trials:
            if not _pit_ok(t, as_of):
                continue
            if not _is_interventional_phase(t):
                continue
            score, signals = _score_trial_design_from_title(t)
            if score > best_score:
                best_score = score
                best_signals = signals

        result[ticker] = {
            "design_quality_score": round(best_score, 4),
            "design_signals": ",".join(best_signals),
        }

    return result


# ---------------------------------------------------------------------------
# Section 5: Endpoint Strength (TA-based proxy)
# ---------------------------------------------------------------------------

# TA-based endpoint quality proxy (without outcomesModule data)
# Higher = harder endpoints more common in that TA
_TA_ENDPOINT_PROXY: Dict[str, float] = {
    "oncology": 0.85,      # hard endpoints (OS, PFS) common
    "cardiovascular": 0.75, # MACE endpoints strong
    "rare_disease": 0.65,   # functional endpoints
    "autoimmune": 0.60,     # validated scales (ACR, PASI)
    "cns": 0.60,            # validated cognitive/functional scales
    "metabolic": 0.55,      # biomarker + clinical
    "infectious_disease": 0.55,
    "respiratory": 0.50,
    "ophthalmology": 0.55,
    "dermatology": 0.50,
    "other": 0.50,
}

_LATE_STAGE_BONUS = 0.10  # bonus for late-stage in strong-endpoint TA


def compute_endpoint_strength(
    trial_records: list,
    as_of_date: str,
) -> Dict[str, dict]:
    """Compute endpoint strength proxy per ticker based on therapeutic area.

    PIT-safe: strict < as_of_date.
    Returns {ticker: {endpoint_strength_score, therapeutic_area}}.
    """
    from common.accuracy_improvements import classify_therapeutic_area

    as_of = _parse_date(as_of_date)
    if as_of is None:
        return {}

    by_ticker: Dict[str, list] = {}
    for t in trial_records:
        tk = (t.get("ticker") or "").upper()
        if tk:
            by_ticker.setdefault(tk, []).append(t)

    result = {}
    for ticker, trials in by_ticker.items():
        all_conditions: List[str] = []
        max_phase = 0.0
        for t in trials:
            if not _pit_ok(t, as_of):
                continue
            conds = t.get("conditions") or []
            if isinstance(conds, str):
                conds = [conds]
            all_conditions.extend(conds)
            max_phase = max(max_phase, _phase_num(str(t.get("phase", ""))))

        ta = classify_therapeutic_area(all_conditions)
        ta_val = ta.value  # string like "oncology"
        base_score = _TA_ENDPOINT_PROXY.get(ta_val, 0.50)

        # Late-stage bonus in strong-endpoint TAs
        if max_phase >= 2.0 and base_score >= 0.65:
            base_score = min(1.0, base_score + _LATE_STAGE_BONUS)

        result[ticker] = {
            "endpoint_strength_score": round(base_score, 4),
            "therapeutic_area": ta_val,
        }

    return result


# ---------------------------------------------------------------------------
# Section 6: Competitive Intensity Wrapper
# ---------------------------------------------------------------------------

def compute_competitive_intensity(
    trial_records: list,
    active_tickers: list,
    as_of_date: str,
) -> Dict[str, dict]:
    """Compute competitive intensity per ticker using CompetitiveIntensityEngine.

    Adapts trial records (status → overall_status).
    Builds landscape ONCE, then scores each ticker in a tight loop.
    Z-scores competitive_intensity_score across universe (ddof=0).

    Returns {ticker: {competitive_intensity_score, competitive_intensity_z,
    crowding_level, competitor_count}}.
    """
    from competitive_intensity_engine import CompetitiveIntensityEngine

    as_of = _parse_date(as_of_date)
    if as_of is None:
        return {}

    # Adapt records: engine expects overall_status, CTgov has status
    adapted = []
    for t in trial_records:
        rec = dict(t)
        if "overall_status" not in rec and "status" in rec:
            rec["overall_status"] = rec["status"]
        adapted.append(rec)

    # Build landscape ONCE
    engine = CompetitiveIntensityEngine()
    engine.build_landscape(adapted, as_of)

    # Score each ticker
    raw_results = {}
    for ticker in active_tickers:
        tk = ticker.upper()
        try:
            score_result = engine.score_ticker(tk, as_of)
            raw_results[tk] = {
                "competitive_intensity_score": float(score_result.get("competitive_intensity_score", 0)),
                "crowding_level": str(score_result.get("crowding_level", "uncrowded")),
                "competitor_count": int(score_result.get("competitor_count", 0)),
            }
        except Exception:
            raw_results[tk] = {
                "competitive_intensity_score": 0.0,
                "crowding_level": "uncrowded",
                "competitor_count": 0,
            }

    # Z-score across universe (ddof=0)
    scores = [v["competitive_intensity_score"] for v in raw_results.values()]
    if scores:
        mean_val = sum(scores) / len(scores)
        var_val = sum((s - mean_val) ** 2 for s in scores) / len(scores)
        std_val = var_val ** 0.5
    else:
        mean_val = 0.0
        std_val = 0.0

    result = {}
    for tk, data in raw_results.items():
        if std_val > 0:
            z = (data["competitive_intensity_score"] - mean_val) / std_val
        else:
            z = 0.0
        result[tk] = {
            "competitive_intensity_score": round(data["competitive_intensity_score"], 4),
            "competitive_intensity_z": round(z, 4),
            "crowding_level": data["crowding_level"],
            "competitor_count": data["competitor_count"],
        }

    return result


# ---------------------------------------------------------------------------
# Section 7: Score Composition + Sizing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalendarAlphaConfig:
    """Configuration for clinical calendar alpha v2 scoring."""
    w_readout_curve: float = 0.15
    w_readout_density: float = 0.10
    w_momentum: float = 0.10
    w_design: float = 0.10
    w_endpoint: float = 0.10
    w_competition: float = 0.05   # negative: penalizes crowded
    max_adjustment: float = 0.35  # cap per-component z contribution
    # Sizing
    enable_sizing: bool = False
    sizing_competition_dampen: float = 0.90
    sizing_design_dampen: float = 0.92
    sizing_density_boost: float = 1.03
    sizing_max_deviation: float = 0.10  # cap total sizing at ±10%


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compose_clinical_score_v2(
    clinical_score: Optional[float],
    features: dict,
    config: CalendarAlphaConfig,
    *,
    z_readout_curve: float = 0.0,
    z_readout_density: float = 0.0,
    z_momentum: float = 0.0,
    z_design: float = 0.0,
    z_endpoint: float = 0.0,
    z_competition: float = 0.0,
) -> Tuple[Optional[float], float, List[str]]:
    """Compose clinical_score_v2 from base score + feature z-scores.

    Args:
        clinical_score: Module 4 clinical_score (0-100) or None.
        features: Merged feature dict for this ticker (unused fields OK).
        config: Weights and caps.
        z_*: Pre-computed cross-sectional z-scores for each component.

    Returns:
        (clinical_score_v2, sizing_multiplier, reason_tags)
    """
    if clinical_score is None:
        return (None, 1.0, [])

    cap = config.max_adjustment
    reason_tags = []

    # Weighted adjustment (z-scores clamped per-component)
    adjustment = 0.0
    adjustment += config.w_readout_curve * _clamp(z_readout_curve, -cap, cap)
    adjustment += config.w_readout_density * _clamp(z_readout_density, -cap, cap)
    adjustment += config.w_momentum * _clamp(z_momentum, -cap, cap)
    adjustment += config.w_design * _clamp(z_design, -cap, cap)
    adjustment += config.w_endpoint * _clamp(z_endpoint, -cap, cap)
    # Competition weight is NEGATIVE (higher crowding → penalty)
    adjustment -= config.w_competition * _clamp(z_competition, -cap, cap)

    # Scale to 0-100 range
    score_v2 = clinical_score + adjustment * 10.0
    score_v2 = _clamp(score_v2, 0.0, 100.0)

    # Build reason tags
    if z_readout_curve > 0.5:
        reason_tags.append("strong_readout_curve")
    if z_momentum > 0.5:
        reason_tags.append("positive_momentum")
    elif z_momentum < -0.5:
        reason_tags.append("negative_momentum")
    if z_design > 0.5:
        reason_tags.append("strong_design")
    if z_competition > 1.0:
        reason_tags.append("high_competition")

    # Sizing multiplier
    sizing = 1.0
    if config.enable_sizing:
        if z_competition > 1.0:
            sizing *= config.sizing_competition_dampen
            reason_tags.append("sizing_competition_dampen")
        dqs = features.get("design_quality_score", 0.5)
        if isinstance(dqs, (int, float)) and dqs < 0.4:
            sizing *= config.sizing_design_dampen
            reason_tags.append("sizing_design_dampen")
        rcs = features.get("readout_curve_score", 0.0)
        eps = features.get("endpoint_strength_score", 0.5)
        if isinstance(rcs, (int, float)) and isinstance(eps, (int, float)):
            if rcs > 0.0 and eps >= 0.65:
                sizing *= config.sizing_density_boost
                reason_tags.append("sizing_density_boost")

        # Clamp total sizing deviation
        lo = 1.0 - config.sizing_max_deviation
        hi = 1.0 + config.sizing_max_deviation
        sizing = _clamp(sizing, lo, hi)

    return (round(score_v2, 4), round(sizing, 4), reason_tags)


# ---------------------------------------------------------------------------
# Cross-sectional z-score utility
# ---------------------------------------------------------------------------

def z_score_dict(
    values: Dict[str, float],
) -> Dict[str, float]:
    """Z-score a dict of ticker → float values (ddof=0).

    Returns dict of ticker → z-score.
    """
    if not values:
        return {}
    vals = list(values.values())
    n = len(vals)
    mean_val = sum(vals) / n
    var_val = sum((v - mean_val) ** 2 for v in vals) / n
    std_val = var_val ** 0.5

    if std_val == 0:
        return {k: 0.0 for k in values}
    return {k: (v - mean_val) / std_val for k, v in values.items()}
