"""Deadline-constrained milestone optionality features (Spec 041).

Phase 1: feature builder only — no DEM integration.

Computes probability-weighted milestone expected value under deadline
pressure, using the bounded prior framework from clinical_pos_prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "milestone_optionality.v1"

# --- Timeline benchmarks (conservative months from phase start to completion) ---
PHASE_DURATION_MONTHS: Dict[str, int] = {
    "phase1": 18,
    "phase2": 30,
    "phase3": 36,
    "phase2_3": 36,
    "nda_review": 12,
    "nda_review_priority": 8,
    "bla_review": 12,
    "bla_review_priority": 8,
}

MILESTONE_VALUE_WEIGHTS: Dict[str, float] = {
    "approval": 1.00,
    "nda_bla_filing": 0.80,
    "phase3_data": 0.70,
    "phase2_data": 0.50,
    "phase1_data": 0.30,
    "preclinical": 0.15,
    "other": 0.25,
}

DESIGNATION_ACCELERATION: Dict[str, float] = {
    "BTD": 0.85,
    "RMAT": 0.88,
    "FT": 0.95,
    "ODD": 1.00,
    "PR": 0.75,
}

_SIGMOID_MIDPOINT_DAYS = 0
_SIGMOID_SCALE_DAYS = 90


@dataclass
class MilestoneInput:
    """A single milestone for a ticker."""

    milestone_type: str
    deadline_date: Optional[date] = None
    estimated_completion_date: Optional[date] = None
    payout_value: Optional[float] = None
    phase: str = "unknown"
    nct_id: str = ""
    is_hard_deadline: bool = False


@dataclass
class MilestoneResult:
    """Output for a single ticker."""

    ticker: str
    milestone_deadline_mode: str = "none"
    milestone_count_active: int = 0
    milestone_primary_type: str = ""
    milestone_primary_days_to_deadline: Optional[int] = None
    milestone_timeline_slack_days: Optional[int] = None
    milestone_timeline_feasible_flag: bool = True
    milestone_safety_delay_flag: bool = False
    milestone_pos_by_deadline_raw: float = 0.0
    milestone_pos_by_deadline_shrunk: float = 0.0
    milestone_value_weight: float = 0.0
    milestone_timeline_weight: float = 0.0
    milestone_corr_penalty: float = 1.0
    milestone_deadline_ev_raw: float = 0.0
    milestone_deadline_ev_pct: float = 0.0
    milestone_deadline_overlay_z: float = 0.0
    clinical_optionality_deadline_overlay_pct: float = 0.0
    milestone_confidence_support: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _sigmoid(x: float, midpoint: float = 0.0, scale: float = 1.0) -> float:
    z = (x - midpoint) / max(scale, 1e-9)
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _estimate_completion_months(
    phase: str, has_btd: bool = False, has_rmat: bool = False, has_ft: bool = False, has_pr: bool = False
) -> float:
    phase_norm = phase.lower().replace(" ", "")

    if phase_norm in ("phase1", "phase 1"):
        months: float = (
            PHASE_DURATION_MONTHS["phase1"]
            + PHASE_DURATION_MONTHS["phase2"]
            + PHASE_DURATION_MONTHS["phase3"]
            + PHASE_DURATION_MONTHS["nda_review"]
        )
    elif phase_norm in ("phase2", "phase 2"):
        months = PHASE_DURATION_MONTHS["phase2"] + PHASE_DURATION_MONTHS["phase3"] + PHASE_DURATION_MONTHS["nda_review"]
    elif phase_norm in ("phase2_3", "phase 2/3"):
        months = PHASE_DURATION_MONTHS["phase2_3"] + PHASE_DURATION_MONTHS["nda_review"]
    elif phase_norm in ("phase3", "phase 3"):
        months = PHASE_DURATION_MONTHS["phase3"] + PHASE_DURATION_MONTHS["nda_review"]
    elif phase_norm in ("nda", "bla", "filed"):
        months = PHASE_DURATION_MONTHS["nda_review"]
    elif phase_norm in ("approved",):
        months = 0
    else:
        months = 60 + PHASE_DURATION_MONTHS["nda_review"]

    accel = 1.0
    if has_pr:
        review_months: float = PHASE_DURATION_MONTHS["nda_review"]
        dev_months = months - review_months
        review_months = review_months * DESIGNATION_ACCELERATION["PR"]
        months = dev_months + review_months
    if has_btd:
        accel = min(accel, DESIGNATION_ACCELERATION["BTD"])
    if has_rmat:
        accel = min(accel, DESIGNATION_ACCELERATION["RMAT"])
    if has_ft:
        accel = min(accel, DESIGNATION_ACCELERATION["FT"])

    return months * accel


def compute_milestone_features(
    ticker: str,
    milestones: List[MilestoneInput],
    as_of_date: date,
    pos_prior_fn=None,
    designations: Optional[List[str]] = None,
    slippage_score: Optional[float] = None,
    corr_alpha: float = 0.5,
    safety_haircut: float = 0.0,
    execution_haircut: float = 0.0,
) -> MilestoneResult:
    """Compute deadline-constrained milestone optionality for one ticker."""
    result = MilestoneResult(ticker=ticker)
    desig_set = set(designations or [])

    if not milestones:
        result.milestone_confidence_support = "no_milestones"
        return result

    active = [m for m in milestones if m.deadline_date and m.deadline_date > as_of_date]
    if not active:
        result.milestone_confidence_support = "no_future_deadlines"
        return result

    result.milestone_count_active = len(active)

    has_hard = any(m.is_hard_deadline for m in active)
    if has_hard:
        result.milestone_deadline_mode = "fixed_deadline"
    else:
        result.milestone_deadline_mode = "dated_event"

    safety_haircut = max(0.0, min(0.25, safety_haircut))
    execution_haircut = max(0.0, min(0.20, execution_haircut))

    if slippage_score is not None and slippage_score < 50:
        execution_haircut = min(0.20, execution_haircut + (50 - slippage_score) / 250)

    w_risk = 1.0 - safety_haircut - execution_haircut

    n_cluster = len(active)
    w_corr = 1.0 / (1.0 + corr_alpha * max(0, n_cluster - 1))
    result.milestone_corr_penalty = round(w_corr, 4)

    evs: List[Dict[str, Any]] = []
    primary_idx = 0
    primary_ev = -1.0

    for i, m in enumerate(active):
        days_to = (m.deadline_date - as_of_date).days

        p_base = 0.10
        if pos_prior_fn:
            try:
                prior = pos_prior_fn(m.phase, "other")
                p_base = prior.get("pos_prior", 0.10)
            except Exception:
                pass

        est_months = _estimate_completion_months(
            m.phase,
            has_btd="BTD" in desig_set,
            has_rmat="RMAT" in desig_set,
            has_ft="FT" in desig_set,
            has_pr="PR" in desig_set,
        )
        est_completion = as_of_date + timedelta(days=int(est_months * 30.44))
        slack_days = (m.deadline_date - est_completion).days

        w_timeline = _sigmoid(slack_days, _SIGMOID_MIDPOINT_DAYS, _SIGMOID_SCALE_DAYS)

        if m.payout_value is not None and m.payout_value > 0:
            w_value = min(1.0, m.payout_value / 10.0)
        else:
            w_value = MILESTONE_VALUE_WEIGHTS.get(m.milestone_type, 0.25)

        ev_i = p_base * w_timeline * w_risk * w_value

        evs.append(
            {
                "milestone_type": m.milestone_type,
                "days_to_deadline": days_to,
                "slack_days": slack_days,
                "p_base": p_base,
                "w_timeline": w_timeline,
                "w_value": w_value,
                "ev": ev_i,
            }
        )

        if ev_i > primary_ev:
            primary_ev = ev_i
            primary_idx = i

    primary = evs[primary_idx]
    primary_m = active[primary_idx]
    result.milestone_primary_type = primary_m.milestone_type
    result.milestone_primary_days_to_deadline = primary["days_to_deadline"]
    result.milestone_timeline_slack_days = primary["slack_days"]
    result.milestone_timeline_feasible_flag = primary["slack_days"] > -90
    result.milestone_safety_delay_flag = safety_haircut > 0
    result.milestone_pos_by_deadline_raw = round(primary["p_base"], 4)
    result.milestone_pos_by_deadline_shrunk = round(primary["p_base"] * primary["w_timeline"], 4)
    result.milestone_value_weight = round(primary["w_value"], 4)
    result.milestone_timeline_weight = round(primary["w_timeline"], 4)

    total_ev = w_corr * sum(e["ev"] for e in evs)
    result.milestone_deadline_ev_raw = round(total_ev, 6)
    result.milestone_deadline_ev_pct = round(min(100.0, total_ev * 100), 4)
    result.clinical_optionality_deadline_overlay_pct = result.milestone_deadline_ev_pct

    support_parts = []
    if len(active) > 0:
        support_parts.append(f"n={len(active)}")
    if has_hard:
        support_parts.append("hard_deadline")
    if desig_set:
        support_parts.append(f"desig={'|'.join(sorted(desig_set))}")
    result.milestone_confidence_support = ";".join(support_parts) or "minimal"

    return result


def compute_universe_milestone_features(
    rankings_rows: List[Dict[str, Any]],
    trial_records: List[Dict[str, Any]],
    pdufa_entries: List[Dict[str, Any]],
    fda_designations: List[Dict[str, Any]],
    as_of_date: date,
    pos_prior_fn=None,
    slippage_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, MilestoneResult]:
    """Compute milestone features for the entire universe."""
    desig_by_ticker: Dict[str, List[str]] = {}
    for d in fda_designations:
        t = d.get("ticker", "")
        dtype = d.get("designation_type", "")
        if t and dtype:
            desig_by_ticker.setdefault(t, []).append(dtype)

    pdufa_by_ticker: Dict[str, List[Dict]] = {}
    for p in pdufa_entries:
        t = p.get("ticker", "")
        if t:
            pdufa_by_ticker.setdefault(t, []).append(p)

    trials_by_ticker: Dict[str, List[Dict]] = {}
    for tr in trial_records:
        t = tr.get("ticker", "")
        if t:
            trials_by_ticker.setdefault(t, []).append(tr)

    phase_map = {}
    for row in rankings_rows:
        t = row.get("ticker", "")
        phase = row.get("clinical_lead_phase", row.get("lead_program_phase", ""))
        phase_map[t] = phase

    results: Dict[str, MilestoneResult] = {}

    for row in rankings_rows:
        ticker = row.get("ticker", "")
        if not ticker:
            continue

        milestones: List[MilestoneInput] = []
        phase = phase_map.get(ticker, "unknown")
        phase_norm = _normalize_phase(phase)

        for p in pdufa_by_ticker.get(ticker, []):
            pdufa_str = p.get("pdufa_date", "")
            if pdufa_str:
                try:
                    pdufa_d = date.fromisoformat(pdufa_str)
                    milestones.append(
                        MilestoneInput(
                            milestone_type="approval",
                            deadline_date=pdufa_d,
                            phase="nda",
                            is_hard_deadline=True,
                        )
                    )
                except ValueError:
                    pass

        for tr in trials_by_ticker.get(ticker, []):
            pcd = tr.get("primary_completion_date", "")
            tr_phase = tr.get("phase", "")
            status = tr.get("status", "")
            if status not in ("RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"):
                continue
            if pcd:
                try:
                    pcd_date = date.fromisoformat(pcd[:10])
                    mtype = _phase_to_milestone_type(tr_phase)
                    milestones.append(
                        MilestoneInput(
                            milestone_type=mtype,
                            deadline_date=pcd_date,
                            phase=_normalize_phase(tr_phase),
                            is_hard_deadline=False,
                            nct_id=tr.get("nct_id", ""),
                        )
                    )
                except ValueError:
                    pass

        if not any(m.milestone_type == "approval" for m in milestones):
            if phase_norm == "phase3" and ("BTD" in desig_by_ticker.get(ticker, [])):
                inferred_deadline = as_of_date + timedelta(days=int(2.5 * 365))
                milestones.append(
                    MilestoneInput(
                        milestone_type="approval",
                        deadline_date=inferred_deadline,
                        phase="phase3",
                        is_hard_deadline=False,
                    )
                )

        result = compute_milestone_features(
            ticker=ticker,
            milestones=milestones,
            as_of_date=as_of_date,
            pos_prior_fn=pos_prior_fn,
            designations=desig_by_ticker.get(ticker, []),
            slippage_score=(slippage_scores or {}).get(ticker),
        )
        results[ticker] = result

    return results


def z_score_overlay(results: Dict[str, MilestoneResult]) -> None:
    """Z-score the milestone_deadline_ev_pct across the universe in-place."""
    vals = [r.milestone_deadline_ev_pct for r in results.values() if r.milestone_count_active > 0]
    if len(vals) < 2:
        return
    mean_val = sum(vals) / len(vals)
    var_val = sum((v - mean_val) ** 2 for v in vals) / len(vals)
    std_val = var_val**0.5
    if std_val < 1e-9:
        return
    for r in results.values():
        if r.milestone_count_active > 0:
            r.milestone_deadline_overlay_z = round((r.milestone_deadline_ev_pct - mean_val) / std_val, 4)


def _normalize_phase(phase: str) -> str:
    if not phase:
        return "unknown"
    p = phase.lower().strip().replace(" ", "")
    mapping = {
        "phase1": "phase1",
        "1": "phase1",
        "1.0": "phase1",
        "phase2": "phase2",
        "2": "phase2",
        "2.0": "phase2",
        "phase3": "phase3",
        "3": "phase3",
        "3.0": "phase3",
        "phase2/3": "phase2_3",
        "phase2_3": "phase2_3",
        "phase4": "phase4",
        "4": "phase4",
        "nda": "nda",
        "bla": "bla",
        "filed": "nda",
        "approved": "approved",
    }
    return mapping.get(p, "unknown")


def _phase_to_milestone_type(phase: str) -> str:
    p = _normalize_phase(phase)
    return {
        "phase1": "phase1_data",
        "phase2": "phase2_data",
        "phase3": "phase3_data",
        "phase2_3": "phase3_data",
        "nda": "nda_bla_filing",
        "approved": "approval",
    }.get(p, "other")
