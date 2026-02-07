#!/usr/bin/env python3
"""
Baker Overlay — CIK-Level Attribution, IC Review Queues, Alignment Metrics

Post-processing overlay that enriches ranked securities with Baker Bros-specific
signals and generates deterministic IC review queues.

Design:
- DETERMINISTIC: No datetime.now(), no randomness
- STDLIB-ONLY: No external dependencies
- Operates on Module 5 output + raw holdings data
- Bonus is computed but NOT applied to composite_score (advisory only)

Queue 1 — "Baker Core Divergence" (IC review, target 15-30 names):
    Include if ANY of:
    - baker_portfolio_weight >= 1.0% AND composite_rank > 75
    - baker_position_value >= $250M AND baker_portfolio_weight >= 0.25% AND composite_rank > 75
    - baker_activity in {new, add} AND baker_portfolio_weight >= 0.25% AND composite_rank > 75

Queue 2 — "Baker Near-Term Catalyst" (actionable watchlist):
    - baker_held == True
    - days_to_catalyst between 15 and 120
    - risk gates pass (no sev2+, no survivability_critical, no runway < 6m)
    - composite_rank > 50

Note on value_kusd: The holdings_detailed.json field "value_kusd" actually stores
values in raw USD (not thousands). This module treats the field as raw USD throughout.

Author: Wake Robin Capital Management
Version: 1.1.0
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BAKER_CIK = "0001263508"

# Activity classification
ACTIVITY_CHANGE_THRESHOLD = 0.10  # 10% = meaningful change

# Queue 1: Baker Core Divergence (IC review)
# Note: value_kusd field stores raw USD despite the name
Q1_WEIGHT_PCT = 0.01           # 1.0% portfolio weight (as fraction)
Q1_WEIGHT_RANK = 75            # composite_rank > 75
Q1_VALUE_USD = 250_000_000     # $250M in raw USD
Q1_VALUE_RANK = 75
Q1_VALUE_MIN_WEIGHT = 0.0025   # 0.25% portfolio weight floor for value trigger
Q1_ACTIVITY_RANK = 75
Q1_ACTIVITY_MIN_WEIGHT = 0.0025  # 0.25% portfolio weight floor for activity trigger

# Queue 2: Baker Near-Term Catalyst
Q2_CATALYST_MIN_DAYS = 15
Q2_CATALYST_MAX_DAYS = 120
Q2_RANK = 50

# Alignment
RECALL_K_VALUES = [50, 100]
BAKER_TOP_N_FOR_RECALL = 20

VERSION = "1.1.0"


def _classify_activity(current_val: float, prior_val: float) -> Tuple[str, Optional[float]]:
    """Classify Baker position activity and compute % change."""
    if prior_val == 0 and current_val > 0:
        return "new", None
    if current_val == 0 and prior_val > 0:
        return "exited", -1.0
    if prior_val == 0 and current_val == 0:
        return "unchanged", 0.0

    pct_change = (current_val - prior_val) / prior_val
    if pct_change > ACTIVITY_CHANGE_THRESHOLD:
        return "add", pct_change
    elif pct_change < -ACTIVITY_CHANGE_THRESHOLD:
        return "trim", pct_change
    else:
        return "unchanged", pct_change


def _risk_gates_pass(security: dict) -> bool:
    """Check if security passes risk gates for Queue 2.

    Mirrors Guardrail A1 conditions: blocks sev2+, survivability_critical, runway<6m.
    """
    severity = security.get("severity", "none")
    if severity in ("sev2", "sev3"):
        return False

    flags = security.get("flags", [])
    for flag in flags:
        if "guardrail_a_survivability_critical" in flag:
            return False
        if "guardrail_a_cash_runway_lt_6m" in flag:
            return False

    return True


def _guardrail_blocks_bonus(security: dict) -> bool:
    """Check if Guardrail A or A2 blocks the additive bonus."""
    flags = security.get("flags", [])
    for flag in flags:
        if flag.startswith("guardrail_a_sev_"):
            return True
        if "guardrail_a_survivability_critical" in flag:
            return True
        if "guardrail_a_cash_runway_lt_6m" in flag:
            return True
        if "guardrail_a_red_flag_eligible" in flag:
            return True
    return False


def _extract_disagreement_reasons(security: dict) -> List[str]:
    """Extract binding penalties/gates that explain why the model ranks this security low.

    Returns a list of short reason strings for IC review.
    """
    reasons = []
    flags = security.get("flags", [])
    severity = security.get("severity", "none")

    # 1. Severity penalties
    if severity in ("sev2", "sev3"):
        reasons.append(f"severity_{severity}")
    elif severity == "sev1":
        reasons.append("severity_sev1")

    # 2. Thesis gate
    if any("thesis_gate_penalty" in f for f in flags):
        reasons.append("thesis_gate")

    # 3. SM reinforcement blocked
    if any("sm_reinforcement_blocked" in f for f in flags):
        reasons.append("sm_reinforcement_blocked")

    # 4. Red flag suppression
    sb = security.get("score_breakdown", {}) or {}
    pg = sb.get("penalties_and_gates", {}) or {}
    if pg.get("red_flag_suppressed"):
        reasons.append("red_flag_suppressed")

    # 5. Low normalized components (< 30)
    components = security.get("component_scores") or sb.get("components", [])
    if isinstance(components, list):
        for comp in components:
            try:
                norm = float(comp.get("normalized", 50))
                if norm < 30:
                    reasons.append(f"low_{comp['name']}={norm:.0f}")
            except (ValueError, TypeError, KeyError):
                pass

    # 6. Low enhancement scores (momentum < 35, valuation < 30)
    enhancements = sb.get("enhancements", {}) or {}
    mom = enhancements.get("momentum", {}) or {}
    try:
        mom_score = float(mom.get("score", 50))
        if mom_score < 35:
            reasons.append(f"neg_momentum={mom_score:.0f}")
    except (ValueError, TypeError):
        pass

    val = enhancements.get("valuation", {}) or {}
    try:
        val_score = float(val.get("score", 50))
        if val_score < 30:
            reasons.append(f"low_valuation={val_score:.0f}")
    except (ValueError, TypeError):
        pass

    # 7. Guardrail A
    for flag in flags:
        if flag.startswith("guardrail_a_"):
            reasons.append(flag)
            break  # only first guardrail_a flag

    # 8. Survivability warning
    if any("survivability_warning" in f for f in flags):
        reasons.append("survivability_warning")

    # 9. Late stage distress
    if any("late_stage_distress" in f for f in flags):
        reasons.append("late_stage_distress")

    return reasons


def compute_baker_overlay(
    ranked_securities: List[Dict[str, Any]],
    holdings_snapshots: Dict[str, Any],
    catalyst_summaries: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute Baker Bros overlay: per-ticker signals, IC review queues, alignment metrics.

    Args:
        ranked_securities: Module 5 ranked_securities output (list of security dicts)
        holdings_snapshots: Tickers dict from holdings_detailed.json
            Format: {"TICKER": {"holdings": {"current": {CIK: {value_kusd: N}}, "prior": {...}}}}
            Note: value_kusd field stores raw USD despite the name.
        catalyst_summaries: Module 3 summaries dict (ticker -> summary with integration block)

    Returns:
        dict with securities, queue_1, queue_2, metrics, diagnostics
    """
    if not ranked_securities or not holdings_snapshots:
        return {
            "securities": [],
            "queue_1_core_divergence": [],
            "queue_2_near_term_catalyst": [],
            "metrics": {},
            "diagnostics": {"error": "no data"},
        }

    # =========================================================================
    # Step 1: Extract Baker positions from holdings data
    # =========================================================================
    baker_positions = {}
    baker_total_usd = 0

    for ticker, ticker_data in holdings_snapshots.items():
        if not isinstance(ticker_data, dict):
            continue
        holdings = ticker_data.get("holdings", {})
        current = holdings.get("current", {})
        prior = holdings.get("prior", {})

        baker_current = current.get(BAKER_CIK, {})
        baker_prior = prior.get(BAKER_CIK, {})

        # value_kusd field actually stores raw USD
        current_val = baker_current.get("value_kusd", 0) or 0
        prior_val = baker_prior.get("value_kusd", 0) or 0

        if current_val > 0 or prior_val > 0:
            activity, activity_pct = _classify_activity(current_val, prior_val)
            baker_positions[ticker.upper()] = {
                "current_usd": current_val,
                "prior_usd": prior_val,
                "activity": activity,
                "activity_pct": activity_pct,
            }
            if current_val > 0:
                baker_total_usd += current_val

    # Compute portfolio weights
    for pos in baker_positions.values():
        if baker_total_usd > 0:
            pos["portfolio_weight"] = pos["current_usd"] / baker_total_usd
        else:
            pos["portfolio_weight"] = 0.0

    logger.info(
        f"  Baker overlay: {len(baker_positions)} positions, "
        f"${baker_total_usd / 1_000_000_000:.1f}B total"
    )

    # =========================================================================
    # Step 2: Enrich ranked securities with per-ticker baker signals
    # =========================================================================
    security_baker_data = []

    for sec in ranked_securities:
        ticker = sec["ticker"]
        pos = baker_positions.get(ticker)

        if pos and pos["activity"] != "exited":
            baker_info = {
                "held": True,
                "portfolio_weight_pct": round(pos["portfolio_weight"] * 100, 4),
                "value_usd": pos["current_usd"],
                "activity": pos["activity"],
                "activity_pct": round(pos["activity_pct"], 4) if pos["activity_pct"] is not None else None,
            }
        else:
            baker_info = {
                "held": False,
                "portfolio_weight_pct": 0.0,
                "value_usd": 0,
                "activity": pos["activity"] if pos else None,
                "activity_pct": None,
            }

        security_baker_data.append(baker_info)

    # =========================================================================
    # Step 3: Compute baker_rank_by_weight and baker_rank_by_activity
    # =========================================================================
    held_indices = [
        i for i in range(len(security_baker_data))
        if security_baker_data[i]["held"]
    ]

    # Rank by weight (desc), tiebreak by ticker (asc)
    weight_sorted = sorted(
        held_indices,
        key=lambda i: (
            -security_baker_data[i]["portfolio_weight_pct"],
            ranked_securities[i]["ticker"],
        ),
    )
    for rank, idx in enumerate(weight_sorted, 1):
        security_baker_data[idx]["rank_by_weight"] = rank

    # Rank by activity priority, then weight, then ticker
    activity_order = {"new": 0, "add": 1, "unchanged": 2, "trim": 3, "exited": 4}
    activity_sorted = sorted(
        held_indices,
        key=lambda i: (
            activity_order.get(security_baker_data[i]["activity"], 5),
            -security_baker_data[i]["portfolio_weight_pct"],
            ranked_securities[i]["ticker"],
        ),
    )
    for rank, idx in enumerate(activity_sorted, 1):
        security_baker_data[idx]["rank_by_activity"] = rank

    # Non-held get None ranks
    for bd in security_baker_data:
        bd.setdefault("rank_by_weight", None)
        bd.setdefault("rank_by_activity", None)

    # =========================================================================
    # Step 4: Queue 1 — Baker Core Divergence (IC review)
    # =========================================================================
    queue_1 = []
    for i, sec in enumerate(ranked_securities):
        bd = security_baker_data[i]
        if not bd["held"]:
            continue

        composite_rank = sec.get("composite_rank", 0)
        weight_frac = bd["portfolio_weight_pct"] / 100.0
        value_usd = bd["value_usd"]
        activity = bd["activity"]

        reasons = []
        # Core conviction: weight >= 1% (no min-weight floor needed)
        if weight_frac >= Q1_WEIGHT_PCT and composite_rank > Q1_WEIGHT_RANK:
            reasons.append(f"weight>={Q1_WEIGHT_PCT*100:.0f}%_rank>{Q1_WEIGHT_RANK}")
        # Large position: value >= $250M with 0.25% weight floor
        if (
            value_usd >= Q1_VALUE_USD
            and weight_frac >= Q1_VALUE_MIN_WEIGHT
            and composite_rank > Q1_VALUE_RANK
        ):
            reasons.append(f"value>=$250M_rank>{Q1_VALUE_RANK}")
        # Active building: new/add with 0.25% weight floor
        if (
            activity in ("new", "add")
            and weight_frac >= Q1_ACTIVITY_MIN_WEIGHT
            and composite_rank > Q1_ACTIVITY_RANK
        ):
            reasons.append(f"activity={activity}_rank>{Q1_ACTIVITY_RANK}")

        if reasons:
            # Extract disagreement reasons from model output
            disagreement = _extract_disagreement_reasons(sec)

            queue_1.append({
                "ticker": sec["ticker"],
                "composite_rank": composite_rank,
                "baker_rank_by_weight": bd["rank_by_weight"],
                "baker_portfolio_weight_pct": bd["portfolio_weight_pct"],
                "baker_value_usd": value_usd,
                "baker_activity": activity,
                "queue_reasons": reasons,
                "disagreement_reasons": disagreement,
                "severity": sec.get("severity", "none"),
            })

    # Sort by baker weight desc
    queue_1.sort(key=lambda x: (-x["baker_portfolio_weight_pct"], x["ticker"]))

    # =========================================================================
    # Step 5: Queue 2 — Baker Near-Term Catalyst
    # =========================================================================
    queue_2 = []
    for i, sec in enumerate(ranked_securities):
        bd = security_baker_data[i]
        if not bd["held"]:
            continue

        composite_rank = sec.get("composite_rank", 0)
        if composite_rank <= Q2_RANK:
            continue

        if not _risk_gates_pass(sec):
            continue

        # Get days_to_catalyst from catalyst_decay (module 5 output)
        catalyst_decay = sec.get("catalyst_decay") or {}
        days_to_catalyst = catalyst_decay.get("days_to_catalyst")

        # Fallback: check module 3 summaries
        if days_to_catalyst is None and catalyst_summaries:
            ticker_summary = catalyst_summaries.get(sec["ticker"])
            if isinstance(ticker_summary, dict):
                integration = ticker_summary.get("integration", {})
                days_to_catalyst = integration.get("catalyst_window_days")

        if (
            days_to_catalyst is not None
            and Q2_CATALYST_MIN_DAYS <= days_to_catalyst <= Q2_CATALYST_MAX_DAYS
        ):
            queue_2.append({
                "ticker": sec["ticker"],
                "composite_rank": composite_rank,
                "baker_rank_by_weight": bd["rank_by_weight"],
                "baker_portfolio_weight_pct": bd["portfolio_weight_pct"],
                "baker_activity": bd["activity"],
                "days_to_catalyst": days_to_catalyst,
                "severity": sec.get("severity", "none"),
            })

    # Sort by days_to_catalyst asc (soonest first)
    queue_2.sort(key=lambda x: (x["days_to_catalyst"], x["ticker"]))

    # =========================================================================
    # Step 6: Alignment metrics
    # =========================================================================
    baker_held_count = sum(1 for bd in security_baker_data if bd["held"])

    # Baker top-N by weight (for Recall@K)
    baker_top_n_tickers = set()
    for rank_pos, idx in enumerate(weight_sorted):
        if rank_pos >= BAKER_TOP_N_FOR_RECALL:
            break
        baker_top_n_tickers.add(ranked_securities[idx]["ticker"])

    # Recall@K: fraction of Baker top-20 in composite top-K
    recall_metrics = {}
    for k in RECALL_K_VALUES:
        composite_top_k = set(
            sec["ticker"] for sec in ranked_securities
            if sec.get("composite_rank", 999) <= k
        )
        overlap = baker_top_n_tickers & composite_top_k
        recall = len(overlap) / max(1, len(baker_top_n_tickers))
        recall_metrics[f"recall_at_{k}"] = round(recall, 4)
        recall_metrics[f"recall_at_{k}_overlap"] = sorted(overlap)
        recall_metrics[f"recall_at_{k}_missing"] = sorted(baker_top_n_tickers - composite_top_k)

    # Baker divergence: mean |composite_rank - baker_rank_by_weight| / n_total
    divergence_sum = 0
    divergence_count = 0
    for i, bd in enumerate(security_baker_data):
        if bd["held"] and bd["rank_by_weight"] is not None:
            composite_rank = ranked_securities[i].get("composite_rank", 0)
            divergence_sum += abs(composite_rank - bd["rank_by_weight"])
            divergence_count += 1

    n_total = len(ranked_securities)
    avg_divergence = divergence_sum / max(1, divergence_count)
    baker_divergence_pct = round(avg_divergence / max(1, n_total) * 100, 2)

    metrics = {
        "baker_held_count": baker_held_count,
        "baker_total_value_usd": baker_total_usd,
        "baker_total_value_b": round(baker_total_usd / 1_000_000_000, 2),
        "baker_core_divergence_count": len(queue_1),
        "baker_near_term_catalyst_count": len(queue_2),
        "baker_divergence_pct": baker_divergence_pct,
        "baker_avg_rank_divergence": round(avg_divergence, 1),
        **recall_metrics,
    }

    # =========================================================================
    # Step 7: Optional additive bonus (computed, NOT applied to composite_score)
    # =========================================================================
    bonus_count = 0
    for i, sec in enumerate(ranked_securities):
        bd = security_baker_data[i]
        if not bd["held"]:
            bd["bonus"] = "0"
            continue

        if _guardrail_blocks_bonus(sec):
            bd["bonus"] = "0"
            bd["bonus_blocked"] = True
            continue

        # Bonus: 1% weight = 1pt, capped at 2pt
        weight_pct = bd["portfolio_weight_pct"]
        raw_bonus = min(Decimal(str(weight_pct)), Decimal("2"))
        bd["bonus"] = str(raw_bonus.quantize(Decimal("0.01")))
        if raw_bonus > Decimal("0"):
            bonus_count += 1

    metrics["baker_bonus_eligible_count"] = bonus_count

    # =========================================================================
    # Diagnostics
    # =========================================================================
    activity_counts = {}
    for pos in baker_positions.values():
        a = pos["activity"]
        activity_counts[a] = activity_counts.get(a, 0) + 1

    diagnostics = {
        "baker_cik": BAKER_CIK,
        "baker_positions_total": len(baker_positions),
        "baker_activity_counts": activity_counts,
        "queue_1_thresholds": {
            "weight_pct": Q1_WEIGHT_PCT * 100,
            "weight_rank": Q1_WEIGHT_RANK,
            "value_usd": Q1_VALUE_USD,
            "value_rank": Q1_VALUE_RANK,
            "value_min_weight_pct": Q1_VALUE_MIN_WEIGHT * 100,
            "activity_rank": Q1_ACTIVITY_RANK,
            "activity_min_weight_pct": Q1_ACTIVITY_MIN_WEIGHT * 100,
        },
        "queue_2_thresholds": {
            "catalyst_min_days": Q2_CATALYST_MIN_DAYS,
            "catalyst_max_days": Q2_CATALYST_MAX_DAYS,
            "rank": Q2_RANK,
        },
        "recall_top_n": BAKER_TOP_N_FOR_RECALL,
        "version": VERSION,
    }

    return {
        "securities": security_baker_data,
        "queue_1_core_divergence": queue_1,
        "queue_2_near_term_catalyst": queue_2,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
