"""Expression Attribution Engine (Spec 062, Phase 2).

Forward-only evidence accumulation for the options expression overlay.
Three responsibilities:

1. ``log_recommendation`` — append one JSONL record per recommendation
2. ``log_decision`` — append one JSONL record per evaluated name (all outcomes)
3. ``resolve_attributions`` — join with CRT outcomes + price data (separate pass)
4. ``evaluate_kill_switches`` — automatic disable on bad evidence

Design principle: attribution is evidence, not feedback. It observes,
records, and evaluates. It does NOT adjust thresholds, tune parameters,
or change behavior.

Hard constraints:
  - No backfilling
  - No recomputation of past records
  - No overwriting of resolved records
  - No dependency on future data
  - No optimization loops
  - Pure forward accumulation

Idempotency (issue #495): the pipeline may run several times per day.
Writes are keyed by (ticker, node_id): a re-run replaces its own pending
records for the same node instead of appending duplicates, and a resolved
record is never displaced by a pending one. Without this, kill-switch
metrics were computed over records duplicated once per re-run.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from event_ev.expression_layer import ExpressionRecommendation

logger = logging.getLogger(__name__)

# Default log paths (relative to project root)
DEFAULT_ATTRIBUTION_LOG = Path("data/expression_attribution_log.jsonl")
DEFAULT_DECISION_LOG = Path("data/expression_decision_log.jsonl")


def _record_key(record: Dict[str, Any]) -> Optional[tuple]:
    """Identity key for a log record; None when it cannot be keyed."""
    ticker = record.get("ticker")
    node_id = record.get("node_id")
    if ticker and node_id:
        return (str(ticker), str(node_id))
    return None


def dedup_keep_last(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate (ticker, node_id) records, keeping the last.

    A resolved record wins over a later pending one, so a stale re-append
    can never shadow a resolution. Keyless records pass through unchanged.
    """
    out: List[Dict[str, Any]] = []
    pos: Dict[tuple, int] = {}
    for rec in records:
        key = _record_key(rec) if isinstance(rec, dict) else None
        if key is None:
            out.append(rec)
            continue
        if key in pos:
            existing = out[pos[key]]
            if existing.get("attribution_status") == "resolved" and rec.get("attribution_status") != "resolved":
                continue
            out[pos[key]] = rec
        else:
            pos[key] = len(out)
            out.append(rec)
    return out


def append_records_idempotent(
    path: Path,
    new_records: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Append records, replacing same-key rows instead of duplicating them.

    Existing records whose (ticker, node_id) matches an incoming record are
    dropped and the incoming record is appended at the end — except that an
    existing *resolved* record is kept and the incoming pending duplicate is
    discarded. Keyless or malformed existing lines are preserved verbatim.
    The rewrite is atomic (temp file + rename).

    Returns counts: {"appended", "replaced", "kept_resolved"}.
    """
    path = Path(path)
    if not new_records:
        return {"appended": 0, "replaced": 0, "kept_resolved": 0}

    new_by_key: Dict[tuple, Dict[str, Any]] = {}
    keyless_new: List[Dict[str, Any]] = []
    for rec in new_records:
        key = _record_key(rec)
        if key is None:
            keyless_new.append(rec)
        else:
            new_by_key[key] = rec

    out_lines: List[str] = []
    replaced = 0
    kept_resolved = 0
    if path.exists():
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    out_lines.append(s)
                    continue
                key = _record_key(rec) if isinstance(rec, dict) else None
                if key is None or key not in new_by_key:
                    out_lines.append(s)
                elif (
                    rec.get("attribution_status") == "resolved"
                    and new_by_key[key].get("attribution_status") != "resolved"
                ):
                    kept_resolved += 1
                    del new_by_key[key]
                    out_lines.append(s)
                else:
                    replaced += 1

    appended = list(new_by_key.values()) + keyless_new
    for rec in appended:
        out_lines.append(json.dumps(rec, default=str))

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "w") as f:
        for s in out_lines:
            f.write(s + "\n")
    tmp.replace(path)

    return {"appended": len(appended), "replaced": replaced, "kept_resolved": kept_resolved}


# ============================================================================
# Logging — forward-only append (idempotent per node)
# ============================================================================


def _recommendation_to_attribution_record(
    rec: ExpressionRecommendation,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Flatten an ExpressionRecommendation into a JSONL-ready attribution record.

    Captures the economic snapshot at recommendation time. Resolution
    fields are null — filled in later by resolve_attributions().
    """
    return {
        # Metadata
        "timestamp": timestamp or datetime.utcnow().isoformat(timespec="seconds"),
        "ticker": rec.ticker,
        "node_id": rec.node_id,
        "as_of_date": rec.as_of_date,
        # Classification
        "mispricing_type": rec.mispricing_type,
        "mispricing_subtype": rec.mispricing_subtype,
        "overlay_class": rec.overlay_class,
        # Confidence
        "belief_strength": round(rec.belief_strength, 4),
        "permission_to_express": round(rec.permission_to_express, 4),
        "mispricing_confidence": round(rec.mispricing_confidence, 4),
        # Surface & execution
        "surface_quality_score": round(rec.surface_quality_score, 2),
        "execution_risk": rec.execution_risk,
        # Tradeability
        "is_tradeable": rec.is_tradeable,
        "gate_failures": list(rec.gate_failures),
        # Sizing
        "max_premium_pct_nav": round(rec.max_premium_pct_nav, 4),
        "sizing_basis": rec.sizing_basis,
        # Economic snapshot (locked at recommendation time)
        "priced_move_pct": round(rec.priced_move_pct, 4) if rec.priced_move_pct is not None else None,
        "scenario_ev": round(rec.scenario_ev, 4),
        "opt_atm_iv": round(rec.opt_atm_iv, 4) if rec.opt_atm_iv is not None else None,
        # Governance
        "model_version": rec.model_version,
        "governance_class": rec.governance_class,
        "policy_flags": list(rec.policy_flags),
        # Resolution (filled later — never at log time)
        "resolved_date": None,
        "realized_outcome": None,
        "realized_move_1d_pct": None,
        "realized_move_5d_pct": None,
        "realized_iv_post": None,
        "pnl_estimate": None,
        "attribution_status": "pending",
    }


def _recommendation_to_decision_record(
    rec: ExpressionRecommendation,
    timestamp: Optional[str] = None,
    kill_switch_active: bool = False,
    kill_switch_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Flatten into a decision log record (ALL evaluated names)."""
    if rec.is_tradeable and not kill_switch_active:
        decision = "tradeable"
    elif kill_switch_active:
        decision = "kill_switched"
    elif rec.gate_failures:
        decision = "rejected"
    else:
        decision = "rejected"

    return {
        "timestamp": timestamp or datetime.utcnow().isoformat(timespec="seconds"),
        "ticker": rec.ticker,
        "node_id": rec.node_id,
        "as_of_date": rec.as_of_date,
        # Decision
        "decision": decision,
        "mispricing_type": rec.mispricing_type,
        "overlay_class": rec.overlay_class,
        # Gate details
        "gate_failures": list(rec.gate_failures),
        "belief_strength": round(rec.belief_strength, 4),
        "permission_to_express": round(rec.permission_to_express, 4),
        "surface_quality_score": round(rec.surface_quality_score, 2),
        # Kill switch
        "kill_switch_active": kill_switch_active,
        "kill_switch_reason": kill_switch_reason,
    }


# Public record builders — batch callers (run_screen) construct records in
# their evaluation loop, then persist once via append_records_idempotent.
build_attribution_record = _recommendation_to_attribution_record
build_decision_record = _recommendation_to_decision_record


def log_recommendation(
    rec: ExpressionRecommendation,
    log_path: Optional[Path] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Log one attribution record for a tradeable recommendation.

    Idempotent per (ticker, node_id): re-logging the same node replaces the
    prior pending record instead of duplicating it; a resolved record is
    never displaced. Returns the record dict (for testing / caller
    inspection). Only logs tradeable recommendations — non-tradeable go to
    decision log only. Callers logging many records per run should build
    them with build_attribution_record and write once via
    append_records_idempotent instead.
    """
    path = log_path or DEFAULT_ATTRIBUTION_LOG
    record = _recommendation_to_attribution_record(rec, timestamp)
    append_records_idempotent(path, [record])
    logger.debug("Attribution logged: %s %s → %s", rec.ticker, rec.overlay_class, path)
    return record


def log_decision(
    rec: ExpressionRecommendation,
    log_path: Optional[Path] = None,
    timestamp: Optional[str] = None,
    kill_switch_active: bool = False,
    kill_switch_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Log one decision record for ANY evaluated name.

    Covers: tradeable, rejected, demoted, kill-switched. Idempotent per
    (ticker, node_id) — see log_recommendation. Returns the record dict.
    Callers logging many records per run should build them with
    build_decision_record and write once via append_records_idempotent.
    """
    path = log_path or DEFAULT_DECISION_LOG
    record = _recommendation_to_decision_record(rec, timestamp, kill_switch_active, kill_switch_reason)
    append_records_idempotent(path, [record])
    logger.debug("Decision logged: %s → %s", rec.ticker, record["decision"])
    return record


# ============================================================================
# Resolution — join with CRT outcomes
# ============================================================================


def _compute_pnl_estimate(
    overlay_class: str,
    priced_move_pct: Optional[float],
    realized_move_pct: Optional[float],
) -> Optional[float]:
    """Simple hypothetical P&L estimate.

    Directional: realized - priced (did the direction play out?)
    Variance: |realized| - priced (did the magnitude exceed implied?)
    Credit: priced - |realized| (did premium decay work?)
    Calendar/Manual: None (too structure-dependent)
    """
    if priced_move_pct is None or realized_move_pct is None:
        return None

    if overlay_class == "DIRECTIONAL_DEBIT":
        return round(realized_move_pct - priced_move_pct, 4)
    elif overlay_class == "VARIANCE_DEBIT":
        return round(abs(realized_move_pct) - priced_move_pct, 4)
    elif overlay_class == "DEFINED_RISK_CREDIT":
        return round(priced_move_pct - abs(realized_move_pct), 4)
    else:
        return None


def load_attribution_log(log_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load attribution records from JSONL, deduplicated keep-last per
    (ticker, node_id) so metrics are never computed over re-run duplicates
    left by pre-#495 appends."""
    path = log_path or DEFAULT_ATTRIBUTION_LOG
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed attribution line")
    return dedup_keep_last(records)


def load_decision_log(log_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load decision records from JSONL, deduplicated keep-last per
    (ticker, node_id)."""
    path = log_path or DEFAULT_DECISION_LOG
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed decision line")
    return dedup_keep_last(records)


def resolve_attributions(
    log_path: Optional[Path] = None,
    resolutions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Join pending attribution records with CRT resolution outcomes.

    Matches by (ticker, node_id). Fills in realized fields. Rewrites
    the JSONL file with updated records. Returns count of newly resolved.

    Resolution dict expected keys:
      - ticker, node_id (match keys)
      - outcome: HIT | MISS | MIXED
      - resolved_date: ISO date
      - price_t_minus_1, price_t_0, price_t_plus_5 (for move computation)
      - post_event_iv (optional)
    """
    if not resolutions:
        return 0

    path = log_path or DEFAULT_ATTRIBUTION_LOG
    records = load_attribution_log(path)
    if not records:
        return 0

    # Index resolutions by (ticker, node_id)
    res_index: Dict[tuple, Dict[str, Any]] = {}
    for r in resolutions:
        key = (r.get("ticker", ""), r.get("node_id", ""))
        if key[0] and key[1]:
            res_index[key] = r

    resolved_count = 0
    for rec in records:
        if rec.get("attribution_status") != "pending":
            continue

        key = (rec.get("ticker", ""), rec.get("node_id", ""))
        if key not in res_index:
            continue

        res = res_index[key]
        outcome = res.get("outcome")
        if outcome not in ("HIT", "MISS", "MIXED"):
            continue

        # Compute realized moves
        pt_minus_1 = res.get("price_t_minus_1")
        pt_0 = res.get("price_t_0")
        pt_5 = res.get("price_t_plus_5")

        move_1d = None
        if pt_minus_1 and pt_0 and pt_minus_1 > 0:
            move_1d = round((pt_0 - pt_minus_1) / pt_minus_1 * 100, 4)

        move_5d = None
        if pt_minus_1 and pt_5 and pt_minus_1 > 0:
            move_5d = round((pt_5 - pt_minus_1) / pt_minus_1 * 100, 4)

        # Update record
        rec["resolved_date"] = res.get("resolved_date")
        rec["realized_outcome"] = outcome
        rec["realized_move_1d_pct"] = move_1d
        rec["realized_move_5d_pct"] = move_5d
        rec["realized_iv_post"] = res.get("post_event_iv")
        rec["pnl_estimate"] = _compute_pnl_estimate(
            rec.get("overlay_class", ""),
            rec.get("priced_move_pct"),
            move_1d,
        )
        rec["attribution_status"] = "resolved"
        resolved_count += 1

    # Rewrite (atomic: write to temp, rename)
    if resolved_count > 0:
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        tmp.replace(path)
        logger.info("Resolved %d attribution records in %s", resolved_count, path)

    return resolved_count


# ============================================================================
# Kill switches — automatic disable on bad evidence
# ============================================================================


def compute_attribution_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute attribution metrics from resolved records.

    Returns aggregate + per-type + per-confidence-bucket metrics.
    Only uses resolved records (attribution_status == "resolved").
    """
    resolved = [r for r in records if r.get("attribution_status") == "resolved"]

    if not resolved:
        return {
            "n_resolved": 0,
            "sufficient": False,
            "aggregate": {},
            "by_type": {},
            "by_confidence": {},
        }

    # Aggregate
    pnls = [r["pnl_estimate"] for r in resolved if r.get("pnl_estimate") is not None]
    wins = [p for p in pnls if p > 0]

    aggregate = {
        "n": len(resolved),
        "n_with_pnl": len(pnls),
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else None,
        "mean_pnl": round(sum(pnls) / len(pnls), 4) if pnls else None,
        "total_pnl": round(sum(pnls), 4) if pnls else None,
    }

    # Sharpe (simple: mean / std)
    if len(pnls) >= 2:
        mean_p = sum(pnls) / len(pnls)
        var_p = sum((p - mean_p) ** 2 for p in pnls) / (len(pnls) - 1)
        std_p = var_p**0.5 if var_p > 0 else 0.0
        aggregate["sharpe"] = round(mean_p / std_p, 4) if std_p > 0 else None
    else:
        aggregate["sharpe"] = None

    # By mispricing type
    by_type: Dict[str, Dict[str, Any]] = {}
    for r in resolved:
        mt = r.get("mispricing_type", "UNKNOWN")
        if mt not in by_type:
            by_type[mt] = {"pnls": [], "n": 0}
        by_type[mt]["n"] += 1
        if r.get("pnl_estimate") is not None:
            by_type[mt]["pnls"].append(r["pnl_estimate"])

    for mt, data in by_type.items():
        pnl_list = data["pnls"]
        w = [p for p in pnl_list if p > 0]
        by_type[mt] = {
            "n": data["n"],
            "n_with_pnl": len(pnl_list),
            "win_rate": round(len(w) / len(pnl_list), 4) if pnl_list else None,
            "mean_pnl": round(sum(pnl_list) / len(pnl_list), 4) if pnl_list else None,
        }

    # By confidence bucket
    buckets = {"low": (0.0, 0.50), "medium": (0.50, 0.70), "high": (0.70, 1.01)}
    by_confidence: Dict[str, Dict[str, Any]] = {}
    for label, (lo, hi) in buckets.items():
        bucket_recs = [
            r
            for r in resolved
            if lo <= (r.get("mispricing_confidence") or 0) < hi and r.get("pnl_estimate") is not None
        ]
        pnl_list = [r["pnl_estimate"] for r in bucket_recs]
        w = [p for p in pnl_list if p > 0]
        by_confidence[label] = {
            "n": len(bucket_recs),
            "win_rate": round(len(w) / len(pnl_list), 4) if pnl_list else None,
            "mean_pnl": round(sum(pnl_list) / len(pnl_list), 4) if pnl_list else None,
        }

    return {
        "n_resolved": len(resolved),
        "sufficient": len(resolved) >= 20,
        "aggregate": aggregate,
        "by_type": by_type,
        "by_confidence": by_confidence,
    }


def evaluate_kill_switches(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate automatic kill switches based on attribution evidence.

    Returns kill switch state. Only fires with >= 20 resolved records.

    Rules (from spec):
    1. Aggregate win rate < 40% → disable all tradeable
    2. Any type win rate < 30% → disable that type
    3. Confidence monotonicity violated → disable sizing
    4. Sharpe < -0.50 → disable entire overlay
    """
    metrics = compute_attribution_metrics(records)

    result: Dict[str, Any] = {
        "overlay_enabled": True,
        "sizing_enabled": True,
        "disabled_types": [],
        "triggered_rules": [],
        "metrics_summary": metrics,
    }

    if not metrics["sufficient"]:
        result["evaluation_status"] = "insufficient_data"
        return result

    result["evaluation_status"] = "evaluated"
    agg = metrics["aggregate"]

    # Rule 1: aggregate win rate < 40%
    if agg.get("win_rate") is not None and agg["win_rate"] < 0.40:
        result["overlay_enabled"] = False
        result["triggered_rules"].append(f"aggregate_win_rate={agg['win_rate']:.2%} < 40%")

    # Rule 2: per-type win rate < 30%
    for mt, data in metrics["by_type"].items():
        if data.get("n_with_pnl", 0) >= 5 and data.get("win_rate") is not None:
            if data["win_rate"] < 0.30:
                result["disabled_types"].append(mt)
                result["triggered_rules"].append(f"type_{mt}_win_rate={data['win_rate']:.2%} < 30%")

    # Rule 3: confidence monotonicity
    by_conf = metrics["by_confidence"]
    conf_pnls = []
    for label in ("low", "medium", "high"):
        mp = by_conf.get(label, {}).get("mean_pnl")
        if mp is not None and by_conf.get(label, {}).get("n", 0) >= 3:
            conf_pnls.append((label, mp))
    if len(conf_pnls) >= 2:
        # Check if highest confidence bucket performs worst
        if conf_pnls[-1][1] < conf_pnls[0][1]:
            result["sizing_enabled"] = False
            result["triggered_rules"].append(
                f"confidence_monotonicity_violated: "
                f"{conf_pnls[-1][0]}={conf_pnls[-1][1]:.4f} < "
                f"{conf_pnls[0][0]}={conf_pnls[0][1]:.4f}"
            )

    # Rule 4: Sharpe < -0.50
    if agg.get("sharpe") is not None and agg["sharpe"] < -0.50:
        result["overlay_enabled"] = False
        result["triggered_rules"].append(f"sharpe={agg['sharpe']:.4f} < -0.50")

    return result
