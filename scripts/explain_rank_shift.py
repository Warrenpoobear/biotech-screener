#!/usr/bin/env python3
"""Explain why a ticker's rank shifted between two consecutive snapshot dates.

Given a ticker, two dates, and a ruleset, re-ranks both snapshots and produces
a deterministic JSON report with rank delta, module-level attribution, and
key feature differences.

Usage:
    python3 scripts/explain_rank_shift.py \
        --ticker ACAD \
        --current-date 2026-02-11 \
        --prior-date 2026-02-10 \
        --snapshot-dir data/snapshots \
        --ruleset production_data/decision_rulesets/v1.6.1_alpha_modifier_within_tier.json \
        --out-json /tmp/explain_2026-02-11_ACAD.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_engine import DecisionRuleset
from scripts.eval_ruleset import _read_rankings, rerank_rows

VERSION = "1.0.0"
SCHEMA = "explain_rank_shift.v1"

# Curated feature fields for the explain output (category -> CSV columns).
_EXPLAIN_CATEGORIES: Dict[str, List[str]] = {
    "eligibility": [
        "eligible", "tier_dev", "tier_any", "tier_commercial",
        "archetype", "missing_components", "missingness_penalty",
    ],
    "catalyst": [
        "catalyst_days", "catalyst_mode", "catalyst_in_window",
        "cat_priority", "catalyst_strength", "catalyst_event_type",
    ],
    "alpha": [
        "alpha_cohort_raw", "alpha_cohort_pct",
        "clinical_score_z", "clinical_score_z_tier", "clinical_alpha_z",
    ],
    "coinvest": [
        "coinvest_score_z", "inst_delta_z",
        "sponsor_tier1_count", "coinvest_tag",
    ],
    "momentum": ["momentum_score", "mom_state"],
    "composite": [
        "composite_score", "composite_rank", "score_rank_pct", "score_z",
    ],
    "price": [
        "de_alpha_60d", "de_drawdown", "de_beta_xbi_60d",
        "de_rsi_14d", "de_drawdown_rel_xbi",
    ],
    "module_scores": [
        "clinical_score", "financial_score", "catalyst_score",
        "smart_money_score", "valuation_score",
    ],
}

_MODULE_COLS = [
    ("clinical", "clinical_score"),
    ("financial", "financial_score"),
    ("catalyst", "catalyst_score"),
    ("smart_money", "smart_money_score"),
    ("valuation", "valuation_score"),
    ("momentum", "momentum_score"),
]


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _feature_snapshot(row: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Extract curated feature values from a CSV row."""
    result: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(_EXPLAIN_CATEGORIES):
        values: Dict[str, Any] = {}
        for f in _EXPLAIN_CATEGORIES[cat]:
            v = row.get(f)
            if v is not None and v != "":
                values[f] = v
        if values:
            result[cat] = values
    return result


def _feature_deltas(
    current: Dict[str, Dict[str, Any]],
    prior: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Compute deltas between two feature snapshots (only changed fields)."""
    all_cats = sorted(set(current) | set(prior))
    deltas: Dict[str, Dict[str, Any]] = {}
    for cat in all_cats:
        cur = current.get(cat, {})
        pri = prior.get(cat, {})
        all_keys = sorted(set(cur) | set(pri))
        cat_deltas: Dict[str, Any] = {}
        for k in all_keys:
            cv, pv = cur.get(k), pri.get(k)
            if cv == pv:
                continue
            entry: Dict[str, Any] = {"current": cv, "prior": pv}
            cn, pn = _safe_float(cv), _safe_float(pv)
            if cn is not None and pn is not None:
                entry["delta"] = round(cn - pn, 6)
            cat_deltas[k] = entry
        if cat_deltas:
            deltas[cat] = cat_deltas
    return deltas


def _module_attribution(
    current_row: Dict[str, str],
    prior_row: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Compute module-level score attribution (delta per module)."""
    result = []
    for name, col in _MODULE_COLS:
        cur = _safe_float(current_row.get(col))
        pri = _safe_float(prior_row.get(col))
        entry: Dict[str, Any] = {"module": name}
        if cur is not None:
            entry["current"] = round(cur, 4)
        if pri is not None:
            entry["prior"] = round(pri, 4)
        if cur is not None and pri is not None:
            entry["delta"] = round(cur - pri, 4)
        result.append(entry)
    return result


def explain_ticker_shift(
    ticker: str,
    current_date: str,
    prior_date: str,
    snapshot_dir: Path,
    ruleset: DecisionRuleset,
) -> Dict[str, Any]:
    """Explain what changed for a ticker between two dates under a ruleset.

    Re-ranks both snapshots with the given ruleset and produces a
    deterministic report with rank delta, module attribution, and
    key feature differences.
    """
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "ticker": ticker,
        "current_date": current_date,
        "prior_date": prior_date,
        "ruleset_id": ruleset.ruleset_id,
    }
    notes: List[str] = []

    # Load snapshots
    cur_path = snapshot_dir / current_date / "rankings.csv"
    pri_path = snapshot_dir / prior_date / "rankings.csv"

    if not cur_path.exists():
        result["error"] = f"Missing rankings: {cur_path}"
        return result
    if not pri_path.exists():
        result["error"] = f"Missing rankings: {pri_path}"
        return result

    cur_rows = _read_rankings(cur_path)
    pri_rows = _read_rankings(pri_path)

    # Rerank both with candidate ruleset
    _, cur_ranks = rerank_rows(cur_rows, ruleset)
    _, pri_ranks = rerank_rows(pri_rows, ruleset)

    cur_rank = cur_ranks.get(ticker)
    pri_rank = pri_ranks.get(ticker)

    result["current_rank"] = cur_rank
    result["prior_rank"] = pri_rank
    if cur_rank is not None and pri_rank is not None:
        result["delta_rank"] = cur_rank - pri_rank
    elif cur_rank is None and pri_rank is not None:
        notes.append("ticker not eligible on current date")
    elif cur_rank is not None and pri_rank is None:
        notes.append("ticker not eligible on prior date")
    else:
        notes.append("ticker not eligible on either date")

    # Row lookup
    cur_by_ticker = {r.get("ticker", ""): r for r in cur_rows}
    pri_by_ticker = {r.get("ticker", ""): r for r in pri_rows}
    cur_row = cur_by_ticker.get(ticker, {})
    pri_row = pri_by_ticker.get(ticker, {})

    if not cur_row:
        notes.append("ticker not found in current snapshot")
    if not pri_row:
        notes.append("ticker not found in prior snapshot")

    # Module attribution
    result["module_attribution"] = _module_attribution(cur_row, pri_row)

    # Feature deltas
    cur_features = _feature_snapshot(cur_row)
    pri_features = _feature_snapshot(pri_row)
    result["feature_deltas"] = _feature_deltas(cur_features, pri_features)

    result["notes"] = notes
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Explain a rank shift for a ticker between two dates"
    )
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--current-date", required=True)
    parser.add_argument("--prior-date", required=True)
    parser.add_argument("--snapshot-dir", type=Path, default=Path("data/snapshots"))
    parser.add_argument("--ruleset", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    ruleset = DecisionRuleset.from_json(str(args.ruleset))
    result = explain_ticker_shift(
        ticker=args.ticker,
        current_date=args.current_date,
        prior_date=args.prior_date,
        snapshot_dir=args.snapshot_dir,
        ruleset=ruleset,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
