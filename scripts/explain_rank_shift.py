#!/usr/bin/env python3
"""
Explain a single-ticker rank shift between baseline vs candidate.

Core explain logic (steps 2-5):
  2) Rank + score deltas
  3) Module attribution (delta per component)
  4) Key feature deltas (bounded, stable subset)
  5) Deterministic outputs (stable ordering, sorted JSON)

Step (1) — loading baseline/candidate rows + rank maps — is wired via
scripts/eval_ruleset.py (normal + rerank-only modes).

Usage:
  python3 scripts/explain_rank_shift.py \
    --date 2026-02-24 \
    --ticker IBRX \
    --snapshot-dir data/snapshots \
    --candidate-id 0c1129f6 \
    --baseline-id 88d7ae9a \
    --out-json /tmp/explain.json \
    --out-md /tmp/explain.md \
    --rerank-only \
    --candidate-ruleset production_data/decision_rulesets/v1.6.1_alpha_modifier_within_tier.json \
    --baseline-ruleset production_data/decision_rulesets/v1.5.1_coinvest_off.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_engine import DecisionRuleset
from scripts.eval_ruleset import (
    _read_rankings,
    rank_map as _existing_rank_map,
    rerank_rows,
)

VERSION = "1.0.0"
SCHEMA = "explain_rank_shift.v1"


# -----------------------------
# Small utilities
# -----------------------------

def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _as_float(x: Any) -> Optional[float]:
    if x is None or _is_nan(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _norm_key_candidates(key: str) -> List[str]:
    """Allow for mild schema drift (dot vs underscore)."""
    ks = [key]
    if "." in key:
        ks.append(key.replace(".", "_"))
    if "_" in key:
        ks.append(key.replace("_", "."))
    return list(dict.fromkeys(ks))


def _get(row: Dict[str, Any], key: str) -> Any:
    for k in _norm_key_candidates(key):
        if k in row:
            return row.get(k)
    return None


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)


# -----------------------------
# What we consider "modules" and "features"
# -----------------------------

MODULE_KEY_CANDIDATES: Dict[str, List[str]] = {
    "financial": ["financial_score_z", "financial_score", "financial_z", "score_financial"],
    "clinical": ["clinical_score_z", "clinical_score", "clinical_z", "score_clinical"],
    "catalyst": ["catalyst_score_z", "catalyst_score", "catalyst_z", "score_catalyst"],
    "alpha": ["alpha_raw", "alpha_score", "alpha_signal", "alpha_cohort_raw"],
    "momentum": ["momentum_score", "momentum_score_z", "momentum_z"],
    "coinvest": ["coinvest_score_z", "coinvest_adj", "coinvest_score"],
    "institutional": ["inst_delta_z", "inst_adj", "inst_delta"],
    "penalty": ["missingness_penalty", "penalty_total", "penalty"],
    "composite": ["pre_penalty_score", "composite_score", "score", "final_score"],
}

FEATURE_GROUPS: List[Tuple[str, List[str]]] = [
    ("eligibility", [
        "eligible",
        "archetype",
        "missing_count",
        "missing_components",
        "missingness_penalty",
    ]),
    ("tiers", [
        "tier_dev",
        "tier_commercial",
        "tier_any",
    ]),
    ("catalyst", [
        "catalyst_mode",
        "catalyst_days",
        "catalyst_priority",
        "cat_priority",
        "catalyst_strength",
        "catalyst_event_type",
    ]),
    ("alpha", [
        "alpha_raw",
        "alpha_cohort_pct",
        "alpha_cohort_raw",
        "clinical_score_z",
        "clinical_score_z_tier",
        "clinical_alpha_z",
    ]),
    ("coinvest", [
        "coinvest_score_z",
        "coinvest_adj",
        "inst_delta_z",
        "inst_adj",
    ]),
    ("momentum", [
        "momentum_score",
        "momentum_score_z",
        "mom_state",
    ]),
    ("price_regime", [
        "de_alpha_60d",
        "de_drawdown",
        "de_beta_xbi_60d",
        "de_rsi_14d",
        "de_drawdown_rel_xbi",
    ]),
]


# -----------------------------
# Core explain logic (steps 2-5)
# -----------------------------

@dataclass(frozen=True)
class ExplainResult:
    payload: Dict[str, Any]
    markdown: str


def _module_attribution(
    baseline_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Per-module {key_used, baseline, candidate, delta}.

    Only includes modules where at least one side has a numeric value.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for module_name, keys in MODULE_KEY_CANDIDATES.items():
        key_used: Optional[str] = None
        b_val: Optional[float] = None
        c_val: Optional[float] = None

        for k in keys:
            b_raw = _get(baseline_row, k)
            c_raw = _get(candidate_row, k)
            b = _as_float(b_raw)
            c = _as_float(c_raw)
            if b is not None or c is not None:
                key_used = k
                b_val, c_val = b, c
                break

        if key_used is None:
            continue

        delta = None
        if b_val is not None and c_val is not None:
            delta = c_val - b_val

        out[module_name] = {
            "key": key_used,
            "baseline": b_val,
            "candidate": c_val,
            "delta": delta,
        }

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _feature_deltas(
    baseline_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{group_name: {field: {baseline, candidate}}} for changed fields only."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for group_name, keys in FEATURE_GROUPS:
        group_changes: Dict[str, Dict[str, Any]] = {}
        for k in keys:
            b = _get(baseline_row, k)
            c = _get(candidate_row, k)
            if _is_nan(b):
                b = None
            if _is_nan(c):
                c = None
            if b == c:
                continue
            if b is None and c is None:
                continue
            group_changes[k] = {"baseline": b, "candidate": c}

        if group_changes:
            out[group_name] = group_changes

    return out


def explain_rank_shift(
    *,
    as_of_date: str,
    ticker: str,
    baseline_rank: Optional[int],
    candidate_rank: Optional[int],
    baseline_row: Dict[str, Any],
    candidate_row: Dict[str, Any],
    eval_mode: str,
    baseline_id: str,
    candidate_id: str,
) -> ExplainResult:
    """Compute the explain payload + short markdown."""

    delta_rank = None
    if baseline_rank is not None and candidate_rank is not None:
        delta_rank = candidate_rank - baseline_rank

    modules = _module_attribution(baseline_row, candidate_row)
    features = _feature_deltas(baseline_row, candidate_row)

    notes: List[str] = []
    if delta_rank is not None and abs(delta_rank) >= 50:
        notes.append("large_rank_shift")

    for key in ("tier_any", "tier_dev", "eligible", "missing_count",
                "catalyst_days", "alpha_raw"):
        b = _get(baseline_row, key)
        c = _get(candidate_row, key)
        if _is_nan(b):
            b = None
        if _is_nan(c):
            c = None
        if b != c and not (b is None and c is None):
            notes.append(f"field_changed:{key}")

    notes = sorted(set(notes))

    fingerprint_input = {
        "as_of_date": as_of_date,
        "ticker": ticker,
        "baseline_rank": baseline_rank,
        "candidate_rank": candidate_rank,
        "delta_rank": delta_rank,
        "modules": modules,
        "features": features,
        "notes": notes,
    }
    fingerprint = hashlib.sha256(
        _stable_json(fingerprint_input).encode("utf-8")
    ).hexdigest()[:16]

    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "as_of_date": as_of_date,
        "ticker": ticker,
        "eval_mode": eval_mode,
        "baseline": {
            "ruleset_id": baseline_id,
            "rank": baseline_rank,
        },
        "candidate": {
            "ruleset_id": candidate_id,
            "rank": candidate_rank,
        },
        "delta": {
            "rank": delta_rank,
            "modules": modules,
            "features": features,
        },
        "notes": notes,
        "fingerprint": fingerprint,
    }

    # Short markdown
    md_lines: List[str] = []
    md_lines.append(f"# Rank shift explain: {ticker} — {as_of_date}")
    md_lines.append("")
    md_lines.append(f"- eval_mode: `{eval_mode}`")
    md_lines.append(f"- baseline: `{baseline_id}` rank={baseline_rank}")
    md_lines.append(f"- candidate: `{candidate_id}` rank={candidate_rank}")
    md_lines.append(f"- delta_rank: `{delta_rank}`")
    if notes:
        md_lines.append(f"- notes: `{', '.join(notes)}`")
    md_lines.append("")

    rows = []
    for m, d in modules.items():
        dv = d.get("delta")
        if isinstance(dv, (int, float)):
            rows.append((m, float(dv), d.get("key")))
    rows.sort(key=lambda x: (-abs(x[1]), x[0]))

    if rows:
        md_lines.append("## Module deltas (candidate - baseline)")
        md_lines.append("")
        md_lines.append("| module | delta | key |")
        md_lines.append("|---|---:|---|")
        for m, dv, k in rows[:8]:
            md_lines.append(f"| `{m}` | `{dv:.4f}` | `{k}` |")
        md_lines.append("")

    if features:
        md_lines.append("## Key feature changes")
        md_lines.append("")
        for group, changes in features.items():
            md_lines.append(f"### {group}")
            md_lines.append("")
            md_lines.append("| field | baseline | candidate |")
            md_lines.append("|---|---|---|")
            for field, vals in changes.items():
                b = vals.get("baseline")
                c = vals.get("candidate")
                md_lines.append(f"| `{field}` | `{b}` | `{c}` |")
            md_lines.append("")

    return ExplainResult(
        payload=payload,
        markdown="\n".join(md_lines).rstrip() + "\n",
    )


# -----------------------------
# Integration seam (Step 1)
# -----------------------------

def _load_rows_and_ranks(
    as_of_date: str,
    ticker: str,
    snapshot_dir: Path,
    candidate_id: str,
    baseline_id: str,
    rerank_only: bool,
    candidate_ruleset: Optional[DecisionRuleset] = None,
    baseline_ruleset: Optional[DecisionRuleset] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int], Optional[int]]:
    """Load rows and ranks for a ticker on a single date.

    For cross-ruleset comparison on the same date, baseline_row and
    candidate_row are the same CSV row; ranks differ because rulesets
    sort differently.

    Returns (baseline_row, candidate_row, baseline_rank, candidate_rank).
    """
    snap_path = snapshot_dir / as_of_date / "rankings.csv"
    if not snap_path.exists():
        return {}, {}, None, None

    rows = _read_rankings(snap_path)
    row_by_ticker = {r.get("ticker", ""): r for r in rows}
    row = row_by_ticker.get(ticker, {})
    if not row:
        return {}, {}, None, None

    if rerank_only and candidate_ruleset and baseline_ruleset:
        _, cand_ranks = rerank_rows(rows, candidate_ruleset)
        _, base_ranks = rerank_rows(rows, baseline_ruleset)
    elif candidate_ruleset:
        _, cand_ranks = rerank_rows(rows, candidate_ruleset)
        base_ranks = _existing_rank_map(rows)
    else:
        cand_ranks = _existing_rank_map(rows)
        base_ranks = _existing_rank_map(rows)

    return row, row, base_ranks.get(ticker), cand_ranks.get(ticker)


# -----------------------------
# CLI
# -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Explain a single-ticker rank shift between rulesets"
    )
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--snapshot-dir", default="data/snapshots")
    ap.add_argument("--candidate-id", required=True)
    ap.add_argument("--baseline-id", required=True)
    ap.add_argument("--candidate-ruleset", default=None,
                    help="Path to candidate ruleset JSON (for rerank)")
    ap.add_argument("--baseline-ruleset", default=None,
                    help="Path to baseline ruleset JSON (for rerank)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--rerank-only", action="store_true")
    args = ap.parse_args(argv)

    cand_rs = None
    base_rs = None
    if args.candidate_ruleset:
        cand_rs = DecisionRuleset.from_json(args.candidate_ruleset)
    if args.baseline_ruleset:
        base_rs = DecisionRuleset.from_json(args.baseline_ruleset)

    baseline_row, candidate_row, baseline_rank, candidate_rank = (
        _load_rows_and_ranks(
            as_of_date=args.date,
            ticker=args.ticker,
            snapshot_dir=Path(args.snapshot_dir),
            candidate_id=args.candidate_id,
            baseline_id=args.baseline_id,
            rerank_only=args.rerank_only,
            candidate_ruleset=cand_rs,
            baseline_ruleset=base_rs,
        )
    )

    eval_mode = "rerank_only" if args.rerank_only else "normal"
    res = explain_rank_shift(
        as_of_date=args.date,
        ticker=args.ticker,
        baseline_rank=baseline_rank,
        candidate_rank=candidate_rank,
        baseline_row=baseline_row,
        candidate_row=candidate_row,
        eval_mode=eval_mode,
        baseline_id=args.baseline_id,
        candidate_id=args.candidate_id,
    )

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(_stable_json(res.payload) + "\n", encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(res.markdown, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
