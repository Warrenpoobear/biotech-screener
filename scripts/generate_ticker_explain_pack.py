"""Ticker Explain Pack — per-ticker evidence dossiers for drift movers.

Generates a deterministic mini-dossier explaining *why* each ticker was
dropped / added / degraded / improved between two snapshots.  Works with
a single snapshot (dossier only, no deltas) or a pair (dossier + delta).

Outputs:
    ticker_explain_pack.json   — structured data
    ticker_explain_pack.md     — human-readable Markdown
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_phase2_snapshot_delta import (
    SnapshotData,
    _safe_float,
    _safe_int,
    load_snapshot,
)
from run_drift_report import (
    _GATE_KEYS,
    _parse_pipe_separated,
    compute_attribution,
)
from decision_engine_codes import build_explanation, describe_code

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_AUTO_TICKERS = 50
DEFAULT_A_FLOOR = 0.60

# String columns: emit {prior, current} when changed
_STRING_DELTA_COLS = (
    "tier_dev",
    "size_band",
    "eligible",
    "catalyst_strength",
    "mom_state",
)

# Numeric columns: emit {prior, current, change} when changed
_NUMERIC_DELTA_COLS = (
    "actionable_rank",
    "target_weight_pct",
    "composite_score",
    "composite_rank",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get(row: pd.Series, col: str, default: str = "") -> str:
    """Safe column access from a DataFrame row."""
    if col not in row.index:
        return default
    val = row[col]
    if pd.isna(val):
        return default
    return str(val).strip()


def _build_eligibility_map(dev_df: pd.DataFrame) -> Dict[str, bool]:
    """Return {ticker: is_eligible} for dev-stage rows."""
    result: Dict[str, bool] = {}
    for _, row in dev_df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        elig = str(row.get("eligible", "0")).strip()
        if ticker:
            result[ticker] = elig == "1"
    return result


# ---------------------------------------------------------------------------
# Auto-selection
# ---------------------------------------------------------------------------
def auto_select_tickers(
    current: SnapshotData,
    prior: SnapshotData,
) -> List[str]:
    """Select tickers from attribution movers: churn + strength + eligibility.

    Returns alphabetically sorted, deduplicated list capped at MAX_AUTO_TICKERS.
    """
    attribution = compute_attribution([prior, current])
    if attribution is None:
        return []

    selected: set[str] = set()

    # 1. Churn — all dropped + added tickers
    churn = attribution.get("portfolio_churn", {})
    for entry in churn.get("dropped", []):
        selected.add(entry["ticker"])
    for entry in churn.get("added", []):
        selected.add(entry["ticker"])

    # 2. Strength transitions — degraded + improved
    strength = attribution.get("catalyst_strength", {})
    for entry in strength.get("degraded", []):
        selected.add(entry["ticker"])
    for entry in strength.get("improved", []):
        selected.add(entry["ticker"])

    # 3. Eligibility flippers — dev tickers present in both where status changed
    prior_dev = prior.rankings[prior.rankings["archetype"] == "drug_developer"]
    current_dev = current.rankings[current.rankings["archetype"] == "drug_developer"]
    prior_elig = _build_eligibility_map(prior_dev)
    current_elig = _build_eligibility_map(current_dev)
    common = set(prior_elig) & set(current_elig)
    for ticker in common:
        if prior_elig[ticker] != current_elig[ticker]:
            selected.add(ticker)

    result = sorted(selected)
    return result[:MAX_AUTO_TICKERS]


# ---------------------------------------------------------------------------
# Per-ticker dossier
# ---------------------------------------------------------------------------
def extract_dossier(
    ticker: str,
    rankings: pd.DataFrame,
    a_floor: float = DEFAULT_A_FLOOR,
) -> Dict[str, Any]:
    """Extract a structured dossier for one ticker from a rankings DataFrame."""
    dev = rankings[rankings["ticker"] == ticker]
    if dev.empty:
        return {"ticker": ticker, "found": False}

    row = dev.iloc[0]

    # Gates breakdown
    inelig_reasons = _parse_pipe_separated(row.get("ineligible_reasons", ""))
    gates: Dict[str, str] = {}
    for g in _GATE_KEYS:
        gates[g] = "fail" if g in inelig_reasons else "pass"

    # Optionality vs a_floor
    opt_val = _safe_float(row.get("clinical_optionality_pct_dev"), default=None)
    clears_a_floor: Optional[bool] = None
    if opt_val is not None:
        clears_a_floor = opt_val >= a_floor

    # Risk flags
    risk_flags = _parse_pipe_separated(row.get("risk_flags", ""))

    result = {
        "ticker": ticker,
        "found": True,
        "archetype": _get(row, "archetype"),
        "composite_rank": _safe_int(row.get("composite_rank"), default=None),
        "composite_score": _safe_float(row.get("composite_score"), default=None),
        "tier_dev": _get(row, "tier_dev"),
        "tier_reason": _get(row, "tier_reason"),
        "size_band": _get(row, "size_band"),
        "size_reasons": _get(row, "size_reasons"),
        "eligible": _get(row, "eligible") == "1",
        "gates": gates,
        "de_drawdown": _safe_float(row.get("de_drawdown"), default=None),
        "de_drawdown_missing_reason": _get(row, "de_drawdown_missing_reason"),
        "catalyst_strength": _get(row, "catalyst_strength"),
        "catalyst_days": _safe_int(row.get("catalyst_days"), default=None),
        "catalyst_mode": _get(row, "catalyst_mode"),
        "catalyst_in_window": _get(row, "catalyst_in_window"),
        "clinical_optionality_pct_dev": opt_val,
        "clears_a_floor": clears_a_floor,
        "a_floor": a_floor,
        "mom_state": _get(row, "mom_state"),
        "risk_flags": risk_flags,
        "sponsor_tier1_count": _safe_int(row.get("sponsor_tier1_count"), default=None),
        "sponsor_net_buying": _safe_float(row.get("sponsor_net_buying"), default=None),
        "actionable_rank": _safe_int(row.get("actionable_rank"), default=None),
        "target_weight_pct": _safe_float(row.get("target_weight_pct"), default=None),
    }

    # Structured explanation from canonical registry
    explanation_fields = {
        "eligible": _get(row, "eligible"),
        "ineligible_reasons": _get(row, "ineligible_reasons"),
        "tier_dev": _get(row, "tier_dev"),
        "tier_reason": _get(row, "tier_reason"),
        "size_band": _get(row, "size_band"),
        "size_reasons": _get(row, "size_reasons"),
        "risk_flags": "|".join(risk_flags) if risk_flags else "",
        "catalyst_mode": _get(row, "catalyst_mode"),
        "catalyst_strength": _get(row, "catalyst_strength"),
        "catalyst_days": _get(row, "catalyst_days"),
        "mom_state": _get(row, "mom_state"),
        "runway_bucket": _get(row, "runway_bucket"),
    }
    result["explanation"] = build_explanation(explanation_fields)

    return result


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------
def compute_ticker_delta(
    ticker: str,
    current_rankings: pd.DataFrame,
    prior_rankings: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    """Compare same ticker across two snapshots.

    Returns None if ticker absent in both.
    """
    cur_rows = current_rankings[current_rankings["ticker"] == ticker]
    pri_rows = prior_rankings[prior_rankings["ticker"] == ticker]

    in_current = not cur_rows.empty
    in_prior = not pri_rows.empty

    if not in_current and not in_prior:
        return None

    delta: Dict[str, Any] = {
        "in_current": in_current,
        "in_prior": in_prior,
    }

    if not in_current or not in_prior:
        return delta

    cur = cur_rows.iloc[0]
    pri = pri_rows.iloc[0]

    # String deltas — include only when changed
    for col in _STRING_DELTA_COLS:
        if col not in cur.index or col not in pri.index:
            continue
        c_val = str(cur.get(col, "")).strip()
        p_val = str(pri.get(col, "")).strip()
        if c_val != p_val:
            delta[col] = {"prior": p_val, "current": c_val}

    # Numeric deltas — include only when changed
    for col in _NUMERIC_DELTA_COLS:
        if col not in cur.index or col not in pri.index:
            continue
        c_val = _safe_float(cur.get(col), default=None)
        p_val = _safe_float(pri.get(col), default=None)
        if c_val is None and p_val is None:
            continue
        if c_val != p_val:
            change = None
            if c_val is not None and p_val is not None:
                change = round(c_val - p_val, 4)
            delta[col] = {"prior": p_val, "current": c_val, "change": change}

    return delta


# ---------------------------------------------------------------------------
# Pack builder (orchestrator)
# ---------------------------------------------------------------------------
def build_explain_pack(
    current: SnapshotData,
    prior: Optional[SnapshotData],
    tickers: List[str],
    a_floor: float = DEFAULT_A_FLOOR,
) -> Dict[str, Any]:
    """Build the full explain pack for a list of tickers."""
    explains: List[Dict[str, Any]] = []

    for ticker in tickers:
        dossier = extract_dossier(ticker, current.rankings, a_floor)

        if prior is not None:
            delta = compute_ticker_delta(
                ticker, current.rankings, prior.rankings
            )
            if delta is not None:
                dossier["delta"] = delta

        explains.append(dossier)

    pack: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "snapshot_date": current.date,
        "prev_snapshot_date": prior.date if prior else None,
        "a_floor": a_floor,
        "tickers_selected_by": "auto" if not hasattr(build_explain_pack, "_explicit") else "explicit",
        "ticker_count": len(tickers),
        "explains": explains,
    }
    return pack


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _fmt_opt(val) -> str:
    """Format optionality value."""
    if val is None:
        return "n/a"
    return f"{val:.2f}"


def _fmt_num(val) -> str:
    """Format numeric value with fallback."""
    if val is None:
        return "n/a"
    if isinstance(val, float):
        return f"{val:.1f}" if val != int(val) else str(int(val))
    return str(val)


def _fmt_pct(val) -> str:
    """Format percentage value."""
    if val is None:
        return "n/a"
    return f"{val:.1f}%"


def render_dossier_md(dossier: Dict[str, Any]) -> str:
    """Render one ticker's dossier as Markdown."""
    ticker = dossier["ticker"]

    if not dossier.get("found", False):
        return f"### {ticker} (not found in snapshot)\n"

    tier = dossier.get("tier_dev", "?")
    elig = "eligible" if dossier.get("eligible") else "ineligible"
    lines: List[str] = [f"### {ticker} ({tier}-tier, {elig})\n"]

    # Summary table
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Composite rank / score | {_fmt_num(dossier.get('composite_rank'))} / {_fmt_num(dossier.get('composite_score'))} |")
    lines.append(f"| Size band | {dossier.get('size_band', 'n/a')} |")
    lines.append(f"| Actionable rank | {_fmt_num(dossier.get('actionable_rank'))} |")
    lines.append(f"| Target weight | {_fmt_pct(dossier.get('target_weight_pct'))} |")
    lines.append("")

    # Gates — use explanation if available, fallback to legacy
    explanation = dossier.get("explanation")
    if explanation:
        elig_block = explanation["eligibility"]
        if elig_block["failed_gates"]:
            parts = [
                f"{g} ({elig_block['gate_details'].get(g, '')})"
                for g in elig_block["failed_gates"]
            ]
            lines.append(f"**Gates:** {', '.join(parts)} FAIL")
        else:
            lines.append("**Gates:** all pass")
    else:
        gates = dossier.get("gates", {})
        failed_gates = [g for g, v in gates.items() if v == "fail"]
        if failed_gates:
            lines.append(f"**Gates:** {', '.join(failed_gates)} FAIL")
        else:
            lines.append("**Gates:** all pass")

    # Catalyst
    cat_str = dossier.get("catalyst_strength", "n/a")
    cat_days = dossier.get("catalyst_days")
    cat_mode = dossier.get("catalyst_mode", "")
    cat_parts = [cat_str]
    if cat_days is not None:
        cat_parts.append(f"{cat_days} days")
    if cat_mode:
        cat_parts.append(cat_mode)
    lines.append(f"**Catalyst:** {' ('.join(cat_parts[:1])}" + (f" ({', '.join(cat_parts[1:])})" if len(cat_parts) > 1 else ""))

    # Tier reason (from explanation)
    if explanation:
        tier_block = explanation["tier"]
        if tier_block["primary_reasons"]:
            descs = [
                tier_block["reason_descriptions"].get(r, r)
                for r in tier_block["primary_reasons"]
            ]
            lines.append(f"**Tier reason:** {' + '.join(descs)}")

    # Optionality
    opt_val = dossier.get("clinical_optionality_pct_dev")
    clears = dossier.get("clears_a_floor")
    a_fl = dossier.get("a_floor", DEFAULT_A_FLOOR)
    opt_text = _fmt_opt(opt_val)
    if clears is True:
        opt_text += f" (clears a_floor={a_fl})"
    elif clears is False:
        opt_text += f" (below a_floor={a_fl})"
    lines.append(f"**Optionality:** {opt_text}")

    # Momentum + Risk flags
    mom = dossier.get("mom_state", "n/a")
    risk = dossier.get("risk_flags", [])
    risk_text = ", ".join(risk) if risk else "none"
    lines.append(f"**Momentum:** {mom} | **Risk flags:** {risk_text}")

    # Drawdown
    dd = dossier.get("de_drawdown")
    dd_reason = dossier.get("de_drawdown_missing_reason", "")
    if dd is not None:
        lines.append(f"**Drawdown:** {dd:.2f}")
    elif dd_reason:
        lines.append(f"**Drawdown:** missing ({dd_reason})")
    else:
        lines.append("**Drawdown:** n/a")

    # Delta (if present)
    delta = dossier.get("delta")
    if delta is not None:
        delta_parts: List[str] = []
        if not delta.get("in_prior"):
            delta_parts.append("NEW (not in prior snapshot)")
        elif not delta.get("in_current"):
            delta_parts.append("DROPPED (not in current snapshot)")
        else:
            for col in _STRING_DELTA_COLS:
                if col in delta:
                    d = delta[col]
                    delta_parts.append(f"{col} {d['prior']}\u2192{d['current']}")
            for col in _NUMERIC_DELTA_COLS:
                if col in delta:
                    d = delta[col]
                    change = d.get("change")
                    if change is not None:
                        sign = "+" if change > 0 else ""
                        delta_parts.append(
                            f"{col} {_fmt_num(d['prior'])}\u2192{_fmt_num(d['current'])} ({sign}{_fmt_num(change)})"
                        )
                    else:
                        delta_parts.append(
                            f"{col} {_fmt_num(d['prior'])}\u2192{_fmt_num(d['current'])}"
                        )

        if delta_parts:
            lines.append(f"\n**Delta:** {', '.join(delta_parts)}")

    lines.append("")
    return "\n".join(lines)


def render_explain_pack_md(pack: Dict[str, Any]) -> str:
    """Render the full explain pack as Markdown."""
    lines: List[str] = []
    lines.append("# Ticker Explain Pack")
    lines.append("")
    lines.append(f"**Snapshot:** {pack.get('snapshot_date', 'n/a')}")
    prev = pack.get("prev_snapshot_date")
    if prev:
        lines.append(f"**Previous:** {prev}")
    lines.append(f"**Tickers:** {pack.get('ticker_count', 0)} ({pack.get('tickers_selected_by', 'unknown')})")
    lines.append(f"**a_floor:** {pack.get('a_floor', DEFAULT_A_FLOOR)}")
    lines.append(f"**Generated:** {pack.get('generated_at', '')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for dossier in pack.get("explains", []):
        lines.append(render_dossier_md(dossier))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-ticker explain dossiers for drift movers."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        required=True,
        help="Path to current snapshot directory",
    )
    parser.add_argument(
        "--prev-snapshot-dir",
        type=Path,
        default=None,
        help="Path to previous snapshot directory (optional, enables deltas)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated ticker list; omit for auto-selection from attribution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory (default: artifacts/)",
    )
    parser.add_argument(
        "--a-floor",
        type=float,
        default=DEFAULT_A_FLOOR,
        help=f"Optionality floor for annotation (default: {DEFAULT_A_FLOOR})",
    )

    args = parser.parse_args(argv)

    # Load snapshots
    current = load_snapshot(args.snapshot_dir)
    if current is None:
        print(f"ERROR: Could not load snapshot from {args.snapshot_dir}", file=sys.stderr)
        sys.exit(1)

    prior: Optional[SnapshotData] = None
    if args.prev_snapshot_dir is not None:
        prior = load_snapshot(args.prev_snapshot_dir)
        if prior is None:
            print(
                f"WARNING: Could not load prior snapshot from {args.prev_snapshot_dir}",
                file=sys.stderr,
            )

    # Resolve ticker list
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        selection_mode = "explicit"
    elif prior is not None:
        tickers = auto_select_tickers(current, prior)
        selection_mode = "auto"
    else:
        # Single snapshot, no prior — can't auto-select, produce empty pack
        tickers = []
        selection_mode = "auto"

    # Build pack
    pack = build_explain_pack(current, prior, tickers, args.a_floor)
    pack["tickers_selected_by"] = selection_mode

    # Render
    md = render_explain_pack_md(pack)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ticker_explain_pack.json"
    md_path = args.output_dir / "ticker_explain_pack.md"

    with open(json_path, "w") as f:
        json.dump(pack, f, indent=2)
    with open(md_path, "w") as f:
        f.write(md)

    # Console summary
    print(f"Ticker explain pack: {len(tickers)} tickers ({selection_mode})")
    if tickers:
        print(f"  Tickers: {', '.join(tickers[:20])}", end="")
        if len(tickers) > 20:
            print(f" ... (+{len(tickers) - 20} more)", end="")
        print()
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")


if __name__ == "__main__":
    main()
