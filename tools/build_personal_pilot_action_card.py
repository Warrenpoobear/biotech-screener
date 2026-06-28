#!/usr/bin/env python3
"""Build DEM Personal Pilot Action Card — weekly operator briefing before any rebalance.

Reads the latest ranked snapshot, blotter, and available shadow guard outputs to produce:
  - artifacts/live_pilot/ACTION_CARD_{date}.md   (human-readable decision card)
  - artifacts/live_pilot/ACTION_CARD_{date}.json (machine-readable)

Does NOT place trades. Does NOT modify any production artifact.

Classification: PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_RANKER_CHANGE /
                NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_AUTONOMOUS_TRADING

Usage:
    python3 tools/build_personal_pilot_action_card.py --as-of-date 2026-06-30
    python3 tools/build_personal_pilot_action_card.py --as-of-date 2026-06-30 --account-usd 2500
    python3 tools/build_personal_pilot_action_card.py --as-of-date 2026-06-30 --snap-date 2026-06-26
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SNAPSHOTS_ROOT = REPO_ROOT / "data" / "snapshots"
PRODUCTION_DATA = REPO_ROOT / "production_data"
PILOT_ROOT = REPO_ROOT / "artifacts" / "live_pilot"
BLOTTER_CSV = PILOT_ROOT / "dem_personal_pilot_blotter.csv"
SHADOW_ROOT = REPO_ROOT / "artifacts" / "shadow"
STRESS_WRAPPER_DIR = REPO_ROOT / "artifacts" / "validation" / "stress_wrapper"
FAILURE_GUARDS_DIR = REPO_ROOT / "artifacts" / "failure_mode_guards"
PRICE_HISTORY = PRODUCTION_DATA / "price_history_split_adj.csv"
if not PRICE_HISTORY.exists():
    PRICE_HISTORY = PRODUCTION_DATA / "price_history.csv"

SCHEMA_VERSION = "dem_personal_pilot_action_card.v2"
GOVERNANCE = (
    "PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_RANKER_CHANGE / "
    "NO_SELECTOR_CHANGE / NO_SIZING_CHANGE / NO_AUTONOMOUS_TRADING"
)

TOP_N = 30
BENCH_N = 60
INELIGIBLE_NAMES = {"ABVX"}  # sell-only; never buy

# Drawdown rails (relative vs XBI, pp)
RAIL_REVIEW = -5.0
RAIL_PAUSE = -7.5
RAIL_FREEZE = -10.0


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def find_latest_snapshot(before_date: Optional[str] = None) -> Optional[Path]:
    candidates = sorted(
        p for p in SNAPSHOTS_ROOT.iterdir() if p.is_dir() and p.name[:4].isdigit() and "__" not in p.name
    )
    if before_date:
        candidates = [p for p in candidates if p.name <= before_date]
    return candidates[-1] if candidates else None


def load_rankings(snap_dir: Path) -> List[Dict[str, Any]]:
    path = snap_dir / "rankings.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_metadata(snap_dir: Path) -> Dict[str, Any]:
    path = snap_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_blotter() -> List[Dict[str, str]]:
    if not BLOTTER_CSV.exists():
        return []
    rows = []
    with open(BLOTTER_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ticker"):
                rows.append(row)
    return rows


def load_stress_wrapper() -> Dict[str, Any]:
    card_json = STRESS_WRAPPER_DIR / "stress_wrapper_card.json"
    if card_json.exists():
        try:
            return json.loads(card_json.read_text())
        except Exception:
            pass
    return {}


def load_ees_shadow() -> Dict[str, Any]:
    cards = sorted(SHADOW_ROOT.glob("ees_v3_raw_veto_shadow_*.json"))
    if cards:
        try:
            return json.loads(cards[-1].read_text())
        except Exception:
            pass
    cards = sorted(SHADOW_ROOT.glob("ees_v3_daily_card_*.json"))
    if cards:
        try:
            return json.loads(cards[-1].read_text())
        except Exception:
            pass
    return {}


def load_rh_feed(as_of: str) -> Dict[str, Any]:
    try:
        from tools.rh_feed_sync import load_rh_feed as _load

        return _load(as_of)
    except ImportError:
        return {"as_of_date": as_of, "available": []}


def load_xbi_price(as_of: str) -> Optional[float]:
    """Find XBI close for as_of in the long-format price history (date, ticker, close, ...)."""
    for ph in (PRICE_HISTORY, PRODUCTION_DATA / "price_history.csv", PRODUCTION_DATA / "price_history_split_adj.csv"):
        if not ph.exists():
            continue
        try:
            with open(ph, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = [c.lower() for c in (reader.fieldnames or [])]
                # Wide-format: columns are tickers
                if "xbi" in cols:
                    date_col_idx = next(
                        (i for i, c in enumerate(reader.fieldnames or []) if c.lower() in ("date", "as_of_date")), None
                    )
                    xbi_col = next(c for c in (reader.fieldnames or []) if c.upper() == "XBI")
                    for row in reader:
                        d = list(row.values())[date_col_idx] if date_col_idx is not None else ""
                        if d.startswith(as_of):
                            v = row.get(xbi_col, "")
                            return float(v) if v else None
                # Long-format: rows have a 'ticker' column
                elif "ticker" in cols:
                    close_col = next(
                        (c for c in (reader.fieldnames or []) if c.lower() in ("close", "close_price", "adj_close")),
                        None,
                    )
                    if not close_col:
                        continue
                    best: Optional[float] = None
                    for row in reader:
                        if row.get("ticker", "").upper() != "XBI":
                            continue
                        d = row.get("date", row.get("Date", ""))
                        if d.startswith(as_of):
                            v = row.get(close_col, "")
                            return float(v) if v else None
                        if d <= as_of:
                            v = row.get(close_col, "")
                            best = float(v) if v else best
                    if best is not None:
                        return best
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Pre-trade checklist
# ---------------------------------------------------------------------------


def _check(name: str, passed: bool, detail: str = "") -> Dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def run_pretrade_checklist(
    snap_dir: Path,
    rankings: List[Dict],
    metadata: Dict,
    as_of: str,
    rh_feed: Optional[Dict] = None,
) -> Tuple[List[Dict], bool]:
    checks = []
    xbi_ok = load_xbi_price(as_of) is not None

    eligible = [r for r in rankings if r.get("eligible", "1") != "0"]
    actionable = [
        r
        for r in eligible
        if _safe_int(r.get("actionable_rank")) is not None and r.get("ticker") not in INELIGIBLE_NAMES
    ]
    actionable_sorted = sorted(actionable, key=lambda r: _safe_int(r.get("actionable_rank"), 9999))
    top30 = actionable_sorted[:TOP_N]

    # 1. Price coverage
    priced = sum(1 for r in top30 if r.get("close_price") or r.get("de_alpha_60d_source") == "price_history")
    price_ok = len(top30) > 0 and (priced / len(top30)) >= 0.95
    checks.append(_check("price_coverage", price_ok, f"{priced}/{len(top30)} Top-30 names have price data"))

    # 2. XBI endpoint
    checks.append(
        _check(
            "xbi_endpoint", xbi_ok, "XBI price found in price_history" if xbi_ok else "XBI not found in price_history"
        )
    )

    # 3. Split-adjusted price source
    split_adj = (PRODUCTION_DATA / "price_history_split_adj.csv").exists() or any(
        r.get("de_alpha_60d_source") == "price_history" for r in top30
    )
    checks.append(
        _check("split_adjusted_source", split_adj, "price_history_split_adj.csv present or source=price_history")
    )

    # 4. Model hash unchanged
    model_hash_current = _get_model_hash()
    model_hash_last = _get_last_blotter_field("model_hash")
    model_ok = model_hash_last is None or model_hash_last == model_hash_current
    checks.append(
        _check(
            "model_hash_unchanged",
            model_ok,
            f"hash={model_hash_current}" + ("" if model_ok else f" (was {model_hash_last})"),
        )
    )

    # 5. Ruleset hash unchanged
    ruleset_hash_current = metadata.get("ruleset_hash") or _get_ruleset_hash(rankings)
    ruleset_hash_last = _get_last_blotter_field("ruleset_hash")
    ruleset_ok = ruleset_hash_last is None or ruleset_hash_last == ruleset_hash_current
    checks.append(
        _check(
            "ruleset_hash_unchanged",
            ruleset_ok,
            f"hash={ruleset_hash_current}" + ("" if ruleset_ok else f" (was {ruleset_hash_last})"),
        )
    )

    # 6. Rankings snapshot complete
    snap_ok = len(top30) >= TOP_N
    checks.append(
        _check("rankings_snapshot_complete", snap_ok, f"{len(top30)} eligible actionable names (need {TOP_N})")
    )

    # Soft checks (WARN only — do not block)
    ees_data = load_ees_shadow()
    ees_false_names = set(ees_data.get("veto_names", []) or ees_data.get("false_names", []))
    ees_false_in_top30 = [r["ticker"] for r in top30 if r.get("ticker") in ees_false_names]
    ees_gate_false = [r["ticker"] for r in top30 if r.get("ees_v3_gate", "").lower() in ("false", "0", "fail")]
    ees_flagged = list(set(ees_false_in_top30 + ees_gate_false))
    checks.append(
        _check(
            "ees_false_count_recorded",
            True,
            f"{len(ees_flagged)} EES-false names in Top-30: {', '.join(ees_flagged) or 'none'}",
        )
    )

    stress_data = load_stress_wrapper()
    repeat_offenders = [
        name
        for name, info in (stress_data.get("repeat_offenders") or {}).items()
        if name in {r["ticker"] for r in top30}
    ]
    checks.append(
        _check(
            "repeat_offender_count_recorded",
            True,
            f"{len(repeat_offenders)} repeat-offenders in Top-30: {', '.join(repeat_offenders) or 'none'}",
        )
    )

    bench = actionable_sorted[TOP_N:BENCH_N]
    checks.append(_check("replacement_bench_recorded", True, f"{len(bench)} names available in ranks 31–60"))

    cost_est = _estimate_costs(top30)
    checks.append(_check("cost_estimate_recorded", True, f"estimated cost drag: {cost_est:.1f}bps"))

    # Soft check 11: earnings cross-check via RH feed
    if rh_feed and "earnings" in rh_feed.get("available", []):
        try:
            from tools.rh_feed_sync import cross_check_earnings

            discrepancies = cross_check_earnings(as_of, snap_dir / "rankings.csv")
            checks.append(
                _check(
                    "earnings_crosscheck",
                    True,
                    f"RH earnings cross-check: {len(discrepancies)} discrepancies "
                    f"({', '.join(d['ticker'] for d in discrepancies[:4]) or 'none'})",
                )
            )
        except Exception as e:
            checks.append(_check("earnings_crosscheck", True, f"cross-check unavailable: {e}"))
    else:
        checks.append(_check("earnings_crosscheck", True, "RH earnings cache absent — run rh_feed_sync to enable"))

    # Soft check 12: tradability — flag untradeable names in Top-30
    if rh_feed and "tradability" in rh_feed.get("available", []):
        trad_data = rh_feed.get("tradability", {})
        tickers_data = trad_data.get("tickers", {})
        untradeable_top30 = [r["ticker"] for r in top30 if not tickers_data.get(r["ticker"], {}).get("tradeable", True)]
        checks.append(
            _check(
                "tradability_checked",
                True,
                f"{len(untradeable_top30)} untradeable in Top-30"
                + (f": {', '.join(untradeable_top30)}" if untradeable_top30 else ""),
            )
        )
    else:
        checks.append(
            _check("tradability_checked", True, "RH tradability cache absent — run write_tradability() to enable")
        )

    hard_pass = all(c["passed"] for c in checks[:6])
    return checks, hard_pass


def _safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _get_model_hash() -> str:
    model_path = PRODUCTION_DATA / "ranker_v2_model.json"
    if model_path.exists():
        import hashlib

        return hashlib.md5(model_path.read_bytes()).hexdigest()[:12]
    return "no_model_file"


def _get_ruleset_hash(rankings: List[Dict]) -> str:
    for r in rankings[:3]:
        h = r.get("decision_engine_ruleset_id") or r.get("ruleset_hash")
        if h:
            return str(h)
    return "unknown"


def _get_last_blotter_field(field: str) -> Optional[str]:
    rows = load_blotter()
    if not rows:
        return None
    last = rows[-1]
    return last.get(field) or None


def _estimate_costs(top30: List[Dict]) -> float:
    costs = [float(r.get("est_cost_bps", 0) or 0) for r in top30]
    return sum(costs) / len(costs) if costs else 0.0


# ---------------------------------------------------------------------------
# Holdings and delta computation
# ---------------------------------------------------------------------------


def current_blotter_holdings() -> Dict[str, Dict]:
    rows = load_blotter()
    held: Dict[str, Dict] = {}
    for row in rows:
        ticker = row.get("ticker", "")
        if not ticker:
            continue
        action = row.get("action", "").upper()
        if action == "SELL":
            held.pop(ticker, None)
        elif action in ("BUY", "REBALANCE", "HOLD", "UNAVAILABLE"):
            held[ticker] = row
    return held


def compute_deltas(
    top30_tickers: List[str],
    current_held: Dict[str, Dict],
    account_usd: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    target = set(top30_tickers)
    current = set(current_held.keys())

    equal_weight = 1.0 / len(top30_tickers) if top30_tickers else 0.0
    target_notional = equal_weight * account_usd

    buys = [
        {"ticker": t, "action": "BUY", "target_weight": equal_weight, "target_notional": target_notional}
        for t in top30_tickers
        if t not in current
    ]
    sells = [
        {"ticker": t, "action": "SELL", "target_weight": 0.0, "target_notional": 0.0}
        for t in current
        if t not in target
    ]
    rebalances = []
    for t in top30_tickers:
        if t in current:
            last = current_held[t]
            actual_w = float(last.get("actual_weight") or 0)
            drift = abs(actual_w - equal_weight)
            if drift > 0.25 * equal_weight:
                rebalances.append(
                    {
                        "ticker": t,
                        "action": "REBALANCE",
                        "current_weight": actual_w,
                        "target_weight": equal_weight,
                        "target_notional": target_notional,
                    }
                )
    return buys, sells, rebalances


# ---------------------------------------------------------------------------
# Drawdown check
# ---------------------------------------------------------------------------


def compute_relative_drawdown(blotter_rows: List[Dict]) -> Optional[float]:
    # Minimal: reads inception_xs from blotter summary entries if available
    xs_rows = [r for r in blotter_rows if r.get("action") == "PERF_RECORD"]
    if not xs_rows:
        return None
    try:
        latest = xs_rows[-1]
        return float(latest.get("operator_note", "").split("rel_xs=")[-1].split()[0])
    except Exception:
        return None


def drawdown_verdict(rel_dd: Optional[float]) -> str:
    if rel_dd is None:
        return "UNKNOWN (no drawdown data yet)"
    if rel_dd <= RAIL_FREEZE:
        return f"FREEZE_PILOT ({rel_dd:.1f}pp vs XBI)"
    if rel_dd <= RAIL_PAUSE:
        return f"PAUSE ({rel_dd:.1f}pp vs XBI)"
    if rel_dd <= RAIL_REVIEW:
        return f"REVIEW ({rel_dd:.1f}pp vs XBI)"
    return f"OK ({rel_dd:+.1f}pp vs XBI)"


# ---------------------------------------------------------------------------
# Status recommendation
# ---------------------------------------------------------------------------


def recommend_status(checklist_pass: bool, rel_dd: Optional[float]) -> str:
    if not checklist_pass:
        return "NO_REBALANCE — pre-trade checklist FAIL"
    if rel_dd is not None and rel_dd <= RAIL_FREEZE:
        return "FREEZE — drawdown beyond -10pp vs XBI; liquidate and run autopsy"
    if rel_dd is not None and rel_dd <= RAIL_PAUSE:
        return "PAUSE — drawdown beyond -7.5pp vs XBI; hold positions"
    if rel_dd is not None and rel_dd <= RAIL_REVIEW:
        return "REVIEW — drawdown beyond -5pp vs XBI; no new capital; identify cause"
    return "REBALANCE / HOLD"


# ---------------------------------------------------------------------------
# Card rendering
# ---------------------------------------------------------------------------


def build_card(
    as_of: str,
    snap_dir: Path,
    rankings: List[Dict],
    metadata: Dict,
    account_usd: float,
    checklist: List[Dict],
    checklist_pass: bool,
    top30: List[Dict],
    bench: List[Dict],
    buys: List[Dict],
    sells: List[Dict],
    rebalances: List[Dict],
    rel_dd: Optional[float],
    ees_flagged: List[str],
    repeat_offenders: List[str],
    status_rec: str,
    rh_feed: Optional[Dict] = None,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ruleset = metadata.get("ruleset_hash") or _get_ruleset_hash(rankings)
    model_hash = _get_model_hash()

    lines: List[str] = []

    # Index quotes live header
    rh = rh_feed or {}
    index_quotes = rh.get("index_quotes", {})
    xbi_q = index_quotes.get("xbi", {})
    spy_q = index_quotes.get("spy", {})
    xbi_str = (
        f"${xbi_q['last']:,.2f} ({xbi_q['daily_move_pct']:+.2f}%)"
        if xbi_q.get("last") is not None and xbi_q.get("daily_move_pct") is not None
        else "n/a"
    )
    spy_str = (
        f"${spy_q['last']:,.2f} ({spy_q['daily_move_pct']:+.2f}%)"
        if spy_q.get("last") is not None and spy_q.get("daily_move_pct") is not None
        else "n/a"
    )
    ratio_str = f"{index_quotes['xbi_spy_ratio']:.4f}" if index_quotes.get("xbi_spy_ratio") is not None else "n/a"

    lines += [
        "# DEM Personal Pilot — Action Card",
        "",
        f"**As-of date**: {as_of}  ",
        f"**Generated**: {now}  ",
        f"**Model hash**: {model_hash}  ",
        f"**Ruleset hash**: {ruleset}  ",
        f"**Account estimate**: ${account_usd:,.0f}  ",
        f"**XBI**: {xbi_str}  |  **SPY**: {spy_str}  |  **XBI/SPY ratio**: {ratio_str}  ",
        "",
        "---",
        "",
        "## Current Status",
        "",
        f"**Recommendation**: {status_rec}",
        "",
        f"**Drawdown vs XBI**: {drawdown_verdict(rel_dd)}",
        "",
    ]

    # Pre-trade checklist
    lines += ["## Pre-trade Checklist", ""]
    lines += ["| # | Check | Result | Detail |", "|---|-------|--------|--------|"]
    for i, c in enumerate(checklist, 1):
        icon = "PASS" if c["passed"] else "FAIL"
        soft = "" if i <= 6 else " *(warn)*"
        lines.append(f"| {i} | {c['name']}{soft} | {icon} | {c['detail']} |")
    lines += [
        "",
        f"**Hard gates (1–6)**: {'ALL PASS' if checklist_pass else 'FAIL — no rebalance'}",
        "",
        "---",
        "",
    ]

    # Model — Top-30 basket
    lines += ["## Model — Theoretical Top-30", ""]
    lines += [
        "| Rank | Ticker | Cat Bucket | EES | Repeat Offender |",
        "|------|--------|------------|-----|-----------------|",
    ]
    for r in top30:
        ticker = r.get("ticker", "")
        rank = _safe_int(r.get("actionable_rank"), 0)
        bucket = r.get("catalyst_bucket") or r.get("cat_priority") or ""
        ees = "WARN" if ticker in ees_flagged else "ok"
        ro = "WARN" if ticker in repeat_offenders else "ok"
        lines.append(f"| {rank} | {ticker} | {bucket} | {ees} | {ro} |")
    lines += [""]

    ees_ro_overlap = [t for t in ees_flagged if t in repeat_offenders]
    if ees_ro_overlap:
        lines += [
            f"**OPERATOR REVIEW required**: {', '.join(ees_ro_overlap)} "
            "(EES-False + repeat-offender — do not trade these without operator decision)",
            "",
        ]

    # Execution
    lines += ["## Execution — Required Trades", ""]

    total_actions = len(buys) + len(sells) + len(rebalances)
    if not checklist_pass:
        lines += ["**BLOCKED — pre-trade checklist failed. No trades.**", ""]
    elif total_actions == 0:
        lines += ["**HOLD** — no trades required (basket unchanged, no drift triggers)", ""]
    else:
        if sells:
            lines += [f"**Sells** ({len(sells)}):"]
            lines += [f"- {s['ticker']}" for s in sells]
            lines += [""]
        if buys:
            lines += [f"**Buys** ({len(buys)}):"]
            eq_w = 1.0 / len(top30) if top30 else 0.0
            trad_tickers = rh.get("tradability", {}).get("tickers", {})
            for b in buys:
                notional = b.get("target_notional", eq_w * account_usd)
                flags = []
                if b["ticker"] in ees_ro_overlap:
                    flags.append("⚠ EES+RO — OPERATOR REVIEW")
                if trad_tickers and not trad_tickers.get(b["ticker"], {}).get("tradeable", True):
                    flags.append("⚠ UNTRADEABLE")
                flag_str = ("  " + "  ".join(flags)) if flags else ""
                lines.append(f"- {b['ticker']}  ~${notional:,.0f}{flag_str}")
            lines += [""]
        if rebalances:
            lines += [f"**Rebalances** ({len(rebalances)}):"]
            for rb in rebalances:
                lines.append(
                    f"- {rb['ticker']}  " f"current={rb['current_weight']:.1%} → target={rb['target_weight']:.1%}"
                )
            lines += [""]

    # Unavailable names
    uo_names = [t for t in {r["ticker"] for r in top30} if t in INELIGIBLE_NAMES]
    if uo_names:
        lines += [f"**Ineligible (excluded)**: {', '.join(uo_names)}", ""]

    # Shadow bench
    bench_tickers = [r.get("ticker", "") for r in bench]
    lines += [
        "## Shadow Bench (Ranks 31–60)",
        "",
        f"Available replacement names: {', '.join(bench_tickers[:20])}{'…' if len(bench_tickers) > 20 else ''}",
        "",
        "Shadow-only — no automatic substitution.",
        "",
        "---",
        "",
    ]

    # Risk summary
    fund_data = rh.get("fundamentals", {})
    tier_drifts = fund_data.get("tier_drifts", [])
    drift_data = rh.get("price_drift", {})
    price_alerts = drift_data.get("alerts", [])
    trad_data = rh.get("tradability", {})
    n_untradeable = trad_data.get("n_untradeable", "n/a" if "tradability" not in rh.get("available", []) else 0)

    lines += [
        "## Risk Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Relative drawdown vs XBI | {f'{rel_dd:.1f}pp' if rel_dd is not None else 'unknown'} |",
        f"| EES-false in Top-30 | {len(ees_flagged)} |",
        f"| Repeat-offenders in Top-30 | {len(repeat_offenders)} |",
        f"| EES+RO overlap (review required) | {len(ees_ro_overlap)} |",
        f"| Bench names available | {len(bench)} |",
        f"| Untradeable in Top-30 | {n_untradeable} |",
        f"| Price drift HIGH/MEDIUM alerts | {len(price_alerts)} |",
        f"| Market cap tier drifts | {len(tier_drifts)} |",
        "",
        "---",
        "",
    ]

    # RH Feed sections
    available = rh.get("available", [])

    # Fills — pending blotter update
    if "fills" in available:
        try:
            from tools.rh_feed_sync import pending_blotter_fills

            pending = pending_blotter_fills(as_of)
        except ImportError:
            pending = []
        fills_data = rh.get("fills", {})
        n_filled = sum(1 for o in fills_data.get("orders", []) if o.get("state") == "filled")
        lines += [
            "## RH Fills",
            "",
            f"Filled orders: {n_filled} | Pending blotter update: {len(pending)}",
            "",
        ]
        if pending:
            lines += ["**Unlogged fills — call write_fills_to_blotter() to record:**", ""]
            lines += ["| Ticker | Side | Qty | Avg Price | Notional |", "|--------|------|-----|-----------|---------|"]
            for o in pending:
                lines.append(
                    f"| {o['ticker']} | {o['side']} | {o['filled_qty']} "
                    f"| ${o['average_price']} | ${o['notional']} |"
                )
            lines += [""]
        lines += ["---", ""]

    # Realized P&L
    if "pnl" in available:
        pnl = rh.get("pnl", {})
        gain = pnl.get("total_realized_gain_usd")
        gain_pct = pnl.get("total_realized_gain_pct")
        lines += [
            "## Realized P&L (Net of Costs)",
            "",
            (
                f"Total realized: **${gain:+,.2f}** ({gain_pct:+.2%})"
                if gain is not None
                else "Realized P&L: data present but unparsed."
            ),
            f"Span: {pnl.get('span', 'all')} | Source: Robinhood confirmed fills",
            "",
            "---",
            "",
        ]

    # Intraday binary-now alerts
    if "intraday" in available:
        intraday = rh.get("intraday", {})
        alerts = intraday.get("alerts", [])
        quotes = intraday.get("quotes", {})
        lines += ["## Intraday Binary-Now Monitor", ""]
        if alerts:
            lines += [f"**MOVE ALERTS ({len(alerts)}) — >5% intraday move, catalyst ≤5d:**", ""]
            for a in alerts:
                lines.append(f"- **{a['ticker']}**: {a['move_pct']:+.1f}% intraday, {a['catalyst_days']}d to catalyst")
            lines += [""]
        lines += ["| Ticker | Last | Move% | Cat Days | Alert |", "|--------|------|-------|----------|-------|"]
        for ticker, q in sorted(quotes.items()):
            move = f"{q['intraday_move_pct']:+.1f}%" if q.get("intraday_move_pct") is not None else "—"
            lines.append(
                f"| {ticker} | ${q.get('last', '—')} | {move} "
                f"| {q.get('catalyst_days', '—')} | {q.get('alert', '—')} |"
            )
        lines += ["", "---", ""]
    else:
        lines += [
            "## Intraday Binary-Now Monitor",
            "",
            "*No intraday cache — run rh_feed_sync write_intraday() for live quotes.*",
            "",
            "---",
            "",
        ]

    # Post-snapshot price drift
    if "price_drift" in available:
        pd_data = rh.get("price_drift", {})
        pd_alerts = pd_data.get("alerts", [])
        pd_drifts = pd_data.get("drifts", {})
        top30_set = {r["ticker"] for r in top30}
        top30_drifts = {t: v for t, v in pd_drifts.items() if t in top30_set}
        lines += ["## Post-Snapshot Price Drift (vs Snap Close)", ""]
        if pd_alerts:
            lines += [f"**{len(pd_alerts)} HIGH/MEDIUM drift alert(s):**", ""]
            for a in pd_alerts:
                if a["ticker"] in top30_set:
                    lines.append(
                        f"- **{a['ticker']}** {a['drift_pct']:+.1f}%  "
                        f"snap=${a['snap_close']}  live=${a['live_close']}  [{a['alert']}]"
                    )
            lines += [""]
        if top30_drifts:
            lines += [
                "| Ticker | Snap Close | Live Close | Drift% | Alert |",
                "|--------|------------|------------|--------|-------|",
            ]
            for t, d in sorted(top30_drifts.items(), key=lambda x: abs(x[1].get("drift_pct") or 0), reverse=True):
                lines.append(
                    f"| {t} | ${d.get('snap_close', '—')} | ${d.get('live_close', '—')} "
                    f"| {d.get('drift_pct', '—')}% | {d.get('alert', '—')} |"
                )
            lines += [""]
        else:
            lines += ["*No drift data for Top-30.*", ""]
        lines += ["---", ""]
    else:
        lines += [
            "## Post-Snapshot Price Drift",
            "",
            "*No price drift cache — run write_price_drift() to enable.*",
            "",
            "---",
            "",
        ]

    # Market cap tier drifts
    if "fundamentals" in available:
        fd = rh.get("fundamentals", {})
        td = fd.get("tier_drifts", [])
        top30_set = {r["ticker"] for r in top30}
        td_top30 = [d for d in td if d["ticker"] in top30_set]
        lines += ["## Market Cap Tier Drifts (vs Snapshot)", ""]
        if td_top30:
            lines += [f"**{len(td_top30)} tier drift(s) in Top-30:**", ""]
            lines += [
                "| Ticker | Snap Bucket | Live Bucket | Snap Cap $mm | Live Cap $mm |",
                "|--------|-------------|-------------|--------------|--------------|",
            ]
            for d in td_top30:
                lines.append(
                    f"| {d['ticker']} | {d['from']} | {d['to']} "
                    f"| ${d.get('snap_cap_mm', '—'):,} | ${d.get('live_cap_mm', '—'):,} |"
                )
            lines += [""]
        else:
            lines += ["No tier drifts in Top-30.", ""]
        lines += ["---", ""]
    elif "fundamentals" not in available:
        lines += [
            "## Market Cap Tier Drifts",
            "",
            "*No fundamentals cache — run write_fundamentals() to enable.*",
            "",
            "---",
            "",
        ]

    # Earnings cross-check summary
    if "earnings" in available:
        try:
            from tools.rh_feed_sync import cross_check_earnings

            discrepancies = cross_check_earnings(as_of, snap_dir / "rankings.csv")
        except ImportError:
            discrepancies = []
        lines += ["## Earnings Calendar Cross-Check", ""]
        if discrepancies:
            lines += [f"**{len(discrepancies)} discrepancies between model and RH calendar:**", ""]
            lines += ["| Ticker | Rank | Type | Detail |", "|--------|------|------|--------|"]
            for d in discrepancies:
                lines.append(f"| {d['ticker']} | {d['rank']} | {d['type']} | {d['detail']} |")
        else:
            lines += ["Catalyst dates consistent between model and RH earnings calendar."]
        lines += ["", "---", ""]

    # Regime (informational only)
    ri = rh.get("regime_inputs", {}) if rh else {}
    regime_label = ri.get("regime") or (rankings[0].get("regime_label") if rankings else None)
    regime_desc = ri.get("regime_description", "")
    regime_conf = ri.get("confidence")
    vix_val = ri.get("vix")
    xbi_10d = ri.get("xbi_10d_momentum")
    spy_10d = ri.get("spy_10d_momentum")
    xbi_vs_spy_30d = ri.get("xbi_vs_spy_30d")
    regime_error = ri.get("regime_error")

    regime_lines = [
        "## Regime (Informational Only — Not a Sizing Input)",
        "",
    ]
    if ri and ri.get("inputs_complete"):
        conf_str = f"  confidence={regime_conf:.2f}" if regime_conf is not None else ""
        regime_lines += [
            f"**{regime_label or 'UNKNOWN'}**{conf_str}",
            "",
            f"{regime_desc}" if regime_desc else "",
            "",
            "| Input | Value |",
            "|-------|-------|",
            f"| VIX | {vix_val:.2f} |" if vix_val is not None else "| VIX | n/a |",
            f"| XBI 10d momentum | {xbi_10d:+.2f}% |" if xbi_10d is not None else "| XBI 10d momentum | n/a |",
            f"| SPY 10d momentum | {spy_10d:+.2f}% |" if spy_10d is not None else "| SPY 10d momentum | n/a |",
            (
                f"| XBI vs SPY 30d | {xbi_vs_spy_30d:+.2f}pp |"
                if xbi_vs_spy_30d is not None
                else "| XBI vs SPY 30d | n/a |"
            ),
        ]
        if regime_error:
            regime_lines += ["", f"*Engine error: {regime_error}*"]
    elif ri:
        regime_lines += [
            f"*Regime inputs incomplete — missing: {'VIX' if vix_val is None else 'XBI/SPY 30d history'}.*",
            "",
            f"| VIX | {vix_val:.2f} |" if vix_val is not None else "",
        ]
    else:
        regime_lines += [
            f"Current regime label: **{regime_label or 'UNKNOWN'}**",
            "",
            "*No regime_inputs cache — run write_regime_inputs() to enable live classification.*",
        ]
    regime_lines += [
        "",
        "*Informational only — not a sizing or ranking input.*",
        "",
        "---",
        "",
    ]
    lines += [r for r in regime_lines if r is not None]

    # Evidence
    lines += [
        "## Evidence Trackers",
        "",
        "| Tracker | Status |",
        "|---------|--------|",
        "| Forward bootstrap percentile | *run eval_forward_returns.py* |",
        "| Net excess vs XBI | *populate from blotter performance records* |",
        "| Actual-vs-theoretical gap | *populate after session* |",
        "| EES shadow delta (Monitor B) | *see artifacts/failure_mode_guards/ees_guarded_returns.jsonl* |",
        "| Repeat-offender shadow (Monitor A) | *see artifacts/failure_mode_guards/repeat_offender_*.json* |",
        "",
        "---",
        "",
    ]

    # Decision
    lines += [
        "## Decision",
        "",
        f"**{status_rec}**",
        "",
        "Operator note: *(fill in before executing)*",
        "",
        "---",
        "",
        f"*Classification: {GOVERNANCE}*  ",
        f"*Schema: {SCHEMA_VERSION}*",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of-date", required=True, help="Card date, YYYY-MM-DD (usually today / Monday)")
    parser.add_argument("--snap-date", help="Override snapshot date (default: latest snapshot on or before as-of-date)")
    parser.add_argument(
        "--account-usd",
        type=float,
        default=None,
        help="Pilot account size in USD (default: read from portfolio_policy.json)",
    )
    args = parser.parse_args()

    as_of = args.as_of_date
    PILOT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load account size
    account_usd = args.account_usd
    if account_usd is None:
        policy_path = PRODUCTION_DATA / "portfolio_policy.json"
        if policy_path.exists():
            try:
                policy = json.loads(policy_path.read_text())
                account_usd = float(
                    policy.get("pilot_max_capital_usd")
                    or policy.get("account_usd")
                    or policy.get("account_size_usd")
                    or 2500
                )
            except Exception:
                account_usd = 2500.0
        else:
            account_usd = 2500.0

    # Find snapshot
    snap_date = args.snap_date or as_of
    snap_dir = find_latest_snapshot(snap_date)
    if snap_dir is None:
        print(f"ERROR: no snapshot found on or before {snap_date}", file=sys.stderr)
        sys.exit(1)
    print(f"Using snapshot: {snap_dir.name}")

    rankings = load_rankings(snap_dir)
    metadata = load_metadata(snap_dir)

    if not rankings:
        print("ERROR: rankings.csv is empty or missing", file=sys.stderr)
        sys.exit(1)

    # Build sorted name lists
    eligible = [r for r in rankings if r.get("eligible", "1") != "0"]
    actionable = [
        r
        for r in eligible
        if _safe_int(r.get("actionable_rank")) is not None and r.get("ticker") not in INELIGIBLE_NAMES
    ]
    actionable_sorted = sorted(actionable, key=lambda r: _safe_int(r.get("actionable_rank"), 9999))
    top30 = actionable_sorted[:TOP_N]
    bench = actionable_sorted[TOP_N:BENCH_N]
    top30_tickers = [r["ticker"] for r in top30]

    # RH feed
    rh_feed = load_rh_feed(as_of)
    if rh_feed["available"]:
        print(f"RH feed loaded: {', '.join(rh_feed['available'])}")
    else:
        print("RH feed: no cache for this date (run rh_feed_sync to populate)")

    # Checklist
    checklist, checklist_pass = run_pretrade_checklist(snap_dir, rankings, metadata, as_of, rh_feed)

    # EES and repeat-offender flags
    ees_data = load_ees_shadow()
    ees_false_names = set(ees_data.get("veto_names", []) or ees_data.get("false_names", []))
    ees_gate_false = {r["ticker"] for r in top30 if r.get("ees_v3_gate", "").lower() in ("false", "0", "fail")}
    ees_flagged = list(ees_false_names.union(ees_gate_false) & set(top30_tickers))

    stress_data = load_stress_wrapper()
    repeat_offenders = [
        name for name in (stress_data.get("repeat_offenders") or {}).keys() if name in set(top30_tickers)
    ]

    # Current holdings and deltas
    current_held = current_blotter_holdings()
    buys, sells, rebalances = compute_deltas(top30_tickers, current_held, account_usd)

    # Drawdown
    blotter_rows = load_blotter()
    rel_dd = compute_relative_drawdown(blotter_rows)

    # Status recommendation
    status_rec = recommend_status(checklist_pass, rel_dd)

    # Build card text
    card_md = build_card(
        as_of=as_of,
        snap_dir=snap_dir,
        rankings=rankings,
        metadata=metadata,
        account_usd=account_usd,
        checklist=checklist,
        checklist_pass=checklist_pass,
        top30=top30,
        bench=bench,
        buys=buys,
        sells=sells,
        rebalances=rebalances,
        rel_dd=rel_dd,
        ees_flagged=ees_flagged,
        repeat_offenders=repeat_offenders,
        status_rec=status_rec,
        rh_feed=rh_feed,
    )

    # Write outputs
    md_out = PILOT_ROOT / f"ACTION_CARD_{as_of}.md"
    md_out.write_text(card_md, encoding="utf-8")
    print(f"Wrote: {md_out}")

    json_out = PILOT_ROOT / f"ACTION_CARD_{as_of}.json"
    ruleset = metadata.get("ruleset_hash") or _get_ruleset_hash(rankings)
    payload = {
        "schema": SCHEMA_VERSION,
        "governance": GOVERNANCE,
        "as_of_date": as_of,
        "snap_date": snap_dir.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_hash": _get_model_hash(),
        "ruleset_hash": ruleset,
        "account_usd": account_usd,
        "status_recommendation": status_rec,
        "checklist_pass": checklist_pass,
        "checklist": checklist,
        "top30_tickers": top30_tickers,
        "bench_tickers": [r["ticker"] for r in bench],
        "buys": buys,
        "sells": sells,
        "rebalances": rebalances,
        "ees_flagged": ees_flagged,
        "repeat_offenders": repeat_offenders,
        "ees_ro_overlap": [t for t in ees_flagged if t in repeat_offenders],
        "rh_feed_available": rh_feed.get("available", []),
        "relative_drawdown_pp": rel_dd,
        "drawdown_verdict": drawdown_verdict(rel_dd),
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {json_out}")

    # Print brief summary to stdout
    print()
    print(f"STATUS: {status_rec}")
    print(f"Checklist: {'PASS' if checklist_pass else 'FAIL'}")
    print(f"Buys: {len(buys)}  Sells: {len(sells)}  Rebalances: {len(rebalances)}")
    print(f"EES-flagged: {len(ees_flagged)}  Repeat-offenders: {len(repeat_offenders)}")
    if not checklist_pass:
        print("\nFailed checklist items:")
        for c in checklist[:6]:
            if not c["passed"]:
                print(f"  FAIL  {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
