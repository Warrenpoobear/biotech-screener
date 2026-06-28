#!/usr/bin/env python3
"""Daily regime monitor — append-only ledger + self-calibrating accuracy tracker.

Reads the regime_inputs cache written by write_regime_inputs() and:
  1. Appends today's entry to artifacts/regime_monitor/regime_history.jsonl
  2. Detects regime transitions
  3. Settles 20-trading-day-old windows against actual XBI/SPY returns
  4. Computes calibration metrics (accuracy rate, mean prediction error)
  5. Generates a daily card at artifacts/regime_monitor/REGIME_CARD_{date}.md

Self-calibration is advisory only — never triggers model changes.

Classification: PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_AUTONOMOUS_TRADING

Usage:
    python3 scripts/run_regime_monitor.py --as-of-date 2026-06-28
    python3 scripts/run_regime_monitor.py  # uses today's date
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

REGIME_DIR = REPO_ROOT / "artifacts" / "regime_monitor"
LEDGER_PATH = REGIME_DIR / "regime_history.jsonl"
RH_CACHE_ROOT = REPO_ROOT / "data" / "caches" / "robinhood"
PRODUCTION_DATA = REPO_ROOT / "production_data"

# Calibration: expected XBI behaviour per regime (direction only)
REGIME_PREDICTIONS: Dict[str, Dict] = {
    "BULL": {"xbi_positive": True, "xbi_vs_spy_positive": True, "label": "XBI>0, XBI>SPY"},
    "BEAR": {"xbi_positive": False, "xbi_vs_spy_positive": False, "label": "XBI<0, XBI<SPY"},
    "VOLATILITY_SPIKE": {"xbi_positive": False, "xbi_vs_spy_positive": None, "label": "XBI<0"},
    "SECTOR_ROTATION": {"xbi_positive": None, "xbi_vs_spy_positive": None, "label": "mixed"},
    "SECTOR_DISLOCATION": {"xbi_positive": True, "xbi_vs_spy_positive": True, "label": "XBI>SPY (diverge)"},
    "RECESSION_RISK": {"xbi_positive": False, "xbi_vs_spy_positive": None, "label": "XBI<0"},
    "CREDIT_CRISIS": {"xbi_positive": False, "xbi_vs_spy_positive": False, "label": "XBI<0, XBI<SPY"},
}

MISCAL_THRESHOLDS = {
    "xbi_positive_miss": 0.0,  # predicted positive but XBI 20d < 0
    "xbi_negative_miss": 0.05,  # predicted negative but XBI 20d > +5%
    "divergence_miss": 5.0,  # predicted XBI>SPY but actual spread < 5pp
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_ledger() -> List[Dict]:
    if not LEDGER_PATH.exists():
        return []
    rows = []
    for line in LEDGER_PATH.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _append_ledger(entry: Dict) -> None:
    LEDGER_DIR = LEDGER_PATH.parent
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _load_xbi_price(as_of: str) -> Optional[float]:
    """Load XBI close for a given date from price history or regime cache."""
    # Try regime cache first (most recent)
    cache_path = RH_CACHE_ROOT / as_of / f"{as_of}_rh_regime_inputs.json"
    cache = _load_json(cache_path)
    if cache and cache.get("xbi_close"):
        return float(cache["xbi_close"])

    # Try index_quotes cache
    iq_path = RH_CACHE_ROOT / as_of / f"{as_of}_rh_index_quotes.json"
    iq = _load_json(iq_path)
    if iq and iq.get("xbi", {}).get("last"):
        return float(iq["xbi"]["last"])

    # Fall back to price_history.csv
    for ph_name in ("price_history_split_adj.csv", "price_history.csv"):
        ph = PRODUCTION_DATA / ph_name
        if not ph.exists():
            continue
        try:
            import csv

            with open(ph, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = [c.lower() for c in (reader.fieldnames or [])]
                if "ticker" not in cols:
                    continue
                close_col = next(
                    (c for c in (reader.fieldnames or []) if c.lower() in ("close", "close_price", "adj_close")), None
                )
                if not close_col:
                    continue
                best: Optional[float] = None
                for row in reader:
                    if row.get("ticker", "").upper() != "XBI":
                        continue
                    d = row.get("date", row.get("Date", ""))
                    v = row.get(close_col, "")
                    val = float(v) if v else None
                    if d == as_of:
                        return val
                    if d < as_of and val is not None:
                        best = val
                if best is not None:
                    return best
        except Exception:
            continue
    return None


def _load_spy_price(as_of: str) -> Optional[float]:
    cache_path = RH_CACHE_ROOT / as_of / f"{as_of}_rh_regime_inputs.json"
    cache = _load_json(cache_path)
    if cache and cache.get("spy_close"):
        return float(cache["spy_close"])
    iq = _load_json(RH_CACHE_ROOT / as_of / f"{as_of}_rh_index_quotes.json")
    if iq and iq.get("spy", {}).get("last"):
        return float(iq["spy"]["last"])
    return None


def _trading_days_ago(ref_date: str, n: int) -> str:
    """Approximate: subtract n*1.4 calendar days then round to nearest Mon-Fri."""
    d = date.fromisoformat(ref_date) - timedelta(days=int(n * 1.4) + 1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


def _check_prediction(regime: str, actual_xbi_20d: float, actual_spy_20d: float) -> Dict:
    pred = REGIME_PREDICTIONS.get(regime, {})
    xbi_pos_pred = pred.get("xbi_positive")
    xbi_vs_spy_pred = pred.get("xbi_vs_spy_positive")
    actual_spread = actual_xbi_20d - actual_spy_20d

    correct = True
    flags = []

    if xbi_pos_pred is True and actual_xbi_20d < MISCAL_THRESHOLDS["xbi_positive_miss"]:
        correct = False
        flags.append(f"BULL_MISS: predicted XBI>0 but actual={actual_xbi_20d:.1f}%")
    if xbi_pos_pred is False and actual_xbi_20d > MISCAL_THRESHOLDS["xbi_negative_miss"] * 100:
        correct = False
        flags.append(f"BEAR_MISS: predicted XBI<0 but actual={actual_xbi_20d:.1f}%")
    if xbi_vs_spy_pred is True and actual_spread < MISCAL_THRESHOLDS["divergence_miss"]:
        correct = False
        flags.append(f"DIVERGENCE_MISS: predicted XBI>SPY but spread={actual_spread:.1f}pp")

    return {
        "correct": correct,
        "flags": flags,
        "actual_xbi_20d": round(actual_xbi_20d, 3),
        "actual_spy_20d": round(actual_spy_20d, 3),
        "actual_spread_20d": round(actual_spread, 3),
    }


# ---------------------------------------------------------------------------
# Core: today's entry + settlement
# ---------------------------------------------------------------------------


def run(as_of: str) -> None:
    REGIME_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load today's regime_inputs cache ────────────────────────────────
    cache_path = RH_CACHE_ROOT / as_of / f"{as_of}_rh_regime_inputs.json"
    cache = _load_json(cache_path)
    if not cache:
        print(f"ERROR: No regime_inputs cache for {as_of}.")
        print(f"  Expected: {cache_path}")
        print("  Run write_regime_inputs() first (see biotech-regime-monitor skill).")
        sys.exit(1)

    regime = cache.get("regime") or "UNKNOWN"
    confidence = cache.get("confidence")
    vix = cache.get("vix")
    xbi_10d = cache.get("xbi_10d_momentum")
    spy_10d = cache.get("spy_10d_momentum")
    xbi_30d = cache.get("xbi_30d_return")
    spy_30d = cache.get("spy_30d_return")
    xbi_vs_spy_30d = cache.get("xbi_vs_spy_30d")
    xbi_close = cache.get("xbi_close")
    spy_close = cache.get("spy_close")
    signal_adj = cache.get("signal_adjustments", {})

    # ── Check ledger for previous regime (transition detection) ─────────
    ledger = _read_ledger()
    prev_regime = ledger[-1].get("regime") if ledger else None
    transition = prev_regime is not None and prev_regime != regime

    # ── Build today's entry ──────────────────────────────────────────────
    entry: Dict[str, Any] = {
        "as_of_date": as_of,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "regime": regime,
        "confidence": confidence,
        "vix": vix,
        "xbi_close": xbi_close,
        "spy_close": spy_close,
        "xbi_10d_momentum": xbi_10d,
        "spy_10d_momentum": spy_10d,
        "xbi_30d_return": xbi_30d,
        "spy_30d_return": spy_30d,
        "xbi_vs_spy_30d": xbi_vs_spy_30d,
        "signal_adjustments": signal_adj,
        "transition_from": prev_regime if transition else None,
        "fwd_20d_settled": False,
        "fwd_20d_xbi": None,
        "fwd_20d_spy": None,
        "fwd_20d_spread": None,
        "calibration_correct": None,
        "calibration_flags": [],
    }

    # Avoid duplicate entries for same date
    existing_dates = {r["as_of_date"] for r in ledger}
    if as_of not in existing_dates:
        _append_ledger(entry)
        ledger.append(entry)
        print(f"Ledger: appended entry for {as_of} (total rows: {len(ledger)})")
    else:
        print(f"Ledger: {as_of} already present — skipping append")

    # ── Settle 20d-old windows ───────────────────────────────────────────

    if len(ledger) >= 20:
        # The entry from ~20 trading days ago
        target_idx = len(ledger) - 21  # 0-indexed; today is last appended
        if target_idx >= 0:
            target_row = ledger[target_idx]
            if not target_row.get("fwd_20d_settled"):
                target_date = target_row["as_of_date"]
                target_xbi = target_row.get("xbi_close")
                # Today's XBI close is the settlement value
                today_xbi = xbi_close
                today_spy = spy_close
                old_spy = target_row.get("spy_close")

                if target_xbi and today_xbi and old_spy and today_spy:
                    fwd_xbi_20d = (today_xbi - target_xbi) / target_xbi * 100
                    fwd_spy_20d = (today_spy - old_spy) / old_spy * 100
                    cal = _check_prediction(target_row["regime"], fwd_xbi_20d, fwd_spy_20d)

                    # Rewrite that row (append updated version; script de-dupes on load)
                    target_row["fwd_20d_settled"] = True
                    target_row["fwd_20d_xbi"] = round(fwd_xbi_20d, 3)
                    target_row["fwd_20d_spy"] = round(fwd_spy_20d, 3)
                    target_row["fwd_20d_spread"] = round(fwd_xbi_20d - fwd_spy_20d, 3)
                    target_row["calibration_correct"] = cal["correct"]
                    target_row["calibration_flags"] = cal["flags"]

                    # Re-write ledger with updated row
                    _rewrite_ledger(ledger)
                    print(
                        f"Settled: {target_date} → XBI 20d={fwd_xbi_20d:.1f}%  SPY 20d={fwd_spy_20d:.1f}%  correct={cal['correct']}"
                    )

    # ── Calibration summary ──────────────────────────────────────────────
    settled = [r for r in ledger if r.get("fwd_20d_settled")]
    n_correct = sum(1 for r in settled if r.get("calibration_correct") is True)
    n_settled = len(settled)
    all_flags = [f for r in settled for f in r.get("calibration_flags", [])]

    # ── Generate daily card ──────────────────────────────────────────────
    _write_card(
        as_of=as_of,
        regime=regime,
        confidence=confidence,
        vix=vix,
        xbi_10d=xbi_10d,
        spy_10d=spy_10d,
        xbi_vs_spy_30d=xbi_vs_spy_30d,
        signal_adj=signal_adj,
        transition=transition,
        prev_regime=prev_regime,
        ledger=ledger,
        n_settled=n_settled,
        n_correct=n_correct,
        all_flags=all_flags,
        cache=cache,
    )


def _rewrite_ledger(rows: List[Dict]) -> None:
    """Rewrite the full ledger file (used after settling a window)."""
    with open(LEDGER_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _write_card(
    as_of: str,
    regime: str,
    confidence: Optional[float],
    vix: Optional[float],
    xbi_10d: Optional[float],
    spy_10d: Optional[float],
    xbi_vs_spy_30d: Optional[float],
    signal_adj: Dict,
    transition: bool,
    prev_regime: Optional[str],
    ledger: List[Dict],
    n_settled: int,
    n_correct: int,
    all_flags: List[str],
    cache: Dict,
) -> None:
    conf_str = f"  confidence={confidence:.2f}" if confidence is not None else ""
    lines = [
        f"# Regime Monitor — {as_of}",
        "",
        f"**Regime**: **{regime}**{conf_str}  ",
        f"**Description**: {cache.get('regime_description', '')}",
        "",
        "## Market Inputs",
        "",
        "| Input | Value |",
        "|-------|-------|",
        f"| VIX | {vix:.2f} |" if vix is not None else "| VIX | n/a |",
        f"| XBI 10d momentum | {xbi_10d:+.2f}% |" if xbi_10d is not None else "| XBI 10d | n/a |",
        f"| SPY 10d momentum | {spy_10d:+.2f}% |" if spy_10d is not None else "| SPY 10d | n/a |",
        f"| XBI vs SPY 30d | {xbi_vs_spy_30d:+.2f}pp |" if xbi_vs_spy_30d is not None else "| XBI vs SPY 30d | n/a |",
        f"| XBI close | ${cache.get('xbi_close', 'n/a')} |",
        f"| SPY close | ${cache.get('spy_close', 'n/a')} |",
        f"| SPX | {cache.get('spx', 'n/a')} |",
        "",
    ]

    # Transition
    if transition:
        lines += [
            "## ⚠ Regime Transition",
            "",
            f"**{prev_regime} → {regime}**",
            "",
            "*Review signal adjustment changes before next rebalance.*",
            "",
        ]
    else:
        # Count days in current streak
        streak = 1
        for row in reversed(ledger[:-1]):
            if row.get("regime") == regime:
                streak += 1
            else:
                break
        lines += [
            "## Regime Continuity",
            "",
            f"No transition — {streak} consecutive day(s) in **{regime}**.",
            "",
        ]

    # Signal adjustments
    if signal_adj:
        lines += ["## Signal Adjustments (Informational)", ""]
        lines += ["| Signal | Multiplier | Direction |", "|--------|------------|-----------|"]
        for k, v in sorted(signal_adj.items()):
            direction = "↑ boost" if float(v) > 1.0 else ("↓ damp" if float(v) < 1.0 else "→ neutral")
            lines.append(f"| {k} | ×{float(v):.2f} | {direction} |")
        lines += [
            "",
            "*These are the regime engine's recommended weight multipliers.*",
            "*They are informational only — the ranker is frozen.*",
            "",
        ]

    # Self-calibration
    lines += [
        "## Self-Calibration",
        "",
        f"Settled 20d windows: **{n_settled}**  ",
        (
            f"Correct direction calls: **{n_correct}/{n_settled}**  "
            if n_settled > 0
            else "Correct direction calls: **—** (no settled windows yet)  "
        ),
    ]
    if n_settled > 0:
        acc = n_correct / n_settled * 100
        lines.append(f"Accuracy rate: **{acc:.1f}%**  ")

        # Mean prediction error (spread)
        settled = [r for r in ledger if r.get("fwd_20d_settled")]
        spreads = [r["fwd_20d_spread"] for r in settled if r.get("fwd_20d_spread") is not None]
        if spreads:
            mean_spread = sum(spreads) / len(spreads)
            lines.append(f"Mean realized XBI-SPY 20d spread: **{mean_spread:+.1f}pp**  ")

    if all_flags:
        lines += ["", "**Miscalibration flags (advisory):**", ""]
        for flag in all_flags[-5:]:  # show last 5
            lines.append(f"- {flag}")
        lines += ["", "*Review flagged windows before any regime threshold adjustment.*"]
    else:
        lines += ["", "*No miscalibration flags.*"]

    lines += [
        "",
        "---",
        "",
        "## Regime History (last 10)",
        "",
        "| Date | Regime | VIX | XBI 10d | SPY 10d | XBI vs SPY 30d | 20d settled |",
        "|------|--------|-----|---------|---------|----------------|-------------|",
    ]
    for row in ledger[-10:]:
        settled_str = "✓" if row.get("fwd_20d_settled") else "—"
        xbi10 = f"{row['xbi_10d_momentum']:+.1f}%" if row.get("xbi_10d_momentum") is not None else "—"
        spy10 = f"{row['spy_10d_momentum']:+.1f}%" if row.get("spy_10d_momentum") is not None else "—"
        xvs = f"{row['xbi_vs_spy_30d']:+.1f}pp" if row.get("xbi_vs_spy_30d") is not None else "—"
        vix_str = f"{row['vix']:.1f}" if row.get("vix") is not None else "—"
        lines.append(
            f"| {row['as_of_date']} | {row.get('regime', '—')} | {vix_str} "
            f"| {xbi10} | {spy10} | {xvs} | {settled_str} |"
        )

    lines += [
        "",
        "---",
        "",
        "*Classification: PILOT_ACTION_INFRASTRUCTURE / NO_MODEL_CHANGE / NO_AUTONOMOUS_TRADING*  ",
        "*Schema: regime_monitor.v1*",
    ]

    card_path = REGIME_DIR / f"REGIME_CARD_{as_of}.md"
    card_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Card: {card_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily regime monitor + self-calibration")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    args = parser.parse_args()
    run(args.as_of_date)


if __name__ == "__main__":
    main()
